"""
Audio-to-guitar-tab helpers and ComfyUI nodes.

This module adds an accuracy-first transcription pipeline:
    AUDIO -> NOTE_EVENTS -> TAB_EVENTS -> TAB_TEXT
"""

from __future__ import annotations

import itertools
import math
import os
import tempfile
import wave
from datetime import datetime
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from scipy.signal import butter, sosfilt, sosfiltfilt

try:
    import librosa
except ModuleNotFoundError as exc:
    librosa = None
    _LIBROSA_IMPORT_ERROR = exc
else:
    _LIBROSA_IMPORT_ERROR = None

try:
    from basic_pitch.inference import predict as basic_pitch_predict
except ModuleNotFoundError as exc:
    basic_pitch_predict = None
    _BASIC_PITCH_IMPORT_ERROR = exc
else:
    _BASIC_PITCH_IMPORT_ERROR = None

try:
    import torchcrepe
except ModuleNotFoundError as exc:
    torchcrepe = None
    _TORCHCREPE_IMPORT_ERROR = exc
else:
    _TORCHCREPE_IMPORT_ERROR = None


TUNINGS: Dict[str, Dict[str, List[int] | List[str]]] = {
    "Standard E": {
        "strings": ["E", "A", "D", "G", "B", "e"],
        "open_midi": [40, 45, 50, 55, 59, 64],
    },
    "Drop D": {
        "strings": ["D", "A", "D", "G", "B", "e"],
        "open_midi": [38, 45, 50, 55, 59, 64],
    },
    "DADGAD": {
        "strings": ["D", "A", "D", "G", "A", "D"],
        "open_midi": [38, 45, 50, 55, 57, 62],
    },
}

DEFAULT_TRANSCRIPTION_HPF_HZ = 40.0
GROUP_ONSET_TOLERANCE_SEC = 0.05
MAX_COMBINATION_SEARCH = 256
TAB_GROUPS_PER_SECTION = 24
TORCHCREPE_SAMPLE_RATE = 16000
TORCHCREPE_HOP_LENGTH = 160
TORCHCREPE_GPU_BATCH_SIZE = 512
TORCHCREPE_CPU_BATCH_SIZE = 128


def _require_transcription_dependencies() -> None:
    install_hint = (
        "Install this node pack's extra transcription dependencies into the same Python "
        "environment that runs ComfyUI. Example:\n"
        "python -m pip install -r ComfyUI/custom_nodes/SongSeperatorNode/requirements.txt"
    )
    if librosa is None:
        raise RuntimeError(
            f"Missing dependency 'librosa'. {install_hint}"
        ) from _LIBROSA_IMPORT_ERROR
    if basic_pitch_predict is None and torchcrepe is None:
        raise RuntimeError(
            "Missing transcription backend. Install either 'basic-pitch' or 'torchcrepe'. "
            + install_hint
        ) from (_BASIC_PITCH_IMPORT_ERROR or _TORCHCREPE_IMPORT_ERROR)


def _extract_audio_input(audio: Dict) -> Tuple[torch.Tensor, int]:
    if not isinstance(audio, dict):
        raise TypeError("Expected 'audio' to be a ComfyUI AUDIO dict.")

    waveform = audio.get("waveform")
    sample_rate = audio.get("sample_rate")
    if waveform is None or sample_rate is None:
        raise ValueError("Input AUDIO must include 'waveform' and 'sample_rate'.")

    if not isinstance(waveform, torch.Tensor):
        waveform = torch.as_tensor(waveform)

    if waveform.ndim == 3:
        if waveform.shape[0] < 1:
            raise ValueError("Input AUDIO waveform batch is empty.")
        waveform = waveform[0]
    elif waveform.ndim != 2:
        raise ValueError(f"Expected waveform with 2 or 3 dims, got {waveform.ndim}.")

    if waveform.dtype != torch.float32:
        waveform = waveform.to(torch.float32)

    return waveform.contiguous(), int(sample_rate)


def _to_mono_float32(waveform: torch.Tensor) -> np.ndarray:
    if waveform.ndim != 2:
        raise ValueError(f"Expected (channels, samples) waveform, got {tuple(waveform.shape)}.")
    mono = waveform.mean(dim=0)
    return np.ascontiguousarray(mono.detach().cpu().numpy(), dtype=np.float32)


def _peak_normalize(audio_np: np.ndarray) -> np.ndarray:
    audio_np = np.asarray(audio_np, dtype=np.float32)
    peak = float(np.max(np.abs(audio_np))) if audio_np.size else 0.0
    if peak <= 1e-9:
        return audio_np
    return (audio_np / peak).astype(np.float32, copy=False)


def _apply_highpass_mono(
    audio_np: np.ndarray,
    sample_rate: int,
    cutoff_hz: float = DEFAULT_TRANSCRIPTION_HPF_HZ,
) -> np.ndarray:
    audio_np = np.asarray(audio_np, dtype=np.float32)
    if audio_np.size == 0 or sample_rate <= 0:
        return audio_np
    nyquist = sample_rate / 2.0
    if cutoff_hz <= 0 or cutoff_hz >= nyquist:
        return audio_np
    sos = butter(4, cutoff_hz / nyquist, btype="highpass", output="sos")
    try:
        filtered = sosfiltfilt(sos, audio_np)
    except ValueError:
        filtered = sosfilt(sos, audio_np)
    return filtered.astype(np.float32, copy=False)


def _apply_silence_gate(audio_np: np.ndarray, threshold: float) -> np.ndarray:
    if threshold <= 0:
        return audio_np.astype(np.float32, copy=False)
    gated = audio_np.astype(np.float32, copy=True)
    gated[np.abs(gated) < float(threshold)] = 0.0
    return gated


def _write_temp_wav(audio_np: np.ndarray, sample_rate: int) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        path = handle.name

    pcm = np.clip(audio_np, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16, copy=False)

    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())

    return path


def _filter_and_normalize_note_events(
    raw_events: Iterable,
    fmin_hz: float,
    fmax_hz: float,
) -> List[Dict]:
    min_midi = float(librosa.hz_to_midi(fmin_hz))
    max_midi = float(librosa.hz_to_midi(fmax_hz))
    events: List[Dict] = []

    for raw_event in raw_events:
        if len(raw_event) < 3:
            continue
        start_sec = float(raw_event[0])
        end_sec = float(raw_event[1])
        midi_note = int(round(float(raw_event[2])))
        amplitude = float(raw_event[3]) if len(raw_event) >= 4 else 0.0
        pitch_bends = raw_event[4] if len(raw_event) >= 5 else []

        if end_sec <= start_sec:
            continue
        if midi_note < min_midi or midi_note > max_midi:
            continue

        events.append(
            {
                "start_sec": start_sec,
                "duration_sec": end_sec - start_sec,
                "midi": midi_note,
                "amplitude": amplitude,
                "pitch_bends": list(pitch_bends) if pitch_bends else [],
            }
        )

    events.sort(key=lambda item: (item["start_sec"], item["midi"], -item["duration_sec"]))
    return events


def _snap_events_to_grid(events: List[Dict], bpm: float) -> List[Dict]:
    if bpm <= 0:
        return events

    grid_step = 60.0 / float(bpm) / 4.0
    snapped: List[Dict] = []
    for event in events:
        start = round(float(event["start_sec"]) / grid_step) * grid_step
        end = round((float(event["start_sec"]) + float(event["duration_sec"])) / grid_step) * grid_step
        if end <= start:
            end = start + grid_step
        snapped.append(
            {
                **event,
                "start_sec": round(start, 4),
                "duration_sec": round(end - start, 4),
            }
        )
    return snapped


def _serialize_note_events(note_events: Dict) -> str:
    events = note_events.get("events", [])
    metadata = note_events.get("metadata", {})
    lines = [
        f"# backend={metadata.get('backend', 'unknown')}",
        f"# tuning={metadata.get('tuning', 'Standard E')}",
        f"# capo={metadata.get('capo', 0)}",
        f"# max_fret={metadata.get('max_fret', 24)}",
        "start_sec,duration_sec,midi,amplitude,pitch_bends",
    ]
    for event in events:
        bend_count = len(event.get("pitch_bends", []))
        lines.append(
            f"{float(event['start_sec']):.4f},"
            f"{float(event['duration_sec']):.4f},"
            f"{int(event['midi'])},"
            f"{float(event.get('amplitude', 0.0)):.4f},"
            f"{bend_count}"
        )
    return "\n".join(lines)


def _format_memory_report(audio_np: np.ndarray, note_events: Dict, sample_rate: int) -> str:
    event_count = len(note_events.get("events", []))
    duration = len(audio_np) / sample_rate if sample_rate > 0 else 0.0
    return "\n".join(
        [
            "[Transcription]",
            f"duration_sec: {duration:.2f}",
            f"audio_mb: {audio_np.nbytes / (1024 * 1024):.2f}",
            f"note_events: {event_count}",
            f"backend: {note_events.get('metadata', {}).get('backend', 'unknown')}",
        ]
    )


def _numpy_to_comfy_audio(waveform: np.ndarray, sample_rate: int) -> Dict:
    """Convert a mono waveform into the standard ComfyUI AUDIO dict."""
    audio = np.asarray(waveform, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError(f"Expected mono 1D waveform, got shape {audio.shape}.")
    tensor = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0)  # (1, 1, T)
    return {"waveform": tensor, "sample_rate": int(sample_rate)}


def _midi_note_to_frequency(midi_note: int) -> float:
    return 440.0 * (2.0 ** ((int(midi_note) - 69) / 12.0))


def _synthesize_note_events_audio(
    note_events: Dict,
    sample_rate: int | None = None,
    amplitude: float = 0.18,
    fade_ms: float = 8.0,
) -> Dict:
    """Render note events to a simple playable reference waveform."""
    events = list(note_events.get("events", []))
    metadata = dict(note_events.get("metadata", {}))
    render_sr = int(sample_rate or metadata.get("sample_rate", 22050) or 22050)
    render_sr = max(8000, render_sr)

    if not events:
        return _numpy_to_comfy_audio(np.zeros(1, dtype=np.float32), render_sr)

    end_time = max(
        float(event["start_sec"]) + float(event["duration_sec"])
        for event in events
        if float(event.get("duration_sec", 0.0)) > 0.0
    )
    total_samples = max(1, int(math.ceil(end_time * render_sr)) + 1)
    output = np.zeros(total_samples, dtype=np.float32)
    fade_samples = max(1, int(render_sr * float(fade_ms) / 1000.0))

    for event in events:
        midi_note = int(event.get("midi", 0))
        start_sec = float(event.get("start_sec", 0.0))
        duration_sec = float(event.get("duration_sec", 0.0))
        if midi_note <= 0 or duration_sec <= 0.0:
            continue

        start_index = max(0, int(round(start_sec * render_sr)))
        duration_samples = max(1, int(round(duration_sec * render_sr)))
        end_index = min(total_samples, start_index + duration_samples)
        actual_length = end_index - start_index
        if actual_length <= 0:
            continue

        time_axis = np.arange(actual_length, dtype=np.float32) / render_sr
        frequency = _midi_note_to_frequency(midi_note)
        wave = np.sin(2.0 * math.pi * frequency * time_axis).astype(np.float32, copy=False)

        fade_length = min(fade_samples, max(1, actual_length // 2))
        if fade_length > 0:
            fade_in = np.linspace(0.0, 1.0, fade_length, dtype=np.float32)
            fade_out = np.linspace(1.0, 0.0, fade_length, dtype=np.float32)
            wave[:fade_length] *= fade_in
            wave[-fade_length:] *= fade_out

        note_amp = float(event.get("amplitude", amplitude))
        note_amp = min(max(note_amp, 0.05), 1.0)
        output[start_index:end_index] += wave * (float(amplitude) * note_amp)

    peak = float(np.max(np.abs(output))) if output.size else 0.0
    if peak > 1.0:
        output /= peak
    return _numpy_to_comfy_audio(output, render_sr)


def _run_basic_pitch(
    audio_np: np.ndarray,
    sample_rate: int,
    onset_threshold: float,
    frame_threshold: float,
    minimum_note_length_ms: float,
    minimum_frequency_hz: float,
    maximum_frequency_hz: float,
    melodia_trick: bool,
) -> List[Dict]:
    wav_path = _write_temp_wav(audio_np, sample_rate)
    try:
        _, _, raw_events = basic_pitch_predict(
            wav_path,
            onset_threshold=float(onset_threshold),
            frame_threshold=float(frame_threshold),
            minimum_note_length=float(minimum_note_length_ms),
            minimum_frequency=float(minimum_frequency_hz),
            maximum_frequency=float(maximum_frequency_hz),
            melodia_trick=bool(melodia_trick),
        )
    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass

    return _filter_and_normalize_note_events(
        raw_events=raw_events,
        fmin_hz=minimum_frequency_hz,
        fmax_hz=maximum_frequency_hz,
    )


def _run_torchcrepe(
    audio_np: np.ndarray,
    sample_rate: int,
    confidence_threshold: float,
    minimum_note_length_ms: float,
    minimum_frequency_hz: float,
    maximum_frequency_hz: float,
    model_size: str,
) -> List[Dict]:
    if torchcrepe is None:
        raise RuntimeError("torchcrepe is not installed.")
    if audio_np.size == 0:
        return []

    if sample_rate != TORCHCREPE_SAMPLE_RATE:
        resampled = librosa.resample(
            audio_np.astype(np.float32, copy=False),
            orig_sr=sample_rate,
            target_sr=TORCHCREPE_SAMPLE_RATE,
        )
    else:
        resampled = audio_np.astype(np.float32, copy=False)

    if resampled.size < TORCHCREPE_HOP_LENGTH:
        return []

    audio_tensor = torch.from_numpy(np.ascontiguousarray(resampled)).unsqueeze(0)
    cuda_available = torch.cuda.is_available()

    def _predict(device: str, batch_size: int):
        return torchcrepe.predict(
            audio_tensor,
            TORCHCREPE_SAMPLE_RATE,
            TORCHCREPE_HOP_LENGTH,
            float(minimum_frequency_hz),
            float(maximum_frequency_hz),
            model=model_size,
            batch_size=batch_size,
            device=device,
            return_periodicity=True,
            decoder=torchcrepe.decode.viterbi,
        )

    if cuda_available:
        try:
            pitch_hz, periodicity = _predict("cuda", TORCHCREPE_GPU_BATCH_SIZE)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            pitch_hz, periodicity = _predict("cpu", TORCHCREPE_CPU_BATCH_SIZE)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            torch.cuda.empty_cache()
            pitch_hz, periodicity = _predict("cpu", TORCHCREPE_CPU_BATCH_SIZE)
    else:
        pitch_hz, periodicity = _predict("cpu", TORCHCREPE_CPU_BATCH_SIZE)

    pitch_hz = torchcrepe.filter.median(pitch_hz, 3)
    periodicity = torchcrepe.filter.mean(periodicity, 3)
    pitch_hz = pitch_hz.squeeze(0).detach().cpu().numpy()
    periodicity = periodicity.squeeze(0).detach().cpu().numpy()

    voiced = (periodicity >= float(confidence_threshold)) & (pitch_hz > 0.0)
    midi_frames = np.zeros(len(pitch_hz), dtype=np.int32)
    if voiced.any():
        midi_frames[voiced] = np.round(
            12.0 * np.log2(pitch_hz[voiced] / 440.0) + 69.0
        ).astype(np.int32)

    hop_seconds = TORCHCREPE_HOP_LENGTH / float(TORCHCREPE_SAMPLE_RATE)
    min_frames = max(1, int(round(float(minimum_note_length_ms) / 1000.0 / hop_seconds)))

    events: List[Dict] = []
    index = 0
    while index < len(midi_frames):
        current = int(midi_frames[index])
        end = index
        while end < len(midi_frames) and int(midi_frames[end]) == current:
            end += 1
        if current > 0 and (end - index) >= min_frames:
            start_sec = index * hop_seconds
            duration_sec = (end - index) * hop_seconds
            mean_periodicity = float(np.mean(periodicity[index:end])) if end > index else 0.0
            events.append(
                {
                    "start_sec": start_sec,
                    "duration_sec": duration_sec,
                    "midi": current,
                    "amplitude": mean_periodicity,
                    "pitch_bends": [],
                }
            )
        index = end

    return events


def _candidate_positions(midi_note: int, tuning_name: str, capo: int, max_fret: int) -> List[Dict]:
    open_midis = TUNINGS[tuning_name]["open_midi"]
    candidates = []
    for string_index, open_midi in enumerate(open_midis):
        fret = midi_note - (int(open_midi) + int(capo))
        if 0 <= fret <= max_fret:
            candidates.append(
                {
                    "string": string_index,
                    "fret": int(fret),
                    "midi": int(midi_note),
                }
            )
    return candidates


def _group_note_events(note_events: List[Dict], tolerance_sec: float = GROUP_ONSET_TOLERANCE_SEC) -> List[List[Dict]]:
    if not note_events:
        return []

    ordered = sorted(note_events, key=lambda item: (item["start_sec"], item["midi"]))
    groups: List[List[Dict]] = [[ordered[0]]]
    for event in ordered[1:]:
        last_group = groups[-1]
        reference = min(item["start_sec"] for item in last_group)
        if abs(float(event["start_sec"]) - float(reference)) <= tolerance_sec:
            last_group.append(event)
        else:
            groups.append([event])
    return groups


def _events_total_duration(note_events: List[Dict], fallback_duration_sec: float = 0.0) -> float:
    event_end = max(
        (float(event["start_sec"]) + float(event["duration_sec"]) for event in note_events),
        default=0.0,
    )
    return max(float(fallback_duration_sec), event_end)


def _build_timeline_groups(
    note_groups: List[List[Dict]],
    total_duration_sec: float,
    tolerance_sec: float = GROUP_ONSET_TOLERANCE_SEC,
) -> List[Dict]:
    """Convert note groups into a full-length timeline with explicit rests."""
    timeline: List[Dict] = []
    cursor = 0.0
    safe_total = max(0.0, float(total_duration_sec))

    for group in note_groups:
        group_start = min(float(event["start_sec"]) for event in group)
        group_end = max(float(event["start_sec"]) + float(event["duration_sec"]) for event in group)

        if group_start - cursor > tolerance_sec:
            timeline.append(
                {
                    "start_sec": cursor,
                    "duration_sec": group_start - cursor,
                    "notes": [],
                    "is_rest": True,
                }
            )

        timeline.append(
            {
                "start_sec": group_start,
                "duration_sec": max(0.0, group_end - group_start),
                "notes": list(group),
                "is_rest": False,
            }
        )
        cursor = max(cursor, group_end)

    if safe_total - cursor > tolerance_sec or (not timeline and safe_total > 0):
        timeline.append(
            {
                "start_sec": cursor,
                "duration_sec": max(0.0, safe_total - cursor),
                "notes": [],
                "is_rest": True,
            }
        )

    return timeline


def _combination_cost(positions: List[Dict]) -> float:
    frets = [pos["fret"] for pos in positions]
    strings = [pos["string"] for pos in positions]
    return (
        (max(frets) - min(frets)) * 1.1
        + (sum(frets) / len(frets)) * 0.08
        + (max(strings) - min(strings)) * 0.35
    )


def _transition_cost(prev_positions: List[Dict] | None, curr_positions: List[Dict]) -> float:
    if not prev_positions:
        return _combination_cost(curr_positions)

    prev_avg_fret = sum(item["fret"] for item in prev_positions) / len(prev_positions)
    curr_avg_fret = sum(item["fret"] for item in curr_positions) / len(curr_positions)
    prev_center_string = sum(item["string"] for item in prev_positions) / len(prev_positions)
    curr_center_string = sum(item["string"] for item in curr_positions) / len(curr_positions)

    cost = _combination_cost(curr_positions)
    cost += abs(curr_avg_fret - prev_avg_fret) * 0.7
    cost += abs(curr_center_string - prev_center_string) * 0.6

    prev_strings = {item["string"] for item in prev_positions}
    curr_strings = {item["string"] for item in curr_positions}
    repeated_strings = len(prev_strings & curr_strings)
    cost -= repeated_strings * 0.1
    return cost


def _enumerate_group_assignments(group: List[Dict], tuning_name: str, capo: int, max_fret: int) -> List[List[Dict]]:
    candidate_lists = []
    for event in group:
        candidates = _candidate_positions(int(event["midi"]), tuning_name, capo, max_fret)
        if not candidates:
            candidates = [{"string": -1, "fret": -1, "midi": int(event["midi"])}]
        candidate_lists.append(candidates)

    total_combinations = 1
    for candidates in candidate_lists:
        total_combinations *= len(candidates)
        if total_combinations > MAX_COMBINATION_SEARCH:
            break

    if total_combinations > MAX_COMBINATION_SEARCH:
        combinations = []
        used_strings = set()
        for candidates in candidate_lists:
            selected = None
            for candidate in sorted(candidates, key=lambda item: (item["fret"], item["string"])):
                if candidate["string"] not in used_strings:
                    selected = candidate
                    break
            if selected is None:
                selected = min(candidates, key=lambda item: (item["fret"], item["string"]))
            if selected["string"] >= 0:
                used_strings.add(selected["string"])
            combinations.append(selected)
        return [combinations]

    valid_assignments: List[List[Dict]] = []
    for combo in itertools.product(*candidate_lists):
        strings = [item["string"] for item in combo if item["string"] >= 0]
        if len(strings) != len(set(strings)):
            continue
        valid_assignments.append([dict(item) for item in combo])

    return valid_assignments or [[min(candidates, key=lambda item: (item["fret"], item["string"])) for candidates in candidate_lists]]


def _map_groups_to_fretboard(
    note_events: List[Dict],
    tuning_name: str,
    capo: int,
    max_fret: int,
    total_duration_sec: float = 0.0,
) -> Dict:
    grouped_notes = _group_note_events(note_events)
    safe_total_duration = _events_total_duration(note_events, fallback_duration_sec=total_duration_sec)
    timeline_groups = _build_timeline_groups(grouped_notes, safe_total_duration)
    note_groups = [group["notes"] for group in timeline_groups if not group["is_rest"]]
    if not timeline_groups:
        return {
            "groups": [],
            "metadata": {
                "tuning": tuning_name,
                "capo": capo,
                "max_fret": max_fret,
                "duration_sec": safe_total_duration,
            },
        }

    assignment_options = [
        _enumerate_group_assignments(group, tuning_name, capo, max_fret)
        for group in note_groups
    ]

    dp: List[List[float]] = []
    backpointers: List[List[int | None]] = []

    for group_index, assignments in enumerate(assignment_options):
        group_costs = [float("inf")] * len(assignments)
        group_prev: List[int | None] = [None] * len(assignments)

        for assignment_index, assignment in enumerate(assignments):
            if group_index == 0:
                group_costs[assignment_index] = _transition_cost(None, assignment)
                continue

            prev_assignments = assignment_options[group_index - 1]
            for prev_index, prev_assignment in enumerate(prev_assignments):
                candidate_cost = dp[group_index - 1][prev_index] + _transition_cost(prev_assignment, assignment)
                if candidate_cost < group_costs[assignment_index]:
                    group_costs[assignment_index] = candidate_cost
                    group_prev[assignment_index] = prev_index

        dp.append(group_costs)
        backpointers.append(group_prev)

    last_index = min(range(len(dp[-1])), key=lambda idx: dp[-1][idx])
    chosen_assignments: List[List[Dict]] = []
    for group_index in range(len(note_groups) - 1, -1, -1):
        chosen_assignments.append(assignment_options[group_index][last_index])
        previous = backpointers[group_index][last_index]
        if previous is None:
            break
        last_index = previous
    chosen_assignments.reverse()

    rendered_note_groups = []
    for group, positions in zip(note_groups, chosen_assignments):
        rendered_notes = []
        for event, position in zip(group, positions):
            rendered_notes.append(
                {
                    "start_sec": float(event["start_sec"]),
                    "duration_sec": float(event["duration_sec"]),
                    "midi": int(event["midi"]),
                    "amplitude": float(event.get("amplitude", 0.0)),
                    "string": int(position["string"]),
                    "fret": int(position["fret"]),
                    "pitch_bends": list(event.get("pitch_bends", [])),
                }
            )
        rendered_note_groups.append(rendered_notes)

    rendered_groups = []
    note_group_index = 0
    for group in timeline_groups:
        if group["is_rest"]:
            rendered_groups.append(
                {
                    "start_sec": float(group["start_sec"]),
                    "duration_sec": float(group["duration_sec"]),
                    "notes": [],
                    "is_rest": True,
                }
            )
            continue

        rendered_groups.append(
            {
                "start_sec": float(group["start_sec"]),
                "duration_sec": float(group["duration_sec"]),
                "notes": rendered_note_groups[note_group_index],
                "is_rest": False,
            }
        )
        note_group_index += 1

    return {
        "groups": rendered_groups,
        "metadata": {
            "tuning": tuning_name,
            "capo": int(capo),
            "max_fret": int(max_fret),
            "duration_sec": float(safe_total_duration),
        },
    }


def _format_seconds(seconds: float) -> str:
    total = max(0.0, float(seconds))
    minutes = int(total // 60)
    remainder = total - minutes * 60
    return f"{minutes}:{remainder:05.2f}"


def _render_tab_groups(tab_events: Dict) -> str:
    metadata = tab_events.get("metadata", {})
    tuning_name = metadata.get("tuning", "Standard E")
    capo = int(metadata.get("capo", 0))
    groups = tab_events.get("groups", [])
    string_names = TUNINGS[tuning_name]["strings"]
    num_strings = len(string_names)

    if not groups:
        return "\n".join(f"{name}|---|" for name in reversed(string_names))

    sections: List[str] = []
    total_sections = max(1, math.ceil(len(groups) / TAB_GROUPS_PER_SECTION))

    for section_index, start in enumerate(range(0, len(groups), TAB_GROUPS_PER_SECTION), start=1):
        section_groups = groups[start:start + TAB_GROUPS_PER_SECTION]
        columns = []
        widths = []

        for group in section_groups:
            column = [None] * num_strings
            for note in group.get("notes", []):
                string_index = int(note["string"])
                fret = int(note["fret"])
                if 0 <= string_index < num_strings and fret >= 0:
                    column[string_index] = str(fret)
            columns.append(column)
            widths.append(max((len(cell) for cell in column if cell is not None), default=1))

        section_start = min(float(group["start_sec"]) for group in section_groups)
        section_end = max(float(group["start_sec"]) + float(group["duration_sec"]) for group in section_groups)

        lines = []
        if total_sections > 1:
            lines.append(f"[Section {section_index}/{total_sections}]")
        capo_part = f" | Capo: {capo}" if capo > 0 else ""
        lines.append(
            f"Tuning: {tuning_name}{capo_part} | Time: {_format_seconds(section_start)} - {_format_seconds(section_end)}"
        )

        for string_index in range(num_strings - 1, -1, -1):
            line = [f"{string_names[string_index]}|"]
            for column, width in zip(columns, widths):
                cell = column[string_index]
                if cell is None:
                    line.append("-" * (width + 1))
                else:
                    line.append(cell.ljust(width, "-") + "-")
            line.append("|")
            lines.append("".join(line))

        sections.append("\n".join(lines))

    return "\n\n".join(sections)


def _prepend_title(tab_text: str, title: str) -> str:
    rendered_title = str(title or "").strip() or "guitar_tab"
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"{rendered_title}\nCreated: {created_at}\n\n{tab_text}"


class GuitarAudioToNoteEvents:
    """Transcribe isolated guitar audio into structured note events."""

    CATEGORY = "Audio/Guitar"
    RETURN_TYPES = ("NOTE_EVENTS", "STRING", "STRING")
    RETURN_NAMES = ("NOTE_EVENTS", "EVENT_CSV", "MEMORY_USAGE")
    FUNCTION = "transcribe"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "tuning": (list(TUNINGS.keys()),),
                "capo": ("INT", {"default": 0, "min": 0, "max": 11, "step": 1}),
                "max_fret": ("INT", {"default": 24, "min": 1, "max": 30, "step": 1}),
                "silence_threshold": ("FLOAT", {
                    "default": 0.01,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.001,
                    "display": "number",
                }),
                "onset_threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                }),
                "frame_threshold": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                }),
                "minimum_note_length_ms": ("FLOAT", {
                    "default": 127.0,
                    "min": 10.0,
                    "max": 1000.0,
                    "step": 1.0,
                    "display": "number",
                }),
                "melodia_trick": ("BOOLEAN", {"default": True}),
                "backend": (["auto", "basic-pitch", "torchcrepe"], {"default": "auto"}),
                "torchcrepe_model": (["full", "tiny"], {"default": "full"}),
                "bpm": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 300.0,
                    "step": 1.0,
                    "display": "number",
                }),
            },
        }

    def transcribe(
        self,
        audio: Dict,
        tuning: str,
        capo: int,
        max_fret: int,
        silence_threshold: float,
        onset_threshold: float,
        frame_threshold: float,
        minimum_note_length_ms: float,
        melodia_trick: bool,
        backend: str,
        torchcrepe_model: str,
        bpm: float,
    ):
        _require_transcription_dependencies()

        waveform, sample_rate = _extract_audio_input(audio)
        mono_audio = _to_mono_float32(waveform)
        mono_audio = _peak_normalize(mono_audio)
        mono_audio = _apply_highpass_mono(mono_audio, sample_rate)
        mono_audio = _apply_silence_gate(mono_audio, silence_threshold)

        open_midis = TUNINGS[tuning]["open_midi"]
        fmin_hz = float(librosa.midi_to_hz(min(open_midis) + capo))
        fmax_hz = float(librosa.midi_to_hz(max(open_midis) + capo + max_fret))

        resolved_backend = str(backend)
        if resolved_backend == "auto":
            resolved_backend = "basic-pitch" if basic_pitch_predict is not None else "torchcrepe"

        if resolved_backend == "basic-pitch":
            if basic_pitch_predict is None:
                raise RuntimeError(
                    "The 'basic-pitch' backend was selected, but basic-pitch is not installed "
                    "or is unavailable in this Python environment."
                )
            events = _run_basic_pitch(
                audio_np=mono_audio,
                sample_rate=sample_rate,
                onset_threshold=onset_threshold,
                frame_threshold=frame_threshold,
                minimum_note_length_ms=minimum_note_length_ms,
                minimum_frequency_hz=fmin_hz,
                maximum_frequency_hz=fmax_hz,
                melodia_trick=melodia_trick,
            )
        elif resolved_backend == "torchcrepe":
            events = _run_torchcrepe(
                audio_np=mono_audio,
                sample_rate=sample_rate,
                confidence_threshold=frame_threshold,
                minimum_note_length_ms=minimum_note_length_ms,
                minimum_frequency_hz=fmin_hz,
                maximum_frequency_hz=fmax_hz,
                model_size=str(torchcrepe_model),
            )
        else:
            raise ValueError(f"Unsupported transcription backend: {backend}")

        events = _snap_events_to_grid(events, bpm)

        note_events = {
            "events": events,
            "metadata": {
                "backend": resolved_backend,
                "tuning": tuning,
                "capo": int(capo),
                "max_fret": int(max_fret),
                "sample_rate": int(sample_rate),
                "duration_sec": len(mono_audio) / float(sample_rate) if sample_rate > 0 else 0.0,
                "minimum_frequency_hz": fmin_hz,
                "maximum_frequency_hz": fmax_hz,
                "bpm": float(bpm),
            },
        }
        return (
            note_events,
            _serialize_note_events(note_events),
            _format_memory_report(mono_audio, note_events, sample_rate),
        )


class GuitarNoteEventsToTabEvents:
    """Map transcribed note events onto playable guitar string/fret positions."""

    CATEGORY = "Audio/Guitar"
    RETURN_TYPES = ("TAB_EVENTS", "AUDIO")
    RETURN_NAMES = ("TAB_EVENTS", "REFERENCE_AUDIO")
    FUNCTION = "map_to_fretboard"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "note_events": ("NOTE_EVENTS", {"forceInput": True}),
                "tuning": (list(TUNINGS.keys()),),
                "capo": ("INT", {"default": 0, "min": 0, "max": 11, "step": 1}),
                "max_fret": ("INT", {"default": 24, "min": 1, "max": 30, "step": 1}),
            },
        }

    def map_to_fretboard(
        self,
        note_events: Dict,
        tuning: str,
        capo: int,
        max_fret: int,
    ):
        events = list(note_events.get("events", []))
        source_metadata = dict(note_events.get("metadata", {}))
        mapped = _map_groups_to_fretboard(
            events,
            tuning,
            capo,
            max_fret,
            total_duration_sec=float(source_metadata.get("duration_sec", 0.0)),
        )
        mapped["metadata"].update({"source": source_metadata})
        reference_audio = _synthesize_note_events_audio(note_events)
        return (mapped, reference_audio)


class GuitarAudioToTabEvents:
    """Transcribe guitar audio and map it directly to full-length tab events."""

    CATEGORY = "Audio/Guitar"
    RETURN_TYPES = ("TAB_EVENTS", "NOTE_EVENTS", "AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("TAB_EVENTS", "NOTE_EVENTS", "REFERENCE_AUDIO", "EVENT_CSV", "MEMORY_USAGE")
    FUNCTION = "process"

    @classmethod
    def INPUT_TYPES(cls):
        return GuitarAudioToNoteEvents.INPUT_TYPES()

    def process(
        self,
        audio: Dict,
        tuning: str,
        capo: int,
        max_fret: int,
        silence_threshold: float,
        onset_threshold: float,
        frame_threshold: float,
        minimum_note_length_ms: float,
        melodia_trick: bool,
        backend: str,
        torchcrepe_model: str,
        bpm: float,
    ):
        transcriber = GuitarAudioToNoteEvents()
        note_events, event_csv, memory_usage = transcriber.transcribe(
            audio=audio,
            tuning=tuning,
            capo=capo,
            max_fret=max_fret,
            silence_threshold=silence_threshold,
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
            minimum_note_length_ms=minimum_note_length_ms,
            melodia_trick=melodia_trick,
            backend=backend,
            torchcrepe_model=torchcrepe_model,
            bpm=bpm,
        )

        mapper = GuitarNoteEventsToTabEvents()
        tab_events, reference_audio = mapper.map_to_fretboard(
            note_events=note_events,
            tuning=tuning,
            capo=capo,
            max_fret=max_fret,
        )
        return (tab_events, note_events, reference_audio, event_csv, memory_usage)


class RenderGuitarTab:
    """Render mapped fretboard events as readable ASCII guitar tab."""

    CATEGORY = "Audio/Guitar"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("TAB_TEXT",)
    FUNCTION = "render"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tab_events": ("TAB_EVENTS", {"forceInput": True}),
                "title": ("STRING", {"default": "guitar_tab", "multiline": False}),
            },
        }

    def render(self, tab_events: Dict, title: str):
        tab_text = _render_tab_groups(tab_events)
        return (_prepend_title(tab_text, title),)


NODE_CLASS_MAPPINGS = {
    "GuitarAudioToNoteEvents": GuitarAudioToNoteEvents,
    "GuitarNoteEventsToTabEvents": GuitarNoteEventsToTabEvents,
    "GuitarAudioToTabEvents": GuitarAudioToTabEvents,
    "RenderGuitarTab": RenderGuitarTab,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GuitarAudioToNoteEvents": "Guitar Audio To Note Events",
    "GuitarNoteEventsToTabEvents": "Guitar Note Events To Tab Events",
    "GuitarAudioToTabEvents": "Guitar Audio To Tab Events",
    "RenderGuitarTab": "Render Guitar Tab",
}
