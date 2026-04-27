"""
SongSeparator – ComfyUI custom node
====================================
Separates an audio file into up to 6 stems (vocals, drums, bass, guitar,
piano, other) using the HTDemucs 6-stem model from the *demucs* library,
then applies a 100 Hz high-pass filter to the guitar stem to remove
sub-bass bleed that confuses MIDI / pitch-detection engines.

Input and output follow the standard ComfyUI audio convention:
    {"waveform": torch.Tensor[B, C, T], "sample_rate": int}
"""

import gc
import math
import os
import tempfile
import time
from typing import Dict, List, Tuple
from urllib.parse import quote

import numpy as np
import torch
from aiohttp import web
from scipy.signal import butter, sosfilt, sosfiltfilt
from server import PromptServer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_STEMS: List[str] = ["guitar", "vocals", "drums", "bass", "piano", "other"]
MODEL_NAME: str = "htdemucs_6s"
DEFAULT_GUITAR_HPF_CUTOFF_HZ: int = 100
BASS_SHELF_HZ: float = 120.0
TREBLE_SHELF_HZ: float = 4000.0
PREVIEW_DIRECTORY = os.path.join(tempfile.gettempdir(), "songseparator_previews")
PREVIEW_FILES: Dict[str, str] = {}

os.makedirs(PREVIEW_DIRECTORY, exist_ok=True)


@PromptServer.instance.routes.get("/songseparator/preview_audio")
async def get_preview_audio(request):
    """Serve the latest preview WAV generated for a node instance."""
    node_id = request.query.get("node_id", "")
    path = PREVIEW_FILES.get(node_id)
    if not path or not os.path.isfile(path):
        return web.Response(status=404, text="Preview audio not found.")

    response = web.FileResponse(path)
    response.headers["Cache-Control"] = "no-store, max-age=0"
    return response


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _butter_highpass_sos(cutoff: float, fs: float, order: int = 5):
    """Return second-order-sections for a Butterworth high-pass filter."""
    nyq = fs / 2.0
    normal_cutoff = cutoff / nyq
    sos = butter(order, normal_cutoff, btype="high", analog=False, output="sos")
    return sos


def _apply_highpass(waveform: np.ndarray, sample_rate: int,
                    cutoff_hz: float = DEFAULT_GUITAR_HPF_CUTOFF_HZ) -> np.ndarray:
    """Apply a zero-phase Butterworth high-pass filter channel-wise.

    Args:
        waveform:   NumPy array of shape (channels, samples).
        sample_rate: Sample rate in Hz.
        cutoff_hz:  High-pass cutoff frequency in Hz.

    Returns:
        Filtered waveform with the same shape as the input.
    """
    nyquist_hz = sample_rate / 2.0
    if cutoff_hz <= 0 or cutoff_hz >= nyquist_hz:
        # Invalid filter configuration: return original signal unchanged.
        return waveform.astype(np.float32, copy=False)

    sos = _butter_highpass_sos(cutoff_hz, sample_rate)

    # Prefer zero-phase filtering to avoid phase distortion (especially
    # important for downstream pitch detection). Very short clips can fail
    # with filtfilt due to pad length constraints, so we gracefully fall
    # back to one-pass filtering in that case.
    try:
        filtered = np.stack([sosfiltfilt(sos, ch) for ch in waveform], axis=0)
    except ValueError:
        filtered = np.stack([sosfilt(sos, ch) for ch in waveform], axis=0)
    return filtered.astype(np.float32)


def _sanitize_node_id(node_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in node_id)


def _biquad_shelf_sos(
    sample_rate: int,
    cutoff_hz: float,
    gain_db: float,
    shelf_type: str,
) -> np.ndarray | None:
    """Create a single-section low/high shelf filter in SOS form."""
    nyquist_hz = sample_rate / 2.0
    if abs(gain_db) < 1e-6 or cutoff_hz <= 0 or cutoff_hz >= nyquist_hz:
        return None

    a = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * cutoff_hz / sample_rate
    cos_w0 = math.cos(w0)
    sin_w0 = math.sin(w0)
    sqrt_a = math.sqrt(a)
    alpha = sin_w0 / 2.0 * math.sqrt((a + 1.0 / a) * (1.0 - 1.0) + 2.0)

    if shelf_type == "low":
        b0 = a * ((a + 1) - (a - 1) * cos_w0 + 2 * sqrt_a * alpha)
        b1 = 2 * a * ((a - 1) - (a + 1) * cos_w0)
        b2 = a * ((a + 1) - (a - 1) * cos_w0 - 2 * sqrt_a * alpha)
        a0 = (a + 1) + (a - 1) * cos_w0 + 2 * sqrt_a * alpha
        a1 = -2 * ((a - 1) + (a + 1) * cos_w0)
        a2 = (a + 1) + (a - 1) * cos_w0 - 2 * sqrt_a * alpha
    elif shelf_type == "high":
        b0 = a * ((a + 1) + (a - 1) * cos_w0 + 2 * sqrt_a * alpha)
        b1 = -2 * a * ((a - 1) + (a + 1) * cos_w0)
        b2 = a * ((a + 1) + (a - 1) * cos_w0 - 2 * sqrt_a * alpha)
        a0 = (a + 1) - (a - 1) * cos_w0 + 2 * sqrt_a * alpha
        a1 = 2 * ((a - 1) - (a + 1) * cos_w0)
        a2 = (a + 1) - (a - 1) * cos_w0 - 2 * sqrt_a * alpha
    else:
        raise ValueError(f"Unsupported shelf type: {shelf_type}")

    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]], dtype=np.float64)


def _apply_sos_filter(waveform: np.ndarray, sos: np.ndarray | None) -> np.ndarray:
    if sos is None:
        return waveform.astype(np.float32, copy=False)

    try:
        filtered = np.stack([sosfiltfilt(sos, ch) for ch in waveform], axis=0)
    except ValueError:
        filtered = np.stack([sosfilt(sos, ch) for ch in waveform], axis=0)
    return filtered.astype(np.float32)


def _apply_tone_controls(
    waveform: np.ndarray,
    sample_rate: int,
    bass_gain_db: float,
    treble_gain_db: float,
    volume_db: float,
) -> np.ndarray:
    """Apply musical tone controls and output gain to a waveform."""
    processed = waveform.astype(np.float32, copy=True)
    processed = _apply_sos_filter(
        processed,
        _biquad_shelf_sos(sample_rate, BASS_SHELF_HZ, bass_gain_db, "low"),
    )
    processed = _apply_sos_filter(
        processed,
        _biquad_shelf_sos(sample_rate, TREBLE_SHELF_HZ, treble_gain_db, "high"),
    )
    processed *= 10.0 ** (volume_db / 20.0)
    return np.clip(processed, -1.0, 1.0).astype(np.float32, copy=False)


def _build_waveform_peaks(waveform: np.ndarray, num_points: int = 384) -> List[List[float]]:
    """Return min/max envelope pairs for compact waveform drawing."""
    mono = waveform.mean(axis=0) if waveform.shape[0] > 1 else waveform[0]
    if mono.size == 0:
        return [[0.0, 0.0] for _ in range(num_points)]

    chunk_size = max(1, int(math.ceil(mono.size / num_points)))
    peaks: List[List[float]] = []
    for start in range(0, mono.size, chunk_size):
        chunk = mono[start:start + chunk_size]
        peaks.append([float(np.min(chunk)), float(np.max(chunk))])

    if len(peaks) < num_points:
        peaks.extend([[0.0, 0.0]] * (num_points - len(peaks)))
    return peaks[:num_points]


def _write_preview_wav(path: str, waveform: np.ndarray, sample_rate: int) -> None:
    """Write a float waveform in [-1, 1] to a 16-bit PCM WAV file."""
    import wave

    pcm = np.clip(waveform, -1.0, 1.0)
    pcm = (pcm.T * 32767.0).astype(np.int16, copy=False)

    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(int(waveform.shape[0]))
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def _store_preview_audio(node_id: str, waveform: np.ndarray, sample_rate: int) -> str:
    safe_node_id = _sanitize_node_id(str(node_id))
    preview_path = os.path.join(PREVIEW_DIRECTORY, f"{safe_node_id}.wav")
    _write_preview_wav(preview_path, waveform, sample_rate)
    PREVIEW_FILES[str(node_id)] = preview_path
    version = int(time.time() * 1000)
    return f"/songseparator/preview_audio?node_id={quote(str(node_id), safe='')}&v={version}"


def _extract_audio_input(audio: Dict) -> Tuple[torch.Tensor, int]:
    """Extract ``(channels, samples)`` waveform data from a ComfyUI AUDIO input."""
    if not isinstance(audio, dict):
        raise TypeError("SongSeparator: expected 'audio' to be a ComfyUI AUDIO dict.")

    waveform = audio.get("waveform")
    sample_rate = audio.get("sample_rate")

    if waveform is None or sample_rate is None:
        raise ValueError(
            "SongSeparator: input 'audio' must include 'waveform' and 'sample_rate'."
        )

    if not isinstance(waveform, torch.Tensor):
        waveform = torch.as_tensor(waveform)

    if waveform.ndim == 3:
        # ComfyUI audio typically arrives as (batch, channels, samples).
        if waveform.shape[0] < 1:
            raise ValueError("SongSeparator: input 'audio' waveform batch is empty.")
        waveform = waveform[0]
    elif waveform.ndim != 2:
        raise ValueError(
            f"SongSeparator: expected audio waveform with 2 or 3 dims, got {waveform.ndim}."
        )

    if waveform.dtype != torch.float32:
        waveform = waveform.to(torch.float32)

    return waveform.contiguous(), int(sample_rate)


def _numpy_to_comfy(array: np.ndarray, sample_rate: int) -> Dict:
    """Convert a (C, T) float32 NumPy array to the ComfyUI audio dict format.

    ComfyUI audio nodes expect:
        {"waveform": Tensor[1, C, T], "sample_rate": int}
    """
    tensor = torch.from_numpy(array).unsqueeze(0)  # (1, C, T)
    return {"waveform": tensor, "sample_rate": sample_rate}


class SongSeparator:
    """ComfyUI node that isolates audio stems using the HTDemucs 6-stem model.

    Designed to prep audio for MIDI transcription, with particular focus on
    clean guitar isolation and VRAM safety for mixed image/audio workflows.
    """

    CATEGORY = "Audio"
    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "AUDIO", "AUDIO", "AUDIO")
    RETURN_NAMES = (
        "GUITAR_AUDIO",
        "VOCALS_AUDIO",
        "DRUMS_AUDIO",
        "BASS_AUDIO",
        "PIANO_AUDIO",
        "OTHER_AUDIO",
    )
    FUNCTION = "separate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "overlap": ("FLOAT", {
                    "default": 0.75,
                    "min": 0.0,
                    "max": 0.99,
                    "step": 0.01,
                    "display": "number",
                }),
                "shifts": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 8,
                    "step": 1,
                    "display": "number",
                }),
                "segment_seconds": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 30.0,
                    "step": 0.1,
                    "display": "number",
                }),
                "transition_power": ("FLOAT", {
                    "default": 1.0,
                    "min": 1.0,
                    "max": 8.0,
                    "step": 0.1,
                    "display": "number",
                }),
                "guitar_highpass_hz": ("FLOAT", {
                    "default": float(DEFAULT_GUITAR_HPF_CUTOFF_HZ),
                    "min": 0.0,
                    "max": 1000.0,
                    "step": 1.0,
                    "display": "number",
                }),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
            },
        }

    # ------------------------------------------------------------------
    # Main processing method
    # ------------------------------------------------------------------

    def separate(
        self,
        audio: Dict,
        overlap: float,
        shifts: int,
        segment_seconds: float,
        transition_power: float,
        guitar_highpass_hz: float,
        device: str,
    ):
        """Run HTDemucs 6s separation and return all six isolated stems."""
        # ----------------------------------------------------------------
        # 0. Validate inputs
        # ----------------------------------------------------------------
        if audio is None:
            raise ValueError("SongSeparator: missing required 'audio' input.")

        # Fall back to CPU when CUDA is requested but not available
        if device == "cuda" and not torch.cuda.is_available():
            print("[SongSeparator] WARNING: CUDA requested but not available. "
                  "Falling back to CPU.")
            device = "cpu"

        torch_device = torch.device(device)

        # ----------------------------------------------------------------
        # 1. Load audio
        # ----------------------------------------------------------------
        waveform, sample_rate = _extract_audio_input(audio)

        # ----------------------------------------------------------------
        # 2. Load model
        # ----------------------------------------------------------------
        model = self._load_model(torch_device)

        # ----------------------------------------------------------------
        # 3. Run separation
        # ----------------------------------------------------------------
        try:
            stems = self._run_separation(
                model=model,
                waveform=waveform,
                sample_rate=sample_rate,
                torch_device=torch_device,
                overlap=overlap,
                shifts=shifts,
                transition_power=transition_power,
                segment_seconds=segment_seconds,
            )
        finally:
            # ----------------------------------------------------------------
            # 4. VRAM Safety Guard – always release GPU memory after separation
            # ----------------------------------------------------------------
            model.cpu()
            del model
            gc.collect()
            if torch_device.type == "cuda":
                torch.cuda.empty_cache()

        # ----------------------------------------------------------------
        # 5. Apply the optional guitar cleanup filter and package all stems
        # ----------------------------------------------------------------
        guitar_np = stems.get("guitar")
        if guitar_np is None:
            raise RuntimeError(
                "SongSeparator: Demucs output does not include a 'guitar' stem."
            )
        guitar_filtered = _apply_highpass(
            guitar_np,
            sample_rate,
            cutoff_hz=guitar_highpass_hz,
        )

        outputs = []
        for stem_name in ALL_STEMS:
            stem_array = stems.get(stem_name)
            if stem_array is None:
                raise RuntimeError(
                    f"SongSeparator: Demucs output does not include a '{stem_name}' stem."
                )
            if stem_name == "guitar":
                stem_array = guitar_filtered
            outputs.append(_numpy_to_comfy(stem_array, sample_rate))

        return tuple(outputs)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_model(torch_device: torch.device):
        """Load the HTDemucs 6-stem model onto *torch_device*."""
        # Import inside the method so demucs is only required at runtime
        from demucs.pretrained import get_model

        model = get_model(MODEL_NAME)

        model.to(torch_device)
        model.eval()
        return model

    @staticmethod
    def _run_separation(
        model,
        waveform: torch.Tensor,
        sample_rate: int,
        torch_device: torch.device,
        overlap: float,
        shifts: int,
        transition_power: float,
        segment_seconds: float,
    ) -> Dict[str, np.ndarray]:
        """Separate *waveform* and return a dict of {stem_name: np.ndarray}.

        Each array has shape (channels, samples) and float32 dtype.
        """
        from demucs.apply import apply_model
        from demucs.audio import convert_audio

        # Demucs expects (batch, channels, samples)
        # Resample / remix to the model's expected sample rate & channels
        wav = convert_audio(
            waveform,
            sample_rate,
            model.samplerate,
            model.audio_channels,
        )
        wav = wav.unsqueeze(0).to(torch_device)  # (1, C, T)
        segment = segment_seconds if segment_seconds > 0 else None

        with torch.inference_mode():
            raw = apply_model(
                model,
                wav,
                shifts=shifts,
                split=True,
                overlap=overlap,
                transition_power=transition_power,
                device=torch_device,
                segment=segment,
            )
        # raw shape: (batch=1, stems, channels, samples)
        raw = raw.squeeze(0).cpu()  # (stems, channels, samples)

        stem_names: List[str] = model.sources
        result: Dict[str, np.ndarray] = {}
        for idx, name in enumerate(stem_names):
            result[name] = raw[idx].numpy().astype(np.float32, copy=False)

        return result


class AudioPreviewEQ:
    """Preview and tone-shape audio with in-node waveform and playback UI."""

    CATEGORY = "Audio"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("AUDIO",)
    FUNCTION = "process_audio"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "bass_db": ("FLOAT", {
                    "default": 0.0,
                    "min": -18.0,
                    "max": 18.0,
                    "step": 0.5,
                    "display": "slider",
                }),
                "treble_db": ("FLOAT", {
                    "default": 0.0,
                    "min": -18.0,
                    "max": 18.0,
                    "step": 0.5,
                    "display": "slider",
                }),
                "volume_db": ("FLOAT", {
                    "default": 0.0,
                    "min": -24.0,
                    "max": 24.0,
                    "step": 0.5,
                    "display": "slider",
                }),
                "emit_audio": ("BOOLEAN", {"default": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    def process_audio(
        self,
        audio: Dict,
        bass_db: float,
        treble_db: float,
        volume_db: float,
        emit_audio: bool,
        unique_id: str,
    ):
        waveform, sample_rate = _extract_audio_input(audio)
        waveform_np = waveform.detach().cpu().numpy().astype(np.float32, copy=False)

        processed = _apply_tone_controls(
            waveform=waveform_np,
            sample_rate=sample_rate,
            bass_gain_db=bass_db,
            treble_gain_db=treble_db,
            volume_db=volume_db,
        )

        preview_url = _store_preview_audio(unique_id, processed, sample_rate)
        duration_sec = processed.shape[1] / float(sample_rate) if sample_rate > 0 else 0.0
        if emit_audio:
            output_audio = _numpy_to_comfy(processed, sample_rate)
        else:
            from comfy_execution.graph import ExecutionBlocker

            output_audio = ExecutionBlocker(None)

        return {
            "ui": {
                "audio_url": [preview_url],
                "waveform_peaks": [_build_waveform_peaks(processed)],
                "sample_rate": [sample_rate],
                "duration_sec": [round(duration_sec, 2)],
                "channels": [int(processed.shape[0])],
            },
            "result": (output_audio,),
        }


# ---------------------------------------------------------------------------
# Node registration helpers (consumed by __init__.py)
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS: Dict[str, type] = {
    "SongSeparator": SongSeparator,
    "AudioPreviewEQ": AudioPreviewEQ,
}

NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = {
    "SongSeparator": "Song Separator (HTDemucs 6s)",
    "AudioPreviewEQ": "Audio Preview EQ",
}
