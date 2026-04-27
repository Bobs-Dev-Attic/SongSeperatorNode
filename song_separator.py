"""
SongSeparator – ComfyUI custom node
====================================
Separates an audio file into up to 6 stems (vocals, drums, bass, guitar,
piano, other) using the HTDemucs 6-stem model from the *demucs* library,
then applies a 100 Hz high-pass filter to the guitar stem to remove
sub-bass bleed that confuses MIDI / pitch-detection engines.

Output format follows the standard ComfyUI audio convention:
    {"waveform": torch.Tensor[B, C, T], "sample_rate": int}
"""

import os
import gc
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchaudio
from scipy.signal import butter, sosfilt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_STEMS: List[str] = ["guitar", "vocals", "drums", "bass", "piano", "other"]
MODEL_NAME: str = "htdemucs_6s"
HPF_CUTOFF_HZ: int = 100  # high-pass cutoff for the guitar stem


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
                    cutoff_hz: float = HPF_CUTOFF_HZ) -> np.ndarray:
    """Apply a zero-phase Butterworth high-pass filter channel-wise.

    Args:
        waveform:   NumPy array of shape (channels, samples).
        sample_rate: Sample rate in Hz.
        cutoff_hz:  High-pass cutoff frequency in Hz.

    Returns:
        Filtered waveform with the same shape as the input.
    """
    sos = _butter_highpass_sos(cutoff_hz, sample_rate)
    filtered = np.stack(
        [sosfilt(sos, ch) for ch in waveform],
        axis=0,
    )
    return filtered.astype(np.float32)


def _load_audio(audio_path: str) -> Tuple[torch.Tensor, int]:
    """Load audio from *audio_path* and return (waveform, sample_rate).

    The waveform tensor has shape (channels, samples) and float32 dtype.
    """
    waveform, sr = torchaudio.load(audio_path)
    if waveform.dtype != torch.float32:
        waveform = waveform.to(torch.float32)
    return waveform, sr


def _numpy_to_comfy(array: np.ndarray, sample_rate: int) -> Dict:
    """Convert a (C, T) float32 NumPy array to the ComfyUI audio dict format.

    ComfyUI audio nodes expect:
        {"waveform": Tensor[1, C, T], "sample_rate": int}
    """
    tensor = torch.from_numpy(array).unsqueeze(0)  # (1, C, T)
    return {"waveform": tensor, "sample_rate": sample_rate}


def _mix_stems(stems: Dict[str, np.ndarray]) -> np.ndarray:
    """Sum multiple stem arrays (each shape (C, T)) into one mix."""
    arrays = list(stems.values())
    if not arrays:
        raise ValueError("No stems provided for mixing.")
    return np.sum(arrays, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# ComfyUI Node
# ---------------------------------------------------------------------------

class SongSeparator:
    """ComfyUI node that isolates audio stems using the HTDemucs 6-stem model.

    Designed to prep audio for MIDI transcription, with particular focus on
    clean guitar isolation and VRAM safety for mixed image/audio workflows.
    """

    CATEGORY = "Audio"
    RETURN_TYPES = ("AUDIO", "AUDIO")
    RETURN_NAMES = ("GUITAR_AUDIO", "OTHER_AUDIO")
    FUNCTION = "separate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "/path/to/song.wav",
                }),
                "overlap": ("FLOAT", {
                    "default": 0.75,
                    "min": 0.0,
                    "max": 0.99,
                    "step": 0.01,
                    "display": "number",
                }),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
            },
            "optional": {
                # Each stem flag is an independent boolean toggle
                "keep_guitar":  ("BOOLEAN", {"default": True}),
                "keep_vocals":  ("BOOLEAN", {"default": False}),
                "keep_drums":   ("BOOLEAN", {"default": False}),
                "keep_bass":    ("BOOLEAN", {"default": False}),
                "keep_piano":   ("BOOLEAN", {"default": False}),
                "keep_other":   ("BOOLEAN", {"default": False}),
            },
        }

    # ------------------------------------------------------------------
    # Main processing method
    # ------------------------------------------------------------------

    def separate(
        self,
        audio_path: str,
        overlap: float,
        device: str,
        keep_guitar: bool = True,
        keep_vocals: bool = False,
        keep_drums: bool = False,
        keep_bass: bool = False,
        keep_piano: bool = False,
        keep_other: bool = False,
    ):
        """Run HTDemucs 6s separation and return (GUITAR_AUDIO, OTHER_AUDIO).

        GUITAR_AUDIO is the 100 Hz high-pass-filtered guitar stem when
        *keep_guitar* is True, otherwise silence.
        OTHER_AUDIO contains the mix of all *other* selected stems (those
        enabled via their keep_* flag, minus guitar).  If no other stems
        are selected OTHER_AUDIO is silent (zeros at the same shape).
        """
        # ----------------------------------------------------------------
        # 0. Validate inputs
        # ----------------------------------------------------------------
        if not audio_path or not os.path.isfile(audio_path):
            raise FileNotFoundError(
                f"SongSeparator: audio file not found: '{audio_path}'"
            )

        # Fall back to CPU when CUDA is requested but not available
        if device == "cuda" and not torch.cuda.is_available():
            print("[SongSeparator] WARNING: CUDA requested but not available. "
                  "Falling back to CPU.")
            device = "cpu"

        torch_device = torch.device(device)

        # ----------------------------------------------------------------
        # 1. Load audio
        # ----------------------------------------------------------------
        waveform, sample_rate = _load_audio(audio_path)

        # ----------------------------------------------------------------
        # 2. Load model
        # ----------------------------------------------------------------
        model = self._load_model(torch_device, overlap)

        # ----------------------------------------------------------------
        # 3. Run separation
        # ----------------------------------------------------------------
        try:
            stems = self._run_separation(model, waveform, sample_rate,
                                         torch_device)
        finally:
            # ----------------------------------------------------------------
            # 4. VRAM Safety Guard – always release GPU memory after separation
            # ----------------------------------------------------------------
            model.cpu()
            del model
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

        # ----------------------------------------------------------------
        # 5. Pre-Transcription Filter – 100 Hz HPF on guitar stem
        # ----------------------------------------------------------------
        guitar_np = stems["guitar"]  # shape (C, T)
        if keep_guitar:
            guitar_filtered = _apply_highpass(guitar_np, sample_rate,
                                              cutoff_hz=HPF_CUTOFF_HZ)
        else:
            guitar_filtered = np.zeros_like(guitar_np)
        guitar_audio = _numpy_to_comfy(guitar_filtered, sample_rate)

        # ----------------------------------------------------------------
        # 6. Build OTHER_AUDIO from selected non-guitar stems
        # ----------------------------------------------------------------
        keep_flags = {
            "vocals": keep_vocals,
            "drums":  keep_drums,
            "bass":   keep_bass,
            "piano":  keep_piano,
            "other":  keep_other,
        }
        other_stems = {
            name: arr
            for name, arr in stems.items()
            if name != "guitar" and keep_flags.get(name, False)
        }

        if other_stems:
            other_mix = _mix_stems(other_stems)
        else:
            # Return silence with the same shape as the guitar stem
            other_mix = np.zeros_like(guitar_filtered)

        other_audio = _numpy_to_comfy(other_mix, sample_rate)

        return (guitar_audio, other_audio)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_model(torch_device: torch.device, overlap: float):
        """Load the HTDemucs 6-stem model onto *torch_device*."""
        # Import inside the method so demucs is only required at runtime
        from demucs.pretrained import get_model
        from demucs.apply import BagOfModels

        model = get_model(MODEL_NAME)

        # BagOfModels wraps an ensemble; set overlap on all sub-models
        if isinstance(model, BagOfModels):
            for sub in model.models:
                if hasattr(sub, "overlap"):
                    sub.overlap = overlap
        elif hasattr(model, "overlap"):
            model.overlap = overlap

        model.to(torch_device)
        model.eval()
        return model

    @staticmethod
    def _run_separation(
        model,
        waveform: torch.Tensor,
        sample_rate: int,
        torch_device: torch.device,
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

        with torch.no_grad():
            raw = apply_model(model, wav, device=torch_device)
        # raw shape: (batch=1, stems, channels, samples)
        raw = raw.squeeze(0).cpu()  # (stems, channels, samples)

        stem_names: List[str] = model.sources
        result: Dict[str, np.ndarray] = {}
        for idx, name in enumerate(stem_names):
            result[name] = raw[idx].numpy().astype(np.float32)

        return result


# ---------------------------------------------------------------------------
# Node registration helpers (consumed by __init__.py)
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS: Dict[str, type] = {
    "SongSeparator": SongSeparator,
}

NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = {
    "SongSeparator": "Song Separator (HTDemucs 6s)",
}
