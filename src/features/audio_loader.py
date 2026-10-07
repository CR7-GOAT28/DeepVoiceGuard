"""
Audio loading, resampling, normalization, and segmentation utility.
"""

import os
import io
import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, List, Optional, Union

DEFAULT_SAMPLE_RATE = 16000

class AudioLoader:
    """
    Audio loader supporting path, raw bytes, BytesIO, or numpy arrays.
    Resamples to standard target sample rate (default 16kHz) and normalizes volume.
    """
    def __init__(self, target_sr: int = DEFAULT_SAMPLE_RATE, top_db: float = 30.0):
        self.target_sr = target_sr
        self.top_db = top_db

    def load_audio(
        self,
        audio_source: Union[str, bytes, io.BytesIO, np.ndarray],
        sr: Optional[int] = None,
        trim_silence: bool = True
    ) -> Tuple[np.ndarray, int]:
        """
        Loads audio, normalizes amplitude, and converts to mono.
        """
        if isinstance(audio_source, (bytes, bytearray)):
            buffer = io.BytesIO(audio_source)
            y, _ = librosa.load(buffer, sr=self.target_sr, mono=True)
        elif isinstance(audio_source, io.BytesIO):
            audio_source.seek(0)
            y, _ = librosa.load(audio_source, sr=self.target_sr, mono=True)
        elif isinstance(audio_source, np.ndarray):
            y = audio_source.astype(np.float32)
            if y.ndim > 1:
                y = np.mean(y, axis=1 if y.shape[1] < y.shape[0] else 0)
            if sr is not None and sr != self.target_sr:
                y = librosa.resample(y, orig_sr=sr, target_sr=self.target_sr)
        elif isinstance(audio_source, str):
            if not os.path.exists(audio_source):
                raise FileNotFoundError(f"Audio file not found: {audio_source}")
            y, _ = librosa.load(audio_source, sr=self.target_sr, mono=True)
        else:
            raise ValueError(f"Unsupported audio source type: {type(audio_source)}")

        if len(y) == 0:
            y = np.zeros(self.target_sr, dtype=np.float32)

        # Peak normalization
        max_val = np.max(np.abs(y))
        if max_val > 1e-6:
            y = y / max_val

        # Voice Activity Detection (VAD) / Silence trimming
        if trim_silence and len(y) > int(0.25 * self.target_sr):
            trimmed, _ = librosa.effects.trim(y, top_db=self.top_db)
            if len(trimmed) > int(0.1 * self.target_sr):
                y = trimmed

        return y.astype(np.float32), self.target_sr

    def segment_audio(
        self,
        y: np.ndarray,
        chunk_duration: float = 3.0,
        overlap: float = 0.5
    ) -> List[np.ndarray]:
        """
        Splits audio into overlapping segments of chunk_duration seconds.
        """
        chunk_len = int(chunk_duration * self.target_sr)
        hop_len = int(chunk_len * (1.0 - overlap))

        if len(y) <= chunk_len:
            reps = int(np.ceil(chunk_len / max(1, len(y))))
            tiled = np.tile(y, reps)[:chunk_len]
            return [tiled.astype(np.float32)]

        segments = []
        for start in range(0, len(y) - chunk_len + 1, hop_len):
            segments.append(y[start : start + chunk_len].astype(np.float32))

        # Include tail remainder if significant
        if len(segments) > 0 and len(y) > chunk_len:
            last_segment = y[-chunk_len:].astype(np.float32)
            if not np.array_equal(last_segment, segments[-1]):
                segments.append(last_segment)

        return segments

    def get_duration(self, y: np.ndarray) -> float:
        return float(len(y)) / float(self.target_sr)
