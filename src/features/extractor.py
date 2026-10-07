"""
Unified Feature Extraction Engine.
Coordinates extraction of tabular acoustic descriptors, 2D Mel spectrograms,
2D LFCC tensors, and forensic diagnostic curves.
"""

import numpy as np
import librosa
from typing import Dict, Any, Tuple, List
from .audio_loader import AudioLoader, DEFAULT_SAMPLE_RATE
from .lfcc import extract_lfcc
from .spectral_forensics import SpectralForensics

class FeatureExtractor:
    def __init__(self, sr: int = DEFAULT_SAMPLE_RATE):
        self.sr = sr
        self.audio_loader = AudioLoader(target_sr=sr)
        self.forensics = SpectralForensics(sr=sr)
        self.feature_names_: List[str] = []

    def extract_tabular(self, y: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """
        Extracts high-dimensional 1D acoustic descriptor vector.
        """
        stats_dict = self.forensics.analyze(y)
        
        # Add LFCC summary statistics
        lfcc_full = extract_lfcc(y, sr=self.sr, n_lfcc=20, with_deltas=True)
        for i in range(20):
            stats_dict[f'lfcc_{i+1}_mean'] = float(np.mean(lfcc_full[i]))
            stats_dict[f'lfcc_{i+1}_std'] = float(np.std(lfcc_full[i]))
            stats_dict[f'lfcc_delta_{i+1}_mean'] = float(np.mean(lfcc_full[20 + i]))
            stats_dict[f'lfcc_delta_{i+1}_std'] = float(np.std(lfcc_full[20 + i]))

        feature_names = sorted(list(stats_dict.keys()))
        feature_vector = np.array([stats_dict[k] for k in feature_names], dtype=np.float32)
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=1.0, neginf=-1.0)
        self.feature_names_ = feature_names
        return feature_vector, feature_names

    def extract_mel_spectrogram(
        self,
        y: np.ndarray,
        n_mels: int = 128,
        n_fft: int = 1024,
        hop_length: int = 256,
        target_frames: int = 188
    ) -> np.ndarray:
        """
        Extracts normalized Log-Mel spectrogram [n_mels, target_frames].
        """
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=self.sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        log_mel = (log_mel + 80.0) / 80.0
        log_mel = np.clip(log_mel, 0.0, 1.0)

        if log_mel.shape[1] < target_frames:
            pad_width = target_frames - log_mel.shape[1]
            log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode='wrap')
        else:
            log_mel = log_mel[:, :target_frames]

        return log_mel.astype(np.float32)

    def extract_lfcc_tensor(
        self,
        y: np.ndarray,
        target_frames: int = 300
    ) -> np.ndarray:
        """
        Extracts LFCC + Deltas tensor [60, target_frames].
        """
        lfcc = extract_lfcc(y, sr=self.sr, n_lfcc=20, with_deltas=True)
        if lfcc.shape[1] < target_frames:
            pad_w = target_frames - lfcc.shape[1]
            lfcc = np.pad(lfcc, ((0, 0), (0, pad_w)), mode='wrap')
        else:
            lfcc = lfcc[:, :target_frames]
        return lfcc.astype(np.float32)

    def extract_all(self, y: np.ndarray) -> Dict[str, Any]:
        """
        Extracts all modalities for an audio segment.
        """
        tab_vec, feat_names = self.extract_tabular(y)
        mel_tensor = self.extract_mel_spectrogram(y)
        lfcc_tensor = self.extract_lfcc_tensor(y)
        forensic_dict = self.forensics.analyze(y)

        return {
            'tabular_vector': tab_vec,
            'feature_names': feat_names,
            'mel_spectrogram': mel_tensor,
            'lfcc_tensor': lfcc_tensor,
            'forensics': forensic_dict,
            'raw_audio': y,
            'duration': len(y) / float(self.sr)
        }
