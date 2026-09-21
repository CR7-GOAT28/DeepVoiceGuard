"""
Forensic Spectral & Acoustic Artifact Analysis.
Extracts physical and statistical markers that distinguish natural human vocal tract
mechanics from neural vocoder / speech synthesis artifacts.
"""

import numpy as np
import librosa
import scipy.stats as stats
from typing import Dict, Any

class SpectralForensics:
    def __init__(self, sr: int = 16000):
        self.sr = sr

    def analyze(self, y: np.ndarray) -> Dict[str, Any]:
        """
        Extracts comprehensive acoustic and forensic descriptors from audio array.
        """
        if len(y) < int(0.1 * self.sr):
            y = np.pad(y, (0, int(0.1 * self.sr) - len(y)), mode='constant')

        results = {}

        # 1. Robust Pitch / F0 & Glottal Dynamics Tracking (YIN Algorithm)
        try:
            f0 = librosa.yin(
                y,
                fmin=65,
                fmax=400,
                sr=self.sr,
                frame_length=1024,
                hop_length=256
            )
            voiced_mask = (f0 >= 70.0) & (f0 <= 380.0)
            voiced_f0 = f0[voiced_mask]
        except Exception:
            f0 = np.zeros(10)
            voiced_f0 = np.array([])
        
        if len(voiced_f0) > 3:
            results['f0_mean'] = float(np.mean(voiced_f0))
            results['f0_std'] = float(np.std(voiced_f0))
            diffs = np.abs(np.diff(voiced_f0))
            results['pitch_jitter'] = float(np.mean(diffs) / (np.mean(voiced_f0) + 1e-6))
            results['voicing_rate'] = float(len(voiced_f0) / (len(f0) + 1e-6))
        else:
            results['f0_mean'] = 0.0
            results['f0_std'] = 0.0
            results['pitch_jitter'] = 0.0
            results['voicing_rate'] = 0.0

        # Harmonic-to-Noise Ratio (HNR) proxy via Harmonic-Percussive Separation
        try:
            y_harm, y_perc = librosa.effects.hpss(y)
            harm_energy = np.sum(y_harm ** 2) + 1e-10
            perc_energy = np.sum(y_perc ** 2) + 1e-10
            results['hnr_db'] = float(10.0 * np.log10(harm_energy / perc_energy))
        except Exception:
            results['hnr_db'] = 0.0

        # 2. Spectral Descriptors
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=self.sr)[0]
        spec_rolloff_85 = librosa.feature.spectral_rolloff(y=y, sr=self.sr, roll_percent=0.85)[0]
        spec_rolloff_95 = librosa.feature.spectral_rolloff(y=y, sr=self.sr, roll_percent=0.95)[0]
        spec_flatness = librosa.feature.spectral_flatness(y=y)[0]
        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=self.sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=self.sr)[0]
        zcr = librosa.feature.zero_crossing_rate(y=y)[0]

        results['spectral_centroid_mean'] = float(np.mean(spec_cent))
        results['spectral_centroid_std'] = float(np.std(spec_cent))
        results['spectral_rolloff_85_mean'] = float(np.mean(spec_rolloff_85))
        results['spectral_rolloff_95_mean'] = float(np.mean(spec_rolloff_95))
        results['spectral_flatness_mean'] = float(np.mean(spec_flatness))
        results['spectral_flatness_std'] = float(np.std(spec_flatness))
        results['spectral_bandwidth_mean'] = float(np.mean(spec_bw))
        results['zcr_mean'] = float(np.mean(zcr))
        results['zcr_std'] = float(np.std(zcr))
        results['spectral_contrast_mean'] = float(np.mean(spec_contrast))

        # 3. High-Frequency Anomaly Ratio
        stft_mag = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
        freq_bins = librosa.fft_frequencies(sr=self.sr, n_fft=1024)

        low_mask = freq_bins < 1000
        mid_mask = (freq_bins >= 1000) & (freq_bins < 4000)
        high_mask = (freq_bins >= 4000) & (freq_bins < 7000)
        ultra_mask = freq_bins >= 7000

        total_power = np.sum(stft_mag ** 2) + 1e-10
        low_power = np.sum(stft_mag[low_mask, :] ** 2)
        mid_power = np.sum(stft_mag[mid_mask, :] ** 2)
        high_power = np.sum(stft_mag[high_mask, :] ** 2)
        ultra_power = np.sum(stft_mag[ultra_mask, :] ** 2)

        results['band_low_ratio'] = float(low_power / total_power)
        results['band_mid_ratio'] = float(mid_power / total_power)
        results['band_high_ratio'] = float(high_power / total_power)
        results['band_ultra_ratio'] = float(ultra_power / total_power)

        # 4. Spectral Statistics (Skewness, Kurtosis)
        flat_spec = stft_mag.flatten()
        results['spectral_skewness'] = float(stats.skew(flat_spec))
        results['spectral_kurtosis'] = float(stats.kurtosis(flat_spec))

        # 5. MFCC Statistics
        mfccs = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=20)
        for i in range(20):
            results[f'mfcc_{i+1}_mean'] = float(np.mean(mfccs[i]))
            results[f'mfcc_{i+1}_std'] = float(np.std(mfccs[i]))

        mfcc_delta = librosa.feature.delta(mfccs)
        for i in range(10):
            results[f'mfcc_delta_{i+1}_mean'] = float(np.mean(mfcc_delta[i]))
            results[f'mfcc_delta_{i+1}_std'] = float(np.std(mfcc_delta[i]))

        return results
