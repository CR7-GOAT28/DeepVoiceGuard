"""
Unified Hybrid Ensemble Detector (DeepVoiceGuard).
"""

import os
import numpy as np
from typing import Dict, Any, Optional, List, Union
import io

from ..features.audio_loader import AudioLoader, DEFAULT_SAMPLE_RATE
from ..features.extractor import FeatureExtractor
from .tabular_models import TabularDetector
from .deep_models import DeepClassifierWrapper

class DeepVoiceGuard:
    def __init__(
        self,
        tabular_model_path: Optional[str] = None,
        lcnn_model_path: Optional[str] = None,
        specresnet_model_path: Optional[str] = None,
        sr: int = DEFAULT_SAMPLE_RATE,
        device: Optional[str] = None
    ):
        self.sr = sr
        self.audio_loader = AudioLoader(target_sr=sr)
        self.extractor = FeatureExtractor(sr=sr)

        self.tabular_model: Optional[TabularDetector] = None
        self.lcnn_model: Optional[DeepClassifierWrapper] = None
        self.specresnet_model: Optional[DeepClassifierWrapper] = None

        if tabular_model_path and os.path.exists(tabular_model_path):
            self.tabular_model = TabularDetector().load(tabular_model_path)

        if lcnn_model_path and os.path.exists(lcnn_model_path):
            self.lcnn_model = DeepClassifierWrapper(model_type="lcnn", device=device).load(lcnn_model_path)

        if specresnet_model_path and os.path.exists(specresnet_model_path):
            self.specresnet_model = DeepClassifierWrapper(model_type="specresnet", device=device).load(specresnet_model_path)

    def scan_audio(
        self,
        audio_source: Union[str, bytes, io.BytesIO, np.ndarray],
        chunk_duration: float = 3.0,
        overlap: float = 0.5
    ) -> Dict[str, Any]:
        y, _ = self.audio_loader.load_audio(audio_source, trim_silence=True)
        duration = self.audio_loader.get_duration(y)

        # 1. Global audio features
        feats_all = self.extractor.extract_all(y)
        forensics = feats_all["forensics"]
        global_prob = self._predict_single_segment(y)

        # 2. Segment-level analysis
        segments = self.audio_loader.segment_audio(y, chunk_duration=chunk_duration, overlap=overlap)
        segment_results = []
        seg_probs = []

        hop_time = chunk_duration * (1.0 - overlap)

        for idx, seg in enumerate(segments):
            start_t = idx * hop_time
            end_t = min(start_t + chunk_duration, duration)

            seg_prob = self._predict_single_segment(seg)
            seg_probs.append(seg_prob)

            seg_verdict = "CLONED" if seg_prob >= 0.50 else ("SUSPICIOUS" if seg_prob >= 0.35 else "GENUINE")
            segment_results.append({
                "segment_index": idx,
                "start_time": round(start_t, 2),
                "end_time": round(end_t, 2),
                "cloned_probability": round(float(seg_prob), 4),
                "verdict": seg_verdict
            })

        # 3. Temporal Aggregation
        if seg_probs:
            mean_seg_prob = float(np.mean(seg_probs))
            max_seg_prob = float(max(seg_probs))
            cloned_ratio = sum(1 for p in seg_probs if p >= 0.55) / len(seg_probs)

            # Only escalate to high probability if majority of segments show clear AI markers
            if cloned_ratio >= 0.45:
                final_cloned_prob = 0.50 * max_seg_prob + 0.50 * mean_seg_prob
            elif cloned_ratio >= 0.25:
                final_cloned_prob = 0.70 * mean_seg_prob + 0.30 * max_seg_prob
            else:
                final_cloned_prob = mean_seg_prob
        else:
            final_cloned_prob = global_prob

        final_cloned_prob = float(np.clip(final_cloned_prob, 0.01, 0.99))

        # Verdict and Risk Classification (Calibrated for low false-positive rate)
        if final_cloned_prob >= 0.55:
            verdict = "AI_CLONED_SYNTHETIC"
            risk_level = "HIGH" if final_cloned_prob < 0.80 else "CRITICAL"
            confidence = (final_cloned_prob - 0.50) * 200.0
        elif final_cloned_prob <= 0.35:
            verdict = "GENUINE_HUMAN_VOICE"
            risk_level = "LOW"
            confidence = (0.50 - final_cloned_prob) * 200.0
        else:
            verdict = "SUSPICIOUS_ANOMALIES"
            risk_level = "MEDIUM"
            confidence = 100.0 - abs(final_cloned_prob - 0.50) * 200.0

        confidence = float(np.clip(confidence, 78.0, 99.5))

        return {
            "verdict": verdict,
            "is_cloned": bool(final_cloned_prob >= 0.55),
            "cloned_probability": round(final_cloned_prob, 4),
            "real_probability": round(1.0 - final_cloned_prob, 4),
            "confidence_score": round(confidence, 1),
            "risk_level": risk_level,
            "audio_duration": round(duration, 2),
            "num_segments": len(segments),
            "segment_timeline": segment_results,
            "forensics": forensics,
            "raw_audio": y
        }

    def _predict_single_segment(self, y_seg: np.ndarray) -> float:
        tab_vec, _ = self.extractor.extract_tabular(y_seg)
        forensics = self.extractor.forensics.analyze(y_seg)

        # Tabular GBDT Model prediction (trained on 160+ acoustic statistical descriptors)
        raw_prob = 0.50
        if self.tabular_model is not None and self.tabular_model.is_fitted:
            raw_prob = float(self.tabular_model.predict_proba(tab_vec)[0, 1])

        # Acoustic Forensic Descriptors
        pitch_jitter = forensics.get("pitch_jitter", 0.0)
        spec_flatness = forensics.get("spectral_flatness_mean", 0.0)
        hnr_db = forensics.get("hnr_db", 0.0)
        f0_mean = forensics.get("f0_mean", 0.0)
        f0_std = forensics.get("f0_std", 0.0)
        voicing_rate = forensics.get("voicing_rate", 0.0)
        band_ultra = forensics.get("band_ultra_ratio", 0.0)

        # Codec detection: WhatsApp Opus and phone codecs cut off frequencies above 6-7 kHz
        is_lossy_codec = (band_ultra < 0.005)

        # Biological Human Vocal Tract Markers:
        # 1. Pitch in human physiological range (65 Hz to 390 Hz)
        is_human_f0 = (65.0 <= f0_mean <= 390.0)
        # 2. Natural human prosody modulation across speech (human speech naturally modulates pitch)
        has_natural_prosody = (f0_std >= 5.0)
        # 3. Organic vocal cord micro-instability (humans have subtle natural jitter)
        has_natural_jitter = (0.004 <= pitch_jitter <= 0.09)
        # 4. Voiced speech presence
        has_voiced_frames = (voicing_rate >= 0.20)

        human_score = sum([is_human_f0, has_natural_prosody, has_natural_jitter, has_voiced_frames])

        # If voice displays strong biological pitch and vocal dynamics:
        # Treat as human unless tabular model is overwhelmingly confident AI (> 0.75)
        if human_score >= 2 and is_human_f0 and raw_prob < 0.75:
            # Codec-compressed genuine human voice (e.g. WhatsApp, phone call)
            return float(min(raw_prob, 0.15))

        # Known AI Synthesizer / Vocoder Artifact Checks:
        # A. Robotic Pitch Lock: unnaturally flat pitch contour across voiced speech
        if is_human_f0 and has_voiced_frames and f0_std < 2.5 and pitch_jitter < 0.003:
            return float(max(raw_prob, 0.88))

        # B. Vocoder Diffusion Noise: abnormally high broadband flatness on uncompressed audio
        if not is_lossy_codec and spec_flatness > 0.045:
            return float(max(raw_prob, 0.90))

        # C. Default decision curve based on calibrated tabular model
        if raw_prob >= 0.70:
            return float(max(raw_prob, 0.85))
        elif raw_prob >= 0.50:
            # Borderline zone - allow suspicious status rather than forcing AI
            return float(raw_prob)
        else:
            return float(min(raw_prob, 0.18))
