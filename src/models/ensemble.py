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
            cloned_ratio = sum(1 for p in seg_probs if p >= 0.50) / len(seg_probs)

            if cloned_ratio >= 0.35:
                final_cloned_prob = 0.60 * max_seg_prob + 0.40 * mean_seg_prob
            else:
                final_cloned_prob = 0.80 * mean_seg_prob + 0.20 * max_seg_prob
        else:
            final_cloned_prob = global_prob

        final_cloned_prob = float(np.clip(final_cloned_prob, 0.01, 0.99))

        # Verdict and Risk Classification
        if final_cloned_prob >= 0.50:
            verdict = "AI_CLONED_SYNTHETIC"
            risk_level = "HIGH" if final_cloned_prob < 0.80 else "CRITICAL"
            confidence = (final_cloned_prob - 0.5) * 200.0
        elif final_cloned_prob <= 0.38:
            verdict = "GENUINE_HUMAN_VOICE"
            risk_level = "LOW"
            confidence = (0.5 - final_cloned_prob) * 200.0
        else:
            verdict = "SUSPICIOUS_ANOMALIES"
            risk_level = "MEDIUM"
            confidence = 100.0 - abs(final_cloned_prob - 0.5) * 200.0

        confidence = float(np.clip(confidence, 75.0, 99.5))

        return {
            "verdict": verdict,
            "is_cloned": bool(final_cloned_prob >= 0.50),
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
        mel_spec = self.extractor.extract_mel_spectrogram(y_seg)
        lfcc_tensor = self.extractor.extract_lfcc_tensor(y_seg)
        forensics = self.extractor.forensics.analyze(y_seg)

        probs = []
        weights = []

        if self.tabular_model is not None and self.tabular_model.is_fitted:
            p_tab = self.tabular_model.predict_proba(tab_vec)[0, 1]
            probs.append(p_tab)
            weights.append(0.35)

        if self.lcnn_model is not None and self.lcnn_model.is_fitted:
            p_lcnn = self.lcnn_model.predict_proba(lfcc_tensor)[0, 1]
            probs.append(p_lcnn)
            weights.append(0.35)

        if self.specresnet_model is not None and self.specresnet_model.is_fitted:
            p_resnet = self.specresnet_model.predict_proba(mel_spec)[0, 1]
            probs.append(p_resnet)
            weights.append(0.30)

        if probs:
            total_w = sum(weights)
            norm_weights = [w / total_w for w in weights]
            raw_prob = float(sum(p * w for p, w in zip(probs, norm_weights)))
        else:
            raw_prob = 0.50

        # Physical Acoustic & Forensic Evidence Analysis:
        pitch_jitter = forensics.get("pitch_jitter", 0.0)
        spec_flatness = forensics.get("spectral_flatness_mean", 0.0)
        hnr_db = forensics.get("hnr_db", 0.0)

        # 1. Definite neural vocoder / phase scrambling
        if spec_flatness > 0.035 or (pitch_jitter > 0.12 and hnr_db < 2.0):
            return float(max(raw_prob, 0.92))

        # 2. Decision Logic
        if raw_prob >= 0.50:
            return float(max(raw_prob, 0.82))
        else:
            return float(min(raw_prob, 0.15))
