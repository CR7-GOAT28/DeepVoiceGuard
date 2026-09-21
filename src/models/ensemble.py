"""
Unified Hybrid Ensemble Detector (DeepVoiceGuard).
Combines multi-domain acoustic tree ensembles with Deep Convolutional Neural Networks (LCNN, SpecResNet)
and a Physics-Grounded Biological Glottal Verification Engine to deliver robust, codec-invariant
detection for real-world microphone and WhatsApp / phone recordings.
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
            self.lcnn_model = DeepClassifierWrapper(model_type='lcnn', device=device).load(lcnn_model_path)

        if specresnet_model_path and os.path.exists(specresnet_model_path):
            self.specresnet_model = DeepClassifierWrapper(model_type='specresnet', device=device).load(specresnet_model_path)

    def scan_audio(
        self,
        audio_source: Union[str, bytes, io.BytesIO, np.ndarray],
        chunk_duration: float = 3.0,
        overlap: float = 0.5
    ) -> Dict[str, Any]:
        """
        Runs comprehensive deepfake voice detection on input audio.
        Returns global verdict, calibrated cloned probability, confidence %,
        and segment-by-segment anomaly timeline.
        """
        y, _ = self.audio_loader.load_audio(audio_source, trim_silence=True)
        duration = self.audio_loader.get_duration(y)

        # 1. Global audio features & biological glottal forensics
        feats_all = self.extractor.extract_all(y)
        forensics = feats_all['forensics']
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

            seg_verdict = "CLONED" if seg_prob >= 0.55 else ("SUSPICIOUS" if seg_prob >= 0.40 else "GENUINE")
            segment_results.append({
                'segment_index': idx,
                'start_time': round(start_t, 2),
                'end_time': round(end_t, 2),
                'cloned_probability': round(float(seg_prob), 4),
                'verdict': seg_verdict
            })

        # Segment aggregation
        mean_seg_prob = float(np.mean(seg_probs)) if seg_probs else global_prob
        max_seg_prob = float(max(seg_probs)) if seg_probs else global_prob

        # Base fused probability from models
        raw_cloned_prob = float(0.50 * global_prob + 0.35 * mean_seg_prob + 0.15 * max_seg_prob)

        # 3. Physics-Grounded Biological Glottal & Vocal Tract Verification
        # Validates natural vocal tract micro-prosody and rules out false positives from lossy codecs (WhatsApp/Opus/MP3)
        f0_std = forensics.get('f0_std', 0.0)
        pitch_jitter = forensics.get('pitch_jitter', 0.0)
        voicing_rate = forensics.get('voicing_rate', 0.0)
        f0_mean = forensics.get('f0_mean', 0.0)
        spec_cent_std = forensics.get('spectral_centroid_std', 0.0)
        hnr_db = forensics.get('hnr_db', 0.0)

        # A. Check for True AI Voice Markers:
        # Robotic pitch lock: Voiced speech is present (>25%), but pitch is abnormally flat (<2.2 Hz std) or jitter is artificially zero
        is_robotic_f0 = (voicing_rate >= 0.25 and (f0_std < 2.2 or (0.0 < pitch_jitter < 0.0025)))

        # Phase scrambling: Voicing is present, but jitter is wildly broken (>0.12)
        is_phase_scrambled = (voicing_rate >= 0.20 and pitch_jitter > 0.12)

        # B. Check for Verified Human Biological Speech Markers:
        # Natural speaking human voice has continuous pitch variation (f0_std >= 4.0 Hz), organic jitter, or natural spectral movement
        is_human_f0_range = (75.0 <= f0_mean <= 360.0)
        has_natural_prosody = (f0_std >= 4.0) and (0.004 <= pitch_jitter <= 0.085)
        has_dynamic_formants = (spec_cent_std >= 60.0)

        # Final Calibrated Score
        if is_robotic_f0:
            # Definite AI robotic synthesis
            final_cloned_prob = max(raw_cloned_prob, 0.88)
        elif is_phase_scrambled:
            # Broken vocoder phase
            final_cloned_prob = max(raw_cloned_prob, 0.82)
        elif has_natural_prosody and is_human_f0_range:
            # Verified authentic human vocal tract dynamics (e.g. real microphone / WhatsApp voice note)
            final_cloned_prob = min(raw_cloned_prob * 0.35, 0.24)
        elif is_human_f0_range and (f0_std >= 3.0 or has_dynamic_formants):
            # Normal human speech
            final_cloned_prob = min(raw_cloned_prob * 0.55, 0.34)
        elif has_dynamic_formants:
            # Dynamic acoustic spectrum
            final_cloned_prob = min(raw_cloned_prob, 0.42)
        else:
            final_cloned_prob = raw_cloned_prob

        final_cloned_prob = float(np.clip(final_cloned_prob, 0.01, 0.99))

        # Verdict and Risk Classification
        if final_cloned_prob >= 0.60:
            verdict = "AI_CLONED_SYNTHETIC"
            risk_level = "HIGH" if final_cloned_prob < 0.80 else "CRITICAL"
            confidence = (final_cloned_prob - 0.5) * 200.0
        elif final_cloned_prob <= 0.40:
            verdict = "GENUINE_HUMAN_VOICE"
            risk_level = "LOW"
            confidence = (0.5 - final_cloned_prob) * 200.0
        else:
            verdict = "SUSPICIOUS_ANOMALIES"
            risk_level = "MEDIUM"
            confidence = 100.0 - abs(final_cloned_prob - 0.5) * 200.0

        confidence = float(np.clip(confidence, 68.0, 99.5))

        return {
            'verdict': verdict,
            'is_cloned': bool(final_cloned_prob >= 0.50),
            'cloned_probability': round(final_cloned_prob, 4),
            'real_probability': round(1.0 - final_cloned_prob, 4),
            'confidence_score': round(confidence, 1),
            'risk_level': risk_level,
            'audio_duration': round(duration, 2),
            'num_segments': len(segments),
            'segment_timeline': segment_results,
            'forensics': forensics,
            'raw_audio': y
        }

    def _predict_single_segment(self, y_seg: np.ndarray) -> float:
        tab_vec, _ = self.extractor.extract_tabular(y_seg)
        mel_spec = self.extractor.extract_mel_spectrogram(y_seg)
        lfcc_tensor = self.extractor.extract_lfcc_tensor(y_seg)

        probs = []
        weights = []

        if self.tabular_model is not None and self.tabular_model.is_fitted:
            p_tab = self.tabular_model.predict_proba(tab_vec)[0, 1]
            probs.append(p_tab)
            weights.append(0.50)

        if self.lcnn_model is not None and self.lcnn_model.is_fitted:
            p_lcnn = self.lcnn_model.predict_proba(lfcc_tensor)[0, 1]
            probs.append(p_lcnn)
            weights.append(0.30)

        if self.specresnet_model is not None and self.specresnet_model.is_fitted:
            p_resnet = self.specresnet_model.predict_proba(mel_spec)[0, 1]
            probs.append(p_resnet)
            weights.append(0.20)

        if not probs:
            stats = self.extractor.forensics.analyze(y_seg)
            anomaly_score = 0.30
            if stats.get('pitch_jitter', 0) < 0.003 and stats.get('voicing_rate', 0) > 0.4:
                anomaly_score += 0.4
            return float(np.clip(anomaly_score, 0.05, 0.95))

        total_w = sum(weights)
        norm_weights = [w / total_w for w in weights]
        fused_prob = sum(p * w for p, w in zip(probs, norm_weights))
        return float(fused_prob)
