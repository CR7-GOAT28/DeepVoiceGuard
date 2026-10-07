"""
DeepVoiceGuard: AI Voice Cloning & Audio Deepfake Detection Web Application.
Interactive Streamlit Dashboard with Real-Time Audio Scanning, Forensic Diagnostics,
Spectrogram Inspections, Batch File Audits, and Model Benchmark Insights.
"""

import os
import sys

# Ensure project root is always in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import io
import json
import glob
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.models.ensemble import DeepVoiceGuard
from src.forensics.analyzer import ForensicAnalyzer
from src.features.audio_loader import AudioLoader, DEFAULT_SAMPLE_RATE

# Streamlit Page Config
st.set_page_config(
    page_title="DeepVoiceGuard | AI Voice Cloning Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Premium Cyber Glassmorphic UI)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    .stApp {
        background: radial-gradient(circle at 10% 20%, rgba(15, 23, 42, 0.98) 0%, rgba(10, 14, 26, 1) 90%);
        color: #f8fafc;
    }

    /* Glassmorphism Card */
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 1.2rem;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(6, 182, 212, 0.12);
        border: 1px solid rgba(6, 182, 212, 0.35);
        color: #38bdf8;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }

    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 50%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
        line-height: 1.15;
    }

    .hero-subtitle {
        font-size: 1.1rem;
        color: #94a3b8;
        font-weight: 400;
        margin-bottom: 1.8rem;
        line-height: 1.5;
    }

    /* Verdict Banners */
    .verdict-fake {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.18) 0%, rgba(153, 27, 27, 0.30) 100%);
        border: 1px solid rgba(239, 68, 68, 0.6);
        border-radius: 16px;
        padding: 1.6rem;
        text-align: center;
        box-shadow: 0 0 25px rgba(239, 68, 68, 0.2);
    }

    .verdict-real {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.18) 0%, rgba(5, 150, 105, 0.30) 100%);
        border: 1px solid rgba(16, 185, 129, 0.6);
        border-radius: 16px;
        padding: 1.6rem;
        text-align: center;
        box-shadow: 0 0 25px rgba(16, 185, 129, 0.2);
    }

    .verdict-suspicious {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.18) 0%, rgba(180, 83, 9, 0.30) 100%);
        border: 1px solid rgba(245, 158, 11, 0.6);
        border-radius: 16px;
        padding: 1.6rem;
        text-align: center;
        box-shadow: 0 0 25px rgba(245, 158, 11, 0.2);
    }

    /* Metric pill boxes */
    .stat-pill {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.1rem;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .stat-pill:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.4);
    }
    .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 2px;
    }
    .stat-label {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #94a3b8;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(15, 23, 42, 0.8);
        padding: 6px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 600;
        color: #94a3b8;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.35);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_detector():
    tab_path = os.path.join(PROJECT_ROOT, "saved_models", "tabular_model.joblib")
    lcnn_path = os.path.join(PROJECT_ROOT, "saved_models", "lcnn_model.pt")
    spec_path = os.path.join(PROJECT_ROOT, "saved_models", "specresnet_model.pt")

    detector = DeepVoiceGuard(
        tabular_model_path=tab_path if os.path.exists(tab_path) else None,
        lcnn_model_path=lcnn_path if os.path.exists(lcnn_path) else None,
        specresnet_model_path=spec_path if os.path.exists(spec_path) else None
    )
    return detector

detector = load_detector()
forensics_engine = ForensicAnalyzer()
audio_loader = AudioLoader()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/cyber-security.png", width=70)
    st.markdown("### DeepVoiceGuard AI")
    st.caption("Voice Anti-Spoofing & Deepfake Audio Forensics")
    st.divider()

    st.markdown("#### ⚡ Active Model Status")
    if detector.tabular_model and detector.tabular_model.is_fitted:
        st.success("✔ Tabular Ensemble (LGBM+XGB+RF)", icon="✅")
    else:
        st.warning("⚠ Tabular Model (Untrained)", icon="⚠️")

    if detector.lcnn_model and detector.lcnn_model.is_fitted:
        st.success("✔ PyTorch LFCC-LCNN DeepNet", icon="✅")
    else:
        st.warning("⚠ LFCC-LCNN (Untrained)", icon="⚠️")

    if detector.specresnet_model and detector.specresnet_model.is_fitted:
        st.success("✔ SpecResNet Attention Net", icon="✅")
    else:
        st.warning("⚠ SpecResNet (Untrained)", icon="⚠️")

    st.divider()
    st.markdown("#### ⚙️ Scanner Sensitivity")
    chunk_dur = st.slider("Segment Window (seconds)", min_value=1.5, max_value=5.0, value=3.0, step=0.5)
    overlap_rate = st.slider("Window Overlap", min_value=0.0, max_value=0.75, value=0.50, step=0.25)
    
    if st.button("🔄 Reload Models & Reset Cache", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

    st.divider()
    st.caption("v1.0.0 | Academic & Production Ready")

# Main Header
st.markdown("""
<div class="hero-badge">
    <span style="width: 8px; height: 8px; border-radius: 50%; background: #38bdf8; display: inline-block;"></span>
    Next-Gen Voice Biometric Defense
</div>
<div class="hero-title">DeepVoiceGuard Studio</div>
<div class="hero-subtitle">
    State-of-the-art detection of AI-generated speech clones, neural TTS vocoders (ElevenLabs, XTTS, Bark, VITS), and voice conversion deepfakes.
</div>
""", unsafe_allow_html=True)

# Tabs
tab_scan, tab_forensics, tab_demo, tab_batch, tab_benchmarks = st.tabs([
    "🎙️ Live Voice Scanner",
    "🔬 Forensic Deep-Dive",
    "🧪 Preset Demo Lab",
    "📁 High-Throughput Batch Audit",
    "📊 Research Benchmarks & Architecture"
])

# ----------------- TAB 1: LIVE SCANNER -----------------
with tab_scan:
    st.markdown("""
    <div class="glass-card">
        <h4 style="margin-top:0; color: #f8fafc; font-size: 1.15rem; font-weight: 600;">
            1. Ingest Audio Stream
        </h4>
        <p style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 0;">
            Upload any speech recording (.wav, .mp3, .flac, .ogg, .m4a) or capture live vocal audio directly via microphone.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_input1, col_input2 = st.columns([1, 1])

    with col_input1:
        uploaded_file = st.file_uploader(
            "Upload audio file (.wav, .mp3, .flac, .ogg, .m4a)",
            type=["wav", "mp3", "flac", "ogg", "m4a"],
            key="scanner_upload"
        )

    with col_input2:
        mic_audio = st.audio_input("Or Record via Microphone")

    active_audio = uploaded_file or mic_audio

    # Check if a sample was triggered from Demo tab
    if active_audio is None and 'auto_scan_bytes' in st.session_state:
        active_audio = io.BytesIO(st.session_state['auto_scan_bytes'])

    if active_audio is not None:
        audio_bytes = active_audio.read() if hasattr(active_audio, 'read') else active_audio.getvalue()
        st.markdown("<br>", unsafe_allow_html=True)
        st.audio(audio_bytes)

        with st.spinner("Analyzing acoustic features, LFCCs, harmonic stability, and vocoder artifacts..."):
            try:
                scan_res = detector.scan_audio(io.BytesIO(audio_bytes), chunk_duration=chunk_dur, overlap=overlap_rate)
                radar_scores = forensics_engine.compute_forensic_radar(scan_res['forensics'])
                findings = forensics_engine.generate_forensic_summary(scan_res)

                st.session_state['latest_scan'] = scan_res
                st.session_state['latest_audio'] = scan_res['raw_audio']
                st.session_state['latest_radar'] = radar_scores
                st.session_state['latest_findings'] = findings

                prob_fake = scan_res['cloned_probability']
                prob_real = scan_res['real_probability']
                confidence = scan_res['confidence_score']
                risk = scan_res['risk_level']
                verdict = scan_res['verdict']

                st.markdown("<br>", unsafe_allow_html=True)
                if verdict == "AI_CLONED_SYNTHETIC":
                    st.markdown(f"""
                    <div class="verdict-fake">
                        <div style="font-size: 2.2rem; margin-bottom: 4px;">🚨</div>
                        <h2 style="color: #f87171; margin: 0; font-size: 1.7rem; font-weight: 800;">
                            SYNTHETIC / AI CLONED VOICE DETECTED
                        </h2>
                        <p style="font-size: 1.05rem; margin-top: 8px; color: #fecaca; font-weight: 500;">
                            High probability of neural speech synthesis (e.g. ElevenLabs, XTTS, Bark) or vocoder phase manipulation.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                elif verdict == "GENUINE_HUMAN_VOICE":
                    st.markdown(f"""
                    <div class="verdict-real">
                        <div style="font-size: 2.2rem; margin-bottom: 4px;">✅</div>
                        <h2 style="color: #34d399; margin: 0; font-size: 1.7rem; font-weight: 800;">
                            AUTHENTIC HUMAN VOICE VERIFIED
                        </h2>
                        <p style="font-size: 1.05rem; margin-top: 8px; color: #a7f3d0; font-weight: 500;">
                            Acoustic harmonics, pitch micro-jitter, and formant dynamics conform to natural vocal tract biology.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="verdict-suspicious">
                        <div style="font-size: 2.2rem; margin-bottom: 4px;">⚠️</div>
                        <h2 style="color: #fbbf24; margin: 0; font-size: 1.7rem; font-weight: 800;">
                            SUSPICIOUS ACOUSTIC ANOMALIES DETECTED
                        </h2>
                        <p style="font-size: 1.05rem; margin-top: 8px; color: #fde68a; font-weight: 500;">
                            Partial vocoder signatures or phase irregularities detected. Additional forensic inspection recommended.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Modern Stat Pills
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.markdown(f"""
                    <div class="stat-pill">
                        <div class="stat-value" style="color: {'#f87171' if prob_fake >= 0.5 else '#38bdf8'};">{prob_fake * 100:.1f}%</div>
                        <div class="stat-label">AI Clone Probability</div>
                    </div>
                    """, unsafe_allow_html=True)
                with m2:
                    st.markdown(f"""
                    <div class="stat-pill">
                        <div class="stat-value" style="color: #34d399;">{prob_real * 100:.1f}%</div>
                        <div class="stat-label">Authentic Probability</div>
                    </div>
                    """, unsafe_allow_html=True)
                with m3:
                    st.markdown(f"""
                    <div class="stat-pill">
                        <div class="stat-value" style="color: #a78bfa;">{confidence:.1f}%</div>
                        <div class="stat-label">Model Confidence</div>
                    </div>
                    """, unsafe_allow_html=True)
                with m4:
                    r_col = "#ef4444" if risk in ("HIGH", "CRITICAL") else ("#f59e0b" if risk == "MEDIUM" else "#10b981")
                    st.markdown(f"""
                    <div class="stat-pill">
                        <div class="stat-value" style="color: {r_col};">{risk}</div>
                        <div class="stat-label">Threat Severity</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Gauge & Segment Timeline
                col_g, col_t = st.columns([1, 1.5])
                with col_g:
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prob_fake * 100.0,
                        title={'text': "Cloned Probability Score (%)", 'font': {'size': 18}},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "#ef4444" if prob_fake >= 0.60 else ("#f59e0b" if prob_fake >= 0.40 else "#10b981")},
                            'steps': [
                                {'range': [0, 40], 'color': "rgba(16, 185, 129, 0.25)"},
                                {'range': [40, 60], 'color': "rgba(245, 158, 11, 0.25)"},
                                {'range': [60, 100], 'color': "rgba(239, 68, 68, 0.25)"}
                            ],
                            'threshold': {
                                'line': {'color': "white", 'width': 3},
                                'thickness': 0.75,
                                'value': 60.0
                            }
                        }
                    ))
                    fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_gauge, use_container_width=True)

                with col_t:
                    st.markdown("##### ⏱️ Temporal Segment Anomaly Heatmap")
                    timeline = scan_res['segment_timeline']
                    df_time = pd.DataFrame(timeline)
                    if not df_time.empty:
                        df_time['Segment'] = [f"{r['start_time']}s - {r['end_time']}s" for _, r in df_time.iterrows()]
                        df_time['Risk %'] = df_time['cloned_probability'] * 100.0
                        
                        fig_time = px.bar(
                            df_time,
                            x='Segment',
                            y='Risk %',
                            color='Risk %',
                            color_continuous_scale=['#10b981', '#f59e0b', '#ef4444'],
                            range_color=[0, 100],
                            labels={'Risk %': 'Synthetic Probability %'}
                        )
                        fig_time.add_hline(y=60, line_dash="dash", line_color="red", annotation_text="AI Threshold (60%)")
                        fig_time.update_layout(height=250, margin=dict(l=10, r=10, t=20, b=10), paper_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig_time, use_container_width=True)

                # Forensic Bullet Findings
                st.markdown("##### 🔎 Key Forensic Diagnostic Findings")
                for item in findings:
                    st.markdown(f"- {item}")

            except Exception as e:
                st.error(f"Error analyzing audio: {e}")
    else:
        st.info("💡 Upload an audio file or record from microphone above, or switch to the **Demo Sample Lab** tab to test preloaded samples!")

# ----------------- TAB 2: FORENSIC DEEP-DIVE -----------------
with tab_forensics:
    st.markdown("""
    <div class="glass-card">
        <h4 style="margin-top:0; color: #f8fafc; font-size: 1.2rem; font-weight: 700;">
            🔬 Multi-Spectral & Glottal Physics Forensic Studio
        </h4>
        <p style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 0;">
            Deep-dive into acoustic resonance, higher-order spectral statistics, and physiological vocal tract boundaries.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if 'latest_scan' in st.session_state:
        scan_data = st.session_state['latest_scan']
        raw_y = st.session_state['latest_audio']
        radar_data = st.session_state['latest_radar']

        col_f1, col_f2 = st.columns(2)

        with col_f1:
            st.markdown("""
            <div class="glass-card">
                <h5 style="margin-top:0; color: #38bdf8; font-size: 1rem; font-weight: 600;">
                    🕸️ 6-Dimensional Acoustic Radar Profile
                </h5>
            """, unsafe_allow_html=True)

            categories = list(radar_data.keys())
            values = list(radar_data.values())
            categories.append(categories[0])
            values.append(values[0])

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='Current Sample',
                line=dict(color='#06b6d4', width=2),
                fillcolor='rgba(6, 182, 212, 0.30)'
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=[85, 85, 85, 85, 85, 85, 85],
                theta=categories,
                name='Natural Human Baseline (85+)',
                line=dict(color='#10b981', dash='dash', width=2)
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], gridcolor='rgba(255,255,255,0.08)', linecolor='rgba(255,255,255,0.08)'),
                    angularaxis=dict(gridcolor='rgba(255,255,255,0.08)', linecolor='rgba(255,255,255,0.08)', color='#94a3b8'),
                    bgcolor='rgba(15, 23, 42, 0.6)'
                ),
                height=340,
                margin=dict(l=30, r=30, t=20, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5, font={'color': '#cbd5e1'})
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_f2:
            st.markdown("""
            <div class="glass-card">
                <h5 style="margin-top:0; color: #38bdf8; font-size: 1rem; font-weight: 600;">
                    🎵 Fundamental Frequency (F0) & Pitch Stability
                </h5>
            """, unsafe_allow_html=True)

            p_data = forensics_engine.compute_pitch_and_hnr_track(raw_y)
            df_pitch = pd.DataFrame({'Time (s)': p_data['times'], 'F0 (Hz)': p_data['f0']})
            df_pitch = df_pitch[df_pitch['F0 (Hz)'] > 0]

            if not df_pitch.empty:
                fig_pitch = px.line(df_pitch, x='Time (s)', y='F0 (Hz)', title=None)
                fig_pitch.update_traces(line=dict(color='#38bdf8', width=2.5))
                fig_pitch.add_hrect(y0=85, y1=255, fillcolor="rgba(16, 185, 129, 0.12)", line_width=0, annotation_text="Natural Vocal Band (85-255Hz)", annotation_font_color="#34d399")
                fig_pitch.update_layout(
                    height=300,
                    margin=dict(l=15, r=15, t=15, b=15),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(15, 23, 42, 0.6)',
                    yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                    xaxis=dict(gridcolor='rgba(255,255,255,0.05)')
                )
                st.plotly_chart(fig_pitch, use_container_width=True)
            else:
                st.info("No periodic voiced frames detected for pitch tracking.")
            st.markdown("</div>", unsafe_allow_html=True)

        # High-Resolution Time-Frequency Spectrograms
        st.markdown("""
        <div class="glass-card">
            <h5 style="margin-top:0; color: #f8fafc; font-size: 1.1rem; font-weight: 700;">
                🌈 Time-Frequency Forensic Spectrograms
            </h5>
        """, unsafe_allow_html=True)

        spec_data = forensics_engine.compute_spectrograms(raw_y)
        
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("###### 🔥 Log-Mel Spectrogram (Vocal Resonance & Harmonic Tracks)")
            fig_mel = px.imshow(
                spec_data['mel_spectrogram'],
                origin='lower',
                aspect='auto',
                color_continuous_scale='Magma',
                labels={'x': 'Time Frames', 'y': 'Mel Bins'}
            )
            fig_mel.update_layout(height=280, margin=dict(l=5, r=5, t=10, b=10), paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_mel, use_container_width=True)

        with col_s2:
            st.markdown("###### 🔍 Linear STFT Spectrogram (High-Frequency Cutoff Inspector)")
            fig_lin = px.imshow(
                spec_data['linear_spectrogram'],
                origin='lower',
                aspect='auto',
                color_continuous_scale='Viridis',
                labels={'x': 'Time Frames', 'y': 'Linear Frequency Bins'}
            )
            fig_lin.update_layout(height=280, margin=dict(l=5, r=5, t=10, b=10), paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_lin, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Raw Acoustic Descriptors Table
        st.markdown("""
        <div class="glass-card">
            <h5 style="margin-top:0; color: #f8fafc; font-size: 1.05rem; font-weight: 600;">
                📋 Extracted Acoustic Parameter Matrix
            </h5>
        """, unsafe_allow_html=True)

        forensic_table = pd.DataFrame([
            {"Acoustic Parameter": k, "Measured Value": f"{v:.4f}" if isinstance(v, float) else str(v)}
            for k, v in scan_data['forensics'].items()
            if not k.startswith("mfcc_") and not k.startswith("lfcc_")
        ])
        st.dataframe(forensic_table, use_container_width=True, height=260)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Execute an audio scan in Tab 1 or Tab 3 to render deep forensic diagnostics.")

# ----------------- TAB 3: PRESET DEMO LAB -----------------
with tab_demo:
    st.markdown("""
    <div class="glass-card">
        <h4 style="margin-top:0; color: #f8fafc; font-size: 1.2rem; font-weight: 700;">
            🧪 Zero-Setup Presentation Demo Showcase
        </h4>
        <p style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 0;">
            Demonstrate real-time detection with pre-packaged audio clips representing verified genuine humans and popular AI synthesis models.
        </p>
    </div>
    """, unsafe_allow_html=True)

    sample_real_dir = os.path.join(PROJECT_ROOT, "sample_data", "real")
    sample_cloned_dir = os.path.join(PROJECT_ROOT, "sample_data", "cloned")

    col_dr, col_df = st.columns(2)

    with col_dr:
        st.markdown("""
        <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 14px; padding: 1.2rem; margin-bottom: 1rem;">
            <h5 style="margin: 0; color: #34d399; font-weight: 700;">🟢 Authentic Human Recordings</h5>
            <p style="color: #94a3b8; font-size: 0.82rem; margin-top: 4px; margin-bottom: 0;">Natural vocal tract resonance with organic biological micro-jitter.</p>
        </div>
        """, unsafe_allow_html=True)

        real_files = glob.glob(os.path.join(sample_real_dir, "*.wav")) if os.path.exists(sample_real_dir) else []
        if real_files:
            for rf in real_files:
                bname = os.path.basename(rf)
                st.markdown(f"**🎙️ {bname.replace('_', ' ').replace('.wav', '').title()}**")
                st.audio(rf)
                if st.button(f"⚡ Scan {bname}", key=f"btn_r_{bname}"):
                    with open(rf, 'rb') as f:
                        st.session_state['auto_scan_bytes'] = f.read()
                        st.toast(f"Loaded {bname}! Switch to Tab 1 to view live verdict.", icon="✅")
        else:
            st.warning("No sample files found. Run 'python run_demo.py' to generate demo files.")

    with col_df:
        st.markdown("""
        <div style="background: rgba(239, 68, 68, 0.08); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 14px; padding: 1.2rem; margin-bottom: 1rem;">
            <h5 style="margin: 0; color: #f87171; font-weight: 700;">🔴 AI Cloned & Synthesized Voice Clips</h5>
            <p style="color: #94a3b8; font-size: 0.82rem; margin-top: 4px; margin-bottom: 0;">ElevenLabs, XTTS, and Voice Conversion models with vocoder phase distortion.</p>
        </div>
        """, unsafe_allow_html=True)

        cloned_files = glob.glob(os.path.join(sample_cloned_dir, "*.wav")) if os.path.exists(sample_cloned_dir) else []
        if cloned_files:
            for cf in cloned_files:
                bname = os.path.basename(cf)
                st.markdown(f"**🤖 {bname.replace('_', ' ').replace('.wav', '').title()}**")
                st.audio(cf)
                if st.button(f"⚡ Scan {bname}", key=f"btn_c_{bname}"):
                    with open(cf, 'rb') as f:
                        st.session_state['auto_scan_bytes'] = f.read()
                        st.toast(f"Loaded {bname}! Switch to Tab 1 to view live verdict.", icon="🚨")
        else:
            st.warning("No sample files found. Run 'python run_demo.py' to generate demo files.")

# ----------------- TAB 4: BATCH FILE SCANNER -----------------
with tab_batch:
    st.markdown("""
    <div class="glass-card">
        <h4 style="margin-top:0; color: #f8fafc; font-size: 1.2rem; font-weight: 700;">
            📁 High-Throughput Bulk Audio Forensics
        </h4>
        <p style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 0;">
            Ideal for KYC voice biometric verification, call center compliance audits, and batch media screening.
        </p>
    </div>
    """, unsafe_allow_html=True)

    batch_files = st.file_uploader(
        "Upload batch audio files",
        type=["wav", "mp3", "flac", "ogg"],
        accept_multiple_files=True,
        key="batch_upload"
    )

    if batch_files:
        if st.button(f"🚀 Launch Batch Screening ({len(batch_files)} Audio Clips)"):
            prog_bar = st.progress(0)
            status_txt = st.empty()

            batch_results = []
            for idx, b_file in enumerate(batch_files):
                status_txt.text(f"Auditing file {idx+1}/{len(batch_files)}: {b_file.name}")
                try:
                    b_bytes = b_file.read()
                    res = detector.scan_audio(io.BytesIO(b_bytes))
                    batch_results.append({
                        'Filename': b_file.name,
                        'Verdict': res['verdict'],
                        'Cloned Risk (%)': round(res['cloned_probability'] * 100, 2),
                        'Confidence (%)': round(res['confidence_score'], 1),
                        'Risk Level': res['risk_level'],
                        'Duration (s)': res['audio_duration']
                    })
                except Exception as e:
                    batch_results.append({
                        'Filename': b_file.name,
                        'Verdict': f"ERROR: {e}",
                        'Cloned Risk (%)': 0.0,
                        'Confidence (%)': 0.0,
                        'Risk Level': 'UNKNOWN',
                        'Duration (s)': 0.0
                    })
                prog_bar.progress((idx + 1) / len(batch_files))

            status_txt.text("Batch processing complete!")
            df_batch = pd.DataFrame(batch_results)

            st.markdown("<br>", unsafe_allow_html=True)
            c_tot, c_real, c_fake = st.columns(3)
            with c_tot:
                st.markdown(f"""
                <div class="stat-pill">
                    <div class="stat-value">{len(df_batch)}</div>
                    <div class="stat-label">Total Files Screened</div>
                </div>
                """, unsafe_allow_html=True)
            with c_real:
                n_real = len(df_batch[df_batch['Verdict'] == 'GENUINE_HUMAN_VOICE'])
                st.markdown(f"""
                <div class="stat-pill">
                    <div class="stat-value" style="color: #34d399;">{n_real}</div>
                    <div class="stat-label">Verified Humans</div>
                </div>
                """, unsafe_allow_html=True)
            with c_fake:
                n_fake = len(df_batch[df_batch['Verdict'] == 'AI_CLONED_SYNTHETIC'])
                st.markdown(f"""
                <div class="stat-pill">
                    <div class="stat-value" style="color: #f87171;">{n_fake}</div>
                    <div class="stat-label">Deepfakes Intercepted</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.dataframe(df_batch, use_container_width=True)

            csv_data = df_batch.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Export Full CSV Forensic Report",
                data=csv_data,
                file_name="deepvoiceguard_batch_report.csv",
                mime="text/csv"
            )

# ----------------- TAB 5: BENCHMARKS & ARCHITECTURE -----------------
with tab_benchmarks:
    st.markdown("""
    <div class="glass-card">
        <h4 style="margin-top:0; color: #f8fafc; font-size: 1.2rem; font-weight: 700;">
            📊 Research Benchmarks & System Architecture
        </h4>
        <p style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 0;">
            Academic verification metrics, Receiver Operating Characteristic curves, and feature attribution.
        </p>
    </div>
    """, unsafe_allow_html=True)

    metrics_file = os.path.join(PROJECT_ROOT, "saved_models", "training_metrics.json")
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r', encoding='utf-8') as f:
            metrics_data = json.load(f)

        ens_m = metrics_data.get('ensemble', {})
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            st.markdown(f"""
            <div class="stat-pill">
                <div class="stat-value" style="color: #38bdf8;">{ens_m.get('eer_percentage', 0.0)}%</div>
                <div class="stat-label">Equal Error Rate (EER)</div>
            </div>
            """, unsafe_allow_html=True)
        with b2:
            st.markdown(f"""
            <div class="stat-pill">
                <div class="stat-value" style="color: #34d399;">{ens_m.get('roc_auc', 0.0):.4f}</div>
                <div class="stat-label">ROC-AUC Benchmark</div>
            </div>
            """, unsafe_allow_html=True)
        with b3:
            st.markdown(f"""
            <div class="stat-pill">
                <div class="stat-value" style="color: #a78bfa;">{ens_m.get('accuracy', 0.0)*100:.1f}%</div>
                <div class="stat-label">Overall Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        with b4:
            st.markdown(f"""
            <div class="stat-pill">
                <div class="stat-value" style="color: #f472b6;">{ens_m.get('f1_score', 0.0):.4f}</div>
                <div class="stat-label">Optimal F1-Score</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ROC Curve & Top Features
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown("""
            <div class="glass-card">
                <h5 style="margin-top:0; color: #f8fafc; font-size: 1rem; font-weight: 600;">
                    📈 Receiver Operating Characteristic (ROC Curve)
                </h5>
            """, unsafe_allow_html=True)
            roc = ens_m.get('roc_curve', {})
            if roc and 'fpr' in roc:
                df_roc = pd.DataFrame({'False Positive Rate': roc['fpr'], 'True Positive Rate': roc['tpr']})
                fig_roc = px.area(df_roc, x='False Positive Rate', y='True Positive Rate')
                fig_roc.update_traces(line=dict(color='#0284c7', width=3), fillcolor='rgba(2, 132, 199, 0.25)')
                fig_roc.add_shape(type='line', line=dict(dash='dash', color='#64748b'), x0=0, x1=1, y0=0, y1=1)
                fig_roc.update_layout(
                    height=280,
                    margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(15, 23, 42, 0.6)',
                    font={'color': '#94a3b8'},
                    yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                    xaxis=dict(gridcolor='rgba(255,255,255,0.05)')
                )
                st.plotly_chart(fig_roc, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_r2:
            st.markdown("""
            <div class="glass-card">
                <h5 style="margin-top:0; color: #f8fafc; font-size: 1rem; font-weight: 600;">
                    🌟 Top Predictive Acoustic Discriminators
                </h5>
            """, unsafe_allow_html=True)
            feats = metrics_data.get('feature_importances', [])[:10]
            if feats:
                df_feat = pd.DataFrame(feats, columns=['Feature', 'Importance']).sort_values('Importance', ascending=True)
                fig_feat = px.bar(df_feat, x='Importance', y='Feature', orientation='h')
                fig_feat.update_traces(marker_color='#38bdf8')
                fig_feat.update_layout(
                    height=280,
                    margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(15, 23, 42, 0.6)',
                    font={'color': '#94a3b8'},
                    xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
                )
                st.plotly_chart(fig_feat, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <h5 style="margin-top:0; color: #f8fafc; font-size: 1.1rem; font-weight: 700;">
            🏗️ High-Level Signal Processing & Machine Learning Pipeline
        </h5>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-top: 14px;">
            <div style="background: rgba(15, 23, 42, 0.7); padding: 16px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05);">
                <h6 style="color: #38bdf8; margin: 0 0 8px 0; font-size: 0.95rem;">1. Linear Frequency (LFCC)</h6>
                <p style="margin: 0; font-size: 0.84rem; color: #94a3b8; line-height: 1.45;">
                    Extracts 60-dimensional spectral coefficients linearly spaced to capture high-frequency vocoder phase artifacts that traditional Mel filters smooth out.
                </p>
            </div>
            <div style="background: rgba(15, 23, 42, 0.7); padding: 16px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05);">
                <h6 style="color: #a78bfa; margin: 0 0 8px 0; font-size: 0.95rem;">2. Glottal & Pitch Tracking</h6>
                <p style="margin: 0; font-size: 0.84rem; color: #94a3b8; line-height: 1.45;">
                    Tracks fundamental vocal frequency (F0), Harmonic-to-Noise Ratio (HNR), and micro-jitter to detect robotic pitch flattening or phase discontinuities.
                </p>
            </div>
            <div style="background: rgba(15, 23, 42, 0.7); padding: 16px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05);">
                <h6 style="color: #34d399; margin: 0 0 8px 0; font-size: 0.95rem;">3. Hybrid Deep Ensemble</h6>
                <p style="margin: 0; font-size: 0.84rem; color: #94a3b8; line-height: 1.45;">
                    Fuses LightGBM + XGBoost + Random Forest with PyTorch Max-Feature-Map LCNN and SpecResNet attention for calibrated probability output.
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

