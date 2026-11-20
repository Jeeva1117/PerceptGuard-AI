"""Top branding header, navigation, and system status banner."""

from __future__ import annotations

import streamlit as st
from src.config.system_settings import get_settings


def render_header(
    transport_mode: str = "IN_PROCESS",
    calibration_status: str = "PROVISIONAL CALIBRATION",
) -> None:
    """Render top branding header, mission statement, and operational telemetry banner."""
    settings = get_settings()
    env_str = settings.app.environment.value.upper()
    version_str = settings.app.version
    gate_str = calibration_status or settings.dashboard.calibration_status_label

    # Transport mode color and badge
    is_rest = "REST" in transport_mode.upper()
    mode_color = "#2a9d8f" if is_rest else "#e76f51"
    mode_label = "REST API" if is_rest else "In-Process Unified"

    st.markdown(
        f"""
        <div style="background: linear-gradient(90deg, #0d1b2a 0%, #1b263b 100%); padding: 18px 24px; border-radius: 8px; border-left: 6px solid #415a77; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
                <div>
                    <h1 style="color: #e0e1dd; margin: 0; font-size: 26px; font-weight: 700; letter-spacing: 0.5px;">
                        🛡️ PerceptGuard AI <span style="font-size: 14px; background: #415a77; color: #fff; padding: 2px 8px; border-radius: 4px; margin-left: 10px;">v{version_str}</span>
                        <span style="font-size: 12px; background: #2a9d8f; color: #fff; padding: 2px 6px; border-radius: 4px; margin-left: 6px;">{env_str}</span>
                        <span style="font-size: 12px; background: {mode_color}; color: #fff; padding: 2px 6px; border-radius: 4px; margin-left: 6px;">{mode_label}</span>
                    </h1>
                    <p style="color: #778da9; margin: 4px 0 0 0; font-size: 13px;">
                        Autonomous Driving Perception Failure Early Warning & Multi-Horizon Reliability Forecasting System
                    </p>
                </div>
                <div style="text-align: right;">
                    <span style="font-size: 11px; color: #a3b18a; font-family: monospace;">Device: {settings.model.device.upper()} (Windows 11 / Intel Core i3 / 8GB)</span><br/>
                    <span style="font-size: 11px; color: #e0a96d; font-family: monospace;">Gate: {gate_str}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Backward compatibility alias
render_dashboard_header = render_header
