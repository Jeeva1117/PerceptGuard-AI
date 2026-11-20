"""Section B: Multi-Horizon Reliability Risk Overview component (H1, H3, H5)."""

from __future__ import annotations

from typing import Any, Dict, Optional
import streamlit as st

from dashboard.api_client import DashboardApiClient, FrontendAPIError
from dashboard.components.state_views import render_empty_state, render_error_boundary


def render_risk_overview(client: DashboardApiClient) -> None:
    """Section B: Multi-Horizon Reliability Risk Overview (H1, H3, H5)."""
    st.subheader("Section B — Multi-Horizon Reliability Risk Overview")

    st.markdown(
        r"""
        > ℹ️ **Notice:** Multi-horizon risk scores $S \in [0.0, 1.0]$ represent derived perception failure risk ($S = 1.0 - R$).
        > Thresholds are currently designated as **PROVISIONAL** due to zero-inflation in single-clip driving data.
        """
    )

    try:
        overview = client.get_risk_overview()
    except FrontendAPIError as err:
        render_error_boundary(
            title="Failed to Load Risk Overview Telemetry",
            error_message=f"Backend communication error: {err.message}",
            retry_callback=st.rerun,
            key_suffix="risk_overview",
        )
        return

    horizons = overview.get("horizons", {})
    if not horizons:
        render_empty_state(
            title="No Risk Horizon Telemetry Available",
            message="No reliability risk forecast data has been ingested into the system yet.",
            icon="⚠️",
        )
        return

    col1, col2, col3 = st.columns(3)

    def _render_horizon_card(col, h_key: str, h_label: str, horizon_int: int):
        data = horizons.get(h_key, {})
        score = data.get("risk_score")
        level = data.get("risk_level", "UNKNOWN")
        trend = data.get("trend_direction", "UNKNOWN")
        rolling_mean = data.get("rolling_mean")
        volatility = data.get("volatility")
        obs_count = data.get("observation_count", 0)

        # STRICT INTEGRITY: If score is None, level must NOT be LOW
        if score is None:
            level = "UNKNOWN"
            trend = "UNKNOWN"

        level_colors = {
            "LOW": "#2a9d8f",
            "MEDIUM": "#e9c46a",
            "HIGH": "#f4a261",
            "CRITICAL": "#e76f51",
            "UNKNOWN": "#6c757d",
        }
        l_color = level_colors.get(level, "#6c757d")

        trend_symbols = {
            "INCREASING": "↗ INCREASING",
            "DECREASING": "↘ DECREASING",
            "STABLE": "→ STABLE",
            "VOLATILE": "↕ VOLATILE",
            "UNKNOWN": "— UNKNOWN",
        }
        t_sym = trend_symbols.get(trend, trend)

        with col:
            st.markdown(
                f"""
                <div style="background: #181926; border: 1px solid #313244; border-radius: 8px; padding: 14px 16px; margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <h4 style="margin: 0; color: #cdd6f4;">{h_label} (H{horizon_int})</h4>
                        <span style="background: {l_color}; color: #11111b; font-weight: 700; padding: 2px 8px; border-radius: 4px; font-size: 11px;">
                            {level}
                        </span>
                    </div>
                    <div style="font-size: 28px; font-weight: 700; color: #ffffff; margin-bottom: 4px;">
                        {f'{score:.4f}' if score is not None else 'N/A'}
                    </div>
                    <div style="font-size: 12px; color: #a6adc8; margin-bottom: 8px;">
                        Trend: <strong>{t_sym}</strong>
                    </div>
                    <hr style="border: 0; border-top: 1px solid #313244; margin: 8px 0;"/>
                    <div style="font-size: 11px; color: #bac2de; line-height: 1.6;">
                        <strong>Rolling Mean:</strong> {f'{rolling_mean:.4f}' if rolling_mean is not None else 'N/A'}<br/>
                        <strong>Volatility (σ):</strong> {f'{volatility:.4f}' if volatility is not None else 'N/A'}<br/>
                        <strong>Observations:</strong> {obs_count} frames
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    _render_horizon_card(col1, "h1", "Immediate Horizon (t+1)", 1)
    _render_horizon_card(col2, "h3", "Intermediate Horizon (t+3)", 3)
    _render_horizon_card(col3, "h5", "Extended Horizon (t+5)", 5)
