"""Section: Reliability Risk Assessment & Multi-Horizon Failure Anticiption Component."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import pandas as pd
import streamlit as st

from dashboard.api_client import DashboardApiClient, FrontendAPIError
from dashboard.components.state_views import render_empty_state, render_error_boundary


def render_risk_overview(client: DashboardApiClient) -> None:
    """Render comprehensive reliability risk assessment, category breakdown, and multi-horizon trajectories."""
    st.markdown("### ⚠️ Perception Failure Risk Assessment")
    st.caption(
        "Anticipating detection failures before they occur. Failure risk $S \\in [0.0, 1.0]$ is derived "
        "from predicted reliability ($S = 1.0 - R$) and classified into discrete operational severity tiers."
    )

    # Scientific calibration notice
    st.markdown(
        r"""
        > ℹ️ **Mathematical & Calibration Notice:** Failure risk is formally derived as $S = 1.0 - R_{t+h}$.
        > Classification thresholds are designated as **PROVISIONAL** pending multi-weather driving evaluations.
        > Stale or unavailable predictions produce `UNKNOWN` and are never coerced to `LOW`.
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

    # Multi-Horizon Risk Cards
    col1, col2, col3 = st.columns(3)

    def _render_horizon_card(col, h_key: str, h_label: str, horizon_int: int):
        data = horizons.get(h_key, {})
        score = data.get("risk_score")
        level = data.get("risk_level", "UNKNOWN")
        trend = data.get("trend_direction", "UNKNOWN")
        rolling_mean = data.get("rolling_mean")
        volatility = data.get("volatility")
        obs_count = data.get("observation_count", 0)

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
        color = level_colors.get(level, "#6c757d")

        col.markdown(
            f"""
            <div style="background: #181926; border-left: 5px solid {color}; border-radius: 8px; padding: 16px; margin-bottom: 12px; border-top: 1px solid #313244; border-right: 1px solid #313244; border-bottom: 1px solid #313244;">
                <div style="font-size: 13px; color: #a6adc8; font-weight: 600;">{h_label}</div>
                <div style="display: flex; justify-content: space-between; align-items: baseline; margin: 8px 0;">
                    <span style="font-size: 32px; font-weight: 700; color: #cdd6f4;">
                        {f"{score:.4f}" if score is not None else "N/A"}
                    </span>
                    <span style="background: {color}; color: #ffffff; padding: 2px 8px; border-radius: 4px; font-weight: 700; font-size: 11px;">
                        {level}
                    </span>
                </div>
                <div style="font-size: 12px; color: #a6adc8;">
                    Trend: <strong>{trend}</strong> &nbsp;|&nbsp;
                    Mean: <strong>{f"{rolling_mean:.4f}" if rolling_mean is not None else "N/A"}</strong>
                </div>
                <div style="font-size: 11px; color: #6c7086; margin-top: 4px;">
                    Observations: {obs_count} frames | Volatility: {f"{volatility:.4f}" if volatility is not None else "N/A"}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    _render_horizon_card(col1, "h1", "Horizon 1 — Immediate (t+1)", 1)
    _render_horizon_card(col2, "h3", "Horizon 3 — Intermediate (t+3)", 3)
    _render_horizon_card(col3, "h5", "Horizon 5 — Strategic (t+5)", 5)

    # --------------------------------------------------------------------------
    # Temporal Risk Trajectories Chart
    # --------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("#### Multi-Horizon Risk Trajectories (Time Series)")

    try:
        temp_data = client.get_temporal_data(limit=100)
        horizons_hist = temp_data.get("horizons", {})
        chart_rows = []
        for h_k, records in horizons_hist.items():
            h_lbl = h_k.upper()
            for r in records:
                if r.get("risk_score") is not None:
                    chart_rows.append({
                        "Timestamp (s)": float(r.get("timestamp_seconds", 0.0)),
                        "Risk Score": float(r["risk_score"]),
                        "Horizon": h_lbl,
                    })

        if chart_rows:
            df_chart = pd.DataFrame(chart_rows).pivot_table(
                index="Timestamp (s)", columns="Horizon", values="Risk Score"
            )
            st.line_chart(df_chart, use_container_width=True)
    except Exception as e:
        st.caption(f"Could not load risk trajectories: {e}")

    # --------------------------------------------------------------------------
    # Risk Classification Threshold Calibration Table
    # --------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("#### Operational Severity Boundaries & Calibration Matrix")
    thresh_data = [
        {"Severity Tier": "LOW", "Risk Interval (S)": "[0.00, 0.35)", "Reliability (R)": "> 65%", "Operational Action": "Nominal perception. Autonomous tracking continues unimpeded."},
        {"Severity Tier": "MEDIUM", "Risk Interval (S)": "[0.35, 0.65)", "Reliability (R)": "35% – 65%", "Operational Action": "Elevated uncertainty. Increase Kalman tracking covariance weights."},
        {"Severity Tier": "HIGH", "Risk Interval (S)": "[0.65, 0.85)", "Reliability (R)": "15% – 35%", "Operational Action": "Probable perception failure. Dispatch P2 early-warning alert."},
        {"Severity Tier": "CRITICAL", "Risk Interval (S)": "[0.85, 1.00]", "Reliability (R)": "< 15%", "Operational Action": "Imminent detection collapse. Trigger P1 critical safety alert."},
    ]
    st.dataframe(pd.DataFrame(thresh_data), use_container_width=True)
