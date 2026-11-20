"""Section C: Temporal Risk Monitoring & Trajectory Visualization component."""

from __future__ import annotations

from typing import Any, Dict, List
import pandas as pd
import streamlit as st

from dashboard.api_client import DashboardApiClient, FrontendAPIError
from dashboard.components.state_views import render_empty_state, render_error_boundary


def render_temporal_monitoring(client: DashboardApiClient) -> None:
    """Section C: Temporal Risk Monitoring & History Visualization."""
    st.subheader("Section C — Temporal Risk Monitoring")

    try:
        temp_data = client.get_temporal_data(limit=100)
    except FrontendAPIError as err:
        render_error_boundary(
            title="Failed to Load Temporal Risk Data",
            error_message=f"Backend communication error: {err.message}",
            retry_callback=st.rerun,
            key_suffix="temporal_monitoring",
        )
        return

    horizons_hist = temp_data.get("horizons", {})
    transitions = temp_data.get("state_transitions", [])

    # Check if we have any historical records
    has_history = any(len(records) > 0 for records in horizons_hist.values())

    if not has_history:
        render_empty_state(
            title="No Temporal Monitoring History Recorded",
            message="No temporal monitoring history recorded yet. Ingest predictions to populate trajectory charts.",
            icon="📈",
        )
        return

    # Prepare DataFrame for line chart
    chart_rows = []
    for h_key, records in horizons_hist.items():
        h_name = h_key.upper()
        for r in records:
            if r.get("risk_score") is not None:
                chart_rows.append({
                    "Timestamp (s)": float(r.get("timestamp_seconds", 0.0)),
                    "Risk Score": float(r["risk_score"]),
                    "Horizon": h_name,
                })

    if chart_rows:
        df_chart = pd.DataFrame(chart_rows)
        df_pivot = df_chart.pivot_table(
            index="Timestamp (s)", columns="Horizon", values="Risk Score"
        )
        st.line_chart(df_pivot, use_container_width=True)
    else:
        st.warning("⚠️ No valid non-null risk observations available for plotting.")

    # State transitions expander
    with st.expander(f"Operational State Transitions ({len(transitions)} events)", expanded=False):
        if transitions:
            df_trans = pd.DataFrame([
                {
                    "Timestamp (s)": f"{float(t.get('timestamp_seconds', 0.0)):.3f}",
                    "Horizon": f"H{t.get('horizon')}",
                    "From Level": t.get("previous_level", "UNKNOWN"),
                    "To Level": t.get("current_level", "UNKNOWN"),
                    "Risk Score": f"{float(t.get('risk_score', 0.0)):.4f}",
                    "Genuine": "Yes" if t.get("is_genuine_transition", True) else "No (Jitter)",
                    "Notes": t.get("notes", ""),
                }
                for t in reversed(transitions[-20:])
            ])
            st.dataframe(df_trans, use_container_width=True)
        else:
            st.caption("No operational state transitions recorded.")
