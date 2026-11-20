"""Section D: Early Warning Alerts Feed, Filtering, and Lifecycle Management component."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import pandas as pd
import streamlit as st

from dashboard.api_client import DashboardApiClient, FrontendAPIError
from dashboard.components.state_views import render_empty_state, render_error_boundary


def render_early_warning_alerts(client: DashboardApiClient) -> None:
    """Section D: Early Warning Alerts Feed, Filtering, and Lifecycle Management."""
    st.subheader("Section D — Early Warning Alerts & Incident Feed")

    # Filter controls
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    with col_f1:
        sel_sev = st.selectbox(
            "Filter Severity",
            options=["ALL", "CRITICAL", "WARNING", "CAUTION", "INFO", "DATA_QUALITY_WARNING"],
            index=0,
            key="filter_sev",
        )
    with col_f2:
        sel_horiz = st.selectbox(
            "Filter Horizon",
            options=["ALL", "H1 (t+1)", "H3 (t+3)", "H5 (t+5)"],
            index=0,
            key="filter_horiz",
        )
    with col_f3:
        sel_type = st.selectbox(
            "Filter Alert Type",
            options=[
                "ALL",
                "RISK_LEVEL_THRESHOLD",
                "RAPID_RISK_INCREASE",
                "VOLATILITY_SPIKE",
                "CONSECUTIVE_DEGRADATION",
                "HORIZON_DIVERGENCE",
                "DATA_QUALITY",
            ],
            index=0,
            key="filter_type",
        )
    with col_f4:
        sel_status = st.selectbox(
            "Filter Lifecycle State",
            options=["ALL", "ACTIVE", "ACKNOWLEDGED", "RESOLVED", "EXPIRED"],
            index=0,
            key="filter_status",
        )

    # Parse filters
    sev_filter = None if sel_sev == "ALL" else sel_sev
    horizon_filter = None
    if sel_horiz == "H1 (t+1)":
        horizon_filter = 1
    elif sel_horiz == "H3 (t+3)":
        horizon_filter = 3
    elif sel_horiz == "H5 (t+5)":
        horizon_filter = 5
    type_filter = None if sel_type == "ALL" else sel_type
    status_filter = None if sel_status == "ALL" else sel_status

    try:
        alerts_res = client.get_alerts(
            severity=sev_filter,
            horizon=horizon_filter,
            alert_type=type_filter,
            status=status_filter,
            page=1,
            page_size=50,
        )
    except FrontendAPIError as err:
        render_error_boundary(
            title="Failed to Load Early Warning Alerts",
            error_message=f"Backend communication error: {err.message}",
            retry_callback=st.rerun,
            key_suffix="alerts_feed",
        )
        return

    alerts = alerts_res.get("alerts", [])
    total_count = alerts_res.get("total_count", 0)
    active_count = alerts_res.get("active_alerts_count", 0)

    st.markdown(
        f"**Showing {len(alerts)} of {total_count} matching alerts** (Active: `{active_count}`)"
    )

    if not alerts:
        render_empty_state(
            title="No Matching Alerts",
            message="No early-warning alerts match the selected filter criteria or no alerts have been generated.",
            icon="🔔",
        )
        return

    # Table view
    table_rows = []
    for a in alerts:
        risk_score = a.get("risk_score")
        score_str = f"{float(risk_score):.4f}" if risk_score is not None else "N/A"
        ts = a.get("timestamp_seconds", 0.0)
        table_rows.append({
            "Alert ID": a.get("alert_id", "UNKNOWN"),
            "Severity": a.get("severity", "UNKNOWN"),
            "Type": a.get("alert_type", "UNKNOWN"),
            "Horizon": f"H{a.get('trigger_horizon', 1)}",
            "Score": score_str,
            "State": a.get("lifecycle_state", "UNKNOWN"),
            "Time (s)": f"{float(ts):.3f}" if ts is not None else "0.000",
            "Message": (a.get("warning_message") or "")[:60] + "...",
        })
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

    # Detail Inspection & Management
    st.markdown("#### Alert Detail Inspection & Operator Actions")
    alert_ids = [a["alert_id"] for a in alerts]
    selected_alert_id = st.selectbox(
        "Select Alert to Inspect / Manage", options=alert_ids, key="inspect_alert_select"
    )

    if selected_alert_id:
        try:
            detail = client.get_alert_detail(selected_alert_id)
        except FrontendAPIError as err:
            render_error_boundary(
                title=f"Failed to Inspect Alert {selected_alert_id}",
                error_message=f"Backend communication error: {err.message}",
                key_suffix=f"alert_detail_{selected_alert_id}",
            )
            return

        if detail:
            col_d1, col_d2 = st.columns([2, 1])
            with col_d1:
                st.markdown(
                    f"""
                    **Alert ID:** `{detail.get('alert_id')}`<br/>
                    **Rule Triggered:** `{detail.get('rule_id')}` | **Type:** `{detail.get('alert_type')}`<br/>
                    **Severity:** `{detail.get('severity')}` | **Status:** `{detail.get('lifecycle_state')}`<br/>
                    **Message:** {detail.get('warning_message')}<br/>
                    **Trigger Rationale:** {detail.get('trigger_rationale', 'N/A')}<br/>
                    **Evidence:** Frame `{detail.get('source_frame_id')}` &rarr; Target `{detail.get('target_frame_id')}` | Model: `{detail.get('model_version')}`
                    """,
                    unsafe_allow_html=True,
                )
            with col_d2:
                st.markdown("**Operator Lifecycle Actions**")
                curr_state = detail.get("lifecycle_state", "UNKNOWN")
                can_ack = curr_state == "ACTIVE"
                can_resolve = curr_state in ("ACTIVE", "ACKNOWLEDGED")

                c_act1, c_act2 = st.columns(2)
                with c_act1:
                    if st.button("Acknowledge", disabled=not can_ack, key=f"btn_ack_{selected_alert_id}"):
                        try:
                            ack_res = client.acknowledge_alert(
                                selected_alert_id, operator_id="dashboard_operator"
                            )
                            if ack_res.get("acknowledged"):
                                st.success("Alert Acknowledged!")
                                st.rerun()
                            else:
                                st.error("Failed to acknowledge alert.")
                        except FrontendAPIError as err:
                            st.error(f"Action failed: {err.message}")

                with c_act2:
                    if st.button("Resolve", disabled=not can_resolve, key=f"btn_res_{selected_alert_id}"):
                        try:
                            res_res = client.resolve_alert(
                                selected_alert_id,
                                reason="Resolved from dashboard",
                                operator_id="dashboard_operator",
                            )
                            if res_res.get("resolved"):
                                st.success("Alert Resolved!")
                                st.rerun()
                            else:
                                st.error("Failed to resolve alert.")
                        except FrontendAPIError as err:
                            st.error(f"Action failed: {err.message}")

            # Lifecycle history audit trail
            events = detail.get("lifecycle_events", [])
            if events:
                with st.expander(f"Audit Trail ({len(events)} lifecycle events)", expanded=False):
                    st.dataframe(pd.DataFrame(events), use_container_width=True)
