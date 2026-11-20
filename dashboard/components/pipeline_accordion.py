"""Pipeline status accordion component (8-stage PerceptGuard AI architecture)."""

from __future__ import annotations

from typing import Any, Dict
import pandas as pd
import streamlit as st

from dashboard.api_client import DashboardApiClient, FrontendAPIError
from dashboard.components.state_views import render_error_boundary


def render_pipeline_status_accordion(client: DashboardApiClient) -> None:
    """Render 8-stage perception pipeline component status accordion."""
    try:
        status_dict = client.get_pipeline_status()
    except FrontendAPIError as err:
        render_error_boundary(
            title="Failed to Load Pipeline Status",
            error_message=f"Backend communication error: {err.message}",
            key_suffix="pipeline_status",
        )
        return

    components = status_dict.get("components", {})

    with st.expander("PerceptGuard AI 8-Stage Architecture Pipeline Status", expanded=False):
        stage_rows = []
        for stage, comp in components.items():
            stage_rows.append({
                "Stage Name": stage.replace("_", " ").title(),
                "Operational Status": comp.get("status", "UNKNOWN"),
                "Details": comp.get("details", ""),
            })
        st.dataframe(pd.DataFrame(stage_rows), use_container_width=True)
