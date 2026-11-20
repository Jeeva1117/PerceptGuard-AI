"""UI components for loading, empty, error, and stale state notifications."""

from __future__ import annotations

import datetime
from typing import Callable, Optional
import streamlit as st


def render_empty_state(title: str, description: str, icon: str = "📭") -> None:
    """Render structured empty state when no data or predictions are available."""
    st.markdown(
        f"""
        <div style="background: #181926; border: 1px dashed #45475a; border-radius: 8px; padding: 24px; text-align: center; margin: 12px 0;">
            <div style="font-size: 32px; margin-bottom: 8px;">{icon}</div>
            <h4 style="color: #cdd6f4; margin: 0 0 6px 0;">{title}</h4>
            <p style="color: #a6adc8; font-size: 13px; margin: 0; line-height: 1.5;">{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_error_boundary(
    title: str,
    error_message: str,
    retry_callback: Optional[Callable[[], None]] = None,
    key_suffix: str = "",
) -> None:
    """Render non-sensitive error notification with optional retry action."""
    st.markdown(
        f"""
        <div style="background: #2a1515; border-left: 4px solid #e76f51; border-radius: 6px; padding: 14px 18px; margin: 12px 0;">
            <h4 style="color: #e76f51; margin: 0 0 4px 0; font-size: 15px;">⚠️ {title}</h4>
            <p style="color: #f8f9fa; font-size: 13px; margin: 0; line-height: 1.5;">{error_message}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if retry_callback:
        if st.button("🔄 Retry Connection", key=f"btn_retry_{key_suffix}"):
            retry_callback()
            st.rerun()


def render_stale_warning(
    last_sync: Optional[datetime.datetime],
    freshness: str = "STALE",
    ttl_seconds: int = 30,
) -> None:
    """Render telemetry staleness indicator if data exceeds freshness threshold."""
    if freshness.upper() == "STALE":
        sync_str = (
            last_sync.strftime("%Y-%m-%d %H:%M:%S UTC")
            if last_sync
            else "Unknown"
        )
        st.markdown(
            f"""
            <div style="background: #332211; border: 1px solid #f4a261; border-radius: 6px; padding: 8px 14px; margin-bottom: 12px; font-size: 12px; color: #f4a261;">
                ⏳ <strong>Telemetry Staleness Notice:</strong> Perception data was not updated within {ttl_seconds}s (Last sync: {sync_str}). Displayed metrics reflect the most recent verified observation.
            </div>
            """,
            unsafe_allow_html=True,
        )
