"""
Tufts Jumbo Weather Forecast — Deep Learning Demo

Usage:
    cd demo && python app.py
"""

import logging
from datetime import timedelta

import gradio as gr

from hrrr_fetch import fetch_hrrr_input
from model_utils import run_forecast, load_model, AVAILABLE_MODELS
from visualization import (
    get_static_maps,
    plot_temperature,
    plot_temperature_placeholder,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── CSS ───────────────────────────────────────────────────────────────

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --font: -apple-system, BlinkMacSystemFont, "SF Pro Display",
            "SF Pro Text", Inter, "Helvetica Neue", Arial, sans-serif;
    --bg: #F2F2F7;
    --card: #FFFFFF;
    --border: #E5E5EA;
    --text: #1D1D1F;
    --muted: #86868B;
    --accent: #0A84FF;
    --dark: #1C1C1E;
}
* { font-family: var(--font) !important; }

.gradio-container {
    max-width: 1320px !important;
    margin: 0 auto !important;
    background: var(--bg) !important;
    padding-bottom: 24px !important;
}

/* ── Top bar ── */
.top-bar {
    background: linear-gradient(135deg, #1C1C1E 0%, #2C2C2E 100%);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.top-bar .title {
    font-size: 24px; font-weight: 700;
    color: #F5F5F7; letter-spacing: -0.3px;
}
.top-bar .subtitle {
    font-size: 13px; color: #98989D;
    margin-top: 2px;
}
.top-bar .location {
    text-align: right;
    font-size: 13px; color: #98989D;
    line-height: 1.6;
}
.top-bar .location b {
    color: #F5F5F7; font-weight: 600;
}

/* ── Hero card ── */
.hero-card {
    background: var(--card);
    border-radius: 16px;
    border: 1px solid var(--border);
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    padding: 32px 36px 28px;
    margin-bottom: 16px;
}
.hero-main {
    display: flex;
    align-items: baseline;
    gap: 20px;
    margin-bottom: 4px;
}
.hero-temp {
    font-size: 64px; font-weight: 300;
    color: var(--text); letter-spacing: -2px;
    line-height: 1;
}
.hero-temp-unit {
    font-size: 28px; font-weight: 400;
    color: var(--muted); margin-left: 2px;
}
.hero-status {
    font-size: 20px; font-weight: 500;
    color: var(--text); padding-left: 8px;
    border-left: 3px solid var(--accent);
}
.hero-metrics {
    display: flex;
    gap: 12px;
    margin: 20px 0 18px;
}
.metric-tile {
    flex: 1;
    background: var(--bg);
    border-radius: 12px;
    padding: 14px 16px;
    text-align: center;
}
.metric-value {
    font-size: 22px; font-weight: 600;
    color: var(--text); line-height: 1.2;
}
.metric-label {
    font-size: 12px; font-weight: 500;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 4px;
}
.hero-meta {
    font-size: 13px; color: var(--muted);
    line-height: 1.6;
}
.hero-meta code {
    background: var(--bg); padding: 2px 6px;
    border-radius: 4px; font-size: 12px;
}
.hero-placeholder {
    text-align: center;
    padding: 36px 0;
    color: var(--muted);
    font-size: 16px; font-weight: 500;
}

/* ── Map section ── */
.maps-heading {
    font-size: 11px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.8px;
    color: var(--muted);
    margin: 8px 0 8px 4px;
}

.map-cell {
    background: var(--card) !important;
    border-radius: 14px !important;
    border: 1px solid var(--border) !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
    overflow: hidden !important;
    min-height: 380px !important;
}

/* ── Controls inside hero ── */
.controls-row {
    display: flex; align-items: end; gap: 10px;
    margin-top: 18px; padding-top: 16px;
    border-top: 1px solid var(--border);
}

/* ── Status ── */
.status-text p, .status-text em {
    font-size: 12px !important; color: var(--muted) !important;
}

/* ── About ── */
.about-section {
    font-size: 13px !important; color: #6E6E73 !important;
    line-height: 1.65 !important;
}

/* ── Button ── */
button.primary {
    background: var(--accent) !important;
    border: none !important; border-radius: 10px !important;
    font-weight: 600 !important; font-size: 15px !important;
    padding: 10px 28px !important;
}
button.primary:hover { background: #0A74E0 !important; }
"""

# ── Helpers ────────────────────────────────────────────────────────────

model_choices = [
    f"{v['display_name']}  ({v['params']})" for v in AVAILABLE_MODELS.values()
]
model_keys = list(AVAILABLE_MODELS.keys())


def _resolve_model(display: str) -> str:
    return model_keys[model_choices.index(display)]


def _hero_placeholder() -> str:
    return (
        '<div class="hero-card">'
        '<div class="hero-placeholder">'
        "Click <b>Run Forecast</b> to fetch real-time HRRR data and generate a 24-hour prediction."
        "</div></div>"
    )


def _hero_html(r: dict, cycle_str: str, forecast_str: str, model_label: str) -> str:
    return (
        '<div class="hero-card">'
        #  temperature + status
        '<div class="hero-main">'
        f'<div><span class="hero-temp">{r["temperature_c"]:.1f}</span>'
        f'<span class="hero-temp-unit">°C</span></div>'
        f'<div class="hero-status">{r["rain_status"]}</div>'
        "</div>"
        #  metric tiles
        '<div class="hero-metrics">'
        f'<div class="metric-tile"><div class="metric-value">{r["temperature_f"]:.0f}°F</div>'
        '<div class="metric-label">Temperature</div></div>'
        f'<div class="metric-tile"><div class="metric-value">{r["humidity_pct"]:.0f}%</div>'
        '<div class="metric-label">Humidity</div></div>'
        f'<div class="metric-tile"><div class="metric-value">{r["wind_speed_ms"]:.1f}</div>'
        f'<div class="metric-label">Wind m/s {r["wind_dir_str"]}</div></div>'
        f'<div class="metric-tile"><div class="metric-value">{r["gust_ms"]:.1f}</div>'
        '<div class="metric-label">Gust m/s</div></div>'
        f'<div class="metric-tile"><div class="metric-value">{r["precipitation_mm"]:.2f}</div>'
        '<div class="metric-label">Precip mm</div></div>'
        "</div>"
        #  meta line
        '<div class="hero-meta">'
        f"Based on &ensp;<code>{cycle_str}</code> &ensp; "
        f"Forecast valid &ensp;<b>{forecast_str}</b> &ensp; "
        f"Model &ensp;<b>{model_label}</b>"
        "</div>"
        "</div>"
    )


# ── Main callback ──────────────────────────────────────────────────────

def do_forecast(model_display: str, progress=gr.Progress()):
    model_name = _resolve_model(model_display)

    progress(0.02, desc="Finding latest HRRR cycle...")
    try:
        input_array, cycle_time = fetch_hrrr_input(
            progress_callback=lambda f, m: progress(f, desc=m),
        )
    except Exception as e:
        raise gr.Error(f"HRRR fetch failed: {e}")

    cycle_str = cycle_time.strftime("%Y-%m-%d %H:%M UTC")
    forecast_time = cycle_time + timedelta(hours=24)
    forecast_str = forecast_time.strftime("%Y-%m-%d %H:%M UTC")

    progress(0.95, desc="Running model inference...")
    try:
        r = run_forecast(model_name, input_array)
    except Exception as e:
        raise gr.Error(f"Inference failed: {e}")

    model_label = model_display.split("(")[0].strip()
    hero = _hero_html(r, cycle_str, forecast_str, model_label)
    temp_fig = plot_temperature(input_array, r, cycle_str, forecast_str)
    status = f"Forecast complete — HRRR cycle {cycle_str}"

    return hero, temp_fig, status


# ── Build UI ──────────────────────────────────────────────────────────

# Pre-render static maps at import time
logger.info("Rendering static basemaps...")
_sat_fig, _street_fig = get_static_maps()
_temp_placeholder = plot_temperature_placeholder()
logger.info("Basemaps ready.")

with gr.Blocks(title="Tufts Jumbo Weather Forecast") as demo:

    # ── Top bar ───────────────────────────────────────────────────
    gr.HTML(
        '<div class="top-bar">'
        '<div>'
        '<div class="title">Tufts Jumbo Weather</div>'
        '<div class="subtitle">Real-time deep-learning forecast</div>'
        "</div>"
        '<div class="location">'
        "<b>Medford, MA</b><br>"
        "42.41°N &ensp; 71.12°W"
        "</div>"
        "</div>"
    )

    # ── Hero card ─────────────────────────────────────────────────
    hero_html = gr.HTML(_hero_placeholder())

    # ── Controls ──────────────────────────────────────────────────
    with gr.Row(elem_classes=["controls-row"]):
        model_dd = gr.Dropdown(
            choices=model_choices, value=model_choices[0],
            label="Model", scale=3,
        )
        run_btn = gr.Button("Run Forecast", variant="primary", scale=1)

    status_bar = gr.Markdown(
        "_Ready — click **Run Forecast**._",
        elem_classes=["status-text"],
    )

    # ── Maps ──────────────────────────────────────────────────────
    gr.HTML('<div class="maps-heading">Coverage Maps — 1 350 km × 1 350 km &ensp; 3 km resolution</div>')

    with gr.Row(equal_height=True):
        sat_plot = gr.Plot(
            value=_sat_fig, label="Satellite",
            elem_classes=["map-cell"],
        )
        street_plot = gr.Plot(
            value=_street_fig, label="Reference Map",
            elem_classes=["map-cell"],
        )
        temp_plot = gr.Plot(
            value=_temp_placeholder, label="Temperature",
            elem_classes=["map-cell"],
        )

    # ── About ─────────────────────────────────────────────────────
    with gr.Accordion("About this demo", open=False):
        gr.Markdown(
            "**Data** &ensp; HRRR 3 km analysis from NOAA (AWS S3, via Herbie). "
            "42 atmospheric channels covering the US Northeast.\n\n"
            "**Models** &ensp; CNN Baseline (11.3 M params) · ResNet-18 (11.2 M params) — "
            "predict 6 weather variables 24 h ahead for a single target point.\n\n"
            "**Course** &ensp; Tufts CS 137 — Deep Neural Networks, Spring 2026",
            elem_classes=["about-section"],
        )

    # ── Callbacks ─────────────────────────────────────────────────
    run_btn.click(
        fn=do_forecast,
        inputs=[model_dd],
        outputs=[hero_html, temp_plot, status_bar],
    )


if __name__ == "__main__":
    logger.info("Pre-loading default model...")
    try:
        load_model(model_keys[0])
        logger.info("Model loaded.")
    except Exception as e:
        logger.warning(f"Pre-load failed: {e}")

    demo.launch(share=False, css=CUSTOM_CSS)
