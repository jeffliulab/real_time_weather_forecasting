"""
Forecast visualization — satellite, street, and temperature maps.

Satellite and reference maps are static (rendered once at startup).
Temperature map updates each time a forecast is run.
"""

import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt

from var_mapping import JUMBO_ROW, JUMBO_COL

logger = logging.getLogger(__name__)

# ── Projection & coordinates (from data_spec.py) ──────────────────────

PROJ = ccrs.LambertConformal(
    central_longitude=262.5,
    central_latitude=38.5,
    standard_parallels=(38.5, 38.5),
    globe=ccrs.Globe(semimajor_axis=6371229, semiminor_axis=6371229),
)

_x = 1352479.8574780696 + np.arange(449) * 3000
_y = 212693.8474433364 + np.arange(450) * 3000
EXTENT = [_x[0], _x[-1], _y[0], _y[-1]]

JUMBO_LON, JUMBO_LAT = -71.1204, 42.4078
X_GRID, Y_GRID = np.meshgrid(_x, _y)

CITIES = [
    ("Boston",      42.36, -71.06),
    ("Providence",  41.82, -71.41),
    ("Hartford",    41.76, -72.68),
    ("Portland",    43.66, -70.26),
    ("Burlington",  44.48, -73.21),
    ("Concord",     43.21, -71.54),
    ("Albany",       42.65, -73.76),
    ("New York",    40.71, -74.01),
    ("Montreal",    45.50, -73.57),
]


# ── Tile sources ──────────────────────────────────────────────────────

class _EsriSatellite(cimgt.GoogleWTS):
    def _image_url(self, tile):
        x, y, z = tile
        return (
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            f"World_Imagery/MapServer/tile/{z}/{y}/{x}"
        )


class _EsriStreetMap(cimgt.GoogleWTS):
    def _image_url(self, tile):
        x, y, z = tile
        return (
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            f"World_Street_Map/MapServer/tile/{z}/{y}/{x}"
        )


# ── Shared style ─────────────────────────────────────────────────────

_MARKER = dict(
    marker="*", color="#FF3B30", markersize=14,
    markeredgecolor="white", markeredgewidth=1.0,
    transform=ccrs.PlateCarree(), zorder=20,
)

_TAG = dict(
    fontsize=8.5, fontweight="bold", fontfamily="sans-serif",
    color="white", transform=ccrs.PlateCarree(), zorder=25,
    bbox=dict(boxstyle="round,pad=0.25", fc="#1C1C1E", ec="none", alpha=0.80),
)


def _make_ax(fig_or_ax=None, figsize=(7.2, 6.8)):
    """Create a single GeoAxes with consistent extent."""
    if fig_or_ax is None:
        fig, ax = plt.subplots(
            figsize=figsize, subplot_kw={"projection": PROJ},
        )
    else:
        fig, ax = fig_or_ax.figure, fig_or_ax
    ax.set_extent(EXTENT, crs=PROJ)
    return fig, ax


# ── Individual map renderers ─────────────────────────────────────────

def plot_satellite() -> Figure:
    """Render satellite basemap (static, no weather data needed)."""
    fig, ax = _make_ax()
    try:
        ax.add_image(_EsriSatellite(), 7)
    except Exception:
        ax.add_feature(cfeature.LAND, facecolor="#5C4A32")
        ax.add_feature(cfeature.OCEAN, facecolor="#1B3A4B")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6, color="white")
        ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor="#aaa")
    ax.plot(JUMBO_LON, JUMBO_LAT, **_MARKER)
    ax.text(JUMBO_LON + 0.35, JUMBO_LAT + 0.25, "Jumbo", **_TAG)
    ax.set_title(
        "Satellite", fontsize=12, fontweight="600",
        fontfamily="sans-serif", pad=8, color="#1D1D1F",
    )
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.93)
    return fig


def plot_street() -> Figure:
    """Render street / reference basemap (static)."""
    fig, ax = _make_ax()
    try:
        ax.add_image(_EsriStreetMap(), 7)
    except Exception:
        ax.add_feature(cfeature.LAND, facecolor="#E8E4D8")
        ax.add_feature(cfeature.OCEAN, facecolor="#AAD3DF")
        ax.add_feature(cfeature.LAKES, facecolor="#AAD3DF", edgecolor="#888", linewidth=0.3)
        ax.add_feature(cfeature.RIVERS, edgecolor="#AAD3DF", linewidth=0.4)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle="--")
        ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor="#888")
        pc = ccrs.PlateCarree()
        for name, lat, lon in CITIES:
            ax.text(
                lon, lat, name,
                fontsize=7, fontfamily="sans-serif", fontweight="500",
                color="#333", transform=pc, zorder=15,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
            )
    ax.plot(JUMBO_LON, JUMBO_LAT, **_MARKER)
    ax.text(JUMBO_LON + 0.35, JUMBO_LAT + 0.25, "Jumbo", **_TAG)
    ax.set_title(
        "Reference Map", fontsize=12, fontweight="600",
        fontfamily="sans-serif", pad=8, color="#1D1D1F",
    )
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.93)
    return fig


def plot_temperature(
    input_array: np.ndarray,
    forecast: dict,
    cycle_str: str,
    forecast_str: str,
) -> Figure:
    """Render 2 m temperature map with forecast annotation."""
    fig, ax = _make_ax()

    temp_field = input_array[:, :, 0] - 273.15
    masked = np.ma.masked_invalid(temp_field)

    im = ax.pcolormesh(
        X_GRID, Y_GRID, masked,
        cmap="RdYlBu_r", shading="auto", transform=PROJ, zorder=5,
    )
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color="#444", zorder=10)
    ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor="#666", zorder=10)

    cbar = fig.colorbar(im, ax=ax, shrink=0.72, pad=0.03, aspect=28)
    cbar.set_label("°C", fontsize=10, fontfamily="sans-serif")
    cbar.ax.tick_params(labelsize=8)

    ax.plot(JUMBO_LON, JUMBO_LAT, **_MARKER)

    temp_c = forecast["temperature_c"]
    temp_f = forecast["temperature_f"]
    label = f"Forecast {forecast_str}\n{temp_c:+.1f} °C  /  {temp_f:.0f} °F"
    ax.text(
        JUMBO_LON + 0.45, JUMBO_LAT + 0.35, label,
        fontsize=8.5, fontweight="bold", fontfamily="sans-serif",
        color="white", transform=ccrs.PlateCarree(), zorder=25,
        bbox=dict(boxstyle="round,pad=0.35", fc="#1C1C1E", ec="white", alpha=0.88, lw=0.8),
    )

    ax.set_title(
        f"2 m Temperature — {cycle_str}",
        fontsize=12, fontweight="600",
        fontfamily="sans-serif", pad=8, color="#1D1D1F",
    )
    fig.subplots_adjust(left=0.02, right=0.95, bottom=0.02, top=0.93)
    return fig


def plot_temperature_placeholder() -> Figure:
    """Empty temperature panel shown before first forecast."""
    fig, ax = _make_ax()
    ax.add_feature(cfeature.LAND, facecolor="#E8E4D8", zorder=1)
    ax.add_feature(cfeature.OCEAN, facecolor="#D6EAF0", zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color="#999", zorder=2)
    ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor="#bbb", zorder=2)
    ax.plot(JUMBO_LON, JUMBO_LAT, **_MARKER)
    ax.text(
        JUMBO_LON + 0.35, JUMBO_LAT + 0.25, "Jumbo", **_TAG,
    )
    ax.text(
        0.5, 0.50, "Click  Run Forecast",
        transform=ax.transAxes, ha="center", va="center",
        fontsize=14, fontweight="600", fontfamily="sans-serif",
        color="#86868B",
    )
    ax.set_title(
        "2 m Temperature", fontsize=12, fontweight="600",
        fontfamily="sans-serif", pad=8, color="#1D1D1F",
    )
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.93)
    return fig


# ── Startup cache ─────────────────────────────────────────────────────

_cache = {}


def get_static_maps() -> tuple[Figure, Figure]:
    """Return cached satellite and street map figures (rendered once)."""
    if "satellite" not in _cache:
        logger.info("Rendering satellite basemap...")
        _cache["satellite"] = plot_satellite()
    if "street" not in _cache:
        logger.info("Rendering reference basemap...")
        _cache["street"] = plot_street()
    return _cache["satellite"], _cache["street"]
