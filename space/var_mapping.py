"""
Mapping from 42-channel VAR_LEVELS to HRRR GRIB2 Herbie search strings.

Each entry corresponds to one channel in the model input, following the exact
order defined in data_preparation/data_spec.py (VAR_LEVELS = TARGET_VARS + ATMOS_VARS).
"""

# fmt: off
HRRR_MAPPING = [
    # ── Target / surface variables (channels 0-6) ──────────────────────
    {"name": "TMP@2m_above_ground",         "search": ":TMP:2 m above ground",            "product": "sfc", "fxx": 0},
    {"name": "RH@2m_above_ground",          "search": ":RH:2 m above ground",             "product": "sfc", "fxx": 0},
    {"name": "UGRD@10m_above_ground",       "search": ":UGRD:10 m above ground",          "product": "sfc", "fxx": 0},
    {"name": "VGRD@10m_above_ground",       "search": ":VGRD:10 m above ground",          "product": "sfc", "fxx": 0},
    {"name": "GUST@surface",                "search": ":GUST:surface",                     "product": "sfc", "fxx": 0},
    {"name": "DSWRF@surface",               "search": ":DSWRF:surface",                    "product": "sfc", "fxx": 0},
    # APCP is accumulated; not in analysis (fxx=0). Use 1-hour forecast from same cycle.
    {"name": "APCP_1hr_acc_fcst@surface",   "search": ":APCP:surface:0-1 hour acc fcst",   "product": "sfc", "fxx": 1},

    # ── Atmospheric variables (channels 7-41) ──────────────────────────
    # CAPE
    {"name": "CAPE@surface",                "search": ":CAPE:surface",                     "product": "sfc", "fxx": 0},

    # Dew point temperature at pressure levels
    {"name": "DPT@1000mb",                  "search": ":DPT:1000 mb",                     "product": "prs", "fxx": 0},
    {"name": "DPT@500mb",                   "search": ":DPT:500 mb",                      "product": "prs", "fxx": 0},
    {"name": "DPT@700mb",                   "search": ":DPT:700 mb",                      "product": "prs", "fxx": 0},
    {"name": "DPT@850mb",                   "search": ":DPT:850 mb",                      "product": "prs", "fxx": 0},
    {"name": "DPT@925mb",                   "search": ":DPT:925 mb",                      "product": "prs", "fxx": 0},

    # Geopotential height at pressure levels + surface
    {"name": "HGT@1000mb",                  "search": ":HGT:1000 mb",                     "product": "prs", "fxx": 0},
    {"name": "HGT@500mb",                   "search": ":HGT:500 mb",                      "product": "prs", "fxx": 0},
    {"name": "HGT@700mb",                   "search": ":HGT:700 mb",                      "product": "prs", "fxx": 0},
    {"name": "HGT@850mb",                   "search": ":HGT:850 mb",                      "product": "prs", "fxx": 0},
    {"name": "HGT@surface",                 "search": ":HGT:surface",                     "product": "sfc", "fxx": 0},

    # Temperature at pressure levels
    {"name": "TMP@1000mb",                  "search": ":TMP:1000 mb",                     "product": "prs", "fxx": 0},
    {"name": "TMP@500mb",                   "search": ":TMP:500 mb",                      "product": "prs", "fxx": 0},
    {"name": "TMP@700mb",                   "search": ":TMP:700 mb",                      "product": "prs", "fxx": 0},
    {"name": "TMP@850mb",                   "search": ":TMP:850 mb",                      "product": "prs", "fxx": 0},
    {"name": "TMP@925mb",                   "search": ":TMP:925 mb",                      "product": "prs", "fxx": 0},

    # U-component wind at pressure levels
    {"name": "UGRD@1000mb",                 "search": ":UGRD:1000 mb",                    "product": "prs", "fxx": 0},
    {"name": "UGRD@250mb",                  "search": ":UGRD:250 mb",                     "product": "prs", "fxx": 0},
    {"name": "UGRD@500mb",                  "search": ":UGRD:500 mb",                     "product": "prs", "fxx": 0},
    {"name": "UGRD@700mb",                  "search": ":UGRD:700 mb",                     "product": "prs", "fxx": 0},
    {"name": "UGRD@850mb",                  "search": ":UGRD:850 mb",                     "product": "prs", "fxx": 0},
    {"name": "UGRD@925mb",                  "search": ":UGRD:925 mb",                     "product": "prs", "fxx": 0},

    # V-component wind at pressure levels
    {"name": "VGRD@1000mb",                 "search": ":VGRD:1000 mb",                    "product": "prs", "fxx": 0},
    {"name": "VGRD@250mb",                  "search": ":VGRD:250 mb",                     "product": "prs", "fxx": 0},
    {"name": "VGRD@500mb",                  "search": ":VGRD:500 mb",                     "product": "prs", "fxx": 0},
    {"name": "VGRD@700mb",                  "search": ":VGRD:700 mb",                     "product": "prs", "fxx": 0},
    {"name": "VGRD@850mb",                  "search": ":VGRD:850 mb",                     "product": "prs", "fxx": 0},
    {"name": "VGRD@925mb",                  "search": ":VGRD:925 mb",                     "product": "prs", "fxx": 0},

    # Cloud cover
    {"name": "TCDC@entire_atmosphere",      "search": ":TCDC:entire atmosphere",           "product": "sfc", "fxx": 0},
    {"name": "HCDC@high_cloud_layer",       "search": ":HCDC:high cloud layer",            "product": "sfc", "fxx": 0},
    {"name": "MCDC@middle_cloud_layer",     "search": ":MCDC:middle cloud layer",          "product": "sfc", "fxx": 0},
    {"name": "LCDC@low_cloud_layer",        "search": ":LCDC:low cloud layer",             "product": "sfc", "fxx": 0},

    # Moisture
    {"name": "PWAT@entire_atmosphere_single_layer", "search": ":PWAT:entire atmosphere",   "product": "sfc", "fxx": 0},
    {"name": "RHPW@entire_atmosphere",      "search": ":RHPW:entire atmosphere",           "product": "sfc", "fxx": 0},
    {"name": "VIL@entire_atmosphere",       "search": ":VIL:entire atmosphere",            "product": "sfc", "fxx": 0},
]
# fmt: on

assert len(HRRR_MAPPING) == 42, f"Expected 42 channels, got {len(HRRR_MAPPING)}"

# New England subgrid slice in the full HRRR CONUS grid (from data_spec.py)
NE_Y_SLICE = slice(600, 1050)   # 450 rows
NE_X_SLICE = slice(1350, 1799)  # 449 columns

# Jumbo Statue grid location within the NE subgrid
JUMBO_ROW = 177
JUMBO_COL = 263
