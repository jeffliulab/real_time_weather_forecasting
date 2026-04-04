---
title: Tufts Jumbo Weather Forecast
emoji: "\U0001F324"
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "5.23.0"
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
models:
- jeffliulab/weather-forecasting-v1
---

# Tufts Jumbo — 24h Weather Forecast

Real-time deep-learning weather prediction for the Jumbo Statue at Tufts University.

## How It Works

1. **Fetches** the latest HRRR 3 km analysis data from NOAA (42 atmospheric channels, 450x449 grid covering the US Northeast)
2. **Runs** a trained CNN through the spatial snapshot
3. **Predicts** 6 weather variables 24 hours ahead at a single target point (Jumbo Statue, Medford MA)

## Models

| Model | Parameters | Architecture |
|-------|-----------|-------------|
| CNN Baseline | 11.3M | 6 residual blocks, progressive spatial downsampling |
| ResNet-18 | 11.2M | Modified torchvision ResNet-18 (42-channel input) |

## Input Channels (42)

Surface: 2m temperature, 2m humidity, 10m U/V wind, surface gust, solar radiation, 1hr precipitation.
Atmospheric: CAPE, dew point (5 levels), geopotential height (5 levels), temperature (5 levels),
U-wind (6 levels), V-wind (6 levels), cloud cover (4 layers), precipitable water, relative humidity, VIL.

## Output Variables

Temperature (K), Relative Humidity (%), U-Wind (m/s), V-Wind (m/s), Wind Gust (m/s), Precipitation (mm).

## Data Source

[HRRR (High-Resolution Rapid Refresh)](https://rapidrefresh.noaa.gov/hrrr/) — NOAA's 3 km hourly weather model, fetched in real-time from AWS S3 via [Herbie](https://herbie.readthedocs.io/).

## Links

- [Model Weights](https://huggingface.co/jeffliulab/weather-forecasting-v1) — CNN Baseline + ResNet-18 checkpoints
- [GitHub Repository](https://github.com/jeffliulab/real_time_weather_forecasting) — Full project code

## Course

Tufts CS 137 — Deep Neural Networks, Spring 2026
