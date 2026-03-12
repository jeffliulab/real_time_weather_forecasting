# Regional Weather Forecasting with Deep Learning

A deep learning system that forecasts weather conditions 24 hours ahead at a target location in New England, using spatial weather snapshots as input.

## Overview

- **Input**: A spatial weather map of shape `(450, 449, c)` covering the New England area at time *t*
- **Output**: 6 weather variables (temperature, humidity, wind, gust, precipitation) and a binary precipitation label at the target point, 24 hours ahead

## Approach

1. **Baseline CNN** — A 2D CNN that extracts spatial patterns from a single weather snapshot to produce 24h forecasts
2. **Saliency Analysis** — Gradient-based visualization to identify which geographic regions drive the model's predictions
3. **Architecture Comparison** — Systematic comparison of single-frame 2D CNN, multi-frame 2D CNN, 3D CNN, and Vision Transformer to study the effects of temporal context and model architecture on forecast accuracy
4. **Real-Time Demo** — A pipeline that fetches live GFS data from NOAA, runs model inference, and displays the forecast
