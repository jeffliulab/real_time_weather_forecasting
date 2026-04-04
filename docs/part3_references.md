# Part 3 Related Work: Multi-Frame and Temporal Deep Learning for Weather Forecasting

## 1. Overview

Our Part 3 investigation explores whether incorporating multiple consecutive time frames improves CNN-based weather forecasting compared to single-frame models. We implemented three multi-frame architectures (MultiFrameCNN, CNN3D, WeatherViT) and compared them against single-frame baselines. Surprisingly, all multi-frame models underperformed the single-frame ResNet-18 (AUC 0.768). Below we survey the state-of-the-art literature to understand why, and how leading approaches handle temporal information differently.

## 2. Key Papers and Their Temporal Modeling Strategies

### 2.1 Pangu-Weather — Single-Frame with Hierarchical Temporal Aggregation

**Citation:** Bi, K., Xie, L., Zhang, H., Chen, X., Gu, X., & Tian, Q. (2023). *Pangu-Weather: A 3D High-Resolution Model for Fast and Accurate Global Weather Forecast.* Nature, 619, 533-538.
**Link:** [https://arxiv.org/abs/2211.02556](https://arxiv.org/abs/2211.02556) | [Nature](https://www.nature.com/articles/s41586-023-06185-3)

**Approach:** Pangu-Weather uses a **single weather state** as input and predicts a future state. It trains four separate models for different lead times (1h, 3h, 6h, 24h), then uses a greedy algorithm to chain them for longer forecasts. The architecture is a 3D Earth-Specific Transformer (3DEST) with ~256M parameters, trained on 43 years of ERA5 reanalysis data.

**Relevance to our work:** Pangu-Weather demonstrates that a single input frame is sufficient for accurate weather prediction when the model is large enough and trained on sufficient data. Rather than stacking multiple input frames, temporal coverage is achieved through **hierarchical temporal aggregation** — calling models with the largest affordable lead time and chaining predictions. This avoids the parameter explosion of multi-frame channel stacking.

---

### 2.2 GraphCast — Two-Frame Input with Tendency Learning

**Citation:** Lam, R., Sanchez-Gonzalez, A., Willson, M., Wirnsberger, P., Fortunato, M., Alet, F., ... & Battaglia, P. (2023). *GraphCast: Learning skillful medium-range global weather forecasting.* Science, 382(6677), 1416-1421.
**Link:** [https://arxiv.org/abs/2212.12794](https://arxiv.org/abs/2212.12794) | [Science](https://www.science.org/doi/10.1126/science.adi2336)

**Approach:** GraphCast takes **exactly two consecutive weather states** as input and predicts 6 hours ahead using a Graph Neural Network (GNN) in an encode-process-decode configuration (36.7M parameters). The model is autoregressive — its predictions are fed back as input for longer forecasts. During training, the number of autoregressive rollout steps is incrementally increased from 1 to 12 (6 hours to 3 days). Importantly, **three input states did not improve performance enough to justify the increased memory footprint**, so only two frames are used.

**Relevance to our work:** GraphCast's design validates the idea that temporal information is useful, but **two frames suffice**. The key insight is that two consecutive states implicitly encode the **tendency (time derivative)** of the atmospheric state. Our 4-frame channel stacking introduces 4x parameters without proportional benefit. A better approach would be to compute `frame[t] - frame[t-1]` explicitly and concatenate it with `frame[t]`, providing tendency information with only 2x channels (84 vs 168).

---

### 2.3 FourCastNet — Single-Frame with Fourier Neural Operators

**Citation:** Pathak, J., Subramanian, S., Harrington, P., Raja, S., Chattopadhyay, A., Mardani, M., ... & Anandkumar, A. (2022). *FourCastNet: A Global Data-driven High-resolution Weather Model using Adaptive Fourier Neural Operators.* arXiv:2202.11214.
**Link:** [https://arxiv.org/abs/2202.11214](https://arxiv.org/abs/2202.11214)

**Approach:** FourCastNet uses a Vision Transformer (ViT) backbone with **Adaptive Fourier Neural Operators (AFNO)** replacing standard self-attention. AFNO performs token mixing in the Fourier domain, providing a global receptive field efficiently. The model operates on a **single frame**, predicting the next frame, then rolling out autoregressively. It generates a week-long forecast in less than 2 seconds at 0.25-degree resolution.

**Relevance to our work:** FourCastNet shows that even with a single-frame input, a ViT-based architecture can capture long-range spatial dependencies crucial for weather prediction. Our WeatherViT uses standard self-attention on patches, which is less efficient than AFNO for weather data. AFNO's global frequency-domain mixing could better capture the large-scale synoptic patterns that our saliency analysis found to be important (saliency concentrated at domain edges/boundaries).

---

### 2.4 GenCast — Two-Frame Diffusion-Based Ensemble Forecasting

**Citation:** Price, I., Sanchez-Gonzalez, A., Alet, F., Andersson, T. R., El-Kadi, A., Masters, D., ... & Willson, M. (2024). *GenCast: Diffusion-based ensemble forecasting for medium-range weather.* Nature, 636, 1038-1045.
**Link:** [https://arxiv.org/abs/2312.15796](https://arxiv.org/abs/2312.15796) | [Nature](https://www.nature.com/articles/s41586-024-08252-9)

**Approach:** GenCast extends GraphCast's approach by using a **conditional diffusion model** on the sphere. Like GraphCast, it conditions on **two consecutive past observations** to generate probabilistic ensemble forecasts. It produces 15-day global forecasts at 12-hour steps, outperforming ECMWF's ENS on 97.4% of 1,320 evaluation targets.

**Relevance to our work:** GenCast further confirms the "two frames are enough" principle. More importantly, it demonstrates that the **uncertainty** in weather prediction is as important as the point forecast. Our models only produce deterministic predictions; probabilistic approaches could better handle the inherent chaos in atmospheric dynamics, especially for precipitation (our most challenging target).

---

### 2.5 HRRRCast — ResNet-Based HRRR Emulation (Most Comparable to Our Work)

**Citation:** Abdi, D., Jankov, I., Madden, P., Vargas, V., Smith, T. A., Frolov, S., Flora, M., & Potvin, C. (2025). *HRRRCast: a data-driven emulator for regional weather forecasting at convection allowing scales.* arXiv:2507.05658.
**Link:** [https://arxiv.org/abs/2507.05658](https://arxiv.org/abs/2507.05658)

**Approach:** HRRRCast is the most directly comparable work to ours, as it also works with **HRRR data** at convection-allowing resolution. It uses a ResNet-based architecture (ResHRRR) enhanced with:
- **Squeeze-and-Excitation (SE) blocks** for channel-wise attention (learning which weather variables matter most)
- **Feature-wise Linear Modulation (FiLM)** for lead-time conditioning
- **Single-step prediction** with greedy rollout for longer forecasts (not multi-frame input stacking)

**Relevance to our work:** HRRRCast validates our finding that ResNet is a strong baseline for HRRR-based forecasting. The SE blocks and FiLM conditioning it adds are lightweight modifications that could significantly improve our ResNet-18 without the complexity and data requirements of multi-frame architectures. Critically, HRRRCast also uses single-step prediction rather than multi-frame input stacking.

---

### 2.6 StormCast — Generative Diffusion for km-Scale HRRR Emulation

**Citation:** Pathak, J., Cohen, Y., Garg, P., Harrington, P., Brenowitz, N., Durran, D., ... & Pritchard, M. (2024). *Kilometer-Scale Convection Allowing Model Emulation using Generative Diffusion Modeling.* arXiv:2408.10958.
**Link:** [https://arxiv.org/abs/2408.10958](https://arxiv.org/abs/2408.10958)

**Approach:** StormCast uses generative diffusion modeling to emulate NOAA's HRRR model at kilometer scale, producing realistic convective dynamics. The model generates probabilistic forecasts with sharp spatial detail, avoiding the blurring artifacts common in deterministic models for precipitation prediction.

**Relevance to our work:** StormCast demonstrates that for km-scale weather prediction (like our HRRR New England domain), generative models can better capture the spatial structure of extreme weather events. Our deterministic models struggle with precipitation forecasting (APCP RMSE remains high across all architectures), which could be partly due to the inherent stochasticity of convective precipitation.

---

### 2.7 MetNet-3 — Lead-Time Conditioned Regional Forecasting

**Citation:** Andrychowicz, M., Espeholt, L., Li, D., Merchant, S., Merose, A., Zyda, F., Agrawal, S., & Kalchbrenner, N. (2023). *Deep Learning for Day Forecasts from Sparse Observations.* arXiv:2306.06079.
**Link:** [https://arxiv.org/abs/2306.06079](https://arxiv.org/abs/2306.06079)

**Approach:** MetNet-3 is a regional weather prediction model (operational in Google Search) that ingests **multiple temporal inputs** from different sources: radar images from the past 90 minutes, satellite data, and station observations from the past 6 hours. Crucially, it encodes the **forecast lead time** directly as a sinusoidal embedding input, allowing a single model to produce forecasts for any lead time from minutes to 24 hours.

**Relevance to our work:** MetNet-3's approach to temporal information is fundamentally different from our channel stacking: rather than blindly concatenating frames, it uses structured temporal encoding and lead-time conditioning. The lead-time embedding technique (even with a fixed 24h target) could condition our network to better understand the prediction horizon.

---

### 2.8 KARINA — Efficient Global Weather Forecast with ConvNeXt + SE

**Citation:** Cheon, M., Choi, Y., Kang, S., Choi, Y., Lee, J., & Kang, D. (2024). *KARINA: An Efficient Deep Learning Model for Global Weather Forecast.* arXiv:2403.10555.
**Link:** [https://arxiv.org/abs/2403.10555](https://arxiv.org/abs/2403.10555)

**Approach:** KARINA uses **ConvNeXt architecture** combined with **SENet (Squeeze-and-Excitation)** blocks and **Geocyclic Padding** for atmospheric continuity. Trained at 2.5-degree resolution on only 4 A100 GPUs in under 12 hours, it matches performance of models trained on 100x higher-resolution data, surpassing ECMWF S2S reforecasts up to 7 days.

**Relevance to our work:** KARINA directly validates our architecture choices (we used ConvNeXt-Tiny as one of our baselines). The addition of SE blocks for channel attention is a simple but effective enhancement that could help our models learn which weather variables are most predictive for each target.

---

### 2.9 CNN-Transformer Spatiotemporal Weather Prediction

**Citation:** Wu, R., Liang, Y., Lin, L., & Zhang, Z. (2024). *Spatiotemporal Multivariate Weather Prediction Network Based on CNN-Transformer.* Sensors, 24(23), 7837.
**Link:** [https://pmc.ncbi.nlm.nih.gov/articles/PMC11644947/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11644947/)

**Approach:** STWPM combines CNN and Transformer architectures with a multi-scale spatiotemporal evolution module that captures patterns from both **intra-frame** and **inter-frame** perspectives. Inter-frame relationships are modeled using adaptive pooling with 1D convolutions, while a hybrid loss (MSE + SSIM) preserves spatial structure.

**Relevance to our work:** This paper shows that effective temporal modeling requires **explicit separation of spatial and temporal processing**. Our channel stacking and 3D CNN approaches treat all dimensions uniformly, which is suboptimal. A factorized approach (spatial CNN + temporal attention/1D conv) would be more parameter-efficient and effective.

---

### 2.10 Survey: Deep Learning and Foundation Models for Weather Prediction

**Citation:** Shi, J., Shirali, A., Jin, B., Zhou, S., Hu, W., Rangaraj, R., ... & Narasimhan, G. (2025). *Deep Learning and Foundation Models for Weather Prediction: A Survey.* arXiv:2501.06907.
**Link:** [https://arxiv.org/abs/2501.06907](https://arxiv.org/abs/2501.06907)

**Overview:** This comprehensive survey organizes existing ML weather prediction approaches into three training frameworks: **deterministic prediction, probabilistic generation, and transfer learning**. It consolidates datasets, code repositories, and evaluation benchmarks. The survey provides a useful taxonomy for understanding where our approach sits in the broader landscape.

---

## 3. Summary: How Our Approach Compares

| Aspect | Our Multi-Frame Models | State-of-the-Art |
|--------|----------------------|-----------------|
| **Temporal input** | 4 frames, channel-stacked or 3D conv | 1-2 frames; explicit tendency/residual encoding |
| **Temporal modeling** | Implicit (mixed with spatial dims) | Explicit (autoregressive rollout, lead-time conditioning, factorized temporal ops) |
| **Architecture** | Custom CNN, 3D CNN, ViT from scratch | Pretrained backbones, AFNO, GNNs, diffusion models |
| **Prediction strategy** | Direct 24h prediction | Hierarchical multi-lead-time or autoregressive |
| **Data scale** | ~14k samples, 2 years, regional | Decades of global ERA5 reanalysis |
| **Channel attention** | None | SE blocks, FiLM conditioning |
| **Training stability** | NaN loss issues, batch size 4 | Gradient clipping, larger batches, curriculum learning |

## 4. Why Multi-Frame Channel Stacking Underperforms

Based on the literature review and our experimental results, we identify the following reasons:

1. **Parameter explosion without proportional data.** Channel stacking multiplies first-layer parameters by k (4x in our case), but our dataset (~14k samples) is insufficient for the increased capacity. GraphCast's finding that 3 frames didn't improve over 2 further supports this.

2. **No temporal inductive bias.** Channel concatenation treats all k*C channels identically — the network must learn from data alone that channels 0-41 correspond to t-3 and channels 42-83 to t-2. This is extremely data-hungry. GraphCast/GenCast instead explicitly use two consecutive states, inherently encoding the tendency.

3. **Consecutive hourly frames are highly redundant.** Our 4 consecutive hourly frames span only 3 hours — just 12.5% of the 24-hour prediction window. The marginal information gain is minimal compared to the added complexity.

4. **3D CNN temporal collapse is too aggressive.** Our CNN3D collapses the temporal dimension from 4 to 1 in just two stride-2 layers, before the network can extract meaningful spatiotemporal features.

5. **Training instability.** Frequent NaN losses, forced batch size of 4 (due to OOM), and 24-hour SLURM time limits all contributed to suboptimal convergence.

## 5. Recommended Improvements

Based on the literature:

1. **Two frames + explicit difference features:** `concat(frame[t], frame[t] - frame[t-1])` → 84 channels instead of 168
2. **Siamese ResNet-18:** Share backbone weights across frames, fuse at feature level
3. **Squeeze-and-Excitation blocks:** Learn channel-wise attention (as in HRRRCast/KARINA)
4. **Gradient clipping** to resolve NaN training instability
5. **Wider temporal spacing:** e.g., frames at t-12h and t instead of t-3h, t-2h, t-1h, t
