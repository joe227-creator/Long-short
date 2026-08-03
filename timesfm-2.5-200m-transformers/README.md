---
license: apache-2.0
library_name: transformers
pipeline_tag: time-series-forecasting
tags:
- transformers
- timesfm
- timesfm2_5
- time-series-forecasting
- arxiv:2310.10688
---

# TimesFM 2.5 (Transformers)

TimesFM (Time Series Foundation Model) is a pretrained decoder-only model for time-series forecasting. This repository contains the **Transformers** port of the official TimesFM 2.5 PyTorch release.

**Resources and Technical Documentation**:
* Original model: [google/timesfm-2.5-200m-pytorch](https://huggingface.co/google/timesfm-2.5-200m-pytorch)
* Paper: [A decoder-only foundation model for time-series forecasting](https://huggingface.co/papers/2310.10688)
* Transformers docs: [TimesFM 2.5](https://huggingface.co/docs/transformers/main/en/model_doc/timesfm_2p5)

## Model description

This model is converted from the official TimesFM 2.5 PyTorch checkpoint and integrated into `transformers` as `TimesFm2_5ModelForPrediction`.

The converted checkpoint preserves the original architecture and forecasting behavior, including:
* patch-based inputs for time-series contexts
* decoder-only self-attention stack
* point and quantile forecasts

## Usage (Transformers)

```python
import torch
from transformers import TimesFm2_5ModelForPrediction

model = TimesFm2_5ModelForPrediction.from_pretrained("google/timesfm-2.5-200m-transformers")
model = model.to(torch.float32).eval()

past_values = [
    torch.linspace(0, 1, 100),
    torch.sin(torch.linspace(0, 20, 67)),
]

with torch.no_grad():
    outputs = model(past_values=past_values, forecast_context_len=1024)

print(outputs.mean_predictions.shape)
print(outputs.full_predictions.shape)
```

## Conversion details

This checkpoint was produced with:
* script: `src/transformers/models/timesfm_2p5/convert_timesfm_2p5_original_to_hf.py`
* source checkpoint: `google/timesfm-2.5-200m-pytorch`
* conversion date (UTC): `2026-02-20`

Weight conversion parity is verified by comparing converted-model forecasts against the official implementation outputs on deterministic inputs.

## Citation

```bibtex
@inproceedings{das2024a,
    title={A decoder-only foundation model for time-series forecasting},
    author={Abhimanyu Das and Weihao Kong and Rajat Sen and Yichen Zhou},
    booktitle={Forty-first International Conference on Machine Learning},
    year={2024},
    url={https://openreview.net/forum?id=jn2iTJas6h}
}
```
