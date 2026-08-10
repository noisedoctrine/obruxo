# Basic Pitch ONNX parity

- Model: `spotify-basic-pitch-icassp-2022-v0.4.0`
- ONNX blob: `c30e5f9438e798604b7177aa26be1fe64482f767`
- Windows: `5` synthetic, `0` private clips
- Runtime: ONNX Runtime CPU and PyTorch CPU, float32

| Output | Max absolute | Mean absolute | RMSE |
| --- | ---: | ---: | ---: |
| contour | 0.0205295756 | 5.7240466e-05 | 0.000510750435 |
| note | 0.00384346396 | 1.39321807e-05 | 8.78979718e-05 |
| onset | 0.208933622 | 0.000168445606 | 0.00279380153 |

Threshold disagreements: note `0`, onset `0`.
Event counts: ONNX `8`, PyTorch `8`; structural disagreements `0`.

The report contains aggregate values only; private validation identities are intentionally omitted.
