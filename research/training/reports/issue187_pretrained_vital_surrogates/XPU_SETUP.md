# Stable Audio checks on Intel XPU

Use the existing Conda `py312` interpreter from the repository root. This setup was exercised with PyTorch 2.12.1+xpu, Transformers 4.57.6 and an Intel Arc 140T (16 GB).

## Source setup

Clone the upstream repositories outside tracked source, check out the indicated commits, and apply the corresponding patch with `git -C <checkout> apply <absolute-patch-path>`.

| Upstream | Source commit | Patch |
|---|---|---|
| [Stable Audio Tools](https://github.com/Stability-AI/stable-audio-tools) | `3241adba4fc2a85cf5b29d9eb68d42f40a28e820` | [Open Small codec](stable-audio-tools-inference.patch) |
| [Stable Audio 3](https://github.com/Stability-AI/stable-audio-3) | `4007b6f2987bcb760626c2b4ae3e3541753984e9` | [XPU attention and loader](stable-audio-3-xpu.patch) |

Use `PYTHONPATH` for these checkouts. For Open Small, install its extra import dependencies into a separate scratch directory and add that directory to `PYTHONPATH` for the probe:

```powershell
$obruxoTorch = Join-Path $env:USERPROFILE 'miniforge3/envs/py312/python.exe'
& $obruxoTorch -B -m pip install --no-deps --target tmp/stable-audio-deps alias-free-torch==0.0.6 einops-exts==0.0.4 PyWavelets==1.10.0 torchsde==0.2.6 trampoline==0.1.2
$env:PYTHONPATH = 'tmp/stable-audio-tools-source;tmp/stable-audio-3-source;tmp/stable-audio-deps'
$env:HF_HUB_OFFLINE = '1'
$probe = 'research/training/reports/scripts/check_issue187_pretrained_xpu.py'
& $obruxoTorch -B $probe open-small-codec <open-small-snapshot>
& $obruxoTorch -B $probe same-codec <same-s-or-same-l-snapshot>
& $obruxoTorch -B $probe stable-audio-3-generator <generator-snapshot>
```

Use the directory that contains `model_config.json` and `model.safetensors` as the snapshot argument. The generator also uses the snapshot's `t5gemma-b-b-ul2` subdirectory. If the identical encoder and tokenizer are already present in another snapshot, pass that full subdirectory path as `--text-encoder`.

| Checkpoint | Revision |
|---|---|
| [Open Small](https://huggingface.co/stabilityai/stable-audio-open-small) | `ea2772170fc06f80dfd02bea6e57a44d9d4bc9d3` |
| [SAME-S](https://huggingface.co/stabilityai/SAME-S) | `63a63b7193ee9b6ee9836776c5f3c52a2f9600df` |
| [SAME-L](https://huggingface.co/stabilityai/SAME-L) | `54b0fe9c9fc27aa1d5eac36402788efea15f73eb` |
| [Small Music](https://huggingface.co/stabilityai/stable-audio-3-small-music) | `8bdfb53c6b0fbc62113cd571fd37373efe15fa14` |
| [Medium](https://huggingface.co/stabilityai/stable-audio-3-medium) | `d4d3579395a8ac0df4ab13b5b1d7138ce7290f81` |
| [Medium Base](https://huggingface.co/stabilityai/stable-audio-3-medium-base) | `c397d9a6912dc8b1f71a80e7b9673f4d307e7f8c` |

## Checks

The codec probe uses one synthetic stereo clip to test waveform reconstruction and verify that the frozen decoder sends finite, nonzero gradients to its input. It accounts for padding before calculating the test loss. The generator probe requests a two-second stereo sample with a fixed seed and two denoising steps, and checks placement, the generator parameter dtype, and the output.

| Checkpoint | Result on `xpu:0` |
|---|---|
| Open Small | Stereo encode and decode, nonzero decoder input gradient |
| SAME-S | Stereo encode and decode, nonzero decoder input gradient |
| SAME-L | Stereo encode and decode, nonzero decoder input gradient |
| Small Music | FP16 generator, 88,200 finite samples per stereo channel |
| Medium | FP16 generator, 88,200 finite samples per stereo channel |
| Medium Base | FP16 generator, 88,200 finite samples per stereo channel |

The [DAC 44.1 kHz checkpoint](https://huggingface.co/descript/dac_44khz) at revision `c1bc521685adf9cfe247bc39a5ca58917eda1ac4` also passed a separate mono decode and latents gradient probe on the same XPU using Transformers. This check differentiated through the decoder from the quantized embedding, with the encoder and quantizer outside the gradient path.

These checks cover runtime compatibility. The Vital comparison and decision gates are defined in [issue #187](https://github.com/noisedoctrine/obruxo/issues/187).
