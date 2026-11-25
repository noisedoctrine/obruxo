"""Local clip adapter for the published Vital JEPA V7 step-16,000 weights."""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torchaudio.transforms import MelSpectrogram


PREPROCESSING = "stereo_polyphase_first_6s_right_zero_pad_mid_side_logmel_v1"
POOLING = "online_encoder_residual_global_patch_mean_v1"


class VitalJepaV7(nn.Module):
    def __init__(self, checkpoint, device):
        super().__init__()
        artifact = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if artifact["step"] != 16_000:
            raise ValueError("Vital JEPA V7 requires the published step 16,000 checkpoint")
        config, weights = artifact["config"], artifact["model"]
        encoder = config["model"]["encoder"]
        dimension = encoder["embed_dim"]
        patch = (config["patching"]["patch_mels"], config["patching"]["patch_frames"])
        self.pos_embed = nn.Parameter(torch.empty_like(weights["pos_embed"]))
        self.global_tokens = nn.Parameter(torch.empty_like(weights["global_tokens"]))
        self.patch_embed = nn.Sequential()
        self.patch_embed.add_module("proj", nn.Conv2d(2, dimension, kernel_size=patch, stride=patch))
        layer = nn.TransformerEncoderLayer(dimension, encoder["heads"], int(dimension * encoder["mlp_ratio"]),
                                           dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, encoder["depth"], norm=nn.LayerNorm(dimension),
                                             enable_nested_tensor=False)
        encoder_weights = {key: value for key, value in weights.items()
                           if key in {"pos_embed", "global_tokens"} or key.startswith(("encoder.", "patch_embed."))}
        self.load_state_dict(encoder_weights, strict=True)
        self.to(device=device, dtype=torch.float32)

        data, features = config["data"], config["features"]
        self.sample_count = int(data["sample_rate"] * data["duration_sec"])
        self.frames = self.pos_embed.shape[1] // (features["n_mels"] // patch[0]) * patch[1]
        self.log_epsilon = features["log_eps"]
        self.mel = MelSpectrogram(sample_rate=data["sample_rate"], n_fft=features["n_fft"],
                                  win_length=features["win_length"], hop_length=features["hop_length"],
                                  f_min=features["f_min"], f_max=features["f_max"], n_mels=features["n_mels"])

    def forward(self, waveform):
        tokens = self.encode_tokens(waveform)
        global_count = self.global_tokens.shape[1]
        return tokens[:, :global_count].mean(dim=1) + tokens[:, global_count:].mean(dim=1)

    def encode_tokens(self, waveform):
        if waveform.ndim != 2 or waveform.shape[1] != 2:
            raise ValueError("Vital JEPA V7 expects a stereo waveform")
        waveform = waveform[:self.sample_count].T
        waveform = F.pad(waveform, (0, self.sample_count - waveform.shape[-1]))
        left, right = waveform.unbind(dim=0)
        mid_side = torch.stack((left + right, left - right)) / math.sqrt(2.0)
        mel = self.mel(mid_side)[..., :self.frames].clamp_min(self.log_epsilon).log()
        patches = self.patch_embed(mel.unsqueeze(0).to(self.pos_embed.device)).flatten(2).transpose(1, 2)
        return self.encoder(torch.cat((self.global_tokens, patches + self.pos_embed), dim=1))
