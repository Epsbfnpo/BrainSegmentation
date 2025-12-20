import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FreqFit(nn.Module):
    """
    Frequency-based Fine-Tuning Module (FreqFiT).
    Optimized with Padding for FP16 support to save memory.
    """

    def __init__(self, channel, rate=0.25):
        super().__init__()
        self.channel = channel
        self.rate = rate
        self.inter_channel = max(int(channel * rate), 1)

        self.fc1 = nn.Linear(channel, self.inter_channel)

        # Learnable frequency filter
        # Shape: (1, inter_channel, 2) representing complex
        self.complex_weight = nn.Parameter(
            torch.randn(1, self.inter_channel, 2, dtype=torch.float32) * 0.02
        )

        self.fc2 = nn.Linear(self.inter_channel, channel)
        self.act = nn.GELU()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc2.weight)
        nn.init.trunc_normal_(self.complex_weight, std=0.02)

    def forward(self, x):
        residual = x

        # 1. Down-project
        x = self.fc1(x)  # (B, ..., inter_channel)

        # 2. FFT stability guard in FP32
        with torch.cuda.amp.autocast(enabled=False):
            x = x.float()

            # 3. Smart Padding for FP16 FFT
            # cuFFT in FP16 requires power-of-2 dimensions
            C = x.shape[-1]
            target_C = 1
            while target_C < C:
                target_C *= 2

            if target_C != C:
                # Pad the last dimension
                x_padded = F.pad(x, (0, target_C - C))
            else:
                x_padded = x

            # 4. FFT (safe in FP32)
            x_fft = torch.fft.rfft(x_padded, dim=-1, norm='ortho')

            # 5. Frequency Modulation
            weight = torch.view_as_complex(self.complex_weight.float())

            # Resize weight to match padded frequency dimension
            # x_fft shape: (..., target_C // 2 + 1)
            if weight.shape[1] != x_fft.shape[-1]:
                w_t = torch.view_as_real(weight).permute(0, 2, 1)  # (1, 2, C)
                w_t = F.interpolate(w_t, size=x_fft.shape[-1], mode='linear', align_corners=False)
                weight = torch.view_as_complex(w_t.permute(0, 2, 1).contiguous())

            x_fft = x_fft * weight

            # 6. Inverse FFT
            x_out = torch.fft.irfft(x_fft, n=target_C, dim=-1, norm='ortho')

            # 7. Slice back if padded
            if target_C != C:
                x_out = x_out[..., :C]

        # 8. Up-project
        x = self.act(x_out)
        x = self.fc2(x)

        return residual + x


class LoRALinear(nn.Module):
    """Standard LoRA Layer."""

    def __init__(self, original_layer: nn.Linear, rank: int = 8, alpha: int = 16):
        super().__init__()
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.scaling = alpha / rank

        self.pretrained = original_layer
        for param in self.pretrained.parameters():
            param.requires_grad = False

        self.lora_A = nn.Linear(self.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, self.out_features, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        with torch.no_grad():
            out_orig = self.pretrained(x)
        out_lora = self.lora_B(self.lora_A(x)) * self.scaling
        return out_orig + out_lora


def inject_freqfit_and_lora(model: nn.Module, target_modules: list[str], rank: int = 8) -> nn.Module:
    print(f"💉 Injecting LoRA (r={rank}) + FreqFiT...")
    modules_to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(t in name for t in target_modules):
                modules_to_replace.append((name, module))

    for name, module in modules_to_replace:
        parent_name, child_name = name.rsplit('.', 1) if '.' in name else ("", name)
        parent = model.get_submodule(parent_name) if parent_name else model

        lora_layer = LoRALinear(module, rank=rank)
        setattr(parent, child_name, lora_layer)

        # Inject FreqFiT AFTER LoRA
        freqfit_layer = FreqFit(module.out_features)
        composite = nn.Sequential(lora_layer, freqfit_layer)
        setattr(parent, child_name, composite)

    print(f"✅ Successfully injected {len(modules_to_replace)} layers.")
    return model
