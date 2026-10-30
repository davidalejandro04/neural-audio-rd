"""Causal baselines and an explicitly experimental hybrid.

State is external: callers own history/hidden tensors. No mutable model-global
state is shared between streams. Controls are constant within an audio block.
"""
import torch
from torch import nn
from torch.nn import functional as F


class CausalBlock(nn.Module):
    def __init__(self, channels, dilation, conditioned):
        super().__init__()
        self.context = 2 * dilation
        self.conv = nn.Conv1d(channels, channels, 3, dilation=dilation)
        self.film = nn.Linear(2, 2 * channels) if conditioned else None

    def forward(self, x, controls):
        z = self.conv(F.pad(x, (self.context, 0)))
        if self.film is not None:
            scale, shift = self.film(controls).chunk(2, dim=1)
            z = z * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        return x + torch.tanh(z)


class AudioModel(nn.Module):
    def __init__(self, name="conditional_tcn", channels=8, dilations=(1, 2, 4), fir_taps=33, wet_only_conditioning=False, zero_preserving=False):
        super().__init__()
        supported = ("conv1d", "tcn", "conditional_tcn", "gru", "hybrid", "hybrid_no_gru", "hybrid_no_fir")
        if name not in supported:
            raise ValueError(f"Unknown model {name}; expected one of {supported}")
        if channels < 1 or fir_taps < 1 or not dilations or any(d < 1 for d in dilations):
            raise ValueError("Positive channels, FIR taps, and dilations required")
        self.name, self.channels = name, channels
        self.wet_only_conditioning = wet_only_conditioning
        if zero_preserving and name not in ("conv1d", "tcn", "conditional_tcn"):
            raise ValueError("Zero-preserving inference currently supports finite-context TCNs only")
        self.zero_preserving = zero_preserving
        self.has_tcn = name != "gru"
        self.has_gru = name in ("gru", "hybrid", "hybrid_no_fir")
        self.has_fir = name in ("hybrid", "hybrid_no_gru")
        self.conditioned = name not in ("conv1d", "tcn")
        dilations = (1,) if name == "conv1d" else dilations
        self.history_samples = max(sum(2 * d for d in dilations) if self.has_tcn else 0,
                                   fir_taps - 1 if self.has_fir else 0, 1)
        if self.has_tcn:
            self.input = nn.Conv1d(1, channels, 1)
            self.blocks = nn.ModuleList([CausalBlock(channels, d, self.conditioned) for d in dilations])
            self.output = nn.Conv1d(channels, 1, 1)
        if self.has_gru:
            self.gru = nn.GRU(3, channels, batch_first=True)
            self.gru_output = nn.Linear(channels, 1)
        if self.has_fir:
            self.fir = nn.Conv1d(1, 4, fir_taps, bias=False)
            nn.init.normal_(self.fir.weight, std=0.01)
            self.fir_controls = nn.Linear(2, 4)
            self.fir_taps = fir_taps
        self.branch_count = int(self.has_tcn) + int(self.has_gru) + int(self.has_fir)
        self.router = nn.Linear(2, self.branch_count) if self.branch_count > 1 else None

    def initial_state(self, batch=1):
        parameter = next(self.parameters())
        return (parameter.new_zeros(batch, 1, self.history_samples),
                parameter.new_zeros(1, batch, self.channels))

    def process(self, x, controls, history, hidden):
        joined = torch.cat((history, x), dim=-1)
        # Prospective v2: infer the fully wet path from drive; apply mix once at output.
        # The default retains exact historical checkpoint behavior.
        conditioned_controls = torch.cat((controls[:, :1], torch.ones_like(controls[:, 1:2])), dim=1) if self.wet_only_conditioning else controls
        outputs = []
        if self.has_tcn:
            def tcn(audio):
                z = self.input(audio)
                for block in self.blocks:
                    z = block(z, conditioned_controls)
                return self.output(z)[..., self.history_samples:]
            z = tcn(joined)
            if self.zero_preserving:
                # Remove the learned control-dependent DC baseline. Silence with
                # zero history is exactly zero, including after drive changes.
                # This is opt-in so historical checkpoints retain their behavior.
                z = z - tcn(torch.zeros_like(joined))
            outputs.append(z)
        if self.has_gru:
            repeated = conditioned_controls.unsqueeze(1).expand(-1, x.shape[-1], -1)
            z, hidden = self.gru(torch.cat((x.transpose(1, 2), repeated), dim=-1), hidden)
            outputs.append(self.gru_output(z).transpose(1, 2))
        if self.has_fir:
            basis = self.fir(F.pad(joined, (self.fir_taps - 1, 0)))[..., self.history_samples:]
            weights = self.fir_controls(conditioned_controls).unsqueeze(-1)
            outputs.append((basis * weights).sum(dim=1, keepdim=True))
        if self.router is not None:
            weights = self.router(conditioned_controls).softmax(dim=-1)
            residual = sum(z * weights[:, i:i+1, None] for i, z in enumerate(outputs))
        else:
            residual = outputs[0]
        # Exact dry/wet bypass, even for untrained weights.
        y = x + controls[:, 1:2, None] * residual
        return y, joined[..., -self.history_samples:], hidden

    def forward(self, x, controls):
        history, hidden = self.initial_state(x.shape[0])
        return self.process(x, controls, history, hidden)[0]


def build_model(spec):
    return AudioModel(**spec)


@torch.inference_mode()
def stream(model, x, controls, block_sizes=(128,)):
    if not block_sizes or any(size < 1 for size in block_sizes):
        raise ValueError("Positive block sizes required")
    history, hidden = model.initial_state(x.shape[0])
    outputs, offset, index = [], 0, 0
    while offset < x.shape[-1]:
        size = block_sizes[index % len(block_sizes)]
        block = x[..., offset:offset + size]
        y, history, hidden = model.process(block, controls, history, hidden)
        outputs.append(y)
        offset += size
        index += 1
    return torch.cat(outputs, dim=-1)
