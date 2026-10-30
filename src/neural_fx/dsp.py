"""Known targets; these are controls, not learned contributions."""
import numpy as np
import torch
from scipy.signal import fftconvolve, firwin, lfilter


def high_gain(x, drive, mix, sample_rate):
    """Causal synthetic British-inspired preamp; no hardware/cabinet capture.

    Tighten bass before two asymmetric gain stages and roll off ultrasonic fizz.
    FIR memory fits the development TCN's receptive field. Cabinet/space remain
    independent native effects, so they are never learned/applied twice.
    """
    if mix == 0:
        return np.asarray(x, dtype=np.float32).copy()
    low = lfilter(firwin(17, 700, fs=sample_rate), [1.0], x)
    tight = x - 0.65 * low
    gain = 10 ** ((6 + 30 * drive) / 20)
    stage = np.tanh(gain * tight + 0.12) - np.tanh(0.12)
    wet = 0.65 * np.tanh((1.5 + 2.5 * drive) * stage)
    wet = lfilter(firwin(33, min(7000, sample_rate * .4), fs=sample_rate), [1.0], wet)
    return ((1 - mix) * x + mix * wet).astype(np.float32)


def distortion(x, drive, mix):
    gain = 1.0 + 9.0 * drive
    wet = np.tanh(gain * x) / np.tanh(gain)
    return ((1.0 - mix) * x + mix * wet).astype(np.float32)


def impulse_response(sample_rate, kind="cabinet"):
    length = 129 if kind == "cabinet" else int(0.2 * sample_rate)
    t = np.arange(length) / sample_rate
    ir = np.exp(-t * (1600 if kind == "cabinet" else 35))
    ir *= np.cos(2 * np.pi * (1800 if kind == "cabinet" else 730) * t)
    ir[0] += 1
    return (ir / np.abs(ir).sum()).astype(np.float32)


def target(x, controls, effect, sample_rate):
    drive, mix = controls
    if effect == "distortion":
        return distortion(x, drive, mix)
    if effect == "high_gain":
        return high_gain(x, drive, mix, sample_rate)
    if effect not in ("cabinet", "reverb"):
        raise ValueError(f"Unknown target effect: {effect}")
    ir = impulse_response(sample_rate, effect)
    wet = fftconvolve(x, ir)[:len(x)].astype(np.float32)
    # drive acts as a continuous coloration amount for these linear targets.
    return ((1 - drive * mix) * x + drive * mix * wet).astype(np.float32)


def mrstft(prediction, reference, sizes=(64, 128, 256)):
    """Mean spectral convergence + mean absolute log-magnitude error."""
    losses = []
    for size in sizes:
        window = torch.hann_window(size, device=prediction.device)
        def magnitude(x):
            return torch.stft(x.squeeze(1), size, hop_length=size // 4,
                              window=window, center=True, pad_mode="constant",
                              return_complex=True).abs().clamp_min(1e-7)
        a, b = magnitude(prediction), magnitude(reference)
        convergence = torch.linalg.vector_norm(a - b, dim=(-2, -1)) / torch.linalg.vector_norm(b, dim=(-2, -1)).clamp_min(1e-7)
        losses.append(convergence.mean() + (a.log() - b.log()).abs().mean())
    return torch.stack(losses).mean()


def metrics(prediction, reference):
    error = prediction - reference
    # ESR on silence has no meaningful signal-energy denominator.
    energy = reference.square().sum().item()
    size = 256
    def spectrum(x):
        return torch.stft(x.squeeze(1), size, hop_length=64,
                          window=torch.hann_window(size), pad_mode="constant",
                          return_complex=True).abs().clamp_min(1e-7)
    lsd = ((20 * torch.log10(spectrum(prediction)) - 20 * torch.log10(spectrum(reference))).square().mean(dim=-2).sqrt().mean())
    return {"mse": error.square().mean().item(), "l1": error.abs().mean().item(),
            "esr": error.square().sum().item() / energy if energy > 1e-12 else None,
            "mrstft": mrstft(prediction, reference).item(), "lsd_db": lsd.item()}
