import torch
import numpy as np
from snac import SNAC

# One global SNAC, placed on GPU if available.
_SNAC = None
_DEVICE = None

def _init_snac():
    global _SNAC, _DEVICE
    if _SNAC is not None:
        return
    if torch.cuda.is_available():
        _DEVICE = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        _DEVICE = torch.device("cpu")  # MPS path not stable for SNAC; keep CPU.
    else:
        _DEVICE = torch.device("cpu")

    _SNAC = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(_DEVICE)

def get_snac_device():
    _init_snac()
    return str(_DEVICE)

def convert_to_audio(multiframe, count):
    """
    multiframe: list[int] flat, length multiple of 7 (codebook layout)
    Returns: bytes of int16 PCM at 24kHz, or None if invalid.
    """
    _init_snac()

    if len(multiframe) < 7:
        return None

    # Lay out codebooks: c0 (1), c1 (2), c2 (4) per 7-tuple frame
    num_frames = len(multiframe) // 7
    frame = multiframe[: num_frames * 7]

    c0 = torch.tensor([], device=_DEVICE, dtype=torch.int32)
    c1 = torch.tensor([], device=_DEVICE, dtype=torch.int32)
    c2 = torch.tensor([], device=_DEVICE, dtype=torch.int32)

    for j in range(num_frames):
        i = 7 * j
        # c0: [i]
        c0 = torch.cat([c0, torch.tensor([frame[i]], device=_DEVICE, dtype=torch.int32)])
        # c1: [i+1, i+4]
        c1 = torch.cat([c1,
                        torch.tensor([frame[i + 1]], device=_DEVICE, dtype=torch.int32),
                        torch.tensor([frame[i + 4]], device=_DEVICE, dtype=torch.int32)])
        # c2: [i+2, i+3, i+5, i+6]
        c2 = torch.cat([c2,
                        torch.tensor([frame[i + 2]], device=_DEVICE, dtype=torch.int32),
                        torch.tensor([frame[i + 3]], device=_DEVICE, dtype=torch.int32),
                        torch.tensor([frame[i + 5]], device=_DEVICE, dtype=torch.int32),
                        torch.tensor([frame[i + 6]], device=_DEVICE, dtype=torch.int32)])

    # Bounds check
    for cb in (c0, c1, c2):
        if torch.any(cb < 0) or torch.any(cb > 4096):
            return None

    codes = [c0.unsqueeze(0), c1.unsqueeze(0), c2.unsqueeze(0)]

    with torch.inference_mode():
        audio_hat = _SNAC.decode(codes)  # (B, 1, n)
    # Empirically use middle slice for stable chunking (as in upstream)
    audio_slice = audio_hat[:, :, 2048:4096]
    audio_np = audio_slice.detach().cpu().numpy()
    pcm16 = (audio_np * 32767.0).astype(np.int16).tobytes()
    return pcm16
