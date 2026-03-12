import numpy as np

def _extract_ampl_phase(fft_im):
    amp = np.abs(fft_im)
    pha = np.angle(fft_im)
    return amp, pha

def _low_freq_mutate(amp_src, amp_trg, L=0.01):
    """
    Replace low-frequency part of source amplitude with target amplitude.
    Works for 2D arrays; for 3D applies per-slice.
    """
    if amp_src.ndim == 2:
        h, w = amp_src.shape
        b = int(np.floor(min(h, w) * L))
        if b < 1:
            return amp_src

        amp_src_ = np.fft.fftshift(amp_src)
        amp_trg_ = np.fft.fftshift(amp_trg)

        c_h, c_w = h // 2, w // 2
        h1, h2 = c_h - b, c_h + b + 1
        w1, w2 = c_w - b, c_w + b + 1

        amp_src_[h1:h2, w1:w2] = amp_trg_[h1:h2, w1:w2]
        amp_src_ = np.fft.ifftshift(amp_src_)
        return amp_src_

    if amp_src.ndim == 3:
        out = amp_src.copy()
        for i in range(amp_src.shape[0]):
            out[i] = _low_freq_mutate(amp_src[i], amp_trg[i], L=L)
        return out

    return amp_src

def FDA_source_to_target_np(src_img, trg_img, L=0.01):
    """
    Fourier Domain Adaptation (FDA): swap low-frequency amplitude from target into source.
    src_img, trg_img: numpy arrays.
    Supports:
      - (H, W)
      - (D, H, W) : per-slice
      - (C, H, W) or (C, D, H, W) : per-channel (and per-slice if 3D)
    """
    src = src_img.astype(np.float32, copy=False)
    trg = trg_img.astype(np.float32, copy=False)

    if src.ndim == 2:
        fft_src = np.fft.fft2(src)
        fft_trg = np.fft.fft2(trg)
        amp_src, pha_src = _extract_ampl_phase(fft_src)
        amp_trg, _ = _extract_ampl_phase(fft_trg)
        amp_src = _low_freq_mutate(amp_src, amp_trg, L=L)
        fft_src_ = amp_src * np.exp(1j * pha_src)
        return np.real(np.fft.ifft2(fft_src_)).astype(np.float32)

    if src.ndim == 3:
        out = np.zeros_like(src, dtype=np.float32)
        for i in range(src.shape[0]):
            out[i] = FDA_source_to_target_np(src[i], trg[i], L=L)
        return out

    if src.ndim == 4:
        out = np.zeros_like(src, dtype=np.float32)
        for c in range(src.shape[0]):
            out[c] = FDA_source_to_target_np(src[c], trg[c], L=L)
        return out

    return src.astype(np.float32)
