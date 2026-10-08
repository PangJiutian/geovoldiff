from typing import Optional

import numpy as np
from scipy.signal import fftconvolve

def _central_diff(log_imp: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Approximate reflectivity from log-impedance via central differences along
    ``axis``: r[i] = 0.5 * (m[i+1] - m[i-1]). Boundary samples are left as
    zero. Works for any array rank; ``axis`` selects the time/depth axis.
    """
    r = np.zeros_like(log_imp, dtype=float)
    ndim = log_imp.ndim
    sl_mid = [slice(None)] * ndim
    sl_hi  = [slice(None)] * ndim
    sl_lo  = [slice(None)] * ndim
    sl_mid[axis] = slice(1, -1)
    sl_hi[axis]  = slice(2, None)
    sl_lo[axis]  = slice(None, -2)
    r[tuple(sl_mid)] = 0.5 * (log_imp[tuple(sl_hi)] - log_imp[tuple(sl_lo)])
    return r


def forward_model(log_imp: np.ndarray, wavelet: np.ndarray,
                  time_axis: Optional[int] = None) -> np.ndarray:
    """
    Convolutional forward model:
        syn = wavelet * d(log_imp)/d(time_axis)

    Supports:
        1D: [nt]
        2D: [nt, nxl]
        3D: [nil, nxl, nt]  (time-last, matching this project's volume layout)

    ``time_axis`` defaults to 0 for ndim<=2 (this module's 2D convention,
    matching estimate_wavelet's (nt, ncdp) layout) and -1 for ndim==3
    (the project's volume_3d time-last layout). Pass it explicitly to
    override either default.
    """
    if time_axis is None:
        time_axis = 0 if log_imp.ndim <= 2 else -1
    time_axis = time_axis % log_imp.ndim

    refl = _central_diff(log_imp, axis=time_axis)

    wav_shape = [1] * log_imp.ndim
    wav_shape[time_axis] = -1
    wavelet_r = wavelet.reshape(wav_shape)

    return fftconvolve(refl, wavelet_r, mode='same', axes=time_axis)