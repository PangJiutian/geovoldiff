import numpy as np
import torch
from pathlib import Path

from pipelines.inversion_pipeline import SeismicInversionPipeline

device = torch.device("cuda:0")
model_path = Path("train_ckpt") / "inversion_pretrain_post_train_on_fieldwav"
pipe = SeismicInversionPipeline.from_pretrained(model_path).to(device)
pipe.unet.eval()

seismic_nor_path = Path("test_data") / "seimic_nor.npy"

seismic_nor = np.load(seismic_nor_path)  # (H, W)
seismic_t = torch.from_numpy(seismic_nor).float()[None, None].to(device)

out = pipe(seismic_t)
pred = out.impedance[0, 0].cpu().numpy()

wells_path = Path("test_data") / "well_logs_imp.npy"
wells = np.load(wells_path)
mask = (wells == 0)
log_wells = np.log(wells[~mask])

def anti_normalize(x, mean: float, std: float):
    """Inverse of :func:`log_normalize`.

    Returns ``exp(x * std + mean)`` in the same backend as the input.
    """
    if isinstance(x, torch.Tensor):
        return torch.exp(x * std + mean)
    return np.exp(x * std + mean)
pred_phys = anti_normalize(pred, np.nanmean(log_wells), np.nanstd(log_wells))


import matplotlib.pyplot as plt  # noqa: E402
plt.figure(figsize=(10, 5))
# seismic
plt.subplot(1, 2, 1)
plt.imshow(seismic_nor, cmap='gray', aspect='auto')
plt.title("Input Seismic")
plt.axis('off')
# physical prediction
plt.subplot(1, 2, 2)
im = plt.imshow(pred_phys, cmap='jet', aspect='auto')
plt.title("Physical Impedance")
plt.axis('off')
plt.colorbar(im, fraction=0.046)

plt.tight_layout()
plt.show()