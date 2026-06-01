import torch
import random
import numpy as np
from pathlib import Path

from pipelines.cond_gvd_pipeline import CondGeoVolDiffPipeline
from diffusers import DPMSolverMultistepScheduler

device = torch.device("cuda:0")

# ==============================================================================
# load weights
# ==============================================================================
model_path = Path("train_ckpt") / "ldm_3d_controlnet"
pipe = CondGeoVolDiffPipeline.from_pretrained(model_path)
pipe.scheduler = DPMSolverMultistepScheduler.from_pretrained(model_path, subfolder="scheduler")
pipe = pipe.to(device)
pipe.unet.eval()

# ==============================================================================
# condition if have
# ==============================================================================
label = np.load(Path("test_data") / "fault_label.npy")
# start_x = random.randint(0, label.shape[0] - 128)
# start_y = random.randint(0, label.shape[0] - 128)
# start_z = random.randint(0, label.shape[0] - 128)
start_x = 69
start_y = 67
start_z = 105
print(f"Random slice starting at: ({start_x}, {start_y}, {start_z})")
label = label[start_x:start_x + 128, start_y:start_y + 128, start_z:start_z + 128]
label_np = (label > 0).astype(np.float32)
label = torch.from_numpy(label_np.copy())
label = label.unsqueeze(0).unsqueeze(0).float().to(device)

# ==============================================================================
# generation
# ==============================================================================
seed = random.randint(0, 1_000_000)
# seed = 695664
print(f"Selected seed: {seed}")
generator = torch.Generator(device=device).manual_seed(seed)
with torch.no_grad():
    images = pipe(
        T=32, H=32, W=32,
        num_inference_steps=25,
        generator=generator,
        show_progress=True,
        controlnet_cond=label, # if have condition, else set to None
        guidance_scale=2,      # if have condition, else set to 1.0
    )
images = (images + 1) / 2
images_np = images.squeeze().cpu().numpy() # (inline, xline, time/depth)
torch.cuda.empty_cache()

# ==============================================================================
# vis
# ==============================================================================
import matplotlib.pyplot as plt  # noqa: E402
def plot_volume_slices(vol: np.ndarray, step: int = 16, cmap: str = 'seismic', title: str = ''):
    D, H, W = vol.shape
    assert D == H == W, f"Expected cubic volume, got {vol.shape}"
    
    idxs = list(range(0, D, step))
    n = len(idxs)
    vmin, vmax = np.percentile(vol, [2, 98])

    fig, axes = plt.subplots(3, n, figsize=(n * 2.5, 8))
    row_labels = ['Inline', 'Crossline', 'Time']

    for i, idx in enumerate(idxs):
        axes[0, i].imshow(vol[idx].T, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
        axes[1, i].imshow(vol[:, idx, :].T, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
        axes[2, i].imshow(vol[:, :, idx], cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
        axes[0, i].set_title(f'{idx}', fontsize=8)
        for row in range(3):
            axes[row, i].axis('off')

    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=9)

    plt.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.show()

plot_volume_slices(images_np, step=16, cmap='jet')
plot_volume_slices(label_np, step=16, cmap='gray')