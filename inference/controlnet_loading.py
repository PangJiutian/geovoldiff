"""Load a local exported ControlNet pipeline for data generation."""
from pathlib import Path

from diffusers import DPMSolverMultistepScheduler

from pipelines.ldm3d_control_pipeline import ControlNetLDMPipeline3D


def load_controlnet_pipeline(model_path):
    """Return the inference pipeline and exported-model provenance."""
    root = Path(model_path).resolve()
    if not (root / "model_index.json").is_file():
        raise FileNotFoundError(
            f"Expected a complete exported pipeline with model_index.json: {root}"
        )

    pipe = ControlNetLDMPipeline3D.from_pretrained(root, local_files_only=True)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    for module in (pipe.unet, pipe.vae, pipe.controlnet):
        if module is not None:
            module.eval()
            module.requires_grad_(False)

    provenance = {"path": str(root), "format": "pipeline", "weights": "exported"}
    return pipe, provenance
