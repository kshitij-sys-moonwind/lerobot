"""Run with .venv/Scripts/python.exe scripts/verify-pusht-migration.py."""

from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.processor.pipeline import DataProcessorPipeline


def main() -> None:
    migrated = Path(__file__).resolve().parents[1] / "outputs/pretrained/diffusion_pusht"
    original = hf_hub_download("lerobot/diffusion_pusht", "model.safetensors", local_files_only=True)
    with (
        safe_open(original, framework="pt") as source,
        safe_open(migrated / "model.safetensors", framework="pt") as destination,
    ):
        normalization_keys = {key for key in source.keys() if key.startswith(("normalize_", "unnormalize_"))}
        weight_keys = set(source.keys()) - normalization_keys
        assert set(destination.keys()) == weight_keys
        for key in weight_keys:
            assert torch.equal(source.get_tensor(key), destination.get_tensor(key)), key

        for filename, prefix in (
            ("policy_preprocessor.json", "normalize_inputs"),
            ("policy_postprocessor.json", "unnormalize_outputs"),
        ):
            pipeline = DataProcessorPipeline.from_pretrained(migrated, config_filename=filename)
            normalizer = next(step for step in pipeline.steps if hasattr(step, "stats"))
            for feature, stats in normalizer.stats.items():
                feature_prefix = "normalize_inputs"
                if feature == "action":
                    feature_prefix = "normalize_targets" if prefix == "normalize_inputs" else prefix
                for statistic, value in stats.items():
                    key = f"{feature_prefix}.buffer_{feature.replace('.', '_')}.{statistic}"
                    assert torch.equal(torch.as_tensor(value), source.get_tensor(key)), key

    config = DiffusionConfig.from_pretrained(migrated)
    assert config.crop_shape == (84, 84)
    assert config.down_dims == (512, 1024, 2048)
    print(f"Verified {len(weight_keys)} unchanged weight tensors and migrated normalization statistics.")


if __name__ == "__main__":
    main()
