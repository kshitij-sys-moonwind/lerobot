"""Check that Path-valued Hub IDs retain forward slashes on Windows."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.processor.pipeline import DataProcessorPipeline


class PretrainedPathTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = ACTConfig(
            device="cpu",
            input_features={
                "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(2,)),
                "observation.environment_state": PolicyFeature(type=FeatureType.ENV, shape=(2,)),
            },
            output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
            chunk_size=2,
            n_action_steps=2,
            use_vae=False,
            dim_model=16,
            n_heads=2,
            dim_feedforward=32,
            n_encoder_layers=1,
            n_decoder_layers=1,
        )
        self.config.save_pretrained(self.root)
        self.policy = ACTPolicy(self.config)
        save_file(self.policy.state_dict(), self.root / "model.safetensors")
        (self.root / "processor.json").write_text(json.dumps({"name": "test", "steps": []}))

    def check_loaders(self, source):
        config = PreTrainedConfig.from_pretrained(source)
        policy = ACTPolicy.from_pretrained(source, config=config)
        for key, expected in self.policy.state_dict().items():
            torch.testing.assert_close(policy.state_dict()[key], expected)
        pipeline = DataProcessorPipeline.from_pretrained(source, config_filename="processor.json")
        self.assertEqual(pipeline.name, "test")
        self.assertEqual(len(pipeline.steps), 0)

    def test_hub_path_uses_forward_slashes_in_all_loaders(self):
        requested = []

        def download(repo_id, filename, **kwargs):
            self.assertEqual(repo_id, "lerobot/test-path")
            requested.append(filename)
            return str(self.root / filename)

        with (
            patch("lerobot.configs.policies.hf_hub_download", side_effect=download),
            patch("lerobot.policies.pretrained.hf_hub_download", side_effect=download),
            patch("lerobot.processor.pipeline.hf_hub_download", side_effect=download),
        ):
            self.check_loaders(Path("lerobot/test-path"))
        self.assertEqual(requested, ["config.json", "model.safetensors", "processor.json"])

    def test_local_checkpoint_does_not_contact_hub(self):
        with (
            patch("lerobot.configs.policies.hf_hub_download", side_effect=AssertionError("network")),
            patch("lerobot.policies.pretrained.hf_hub_download", side_effect=AssertionError("network")),
            patch("lerobot.processor.pipeline.hf_hub_download", side_effect=AssertionError("network")),
        ):
            self.check_loaders(self.root)


if __name__ == "__main__":
    unittest.main()
