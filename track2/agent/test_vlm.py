"""Focused provider-routing tests; no network or credentials required."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from . import vlm
from .config import config


class FireworksRoutingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.values = {
            name: getattr(config, name)
            for name in (
                "vlm_base_url", "vlm_token", "fireworks_api_key",
                "fireworks_gemma_model", "fb_base_url", "fb_api_key", "fb_model",
                "fb_checker_model",
            )
        }
        config.vlm_base_url = "https://radeon.invalid/v1"
        config.vlm_token = ""
        config.fireworks_api_key = "test-key"
        config.fireworks_gemma_model = "accounts/test/deployments/gemma"
        config.fb_base_url = ""
        config.fb_api_key = ""
        config.fb_model = ""
        config.fb_checker_model = ""

    def tearDown(self) -> None:
        for name, value in self.values.items():
            setattr(config, name, value)

    def test_generator_fallback_uses_fireworks_schema(self) -> None:
        schema = {
            "type": "object",
            "properties": {"formal": {"type": "string"}},
            "required": ["formal"],
            "additionalProperties": False,
        }
        with patch("agent.vlm.post_json", side_effect=[
            OSError("primary unavailable"),
            {"choices": [{"message": {"content": '{"formal":"A formal caption."}'}}]},
        ]) as post:
            text = vlm.chat("caption this", images_b64=["aGVsbG8="], json_mode=True,
                            json_schema=schema)

        self.assertEqual(text, '{"formal":"A formal caption."}')
        self.assertEqual(post.call_count, 2)
        url, payload = post.call_args_list[1].args
        self.assertEqual(url, "https://api.fireworks.ai/inference/v1/chat/completions")
        self.assertEqual(payload["model"], "accounts/test/deployments/gemma")
        self.assertEqual(payload["response_format"]["type"], "json_schema")
        self.assertEqual(payload["response_format"]["json_schema"]["schema"], schema)
        self.assertEqual(post.call_args_list[1].kwargs["headers"],
                         {"Authorization": "Bearer test-key"})

    def test_checker_never_uses_fireworks_generator(self) -> None:
        with patch("agent.vlm.post_json", side_effect=OSError("primary unavailable")) as post:
            text = vlm.chat("judge this", model=config.checker_model,
                            allow_fireworks=False)

        self.assertIsNone(text)
        self.assertEqual(post.call_count, 1)


if __name__ == "__main__":
    unittest.main()
