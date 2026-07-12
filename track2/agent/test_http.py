"""HTTP transport regression tests."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from .http import post_json


class HttpHeadersTest(unittest.TestCase):
    def test_post_uses_explicit_user_agent(self) -> None:
        response = MagicMock()
        response.read.return_value = b"{}"
        response.__enter__.return_value = response
        with patch("urllib.request.urlopen", return_value=response) as urlopen:
            post_json("https://example.test/v1", {"model": "test"}, retries=0)

        request = urlopen.call_args.args[0]
        self.assertEqual(request.get_header("User-agent"), "TerraceRoute/1.0")


if __name__ == "__main__":
    unittest.main()
