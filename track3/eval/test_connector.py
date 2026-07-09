"""Pure cloudflared output parsers."""

import unittest

from agent.tools.connector import (
    parse_ha_connections,
    parse_log_signatures,
    parse_metrics_addr,
    parse_quick_tunnel_url,
)


class TestConnectorParsers(unittest.TestCase):
    def test_quick_tunnel_url(self):
        line = ("2026-07-10 |  https://brave-green-tree-1234.trycloudflare.com  |")
        self.assertEqual(parse_quick_tunnel_url(line),
                         "https://brave-green-tree-1234.trycloudflare.com")

    def test_quick_tunnel_url_absent(self):
        self.assertIsNone(parse_quick_tunnel_url("no url here"))

    def test_metrics_addr(self):
        line = "INF Starting metrics server on 127.0.0.1:20241/metrics"
        self.assertEqual(parse_metrics_addr(line), "127.0.0.1:20241")

    def test_log_sig_connection_refused(self):
        line = ('{"level":"error","error":"dial tcp 127.0.0.1:5244: '
                'connect: connection refused"}')
        self.assertIn("connection_refused", parse_log_signatures(line))

    def test_log_sig_x509(self):
        self.assertIn("x509", parse_log_signatures(
            'error="x509: certificate signed by unknown authority"'))

    def test_log_sig_credentials_missing(self):
        line = "Tunnel credentials file '/root/.cloudflared/x.json' doesn't exist"
        self.assertIn("credentials_missing", parse_log_signatures(line))

    def test_log_sig_none(self):
        self.assertEqual(parse_log_signatures("INF Connection registered"), set())

    def test_ha_connections(self):
        metrics = ("# HELP cloudflared_tunnel_ha_connections\n"
                   "cloudflared_tunnel_ha_connections 4\n")
        self.assertEqual(parse_ha_connections(metrics), 4)

    def test_ha_connections_zero(self):
        self.assertEqual(
            parse_ha_connections("cloudflared_tunnel_ha_connections 0"), 0)

    def test_ha_connections_absent(self):
        self.assertIsNone(parse_ha_connections("other_metric 1"))


if __name__ == "__main__":
    unittest.main()
