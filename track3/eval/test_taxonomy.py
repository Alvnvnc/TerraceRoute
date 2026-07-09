"""Pin each failure mode to a distinguishing signal vector.

This is the self-heal demo's contract: every mode must be reachable from a
realistic observation, and the crucial 1033-vs-502 discriminator must hold.
"""

import unittest

from agent.heal.taxonomy import diagnose
from agent.types import FailureMode, RepairAction, SignalVector


class TestTaxonomy(unittest.TestCase):
    def test_healthy(self):
        s = SignalVector(metrics_port_open=True, ready_code=200, ha_connections=4,
                         edge_code=200, api_status="healthy")
        d = diagnose(s)
        self.assertEqual(d.mode, FailureMode.HEALTHY)
        self.assertEqual(d.repair, RepairAction.NONE)

    def test_process_dead_metrics_refused(self):
        s = SignalVector(metrics_port_open=False, api_status="down", edge_code=1033)
        d = diagnose(s)
        self.assertEqual(d.mode, FailureMode.PROCESS_DEAD)
        self.assertEqual(d.repair, RepairAction.RESTART_CONNECTOR)

    def test_process_dead_edge_1033_only(self):
        # No local visibility, edge says no healthy connector -> connector down.
        s = SignalVector(edge_code=1033, api_status="down")
        self.assertEqual(diagnose(s).mode, FailureMode.PROCESS_DEAD)

    def test_edge_detached_ready_503(self):
        s = SignalVector(metrics_port_open=True, ready_code=503, ha_connections=0,
                         edge_code=1033, api_status="down", ever_registered=True)
        d = diagnose(s)
        self.assertEqual(d.mode, FailureMode.EDGE_DETACHED)
        self.assertEqual(d.repair, RepairAction.CHECK_EGRESS_RESTART)

    def test_partial_degraded(self):
        s = SignalVector(metrics_port_open=True, ready_code=200, ha_connections=2,
                         api_status="degraded")
        self.assertEqual(diagnose(s).mode, FailureMode.PARTIAL_DEGRADED)

    def test_origin_down_is_502_not_1033(self):
        # THE key discriminator: healthy tunnel + edge 502 => origin, not tunnel.
        s = SignalVector(metrics_port_open=True, ready_code=200, ha_connections=4,
                         edge_code=502, api_status="healthy",
                         log_signatures={"connection_refused"})
        d = diagnose(s)
        self.assertEqual(d.mode, FailureMode.ORIGIN_DOWN)
        self.assertEqual(d.repair, RepairAction.RESTART_ORIGIN)

    def test_1033_and_502_are_distinct_modes(self):
        connector_down = SignalVector(metrics_port_open=False, edge_code=1033,
                                      api_status="down")
        origin_down = SignalVector(metrics_port_open=True, ready_code=200,
                                   ha_connections=4, edge_code=502,
                                   api_status="healthy")
        self.assertNotEqual(diagnose(connector_down).mode, diagnose(origin_down).mode)

    def test_ingress_mismatch_404(self):
        s = SignalVector(metrics_port_open=True, ready_code=200, ha_connections=4,
                         edge_code=404, api_status="healthy")
        self.assertEqual(diagnose(s).mode, FailureMode.INGRESS_MISMATCH)

    def test_auth_invalid_by_log(self):
        s = SignalVector(metrics_port_open=True, ready_code=503, ha_connections=0,
                         api_status="inactive", ever_registered=False,
                         log_signatures={"unauthorized"})
        d = diagnose(s)
        self.assertEqual(d.mode, FailureMode.AUTH_INVALID)
        self.assertEqual(d.repair, RepairAction.REISSUE_TOKEN)

    def test_auth_invalid_never_registered(self):
        # No auth log line, but never registered + inactive + process alive.
        s = SignalVector(metrics_port_open=True, ready_code=503, ha_connections=0,
                         api_status="inactive", ever_registered=False)
        self.assertEqual(diagnose(s).mode, FailureMode.AUTH_INVALID)

    def test_origin_tls(self):
        s = SignalVector(metrics_port_open=True, ready_code=200, ha_connections=4,
                         edge_code=502, log_signatures={"x509"})
        self.assertEqual(diagnose(s).mode, FailureMode.ORIGIN_TLS)

    def test_origin_slow_524(self):
        s = SignalVector(metrics_port_open=True, ready_code=200, ha_connections=4,
                         edge_code=524, api_status="healthy")
        self.assertEqual(diagnose(s).mode, FailureMode.ORIGIN_SLOW)

    def test_unknown_escalates(self):
        s = SignalVector()  # no signals at all
        d = diagnose(s)
        self.assertEqual(d.mode, FailureMode.UNKNOWN)
        self.assertEqual(d.repair, RepairAction.ESCALATE_HUMAN)

    def test_evidence_is_recorded(self):
        s = SignalVector(metrics_port_open=False, edge_code=1033, api_status="down")
        d = diagnose(s)
        self.assertTrue(any("edge=1033" in e for e in d.evidence))


if __name__ == "__main__":
    unittest.main()
