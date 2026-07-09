"""Edge response classification -> the code the taxonomy consumes."""

import unittest

from agent.tools.probe import classify_edge, is_live


class TestClassifyEdge(unittest.TestCase):
    def test_live_200(self):
        r = classify_edge(200, "<html>ok</html>")
        self.assertTrue(is_live(r))
        self.assertEqual(r.effective_code, 200)

    def test_1033_parsed_from_body_over_530(self):
        body = "<html><h1>Error 1033</h1> Argo Tunnel error</html>"
        r = classify_edge(530, body)
        self.assertEqual(r.cf_error_code, 1033)
        self.assertEqual(r.effective_code, 1033)  # 1033 wins over 530
        self.assertFalse(is_live(r))

    def test_502_origin_down(self):
        r = classify_edge(502, "Bad gateway")
        self.assertEqual(r.effective_code, 502)
        self.assertFalse(is_live(r))

    def test_404_not_live(self):
        r = classify_edge(404, "not found")
        self.assertFalse(is_live(r))

    def test_error_code_lowercase_form(self):
        r = classify_edge(530, "cloudflare error code: 1033")
        self.assertEqual(r.cf_error_code, 1033)

    def test_unreachable(self):
        r = classify_edge(None, "")
        self.assertFalse(r.reachable)
        self.assertFalse(is_live(r))

    def test_non_cf_number_ignored(self):
        # A random 3-digit number in the body must not be read as a CF code.
        r = classify_edge(200, "you have 200 items and error 404 somewhere")
        # 404 is in the 1xxx guard? no -> cf_error_code stays None for <1000.
        self.assertIsNone(r.cf_error_code)


if __name__ == "__main__":
    unittest.main()
