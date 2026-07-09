"""Plan parsing must survive messy small-model output."""

import unittest

from agent.brain.schemas import Plan, normalize_for_compare, parse_plan


class TestParsePlan(unittest.TestCase):
    def test_clean_json(self):
        p = parse_plan('{"reasoning":"x","op":"expose","hostname":"a.example.com",'
                       '"port":8096,"service_scheme":"http"}')
        self.assertIsNotNone(p)
        self.assertEqual(p.op, "expose")
        self.assertEqual(p.port, 8096)

    def test_fenced_json(self):
        raw = "Sure!\n```json\n{\"op\":\"status\",\"hostname\":\"\",\"port\":0," \
              "\"service_scheme\":\"http\",\"reasoning\":\"check\"}\n```\n"
        p = parse_plan(raw)
        self.assertIsNotNone(p)
        self.assertEqual(p.op, "status")

    def test_json_embedded_in_prose(self):
        raw = 'Here is the plan: {"op":"expose","hostname":"m.example.com",' \
              '"port":"8096","service_scheme":"http","reasoning":"go"} done.'
        p = parse_plan(raw)
        self.assertIsNotNone(p)
        self.assertEqual(p.port, 8096)  # coerced from string

    def test_hostname_with_scheme_is_cleaned(self):
        p = parse_plan('{"op":"expose","hostname":"https://Media.Example.com/x",'
                       '"port":8096,"service_scheme":"http","reasoning":""}')
        self.assertEqual(p.hostname, "media.example.com")

    def test_invalid_op_returns_none(self):
        self.assertIsNone(parse_plan('{"op":"nuke","hostname":"","port":0,'
                                     '"service_scheme":"http","reasoning":""}'))

    def test_garbage_returns_none(self):
        self.assertIsNone(parse_plan("I cannot help with that."))

    def test_port_out_of_range_zeroed(self):
        p = parse_plan('{"op":"expose","hostname":"a.example.com","port":99999,'
                       '"service_scheme":"http","reasoning":""}')
        self.assertEqual(p.port, 0)

    def test_bad_scheme_defaults_http(self):
        p = parse_plan('{"op":"expose","hostname":"a.example.com","port":80,'
                       '"service_scheme":"ftp","reasoning":""}')
        self.assertEqual(p.service_scheme, "http")


class TestNormalize(unittest.TestCase):
    def test_expose_normalizes_relevant_fields(self):
        n = normalize_for_compare(Plan(op="expose", hostname="A.example.com",
                                       port=8096, service_scheme="http"))
        self.assertEqual(n, {"op": "expose", "hostname": "a.example.com",
                             "port": 8096, "service_scheme": "http"})

    def test_status_ignores_host_port(self):
        n = normalize_for_compare(Plan(op="status", hostname="ignored", port=1))
        self.assertEqual(n, {"op": "status"})

    def test_none_normalizes_empty(self):
        self.assertEqual(normalize_for_compare(None), {})

    def test_origin_service_string(self):
        self.assertEqual(
            Plan(op="expose", port=8096).origin_service(), "http://localhost:8096"
        )


if __name__ == "__main__":
    unittest.main()
