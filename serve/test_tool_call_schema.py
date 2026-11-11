"""Tool-call arguments that follow the declared schema, and a value whose own </parameter> the model skipped.

Upstream issues 5743 (a boolean written True/False came back as a string) and 5756 (a nested <parameter=..> pair
ended the value early when the model left out the value's own closer).  Both fixes are opt-in settings
(tool_arg_coercion, tool_skipped_closer); off, the parser behaves as before.

    python -m unittest serve.test_tool_call_schema -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serve import frontend  # noqa: E402
from serve.frontend import OutputParser, param_end, parse_tool_call  # noqa: E402

BOOL_TOOL = {"name": "get_prime_factors", "parameters": {"properties": {
    "number": {"type": "integer"}, "formatted": {"type": "boolean"}, "ratio": {"type": "number"},
    "maybe": {"type": ["boolean", "null"]}, "label": {"type": "string"}}}}
EDIT_TOOL = {"name": "edit_file", "parameters": {"properties": {
    "path": {"type": "string"}, "new_string": {"type": "string"}}}}


def call(body, tool):
    return parse_tool_call(body, tool).arguments


def stream(text, tools, step=7):
    p = OutputParser(thinking=False, tools=tools, stream_tools=True)
    evs = []
    for i in range(0, len(text), step):
        evs += p.feed(text[i:i + step])
    evs += p.finish("stop")
    return [e.call.arguments for e in evs if e.kind == "tool_call"]


class Setting:
    def __init__(self, **kw):
        self.kw, self.old = kw, {}

    def __enter__(self):
        for k, v in self.kw.items():
            self.old[k] = getattr(frontend, k)
            setattr(frontend, k, v)

    def __exit__(self, *a):
        for k, v in self.old.items():
            setattr(frontend, k, v)


class Coercion(unittest.TestCase):
    BODY = ("<function=get_prime_factors>\n<parameter=number>\n450\n</parameter>\n"
            "<parameter=formatted>\nTrue\n</parameter>\n</function>")

    def test_off_keeps_the_text(self):
        self.assertEqual(call(self.BODY, BOOL_TOOL), {"number": 450, "formatted": "True"})

    def test_on_follows_the_declared_type(self):
        with Setting(TOOL_ARG_COERCION=True):
            self.assertEqual(call(self.BODY, BOOL_TOOL), {"number": 450, "formatted": True})
            for text, want in (("False", False), (" TRUE ", True), ("true", True), ("false", False)):
                self.assertIs(frontend.coerce_value(text, "boolean"), want)
            self.assertIs(frontend.coerce_value("False", ["boolean", "null"]), False)
            self.assertEqual(frontend.coerce_value('"450"', "integer"), 450)
            self.assertEqual(frontend.coerce_value('"2.5"', "number"), 2.5)

    def test_on_leaves_other_things_alone(self):
        with Setting(TOOL_ARG_COERCION=True):
            self.assertEqual(frontend.coerce_value("True", "string"), "True")    # a string stays a string
            self.assertEqual(frontend.coerce_value("True", None), "True")        # no declared type: as before
            self.assertEqual(frontend.coerce_value("yes", "boolean"), "yes")     # not guessed
            self.assertEqual(frontend.coerce_value("True", ["string", "boolean"]), "True")

    def test_stream_path(self):
        with Setting(TOOL_ARG_COERCION=True):
            self.assertEqual(stream(f"<tool_call>\n{self.BODY}\n</tool_call>", [BOOL_TOOL]),
                             [{"number": 450, "formatted": True}])
        self.assertEqual(stream(f"<tool_call>\n{self.BODY}\n</tool_call>", [BOOL_TOOL]),
                         [{"number": 450, "formatted": "True"}])


class SkippedCloser(unittest.TestCase):
    SKIPPED = ("<function=edit_file>\n<parameter=path>\ndocs/tools.md\n</parameter>\n<parameter=new_string>\n"
               "<parameter=limit>25</parameter>\n</function>")
    CLOSED = ("<function=edit_file>\n<parameter=path>\ndocs/tools.md\n</parameter>\n<parameter=new_string>\n"
              "<parameter=limit>25</parameter>\n</parameter>\n</function>")

    def test_the_fork_has_the_truncation_when_off(self):
        # upstream 5756's own examples: the value ends at the nested closer
        e = param_end("<parameter=limit>25</parameter>\n</function>\n</tool_call>", final=True, name="new_string")
        self.assertEqual("<parameter=limit>25</parameter>\n</function>"[:e], "<parameter=limit>25")
        self.assertEqual(call(self.SKIPPED, EDIT_TOOL)["new_string"], "<parameter=limit>25")

    def test_on_the_value_keeps_the_nested_pair(self):
        with Setting(TOOL_SKIPPED_CLOSER=True):
            self.assertEqual(call(self.SKIPPED, EDIT_TOOL),
                             {"path": "docs/tools.md", "new_string": "<parameter=limit>25</parameter>"})
            self.assertEqual(call(self.CLOSED, EDIT_TOOL),
                             {"path": "docs/tools.md", "new_string": "<parameter=limit>25</parameter>"})

    def test_on_a_plain_value_is_unchanged(self):
        plain = "<function=edit_file>\n<parameter=path>\na\n</parameter>\n<parameter=new_string>\nhi\n</parameter>\n</function>"
        with Setting(TOOL_SKIPPED_CLOSER=True):
            self.assertEqual(call(plain, EDIT_TOOL), {"path": "a", "new_string": "hi"})

    def test_on_in_a_stream_with_the_wrapper(self):
        text = f"<tool_call>\n{self.SKIPPED}\n</tool_call>"
        want = [{"path": "docs/tools.md", "new_string": "<parameter=limit>25</parameter>"}]
        with Setting(TOOL_SKIPPED_CLOSER=True):
            for step in (3, 11, 1000):
                self.assertEqual(stream(text, [EDIT_TOOL], step), want, step)
            self.assertEqual(stream(f"<tool_call>\n{self.CLOSED}\n</tool_call>", [EDIT_TOOL]), want)


if __name__ == "__main__":
    unittest.main()
