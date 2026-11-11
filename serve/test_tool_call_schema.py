"""Tool-call arguments that follow the declared schema, and a value whose own </parameter> the model skipped.

Upstream issues 5743 (a boolean written True/False came back as a string) and 5756 (a nested <parameter=..> pair
ended the value early when the model left out the value's own closer).  Both fixes are settings:
tool_arg_coercion (on by default) and tool_skipped_closer (off by default); with one off, the parser behaves as before.

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
    "path": {"type": "string"}, "new_string": {"type": "string"}, "limit": {"type": "integer"}}}}


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
        with Setting(TOOL_ARG_COERCION=False):
            self.assertEqual(call(self.BODY, BOOL_TOOL), {"number": 450, "formatted": "True"})

    def test_on_is_the_default(self):
        self.assertTrue(frontend.TOOL_ARG_COERCION)
        self.assertEqual(call(self.BODY, BOOL_TOOL), {"number": 450, "formatted": True})

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
        with Setting(TOOL_ARG_COERCION=False):
            self.assertEqual(stream(f"<tool_call>\n{self.BODY}\n</tool_call>", [BOOL_TOOL]),
                             [{"number": 450, "formatted": "True"}])


class CoercionLimits(unittest.TestCase):
    """The reviewer's S1 (non-finite numbers) and S2 (a schema that allows a string keeps the string)."""

    def test_non_finite_and_odd_numbers_stay_text(self):
        with Setting(TOOL_ARG_COERCION=True):
            for text in ('"inf"', '"1e999"', '"nan"', '"-Infinity"', 'NaN', '"1_000"'):
                for kind in ("integer", "number"):
                    got = frontend.coerce_value(text, {"type": kind})
                    self.assertIn(got, (text, text.strip('"')), (text, kind))
                    self.assertIsInstance(got, str)
            self.assertEqual(frontend.coerce_value('"2.5"', {"type": "integer"}), '2.5')    # not an integer: text
            self.assertEqual(frontend.coerce_value('"9007199254740993"', {"type": "number"}), 9007199254740993)
            self.assertEqual(frontend.coerce_value('"3.0"', {"type": "integer"}), 3)

    def test_a_call_with_them_neither_raises_nor_writes_bare_nan(self):
        tool = {"name": "t", "parameters": {"properties": {"n": {"type": "integer"}, "x": {"type": "number"}}}}
        for text in ('"inf"', '"1e999"', '"nan"'):
            body = f"<function=t>\n<parameter=n>\n{text}\n</parameter>\n<parameter=x>\n{text}\n</parameter>\n</function>"
            for step in (1, 1000):
                calls = stream(f"<tool_call>\n{body}\n</tool_call>", [tool], step)
                self.assertEqual(len(calls), 1, (text, step))
                self.assertIsInstance(calls[0]["n"], str)
                self.assertIsInstance(calls[0]["x"], str)

    def test_a_schema_that_allows_a_string_keeps_it(self):
        with Setting(TOOL_ARG_COERCION=True):
            for schema in ({"type": ["string", "null"]}, {"anyOf": [{"type": "string"}, {"type": "null"}]},
                           {"enum": ["true", "123"]}, {"oneOf": [{"type": "integer"}, {"type": "string"}]}):
                for text in ("123", "true", "null"):
                    self.assertEqual(frontend.coerce_value(text, schema), text, (schema, text))
            self.assertEqual(frontend.coerce_value("123", {"anyOf": [{"type": "integer"}, {"type": "null"}]}), 123)
            self.assertEqual(frontend.coerce_value("true", {"anyOf": [{"type": "boolean"}, {"type": "null"}]}), True)
            self.assertEqual(frontend.coerce_value('"true"', {"type": "boolean"}), True)
            self.assertEqual(frontend.coerce_value("123", None), 123)       # no schema at all: as before

    def test_off_is_the_old_reading_everywhere(self):
        with Setting(TOOL_ARG_COERCION=False):
            self.assertEqual(frontend.coerce_value("123", {"type": ["string", "null"]}), 123)
            self.assertEqual(frontend.coerce_value("123", {"type": "string"}), "123")
            self.assertEqual(frontend.coerce_value("True", {"type": "boolean"}), "True")

    def test_string_streams_as_string_when_the_schema_allows_it(self):
        tool = {"name": "t", "parameters": {"properties": {"v": {"anyOf": [{"type": "string"}, {"type": "null"}]}}}}
        body = "<tool_call>\n<function=t>\n<parameter=v>\n123\n</parameter>\n</function>\n</tool_call>"
        self.assertEqual(stream(body, [tool], 3), [{"v": "123"}])


class SkippedCloser(unittest.TestCase):
    SKIPPED = ("<function=edit_file>\n<parameter=path>\ndocs/tools.md\n</parameter>\n<parameter=new_string>\n"
               "<parameter=limit>25</parameter>\n</function>")
    CLOSED = ("<function=edit_file>\n<parameter=path>\ndocs/tools.md\n</parameter>\n<parameter=new_string>\n"
              "<parameter=limit>25</parameter>\n</parameter>\n</function>")

    def test_the_fork_has_the_truncation_when_off(self):
        # upstream 5756's own examples: the value ends at the nested closer
        with Setting(TOOL_SKIPPED_CLOSER=False):
            e = param_end("<parameter=limit>25</parameter>\n</function>\n</tool_call>", final=True, name="new_string")
            self.assertEqual("<parameter=limit>25</parameter>\n</function>"[:e], "<parameter=limit>25")
            self.assertEqual(call(self.SKIPPED, EDIT_TOOL)["new_string"], "<parameter=limit>25")

    def test_off_is_the_default(self):
        self.assertFalse(frontend.TOOL_SKIPPED_CLOSER)
        self.assertEqual(call(self.SKIPPED, EDIT_TOOL)["new_string"], "<parameter=limit>25")

    def test_a_value_that_quotes_a_whole_call_is_left_alone(self):
        # the corpus's 4078 shape: the nested pair belongs to a quoted call that has its own </function></tool_call>
        value = "first\n<tool_call>\n<function=read>\n<parameter=path>\nn.txt\n</parameter>\n</function>\n</tool_call>\nlast"
        body = f"<function=edit_file>\n<parameter=new_string>\n{value}\n</parameter>\n</function>"
        self.assertEqual(call(body, EDIT_TOOL), {"new_string": value})

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

    def test_on_a_value_that_mentions_an_opener_is_not_corrupted(self):
        # reviewer's B1 (a): with the rule on, a value that mentions <parameter=x> once got a stray closer appended
        body = ("<function=edit_file>\n<parameter=path>\na\n</parameter>\n<parameter=new_string>\n"
                "see <parameter=x> here\n</parameter>\n</function>")
        with Setting(TOOL_SKIPPED_CLOSER=True):
            self.assertEqual(call(body, EDIT_TOOL)["new_string"], "see <parameter=x> here")

    def test_on_the_reviewers_inputs(self):
        # (b) the value mentions an opener and the model also left out </function>
        head = "<function=edit_file>\n<parameter=path>\na\n</parameter>\n<parameter=new_string>\nsee <parameter=x> here\n</parameter>\n"
        with Setting(TOOL_SKIPPED_CLOSER=True):
            for text in (f"<tool_call>\n{head}</tool_call>", f"<tool_call>\n{head}</function>\n</tool_call>"):
                for step in (1, 5, 1000):
                    self.assertEqual(stream(text, [EDIT_TOOL], step),
                                     [{"path": "a", "new_string": "see <parameter=x> here"}], (text, step))
            # a declared sibling opened mid-line is prose, not a skipped closer
            prose = "<function=edit_file>\n<parameter=new_string>\nuse <parameter=limit> to cap\n</parameter>\n</function>"
            self.assertEqual(call(prose, EDIT_TOOL), {"new_string": "use <parameter=limit> to cap"})
            # a nested pair with no closer anywhere after it ends where it did before the rule
            pair = "<function=edit_file>\n<parameter=new_string>\n<parameter=limit>1</parameter>\n</tool_call>"
            self.assertEqual(call(pair, EDIT_TOOL)["new_string"], "<parameter=limit>1")

    def test_on_in_a_stream_with_the_wrapper(self):
        text = f"<tool_call>\n{self.SKIPPED}\n</tool_call>"
        want = [{"path": "docs/tools.md", "new_string": "<parameter=limit>25</parameter>"}]
        with Setting(TOOL_SKIPPED_CLOSER=True):
            for step in (3, 11, 1000):
                self.assertEqual(stream(text, [EDIT_TOOL], step), want, step)
            self.assertEqual(stream(f"<tool_call>\n{self.CLOSED}\n</tool_call>", [EDIT_TOOL]), want)


if __name__ == "__main__":
    unittest.main()
