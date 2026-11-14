"""POST /v1/completions: the raw prompt (no chat template), the chat path's sampling fields, stream and not, usage."""
import http.client
import json
from pathlib import Path
import unittest

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, StrataEngine, serve

TEMPLATE = ChatTemplate(Path(__file__).with_name("chat_template.jinja"))


class RecordingEngine(MockEngine):
    """MockEngine that keeps what the server handed to generate()."""

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.last_sampling, self.last_max_new = dict(sampling or {}), max_new
        return super().generate(ids, max_new, sampling, cancel, embeddings)


class Completions(unittest.TestCase):
    def start(self, script="Hello there", max_context=4096):
        self.tok = ByteTokenizer()
        self.engine = RecordingEngine(self.tok, script, max_context=max_context)
        service = Service(self.engine, self.tok, TEMPLATE)
        self.httpd = serve(service, port=0)
        self.addCleanup(self._stop)
        return service

    def _stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def post(self, body, path="/v1/completions"):
        c = http.client.HTTPConnection("127.0.0.1", self.httpd.server_address[1], timeout=10)
        try:
            c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
            r = c.getresponse()
            return r.status, r.read()
        finally:
            c.close()

    def ok(self, body):
        status, raw = self.post(body)
        self.assertEqual(status, 200, raw)
        return json.loads(raw)

    def events(self, raw):
        lines = [x[len("data: "):] for x in raw.decode().split("\n\n") if x.startswith("data: ")]
        self.assertEqual(lines[-1], "[DONE]")
        return [json.loads(x) for x in lines[:-1]]

    def test_the_prompt_goes_to_the_engine_as_it_is(self):
        self.start()
        out = self.ok({"prompt": "def fib(n):", "max_tokens": 32})
        self.assertEqual(self.tok.decode(self.engine.last_prompt), "def fib(n):")   # no <|im_start|>, no template
        self.assertEqual(out["object"], "text_completion")
        self.assertTrue(out["id"].startswith("cmpl-"))
        self.assertEqual(out["choices"][0]["text"], "Hello there")
        self.assertEqual(out["choices"][0]["finish_reason"], "stop")
        self.assertIsNone(out["choices"][0]["logprobs"])

    def test_usage_counts_prompt_and_completion(self):
        self.start()
        out = self.ok({"prompt": "abc", "max_tokens": 32})
        u = out["usage"]
        self.assertEqual(u["prompt_tokens"], 3)
        self.assertGreaterEqual(u["completion_tokens"], len("Hello there"))
        self.assertEqual(u["total_tokens"], u["prompt_tokens"] + u["completion_tokens"])

    def test_max_tokens_cuts_the_reply(self):
        self.start()
        out = self.ok({"prompt": "abc", "max_tokens": 5})
        self.assertEqual(out["choices"][0]["text"], "Hello")
        self.assertEqual(out["choices"][0]["finish_reason"], "length")
        self.assertEqual(out["usage"]["completion_tokens"], 5)
        self.assertEqual(self.engine.last_max_new, 5)

    def test_prompt_as_token_ids_and_a_list_of_one(self):
        self.start()
        self.ok({"prompt": [72, 105], "max_tokens": 4})
        self.assertEqual(self.engine.last_prompt, [72, 105])
        self.ok({"prompt": [[72, 105, 33]], "max_tokens": 4})
        self.assertEqual(self.engine.last_prompt, [72, 105, 33])
        self.ok({"prompt": ["Hi"], "max_tokens": 4})
        self.assertEqual(self.tok.decode(self.engine.last_prompt), "Hi")

    def test_text_is_not_read_as_thinking_or_a_tool_call(self):
        self.start("<think>x</think> y <tool_call><function=f></function></tool_call>")
        out = self.ok({"prompt": "p", "max_tokens": 200})
        self.assertEqual(out["choices"][0]["text"], "<think>x</think> y <tool_call><function=f></function></tool_call>")

    def test_the_chat_sampling_fields_reach_the_engine(self):
        self.start()
        self.ok({"prompt": "p", "max_tokens": 4, "temperature": 0.6, "top_p": 0.9, "top_k": 20, "seed": 7,
                 "presence_penalty": 0.5, "ignore_eos": True})
        s = self.engine.last_sampling
        self.assertEqual((s["temperature"], s["top_p"], s["top_k"], s["seed"], s["presence_penalty"]),
                         (0.6, 0.9, 20, 7, 0.5))

    def test_stop_string_cuts_the_text(self):
        self.start()
        out = self.ok({"prompt": "p", "max_tokens": 32, "stop": "the"})
        self.assertEqual(out["choices"][0]["text"], "Hello ")
        self.assertEqual(out["choices"][0]["finish_reason"], "stop")

    def test_echo_puts_the_prompt_first(self):
        self.start()
        out = self.ok({"prompt": "abc", "max_tokens": 32, "echo": True})
        self.assertEqual(out["choices"][0]["text"], "abcHello there")
        out = self.ok({"prompt": [97, 98], "max_tokens": 3, "echo": True})
        self.assertEqual(out["choices"][0]["text"], "abHel")

    def test_stream_chunks_then_done_without_usage(self):
        self.start()
        status, raw = self.post({"prompt": "abc", "max_tokens": 32, "stream": True})
        self.assertEqual(status, 200)
        ev = self.events(raw)
        self.assertTrue(all(e["object"] == "text_completion" for e in ev))
        self.assertEqual("".join(c["text"] for e in ev for c in e["choices"]), "Hello there")
        self.assertEqual(ev[-1]["choices"][0]["finish_reason"], "stop")
        self.assertTrue(all("usage" not in e for e in ev))
        self.assertEqual(len({e["id"] for e in ev}), 1)

    def test_stream_include_usage_adds_one_last_chunk_without_choices(self):
        self.start()
        status, raw = self.post({"prompt": "abc", "max_tokens": 5, "stream": True,
                                 "stream_options": {"include_usage": True}})
        self.assertEqual(status, 200)
        ev = self.events(raw)
        self.assertEqual(ev[-1]["choices"], [])
        self.assertEqual(ev[-1]["usage"]["prompt_tokens"], 3)
        self.assertEqual(ev[-1]["usage"]["completion_tokens"], 5)
        self.assertEqual(ev[-2]["choices"][0]["finish_reason"], "length")
        self.assertEqual(sum("usage" in e for e in ev), 1)

    def test_bad_requests_are_400(self):
        self.start(max_context=64)
        for body in ({}, {"prompt": ""}, {"prompt": 5}, {"prompt": ["a", "b"]}, {"prompt": [1.5]}, {"prompt": [-1]},
                     {"prompt": "a", "n": 2}, {"prompt": "a", "best_of": 3}, {"prompt": "a", "suffix": "z"},
                     {"prompt": "a", "logprobs": 3}, {"prompt": "a", "max_tokens": "5"}, {"prompt": "a", "echo": "yes"},
                     {"prompt": "a", "stop": 5}, {"prompt": "a", "temperature": "hot"},
                     {"prompt": "a", "stream_options": 1}, {"prompt": "a" * 100, "max_tokens": 5}):
            status, raw = self.post(body)
            self.assertEqual(status, 400, (body, raw))
            self.assertIn("error", json.loads(raw))
        self.ok({"prompt": "a", "max_tokens": 500})          # as the chat path: shortened to the room (fit_max_tokens)
        self.assertEqual(self.engine.last_max_new, 64 - 8 - 1)

    def test_ignore_eos_runs_past_the_end_token_on_both_endpoints(self):
        self.start("a<|im_end|>b")
        out = self.ok({"prompt": "p", "max_tokens": 32})
        self.assertEqual((out["choices"][0]["text"], out["choices"][0]["finish_reason"]), ("a", "stop"))
        out = self.ok({"prompt": "p", "max_tokens": 32, "ignore_eos": True})
        self.assertEqual((out["choices"][0]["text"], out["choices"][0]["finish_reason"]), ("ab", "length"))
        self.assertEqual(out["usage"]["completion_tokens"], 4)          # a, <|im_end|>, b, the script's own end
        chat = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 32, "reasoning_effort": "none"}
        status, raw = self.post(chat, "/v1/chat/completions")
        self.assertEqual(json.loads(raw)["choices"][0]["message"]["content"], "a")
        status, raw = self.post({**chat, "ignore_eos": True}, "/v1/chat/completions")
        body = json.loads(raw)
        self.assertEqual((body["choices"][0]["message"]["content"], body["choices"][0]["finish_reason"]), ("ab", "length"))

    def test_ignore_eos_is_a_key_for_the_engine_only_when_true(self):
        self.assertEqual(StrataEngine.sampling_keys({"ignore_eos": True}), " ignore_eos=1")
        for v in (False, None, "yes", 1):
            self.assertEqual(StrataEngine.sampling_keys({"ignore_eos": v}), "")

    def test_chat_completions_is_untouched(self):
        self.start("ok")
        status, raw = self.post({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8,
                                 "reasoning_effort": "none"}, "/v1/chat/completions")
        self.assertEqual(status, 200, raw)
        body = json.loads(raw)
        self.assertEqual(body["object"], "chat.completion")
        self.assertIn("<|im_start|>", self.tok.decode(self.engine.last_prompt))   # the template, as before
        status, raw = self.post({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8, "stream": True,
                                 "reasoning_effort": "none"}, "/v1/chat/completions")
        self.assertEqual(status, 200)
        self.assertTrue(raw.endswith(b"data: [DONE]\n\n"))


if __name__ == "__main__":
    unittest.main()
