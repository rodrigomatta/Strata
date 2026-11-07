"""Determinism of the dsv41 serve pipeline (TK-0022): same input, same seed, same output — token for token — and
the same answer per request whatever order the requests arrive in.  The mock engine replays a scripted completion,
so everything these tests pin is the PIPELINE around the engine: render, encoding, streaming, DSML translation,
detokenisation, the JSON/SSE shapes (the engine's own run-to-run behaviour is the suite's GPU half — see the note
below).  `near-tied` flips are a different, categorised class: a numeric route change of ~4e-6 of the largest
output can flip a near-tied greedy token (docs/AMD_HIP.md:757) — that is arithmetic-tie sensitivity, not pipeline
nondeterminism, and it is why the engine's sampling takes a per-request seed.

    python -m unittest serve.test_determinism -v
"""
import json
import threading
import unittest
from http.client import HTTPConnection
from pathlib import Path

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, serve

ROOT = Path(__file__).resolve().parent.parent
CRLF = b"\r\n"
RUNS = 5


class PromptKeyedEngine(MockEngine):
    """A mock that answers by PROMPT, not by arrival position: each script is keyed by a marker string the prompt
    carries, so "the same set of requests in a different order" can mean the same thing it means for a real
    engine (same prompt, same answer)."""

    def __init__(self, tok, by_key: dict[str, str], **kw):
        # one script to the parent (with several it re-picks by arrival position inside generate(), which would
        # overwrite the marker pick below); this class swaps the script itself, by prompt.  The replay is locked:
        # the server also generates session titles and the like on its own threads, and the mock's script is
        # shared state (MockEngine's own docstring warns about unplanned requests)
        super().__init__(tok, list(by_key.values())[0], **kw)
        tok_enc = lambda s: tok.encode(s, parse_special=True)  # noqa: E731
        self.answers = [(tok_enc(k), tok_enc(v)) for k, v in by_key.items()]
        self._replay_lock = threading.Lock()

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        with self._replay_lock:
            for marker, answer in self.answers:
                if any(ids[j:j + len(marker)] == marker for j in range(len(ids) - len(marker) + 1)):
                    self.script = self._with_end(answer)
                    break
            yield from super().generate(ids, max_new, sampling, cancel, embeddings)

    def _with_end(self, answer):
        """The parent's scripts carry the end-of-turn token; a swapped-in answer needs it too."""
        return answer if not self.script or self.script[-1] in answer else answer + [self.script[-1]]


class Determinism(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = ByteTokenizer()
        cls.engine = PromptKeyedEngine(cls.tok, {
            # the template's assistant turn opens <think>; the scripts close it, so the reply lands in `content`
            "ZEBRA": "</think>The zebra crossing is white and black.",
            "YAK": "</think>The yak grazes high on the plateau.",
        }, max_context=4096)
        cls.svc = Service(cls.engine, cls.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.port = cls.httpd.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def post(self, path: str, body: dict):
        """-> (status, raw body bytes) of one JSON POST (http.client: chunked streaming bodies come decoded)."""
        conn = HTTPConnection("127.0.0.1", self.port, timeout=30)
        try:
            conn.request("POST", path, body=json.dumps(body),
                         headers={"Content-Type": "application/json", "Connection": "close"})
            resp = conn.getresponse()
            return resp.status, resp.read()
        finally:
            conn.close()

    @staticmethod
    def chat(marker: str, seed=None, stream=False):
        body = {
            "model": "ds4", "temperature": 0, "max_tokens": 64,
            "messages": [{"role": "user", "content": f"Tell me about the {marker}."}],
            "stream": stream,
        }
        if seed is not None:
            body["seed"] = seed
        return body

    @staticmethod
    def content_of(js: dict) -> str:
        return js["choices"][0]["message"]["content"]

    def stream_text(self, raw: bytes) -> str:
        """The text a client would render from an SSE body: every delta's content, concatenated, in order."""
        text = ""
        for line in raw.decode("utf-8", "replace").splitlines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            delta = json.loads(line[6:]).get("choices", [{}])[0].get("delta", {})
            text += delta.get("content", "")
        return text

    def test_render_and_encoding_run_to_run(self):
        """Same messages -> same token ids, every run: the pipeline's front half is a pure function of the input."""
        msgs = [self.chat("ZEBRA"), self.chat("YAK")]
        first = None
        for _ in range(RUNS):
            self.engine.last_prompt = []
            self.post("/v1/chat/completions", msgs[0])
            ids = list(self.engine.last_prompt)
            self.assertTrue(ids)
            if first is None:
                first = ids
            self.assertEqual(first, ids, "the same prompt must encode to the same ids on every run")

    def test_completion_run_to_run(self):
        """Same request -> the same completion content and finish reason on every run (N=%d).""" % RUNS
        seen = set()
        for _ in range(RUNS):
            status, raw = self.post("/v1/chat/completions", self.chat("ZEBRA"))
            self.assertEqual(status, 200)
            js = json.loads(raw)
            content = self.content_of(js)
            self.assertTrue(content, f"the reply must BE content (a <think>-only reply would pass vacuously): {js}")
            seen.add((content, js["choices"][0].get("finish_reason")))
        self.assertEqual(len(seen), 1, f"run-to-run divergence inside the serve pipeline: {seen}")

    def test_streaming_run_to_run(self):
        """Streamed: the concatenated deltas of the same request are identical on every run."""
        seen = set()
        for _ in range(RUNS):
            status, raw = self.post("/v1/chat/completions", self.chat("ZEBRA", stream=True))
            self.assertEqual(status, 200)
            text = self.stream_text(raw)
            self.assertTrue(text, "the streamed deltas must carry text (an empty stream would pass vacuously)")
            seen.add(text)
        self.assertEqual(len(seen), 1, f"streaming divergence inside the serve pipeline: {seen}")

    def test_admission_order(self):
        """The same two requests answered in either order -> the same answer per request (a prompt-keyed mock
        makes 'same request' mean what it means for a real engine: same prompt)."""
        def answer(marker):
            status, raw = self.post("/v1/chat/completions", self.chat(marker))
            self.assertEqual(status, 200)
            return self.content_of(json.loads(raw))
        zebra_first, yak_first = answer("ZEBRA"), answer("YAK")
        yak_second, zebra_second = answer("YAK"), answer("ZEBRA")
        self.assertEqual(zebra_first, zebra_second)
        self.assertEqual(yak_first, yak_second)
        self.assertIn("zebra", zebra_first.lower())
        self.assertIn("yak", yak_first.lower())

    def test_seed_reaches_the_engine(self):
        """A request's seed is the engine's per-request sampling seed (the near-tied class above): the pipeline
        must hand it through, and a request without one must not carry a stale one."""
        got = []

        class Recorder(PromptKeyedEngine):
            def generate(self, ids, max_new, sampling, cancel, embeddings=None):
                got.append(sampling)
                return super().generate(ids, max_new, sampling, cancel, embeddings)

        self.svc.engine = Recorder(self.tok, {"ZEBRA": "zebra", "YAK": "yak"}, max_context=4096)
        try:
            status, _ = self.post("/v1/chat/completions", self.chat("ZEBRA", seed=42))
            self.assertEqual(status, 200)
            # the mock's generate() sees the request's sampling as the parsed body; a real engine gets the same
            # numbers in the engine's own spelling (server.py's "seed=42" sampling line)
            self.assertTrue(got and got[-1].get("seed") == 42,
                            f"seed=42 must reach the engine's sampling, got {got[-1:]}")
            got.clear()
            status, _ = self.post("/v1/chat/completions", self.chat("ZEBRA"))
            self.assertEqual(status, 200)
            self.assertTrue(got and got[-1].get("seed") is None,
                            "a request without a seed must not carry one")
        finally:
            self.svc.engine = self.engine


if __name__ == "__main__":
    unittest.main()
