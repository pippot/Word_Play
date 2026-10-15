"""
Decision models ("System One" models: Jev and its open reproductions) as the
part of an instance that decides.

A decision model writes nothing. Given a state and a question with a fixed
set of answers, it returns a probability for every answer in one forward
pass. So in a team it can only fill the two private calls -- the planning
sheet and the submission -- while posts in the channel are still written by
the generative model. Each instance is then one deployment of both: an LLM
that talks, a decision model that decides, reading the same console.

Two ways to reach a model. `SystemOneDecider` asks a trained model's own
server (POST /v1/systemone): pplx-decider-27B, the Decider family, JevK5's
server -- the model reads its trained layout with its own head and
calibration, so what is tested is the model as shipped. `Decider` reads
answer letters through SGLang: JevK5 (alibiserikbay/JevK5-9B, JevK5 for the 4B) is Qwen3.5 with a
merged LoRA, and its readout is SemIf's protocol, which only needs the
next-token logprobs of the answer letters. It ships text-only, which SGLang
does not load; tools/make_jevk5_sglang.py repackages it, weights unchanged.
That protocol is reproduced here exactly (from
allebee/jevk5's prompt.py, MIT/Apache-2.0):

  system  a fixed instruction to answer with one letter
  user    JSON {"evidence", "criterion", "options": [{"letter", "description"}]}
  reply   one token; softmax over the letters' logprobs / the model's temperature

The same readout works on any chat model (the untrained base reproduces
SemIf's answers), so pointing it at the generative server gives a control:
the same weights, deciding by readout instead of by writing JSON.

How the two private calls map onto decision questions:
  submission     a `choice` question over the three options; the choice is
                 the most likely one, and the distribution is logged.
  planning sheet one `score` question per option, levels 0..10 steps of the
                 domain's scale (0..9 through /v1/systemone, its maximum); the estimate is the expected level, so it is
                 in the domain's units like a generative instance's estimate.
"""

from __future__ import annotations

import json
import math
import urllib.request

from .domains import Domain

LETTERS = "ABCDEFGHIJKLMNOP"  # one pass answers at most 16 options
SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)
# Calibration temperatures from each model's jevk5_config.json (Hub main).
# Anything else -- e.g. an untrained chat model -- is read at 1.0.
TEMPERATURES = {
    "alibiserikbay/JevK5-9B": 1.316,  # v0.3.3
    "alibiserikbay/JevK5": 1.22,      # v0.3 (4B)
}
LEVELS = 11            # planning-sheet levels: 0, 1, ..., 10 steps of the domain's scale
MISSING_MARGIN = 5.0   # a letter outside the top logprobs counts this far below the lowest seen
TOP_LOGPROBS = 20


class Decider:
    """A decision model behind an OpenAI-compatible server (SGLang, vLLM)."""

    def __init__(self, model: str, base_url: str, temperature: float | None = None,
                 timeout: float = 120.0, api_key: str | None = None) -> None:
        self.model = model
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.temperature = temperature if temperature is not None else TEMPERATURES.get(model, 1.0)
        self.timeout = timeout
        self.api_key = api_key

    def describe(self) -> dict:
        return {"model": self.model, "temperature": self.temperature}

    # ------------------------------------------------------------- readout

    def complete(self, body: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(self.url, data=json.dumps(body).encode(), headers=headers)
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read())

    def letter_logprobs(self, messages: list[dict]) -> dict[str, float]:
        """Next-token logprob of every letter among the top ones. Thinking is
        off: the prompt must end in an empty think block, as JevK5 was trained."""
        body = {
            "model": self.model, "messages": messages, "max_tokens": 1, "temperature": 0,
            "logprobs": True, "top_logprobs": TOP_LOGPROBS,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        top = self.complete(body)["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        seen: dict[str, float] = {}
        for entry in top:
            token = entry["token"].strip()
            if token in LETTERS and len(token) == 1:
                seen[token] = max(seen.get(token, -math.inf), entry["logprob"])
        return seen

    def probabilities(self, evidence: str, criterion: str, options: list[str]) -> tuple[list[float], bool]:
        """(calibrated probability per option, whether every letter was among
        the top logprobs). A missing letter is floored below the lowest seen."""
        if not 2 <= len(options) <= len(LETTERS):
            raise ValueError(f"one pass reads 2 to {len(LETTERS)} options, got {len(options)}")
        payload = {
            "evidence": evidence,
            "criterion": criterion,
            "options": [{"letter": LETTERS[i], "description": d} for i, d in enumerate(options)],
        }
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        seen = self.letter_logprobs(messages)
        if not seen:
            raise ValueError("no answer letter among the top logprobs")
        floor = min(seen.values()) - MISSING_MARGIN
        logprobs = [seen.get(LETTERS[i], floor) for i in range(len(options))]
        top = max(logprobs)
        weights = [math.exp((z - top) / self.temperature) for z in logprobs]
        total = sum(weights)
        return [w / total for w in weights], all(LETTERS[i] in seen for i in range(len(options)))

    # ------------------------------------------------------------- the two private calls

    def choose(self, domain: Domain, evidence: str, criterion: str) -> tuple[str, dict]:
        """The submission: (option, record of the distribution)."""
        names = list(domain.names)
        # SemIf's mapping of a choice question: "key: description".
        probs, complete = self.probabilities(evidence, criterion, [f"{n}: {n}" for n in names])
        dist = dict(zip(names, probs))
        return max(dist, key=dist.get), {"probs": dist, "complete": complete}

    def estimate(self, domain: Domain, evidence: str, criterion_of) -> tuple[dict[str, float], dict]:
        """The planning sheet: (estimate per option in the domain's units,
        record). `criterion_of(option)` is the question for one option."""
        levels = [f"{i}: {i * domain.scale} {domain.estimate}" for i in range(LEVELS)]
        estimates, record, complete = {}, {}, True
        for name in domain.names:
            probs, ok = self.probabilities(evidence, criterion_of(name), levels)
            estimates[name] = domain.scale * sum(i * p for i, p in enumerate(probs))
            record[name] = [round(p, 4) for p in probs]
            complete &= ok
        return estimates, {"levels": record, "complete": complete}


# ============================================================================
# A trained decision model behind its own server (POST /v1/systemone)
# ============================================================================
# The trained 27B decision models (pplx-decider / AutoJev, autojev-serve) and
# the Decider family (Mapika, uvicorn decider.serve:app) read decisions through
# their own heads or prompts, which SGLang cannot reproduce. Their servers all
# speak TypeSafe's wire format, so they are queried as shipped: each model sees
# the state and the question in the layout it was trained on, with its own
# calibration. JevK5's server (jevk5-serve) speaks it too.

SYSTEMONE_LEVELS = 10  # the most score levels these servers accept: 0..9 steps of the scale
BUSY = (429, 503, 529)  # single-GPU servers refuse while busy; wait and retry


class SystemOneDecider:
    """A decision model behind a /v1/systemone server."""

    def __init__(self, model: str, base_url: str, timeout: float = 600.0, api_key: str | None = None) -> None:
        self.model = model  # the label in the logs, e.g. perplexity-ai/pplx-decider-v1.1-27b
        root = base_url.rstrip("/")
        self.root = root[: -len("/v1")] if root.endswith("/v1") else root
        self.timeout = timeout
        self.api_key = api_key
        self._served: str | None = None

    def describe(self) -> dict:
        return {"model": self.model, "api": "systemone", "levels": SYSTEMONE_LEVELS}

    def _request(self, path: str, body: dict | None = None) -> dict:
        import time
        import urllib.error

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = None if body is None else json.dumps(body).encode()
        deadline = time.monotonic() + self.timeout
        while True:
            request = urllib.request.Request(self.root + path, data=data, headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read())
            except urllib.error.HTTPError as exc:
                if exc.code not in BUSY or time.monotonic() > deadline:
                    detail = exc.read().decode(errors="replace")[:300]
                    raise RuntimeError(f"HTTP {exc.code} from {path}: {detail}") from exc
                time.sleep(float(exc.headers.get("Retry-After") or 0.2))

    def served_name(self) -> str:
        """The model id the server accepts (autojev-serve refuses any other)."""
        if self._served is None:
            try:
                data = self._request("/v1/models").get("data") or []
                self._served = data[0]["id"] if data else "jev-latest"
            except Exception:
                self._served = "jev-latest"
        return self._served

    def ask(self, evidence: str, questions: dict) -> dict:
        reply = self._request(
            "/v1/systemone", {"model": self.served_name(), "state": evidence, "questions": questions})
        return reply.get("answers", reply)

    def choose(self, domain: Domain, evidence: str, criterion: str) -> tuple[str, dict]:
        names = list(domain.names)
        answer = self.ask(evidence, {"act": {"type": "choice", "instructions": criterion,
                                             "criteria": {n: n for n in names}}})["act"]
        dist = {n: float(answer["probabilities"][n]) for n in names}
        return max(dist, key=dist.get), {"probs": dist, "complete": True}

    def estimate(self, domain: Domain, evidence: str, criterion_of) -> tuple[dict[str, float], dict]:
        levels = [f"{i * domain.scale} {domain.estimate}" for i in range(SYSTEMONE_LEVELS)]
        ids = {f"o{i}": name for i, name in enumerate(domain.names)}
        answers = self.ask(evidence, {
            key: {"type": "score", "instructions": criterion_of(name), "criteria": levels} for key, name in ids.items()
        })
        estimates, record = {}, {}
        for key, name in ids.items():
            probs = [float(answers[key]["probabilities"][str(i)]) for i in range(SYSTEMONE_LEVELS)]
            estimates[name] = domain.scale * sum(i * p for i, p in enumerate(probs))
            record[name] = [round(p, 4) for p in probs]
        return estimates, {"levels": record, "complete": True}
