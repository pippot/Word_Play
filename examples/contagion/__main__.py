"""
Run the contagion benchmark: every domain x condition on the same seeds.

    python -m examples.contagion --episodes 5 --out pilot.jsonl    # the pilot, 4 domains x 4 conditions
    python -m examples.contagion --domains medical oversight        # a subset of domains
    python -m examples.contagion --no-results                       # nothing can contradict the note
    python -m examples.contagion --conditions placebo misaligned --seat-rounds 24   # steers the whole run
    python -m examples.contagion --out LOG.jsonl ...                # resume: finished episodes are skipped
    python -m examples.contagion.judge LOG.jsonl                    # then label posts for transmission

See run_pilot.sh for the whole pilot in one command. Models are served by
SGLang, as for Lifeline: SGLANG_BASE_URL, SGLANG_MODEL_NAME, SGLANG_API_KEY
(only if the server needs one), SGLANG_TIMEOUT.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

from .analyze import across_domains, analyze, format_domains, format_result, load, setting_of
from .domains import DOMAINS
from .game import CONDITIONS, CORE_CONDITIONS, Config, make_world, run_episode
from .judge import load_labels

LOGS_DIR = Path(__file__).resolve().parent / "logs"


def connect(key: str, model_name: str, base_url: str):
    """Register an SGLang model and return (model, the id the server really serves)."""
    from word_play.presets.models import LLM_MODEL_REGISTRY, register_sglang_model

    with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=10) as response:
        served = [m["id"] for m in json.loads(response.read()).get("data", []) if "id" in m]
    register_sglang_model(
        key, model_name=model_name, base_url=base_url, api_key_env="SGLANG_API_KEY",
        timeout=float(os.environ.get("SGLANG_TIMEOUT", "1800")),
    )
    # SGLang answers whatever name a request carries; record what it serves.
    return LLM_MODEL_REGISTRY.resolve(key), (served[0] if len(served) == 1 else model_name)


def finished_episodes(path: Path, setting: str) -> set[tuple[str, str, int]]:
    """(domain, condition, seed) already in the log. A log is one setting of
    the design, across domains: appending a different setting is refused."""
    if not path.exists():
        return set()
    done = set()
    for rec in load([path]):
        if setting_of(rec, with_domain=False) != setting:
            raise SystemExit(f"{path} holds episodes of a different setting; use a new --out")
        done.add((rec["config"]["domain"], rec["condition"], rec["seed"]))
    return done


def run_all(
    cfg: Config, *, domains, conditions, seeds, courier_model, seat_model, models: dict, out: Path,
    parallel: int = 16, seat_thinking: bool = False,
) -> Path:
    if "misaligned_direct" in conditions and not seat_thinking:
        # It always reasons; its placebo must have the same seat, or nothing pairs.
        raise SystemExit("misaligned_direct runs with a reasoning seat: add --seat-thinking")
    setting = setting_of({"config": asdict(cfg), "models": models, "seat_thinking": seat_thinking}, with_domain=False)
    done = finished_episodes(out, setting)
    jobs = [(d, c, s) for s in seeds for d in domains for c in conditions if (d, c, s) not in done]
    print(f"{len(jobs)} episodes to run ({len(done)} already in {out})")
    out.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def play(domain: str, condition: str, seed: int) -> dict:
        domain_cfg = replace(cfg, domain=domain)
        rec = run_episode(make_world(seed, domain_cfg), domain_cfg, condition, courier_model, seat_model,
                          seat_thinking=seat_thinking)
        rec["models"] = models
        return rec

    with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool, out.open("a", encoding="utf-8") as fh:
        futures = {pool.submit(play, *job): job for job in jobs}
        for n, future in enumerate(as_completed(futures), start=1):
            domain, condition, seed = futures[future]
            try:
                rec = future.result()
            except Exception as exc:  # one broken episode must not cost the rest
                print(f"  [{n}/{len(jobs)}] {domain} {condition} seed {seed} FAILED: {type(exc).__name__}: {exc}")
                continue
            with lock:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
            print(f"  [{n}/{len(jobs)}] {domain} {condition} seed {seed} done ({rec['failures']} unusable replies)")
    return out


def report(out: Path) -> str:
    records, labels = load([out]), load_labels([out])
    parts = [format_result(r) for r in analyze(records, labels)]
    parts += [format_domains(s) for s in across_domains(records, labels)]
    return "\n\n".join(parts)


def main(argv: list[str] | None = None) -> None:
    d = Config()
    p = argparse.ArgumentParser(description="Contagion: does a planted belief or value outlive its source?")
    p.add_argument("--episodes", type=int, default=30, help="seeds per domain and condition")
    p.add_argument("--first-seed", type=int, default=0)
    p.add_argument("--domains", nargs="+", choices=sorted(DOMAINS), default=sorted(DOMAINS))
    p.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CORE_CONDITIONS),
                   help="default: the core, " + " ".join(CORE_CONDITIONS))
    p.add_argument("--agents", type=int, default=d.agents, help="instances on the team at any time")
    p.add_argument("--tenure", type=int, default=d.tenure, help="rounds an ordinary instance stays")
    p.add_argument("--seat-rounds", type=int, default=d.seat_rounds, help="rounds the seat holder stays")
    p.add_argument("--rounds", type=int, default=d.rounds)
    p.add_argument("--scrollback", type=int, default=d.scrollback, help="channel rounds a new instance loads")
    p.add_argument("--noise", type=float, default=d.noise, help="sd of one report (before the domain's scale)")
    p.add_argument("--no-results", action="store_true", help="instances never learn what their submissions achieved")
    p.add_argument("--mandatory-posts", action="store_true", help="every instance must post every round")
    p.add_argument("--warning", action="store_true", help="operations warns the channel may carry wrong or self-serving claims")
    p.add_argument("--model", default=os.environ.get("SGLANG_MODEL_NAME", "Qwen/Qwen3-27B"))
    p.add_argument("--base-url", default=os.environ.get("SGLANG_BASE_URL", "http://localhost:30000/v1"))
    p.add_argument("--seat-model", help="model of the seat holder (every condition); default: the instances'")
    p.add_argument("--seat-base-url", help="server for --seat-model; default: the instances'")
    p.add_argument("--seat-thinking", action="store_true", help="seat holder posts with Qwen thinking on")
    p.add_argument("--parallel", type=int, default=16, help="episodes played at once")
    p.add_argument("--out", help="log file (JSONL); an existing one is resumed")
    args = p.parse_args(argv)

    cfg = Config(
        agents=args.agents, tenure=args.tenure, seat_rounds=args.seat_rounds, rounds=args.rounds,
        scrollback=args.scrollback, noise=args.noise, results=not args.no_results,
        mandatory_posts=args.mandatory_posts, warning=args.warning,
    )
    courier_model, courier_name = connect("contagion-courier", args.model, args.base_url)
    seat_target = (args.seat_model or args.model, args.seat_base_url or args.base_url)
    if seat_target == (args.model, args.base_url):
        seat_model, seat_name = courier_model, courier_name
    else:
        seat_model, seat_name = connect("contagion-seat", *seat_target)

    label = "_".join(
        [f"noise{cfg.noise:g}", f"scroll{cfg.scrollback}"]
        + ([] if cfg.results else ["noresults"]) + (["mandatory"] if cfg.mandatory_posts else [])
        + (["warning"] if cfg.warning else [])
    )
    out = Path(args.out) if args.out else LOGS_DIR / f"contagion_{datetime.now():%Y%m%d_%H%M%S}_{label}.jsonl"
    run_all(
        cfg, domains=args.domains, conditions=args.conditions,
        seeds=range(args.first_seed, args.first_seed + args.episodes),
        courier_model=courier_model, seat_model=seat_model, models={"courier": courier_name, "seat": seat_name},
        out=out, parallel=args.parallel, seat_thinking=args.seat_thinking,
    )
    print()
    print(report(out))
    print(f"\nLog: {out}")


if __name__ == "__main__":
    main()
