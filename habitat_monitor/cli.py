"""Command-line interface: generate, replay, detect, evaluate, report, serve."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from habitat_monitor.evaluate import evaluate_ensemble, format_report
from habitat_monitor.genai import generate_report
from habitat_monitor.generate import generate_labeled_stream, generate_split, records_to_jsonl
from habitat_monitor.ingest import load_jsonl, load_labeled_jsonl
from habitat_monitor.optimize import recommend_interval
from habitat_monitor.pipeline import Pipeline
from habitat_monitor.protocol import encode_frame


def _cmd_generate(args: argparse.Namespace) -> int:
    if args.split:
        train, test = generate_split(args.n_train, args.n_test, seed=args.seed)
        records = train + test
    else:
        records = generate_labeled_stream(args.n, seed=args.seed, inject=not args.clean)
    text = records_to_jsonl(records)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    n_pos = sum(1 for r in records if r.label == 1)
    print(f"wrote {len(records)} records ({n_pos} labeled anomalies)", file=sys.stderr)
    return 0


def _cmd_replay(args: argparse.Namespace) -> int:
    frames = load_jsonl(Path(args.path))
    for frame in frames:
        sys.stdout.write(encode_frame(frame) + "\n")
    print(f"replayed {len(frames)} frames", file=sys.stderr)
    return 0


def _cmd_detect(args: argparse.Namespace) -> int:
    records = load_labeled_jsonl(Path(args.path))
    frames = [r.frame for r in records]
    n_fit = min(args.fit, len(frames))
    if n_fit < 16:
        print("need at least 16 frames to fit", file=sys.stderr)
        return 2
    pipe = Pipeline()
    pipe.fit(frames[:n_fit])
    result = pipe.run(frames[n_fit:] if args.holdout else frames)
    for det in result.detections:
        if not args.anomalies_only or det.is_anomaly:
            sys.stdout.write(det.model_dump_json() + "\n")
    print(
        f"anomalies={sum(1 for d in result.detections if d.is_anomaly)} "
        f"rate={result.anomaly_rate:.3f} interval_ms={result.sampling.interval_ms}",
        file=sys.stderr,
    )
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    if args.path:
        records = load_labeled_jsonl(Path(args.path))
        split_at = args.fit if args.fit < len(records) else max(16, len(records) // 2)
        train, test = records[:split_at], records[split_at:]
    else:
        train, test = generate_split(args.n_train, args.n_test, seed=args.seed)
    metrics = evaluate_ensemble(train, test)
    text = format_report(metrics)
    sys.stdout.write(text)
    if args.json:
        Path(args.json).write_text(json.dumps([m.as_dict() for m in metrics], indent=2), encoding="utf-8")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    records = load_labeled_jsonl(Path(args.path))
    frames = [r.frame for r in records]
    n_fit = min(args.fit, len(frames))
    pipe = Pipeline()
    pipe.fit(frames[:n_fit])
    result = pipe.run(frames)
    sampling = result.sampling if args.optimize else recommend_interval(result.anomaly_rate)
    report = generate_report(result.detections, sampling)
    sys.stdout.write(report.model_dump_json(indent=2) + "\n")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("habitat_monitor.api:app", host=args.host, port=args.port, reload=False)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="habitat-monitor",
        description="Habitat / analog-mission environmental monitoring",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    gen = sub.add_parser("generate", help="Write a labeled synthetic JSONL stream")
    gen.add_argument("--n", type=int, default=240)
    gen.add_argument("--n-train", type=int, default=360)
    gen.add_argument("--n-test", type=int, default=240)
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--out", type=str, default="")
    gen.add_argument("--clean", action="store_true", help="Do not inject anomalies")
    gen.add_argument("--split", action="store_true", help="Clean train + labeled test")
    gen.set_defaults(func=_cmd_generate)

    replay = sub.add_parser("replay", help="Print protocol frames from a JSONL file")
    replay.add_argument("path")
    replay.set_defaults(func=_cmd_replay)

    detect = sub.add_parser("detect", help="Fit and score a JSONL stream")
    detect.add_argument("path")
    detect.add_argument("--fit", type=int, default=160, help="Frames used to fit")
    detect.add_argument("--holdout", action="store_true", help="Score only frames after --fit")
    detect.add_argument("--anomalies-only", action="store_true")
    detect.set_defaults(func=_cmd_detect)

    ev = sub.add_parser("evaluate", help="Print P/R/F1 on a labeled split")
    ev.add_argument("path", nargs="?", default="")
    ev.add_argument("--n-train", type=int, default=360)
    ev.add_argument("--n-test", type=int, default=240)
    ev.add_argument("--seed", type=int, default=42)
    ev.add_argument("--fit", type=int, default=360)
    ev.add_argument("--json", dest="json", default="")
    ev.set_defaults(func=_cmd_evaluate)

    rep = sub.add_parser("report", help="Structured report (mock LLM by default)")
    rep.add_argument("path")
    rep.add_argument("--fit", type=int, default=160)
    rep.add_argument("--optimize", action="store_true")
    rep.set_defaults(func=_cmd_report)

    serve = sub.add_parser("serve", help="Run the HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=_cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
