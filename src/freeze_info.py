"""Print the facts that `results/FROZEN_CONFIG.md` must record, without any API call, database access or test-split read.

    python src/freeze_info.py            # human-readable
    python src/freeze_info.py --json     # machine-readable

Hashes are SHA-256 of the files as they are on disk (data files are hashed, never printed). The API key is never read.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import agent  # noqa: E402
import run_agent_eval as harness  # noqa: E402

FILES = ["src/agent.py", "src/run_agent_eval.py", "src/retrieval.py", "src/compare_runs.py", "results/SELECTION.md",
         "data/chunks.json", "data/questions_public.json", "data/questions.json", "data/absent_questions.json"]


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:  # git missing: report it, do not fail
        return "unavailable"


def collect() -> dict:
    return {
        "agent_version": agent.AGENT_VERSION,
        "files_sha256": {f: harness.sha256_file(ROOT / f) if (ROOT / f).exists() else None for f in FILES},
        "prompt_sha256": {name: harness.sha256_text(text) for name, text in agent.PROMPTS.items()},
        "defaults": harness.DEFAULTS,
        "max_test_runs_total": harness.MAX_TEST_RUNS,
        "model_default": "OPENAI_MODEL from .env or gpt-4o-mini (the model name is recorded in each summary.json)",
        "controller_settings_base": agent.OpenAIController(None, "m", prompt="base").settings() | {"system_prompt": "(see prompt_sha256)"},
        "packages": harness.package_versions(),
        "git_head": git("rev-parse", "HEAD"),
        "git_dirty_files_in_src_tests_results": git("status", "--short", "--", "src", "tests", "results/SELECTION.md").splitlines(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    info = collect()
    if ap.parse_args().json:
        print(json.dumps(info, indent=1, ensure_ascii=False))
        return
    print(f"agent version: {info['agent_version']}   defaults: {info['defaults']}   global test cap: {info['max_test_runs_total']}")
    for k, v in info["files_sha256"].items():
        print(f"  {v or 'MISSING':64}  {k}")
    for k, v in info["prompt_sha256"].items():
        print(f"  {v:64}  prompt '{k}'")
    print("packages:", info["packages"])
    print("git head:", info["git_head"])


if __name__ == "__main__":
    main()
