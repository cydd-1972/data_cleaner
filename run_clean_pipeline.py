#!/usr/bin/env python3
"""
一步完成 LoCoMo 清洗流水线：生成答案 → 评估分类 → 去重与按比例抽样。

依赖与分步脚本相同（见 README）。通过子进程依次执行三步，每步结束后释放显存，避免 8B 与 14B 模型同驻 GPU。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "config"


def _pythonpath() -> str:
    sep = os.pathsep
    parts = [str(ROOT), str(CONFIG_DIR)]
    extra = os.environ.get("PYTHONPATH", "")
    if extra:
        parts.append(extra)
    return sep.join(parts)


def run_step(script: Path, extra_args: list[str]) -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = _pythonpath()
    cmd = [sys.executable, str(script), *extra_args]
    print(f"\n{'=' * 60}\n执行: {' '.join(cmd)}\n{'=' * 60}\n", flush=True)
    return subprocess.call(cmd, cwd=str(ROOT), env=env)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="一步清洗：依次运行 step1_generate → step2_evaluate → step3_filter。"
        " 未识别的参数会原样传给 step2_evaluate（如 --batch-size 64）。",
    )
    parser.add_argument(
        "--skip-step1",
        action="store_true",
        help="跳过生成（已有 generated_answers 时使用）",
    )
    parser.add_argument(
        "--skip-step2",
        action="store_true",
        help="跳过评估（已有 easy/medium/hard 时使用）",
    )
    parser.add_argument(
        "--skip-step3",
        action="store_true",
        help="跳过去重与按比例抽样",
    )
    args, step2_extra = parser.parse_known_args()

    steps: list[tuple[str, Path, list[str]]] = []
    if not args.skip_step1:
        steps.append(("step1 生成答案", ROOT / "step1_generate.py", []))
    if not args.skip_step2:
        steps.append(("step2 评估与分类", ROOT / "step2_evaluate.py", step2_extra))
    if not args.skip_step3:
        steps.append(("step3 去重与抽样", ROOT / "step3_filter.py", []))

    if not steps:
        print("未选择任何步骤。", file=sys.stderr)
        sys.exit(2)

    for title, script, extra in steps:
        if not script.is_file():
            print(f"找不到脚本: {script}", file=sys.stderr)
            sys.exit(1)
        code = run_step(script, extra)
        if code != 0:
            print(f"\n失败: {title} 退出码 {code}", file=sys.stderr)
            sys.exit(code)

    print("\n流水线全部完成。")


if __name__ == "__main__":
    main()
