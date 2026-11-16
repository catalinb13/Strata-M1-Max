#!/usr/bin/env python3
"""Apply this fork's M1 Max tuning to a run config, and optionally start the server with it.

`metal/setup_mac.py` writes a Mac run config with the flags that are right for every Metal Mac. The settings that are
right for a 64 GB M1 Max serving a 60-82 GB expert pack are not universal, so they live here instead of in that shared
path: every value below was measured on this fork's M1 Max (docs/DETAILS.md, "Apple silicon: M1 Max (this fork)").

    metal/tools/m1max-serve.py --config strata-ud-iq4_xs-huihui.json            # show what would change
    metal/tools/m1max-serve.py --config ... --apply                            # write it
    metal/tools/m1max-serve.py --config ... --apply --start                    # write it and serve

Idempotent: running it twice changes nothing the second time. It never touches the model paths, the tokenizer, the port
or the vision block - only the engine args and env named below, plus the auto-session file.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

# -- args: flag -> value. A flag already in the config is set to this value; a flag missing is added, except --mtp / --spec,
# which are only pinned when a drafter is configured (a pack without one must not get a --mtp path to nothing).
ARGS = {
    "--prefill": "auto",
    "--spec": "2",
    "--lookup-max-t": "12",
    "--suffix-draft": "1",
    "--mmap-experts": None,            # present, no value
    "--kv": "fp16",                    # measured faster than int8 on decode here, ~3% slower on prefill
    # 1024 (2.73 GiB, 1,183 resident experts). Nine arms at arena 30 on 2026-11-16, two contexts: at 23.6K it is flat and
    # 4096 is worse (25.3 / 23.4 / 22.8 tok/s for 1024 / 2048 / 4096, misses hide behind the GPU there); at 75.4K it is
    # the best (25.6 against 512's 25.3 and cache-off's 24.8), because long-history routing is diverse enough that the
    # resident tier removes real waits. The depth the sessions run decides, and that is 1024.
    "--expert-cache": "1024",
    "--conversation-cache-mib": "8192",
    "--conversation-cache-slots": "6",
    "--conversation-cache-min-tokens": "2048",
}
# -- env: the measured engine switches. 30 GB is the arena: decode at 23.6k context measured 23.8 tok/s at 30 against
# 22.3/24.1 at 32 (the same band), with ~5.5 GB less resident.
ENV = {
    "STRATA_METAL_STREAM": "1",        # the expert arena reads from the SSD (the dense half stays resident)
    "STRATA_METAL_ARENA_GB": "30",
    "STRATA_METAL_LEAVE_SHARE": "0.35",
    "STRATA_METAL_PROMPT_CHUNK": "5120",
    "STRATA_METAL_PROMPT_MOE": "mpp",  # +5-8% warm pp2048, +18% cold, and it holds its speed instead of swinging
    "STRATA_METAL_GDN_SCAN": "scan_v", # +1-2% (changes the greedy continuation: opt-in, as here)
    "STRATA_METAL_NFCHECK": "1",       # the non-finite sentinel; costs ~nothing and #879 needs it
    "STRATA_NAN_CKPT_RESUME": "1",     # ... and this is what resumes after it fires
    "STRATA_METAL_F16": "1",
    # the prompt GEMM's block order: M on the grid's x, N on its y, so the blocks in flight share one N column and its
    # B slice stays in L2 instead of the whole B being walked per row tile (the tuned prompt path this box runs)
    "STRATA_METAL_PROMPT_GEMM_ORDER": "sw",
    # accuracy, not speed: the tail-skip drops missed experts every token routes below a rank. The default 7 measured
    # +16.5% on a 12 GB card but changes the answers, and this box keeps the exact expert set.
    "STRATA_ROUTE_TAIL_SKIP": "0",
    # Deliberately NOT set, and why:
    #  - the #879 debug instruments (STRATA_METAL_PROFILE / GDNMAX / LOGPOS): +11-12% on cold reads when off. GDNMAX=0
    #    reads the same as unset, so a config that carries it as "0" is already there.
    #  - STRATA_METAL_PROMPT_MOE_TM: the engine's own rule is the measured one (32 rows per expert when the chunk gives
    #    an expert 24+, 16 otherwise); pinning 32 forces the taller tile on short chunks too.
    #  - STRATA_METAL_TRACE: it prints a margin line per decode window and scans the vocabulary twice to do it. Fine
    #    for diagnosis, not part of a fast start; this box's config keeps it on for the log the reports read.
}
VALUELESS = {"--mmap-experts"}


def set_arg(args: list[str], flag: str, value: str | None) -> bool:
    """Set (or add) one engine flag in an args list. -> whether anything changed."""
    try:
        i = args.index(flag)
    except ValueError:
        if value is None:
            args.append(flag)
        else:
            args.extend([flag, value])
        return True
    if value is None:
        return False                      # present, valueless: as wanted
    if i + 1 < len(args) and args[i + 1] == value:
        return False
    if i + 1 < len(args) and not args[i + 1].startswith("--"):
        args[i + 1] = value
    else:
        args.insert(i + 1, value)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="a strata-<model>.json written by metal/setup_mac.py or serve/server.py")
    ap.add_argument("--apply", action="store_true", help="write the changes (default: only report them)")
    ap.add_argument("--start", action="store_true", help="start serve/server.py with the config after applying")
    ap.add_argument("--port", type=int, default=None, help="override the config's port for --start")
    ap.add_argument("--python", default=None, help="interpreter for --start (default: the repo's .venv, else python3)")
    a = ap.parse_args()

    cfg = json.load(open(a.config))
    args = list(cfg.get("args", []))
    changed: list[str] = []

    for flag, value in ARGS.items():
        if flag in ("--mtp", "--spec"):
            continue
        if set_arg(args, flag, value):
            changed.append(f"{flag}{' ' + value if value else ''}")
    # the drafter: only pin --spec when the config already names a drafter (setup_mac.py writes --mtp for packs it has one for)
    if "--mtp" in args:
        for flag, value in (("--spec", "2"), ("--lookup-max-t", "12"), ("--suffix-draft", "1")):
            if set_arg(args, flag, value):
                changed.append(f"{flag} {value}")
    if args != cfg.get("args", []):
        cfg["args"] = args

    env = dict(cfg.get("env") or {})
    for key, value in ENV.items():
        if env.get(key) != value:
            env[key] = value
            changed.append(f"{key}={value}")
    cfg["env"] = env

    # long agent sessions: the conversation comes back across a restart, written when it grew or the file aged
    if not cfg.get("auto_session"):
        cfg["auto_session"] = True
        changed.append("auto_session: true")
    if not cfg.get("slot_save_path"):
        where = os.path.join(os.path.dirname(os.path.abspath(a.config)), "sessions", str(cfg.get("model_name", "model")))
        os.makedirs(where, exist_ok=True)
        cfg["slot_save_path"] = where
        changed.append(f"slot_save_path: {where}")

    if not changed:
        print(f"{a.config}: already carries the M1 Max tuning")
    else:
        print(f"{a.config}: {len(changed)} change(s)")
        for c in changed:
            print(f"  {c}")

    if a.apply and changed:
        backup = a.config + ".before-m1max"
        shutil.copy2(a.config, backup)
        with open(a.config, "w") as f:
            json.dump(cfg, f, indent=1)
            f.write("\n")
        print(f"written (previous version kept at {os.path.basename(backup)})")

    if a.start:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        py = a.python or (os.path.join(root, ".venv", "bin", "python") if os.path.exists(os.path.join(root, ".venv", "bin", "python")) else sys.executable)
        cmd = [py, os.path.join(root, "serve", "server.py"), "--engine", "strata", "--config", os.path.abspath(a.config)]
        if a.port is not None:
            cmd += ["--port", str(a.port)]
        print("starting:", " ".join(cmd), flush=True)
        os.execv(cmd[0], cmd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
