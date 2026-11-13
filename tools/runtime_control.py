#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read and change the dynamic subsystems of a running vLLM engine.

Talks to the runtime control routes (`vllm serve ... --enable-runtime-control`,
or `VLLM_SERVER_DEV_MODE=1`). Standard library only. The API key, if the
server has one, is read from `VLLM_API_KEY`.

    runtime_control.py state [--only eplb,kv] [--json]
    runtime_control.py eplb pause | resume | rearrange
    runtime_control.py eplb set [--step-interval N] [--window-size N]
                                [--max-moves N] [--policy default|incremental]
    runtime_control.py eplb save [--path P] [--source last|window]
    runtime_control.py eplb load [--path P]
    runtime_control.py store scrub [--moved-only]
    runtime_control.py store set [--check-every N] [--scrub-every N]
                                 [--stats-interval S]
    runtime_control.py tiers set --host-tier-bound N
    runtime_control.py kv resize N [--deadline S] [--preempt]
    runtime_control.py kv fit | abort
    runtime_control.py kv set [--release-chunk-mib N|none] [--grow-floor-mib N|default]
    runtime_control.py tower promote | demote [--name T]
    runtime_control.py tower set [--name T] [--promote-after N] [--promote-window S]
                                 [--demote-idle S] [--idle-location host|disk]
                                 [--host-floor-gib G] [--card-reserve-mib N]
    runtime_control.py trace on [--path P] [--interval N] [--sample N]
    runtime_control.py trace off
    runtime_control.py trace set [--interval N] [--sample N] [--rotate-mb N] [--keep N]
    runtime_control.py ple set --stats-interval S
    runtime_control.py fault arm --stage load|plan|copy_rows [--rank R] [--count N]
    runtime_control.py fault clear [--stage S]

Global: --url (default http://localhost:8000), --timeout seconds.
Exit status: 0 applied on every rank, 1 refused or not applied, 2 bad usage.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any


def _request(
    base: str, method: str, path: str, body: dict | None, timeout: float
) -> tuple[int, Any]:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(base.rstrip("/") + path, data=data, method=method)
    request.add_header("Content-Type", "application/json")
    key = os.environ.get("VLLM_API_KEY")
    if key:
        request.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except ValueError:
            return e.code, {"error": raw.decode(errors="replace")}


def _set(args: dict, name: str, value: Any) -> None:
    if value is not None:
        args[name] = value


def _null_or_int(text: str, null: str) -> int | None:
    return None if text == null else int(text)


def _op_and_args(ns: argparse.Namespace) -> tuple[str, dict]:
    group, verb = ns.group, getattr(ns, "verb", None)
    a: dict[str, Any] = {}
    if group == "eplb":
        if verb == "set":
            _set(a, "step_interval", ns.step_interval)
            _set(a, "window_size", ns.window_size)
            _set(a, "max_moves_per_rank", ns.max_moves)
            _set(a, "policy", ns.policy)
        elif verb == "save":
            _set(a, "path", ns.path)
            a["source"] = ns.source
        elif verb == "load":
            _set(a, "path", ns.path)
        return f"eplb.{verb}", a
    if group == "store":
        if verb == "scrub":
            a["full"] = not ns.moved_only
        else:
            _set(a, "check_every", ns.check_every)
            _set(a, "scrub_every", ns.scrub_every)
            _set(a, "stats_interval_s", ns.stats_interval)
        return f"store.{verb}", a
    if group == "tiers":
        _set(a, "host_tier_bound", ns.host_tier_bound)
        return "tiers.set", a
    if group == "kv":
        if verb == "resize":
            a["num_blocks"] = ns.num_blocks
            _set(a, "deadline_s", ns.deadline)
            if ns.preempt:
                a["preempt"] = True
        elif verb == "set":
            if ns.release_chunk_mib is not None:
                a["release_chunk_mib"] = _null_or_int(ns.release_chunk_mib, "none")
            if ns.grow_floor_mib is not None:
                a["grow_floor_mib"] = _null_or_int(ns.grow_floor_mib, "default")
        return f"kv.{verb}", a
    if group == "tower":
        _set(a, "name", ns.name)
        if verb == "set":
            _set(a, "promote_after", ns.promote_after)
            _set(a, "promote_window_s", ns.promote_window)
            _set(a, "demote_idle_s", ns.demote_idle)
            _set(a, "idle_location", ns.idle_location)
            _set(a, "host_available_floor_gib", ns.host_floor_gib)
            _set(a, "card_reserve_mib", ns.card_reserve_mib)
        return f"tower.{verb}", a
    if group == "trace":
        if verb == "on":
            a["enabled"] = True
        elif verb == "off":
            a["enabled"] = False
        if verb in ("on", "set"):
            _set(a, "path", getattr(ns, "path", None))
            _set(a, "interval", ns.interval)
            _set(a, "sample", ns.sample)
        if verb == "set":
            _set(a, "rotate_mb", ns.rotate_mb)
            _set(a, "keep", ns.keep)
        return "trace.set", a
    if group == "ple":
        a["stats_interval_s"] = ns.stats_interval
        return "ple.set", a
    if group == "fault":
        if verb == "arm":
            a["stage"] = ns.stage
            a["count"] = ns.count
            _set(a, "rank", ns.rank)
        else:
            _set(a, "stage", ns.stage)
        return f"fault.{verb}", a
    raise SystemExit(2)


def _print_state(state: dict) -> None:
    kv = state.get("kv", {})
    resize = kv.get("resize", {})
    print(
        f"KV: {kv.get('num_blocks')} blocks, release chunk "
        f"{kv.get('release_chunk_mib')} MiB, grow floor {kv.get('grow_floor_mib')} "
        f"MiB, resize {resize.get('state', '-')}"
    )
    for name, tower in state.get("towers", {}).items():
        opts = tower["options"]
        print(
            f"tower {name}: {tower['place']} ({tower['phase']}), idle "
            f"{tower['idle_s']:.0f} s, promote after {opts['promote_after']} in "
            f"{opts['promote_window_s']:.0f} s, demote after "
            f"{opts['demote_idle_s']:.0f} s to {opts['idle_location']}; "
            f"promotions {tower['promotions']} demotions {tower['demotions']} "
            f"failed {tower['failed_promotions']}"
        )
    for rank in state.get("ranks", []):
        head = f"rank {rank['rank']} (EP {rank['ep_rank']})"
        eplb = rank.get("eplb")
        if eplb:
            s = eplb["settings"]
            print(
                f"{head} EPLB: {s['policy']}, every {s['step_interval']} steps, "
                f"window {s['window_size']} (filled {eplb['window_fill']}), max "
                f"moves {s['max_moves_per_rank']}; step {eplb['step']}, next in "
                f"{eplb['steps_until_next']}; {'PAUSED; ' if eplb['paused'] else ''}"
                f"rearrangements {eplb['rearrangements']}, skipped "
                f"{eplb['skipped']}, in flight: {eplb['in_flight'] or 'no'}"
            )
            for m in eplb["models"]:
                bal = m["balancedness"]
                print(
                    f"  {m['model']}: {m['layers']} layers, {m['physical_experts']}"
                    f" slots, balancedness "
                    f"{'-' if bal is None else f'{bal:.3f}'}, async done "
                    f"{m['completed_async']} abandoned {m['abandoned']}, last "
                    f"{m['last_async_seconds']} s"
                )
        tiers = rank.get("tiers")
        if tiers:
            print(
                f"{head} tiers: {tiers['layers']} layers, fast "
                f"{tiers['fast_slots_per_layer']} host {tiers['host_slots_per_layer']}"
                f" per layer, pool rows {tiers['pool_rows_per_layer']}, pinned "
                f"{tiers['pinned_gib']} GiB, moves {tiers['moves_committed']}"
            )
        for store in rank.get("store") or []:
            st = store.get("stats", {})
            print(
                f"{head} store ({store['mode']}): check every "
                f"{store['check_every']}, scrub every {store['scrub_every']}, "
                f"stats every {store['stats_interval_s']} s; reads "
                f"{st.get('reads', '-')}, check failures "
                f"{st.get('check_failures', '-')}"
            )
        rkv = rank.get("kv")
        if rkv:
            print(
                f"{head} KV: {rkv['committed_blocks']} of {rkv['capacity_blocks']}"
                f" blocks, chunk {rkv['release_chunk_bytes']} B"
            )
        trace = rank.get("trace")
        if trace:
            print(
                f"{head} trace: "
                f"{'on ' + str(trace['path']) if trace['enabled'] else 'off'}"
                f", interval {trace['interval']}, sample {trace['sample']}"
            )
        for table in rank.get("ple") or []:
            if table.get("on_disk"):
                rate = table.get("hit_rate")
                print(
                    f"{head} PLE {table['module']}: hit rate "
                    f"{'-' if rate is None else f'{rate:.3f}'} over "
                    f"{table['steps']} steps, misses {table['misses']}, stats "
                    f"every {table['stats_interval_s']} s"
                )
        faults = rank.get("faults") or {}
        if faults.get("armed"):
            print(f"{head} faults armed: {faults['armed']}")
    agree = state.get("ranks_agree", {})
    if agree and not all(agree.values()):
        print(f"RANKS DISAGREE: {agree}")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--timeout", type=float, default=600.0)
    groups = p.add_subparsers(dest="group", required=True)

    s = groups.add_parser("state")
    s.add_argument("--only")
    s.add_argument("--json", action="store_true")

    e = groups.add_parser("eplb").add_subparsers(dest="verb", required=True)
    for verb in ("pause", "resume", "rearrange"):
        e.add_parser(verb)
    es = e.add_parser("set")
    es.add_argument("--step-interval", type=int)
    es.add_argument("--window-size", type=int)
    es.add_argument("--max-moves", type=int)
    es.add_argument("--policy", choices=("default", "incremental"))
    sv = e.add_parser("save")
    sv.add_argument("--path")
    sv.add_argument("--source", choices=("last", "window"), default="last")
    ld = e.add_parser("load")
    ld.add_argument("--path")

    st = groups.add_parser("store").add_subparsers(dest="verb", required=True)
    sc = st.add_parser("scrub")
    sc.add_argument("--moved-only", action="store_true")
    ss = st.add_parser("set")
    ss.add_argument("--check-every", type=int)
    ss.add_argument("--scrub-every", type=int)
    ss.add_argument("--stats-interval", type=float)

    ti = groups.add_parser("tiers").add_subparsers(dest="verb", required=True)
    ti.add_parser("set").add_argument("--host-tier-bound", type=int, required=True)

    kv = groups.add_parser("kv").add_subparsers(dest="verb", required=True)
    kr = kv.add_parser("resize")
    kr.add_argument("num_blocks", type=int)
    kr.add_argument("--deadline", type=float)
    kr.add_argument("--preempt", action="store_true")
    kv.add_parser("fit")
    kv.add_parser("abort")
    ks = kv.add_parser("set")
    ks.add_argument("--release-chunk-mib", help="MiB, or 'none'")
    ks.add_argument("--grow-floor-mib", help="MiB, or 'default'")

    to = groups.add_parser("tower").add_subparsers(dest="verb", required=True)
    for verb in ("promote", "demote", "set"):
        tp = to.add_parser(verb)
        tp.add_argument("--name")
        if verb == "set":
            tp.add_argument("--promote-after", type=int)
            tp.add_argument("--promote-window", type=float)
            tp.add_argument("--demote-idle", type=float)
            tp.add_argument("--idle-location", choices=("host", "disk"))
            tp.add_argument("--host-floor-gib", type=float)
            tp.add_argument("--card-reserve-mib", type=int)

    tr = groups.add_parser("trace").add_subparsers(dest="verb", required=True)
    on = tr.add_parser("on")
    on.add_argument("--path")
    on.add_argument("--interval", type=int)
    on.add_argument("--sample", type=int)
    tr.add_parser("off")
    ts = tr.add_parser("set")
    ts.add_argument("--interval", type=int)
    ts.add_argument("--sample", type=int)
    ts.add_argument("--rotate-mb", type=int)
    ts.add_argument("--keep", type=int)

    pl = groups.add_parser("ple").add_subparsers(dest="verb", required=True)
    pl.add_parser("set").add_argument("--stats-interval", type=float, required=True)

    fa = groups.add_parser("fault").add_subparsers(dest="verb", required=True)
    fa_arm = fa.add_parser("arm")
    fa_arm.add_argument("--stage", choices=("load", "plan", "copy_rows"), required=True)
    fa_arm.add_argument("--rank", type=int)
    fa_arm.add_argument("--count", type=int, default=1)
    fa.add_parser("clear").add_argument("--stage")
    return p


def main(argv: list[str] | None = None) -> int:
    ns = _parser().parse_args(argv)
    if ns.group == "state":
        path = "/runtime_control" + (f"?only={ns.only}" if ns.only else "")
        status, reply = _request(ns.url, "GET", path, None, ns.timeout)
        if status != 200 or ns.json:
            print(json.dumps(reply, indent=2))
            return 0 if status == 200 else 1
        _print_state(reply)
        return 0
    op, body = _op_and_args(ns)
    status, reply = _request(ns.url, "POST", f"/runtime_control/{op}", body, ns.timeout)
    print(json.dumps(reply, indent=2))
    return 0 if status == 200 else 1


if __name__ == "__main__":
    sys.exit(main())
