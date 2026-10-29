"""Shared helpers for the end-to-end tests (pipeline, tiling, python).

The CLI under test is <workspace>/target/release/jupwrap-cli, built by the
test command (`cargo build --offline --release -p jupwrap-cli`). Wall time and
peak child RSS are recorded to judge/results/<test>.json (not gated unless a
test says so)."""
from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

JUDGE = Path(__file__).resolve().parent.parent
ROOT = JUDGE.parent
sys.path.insert(0, str(JUDGE))

from ref import e2e_cases as E  # noqa: E402
from ref.metrics import congruence_error, fraction_correct, psi_of, truth_regions  # noqa: E402

TARGET = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target"))
if not TARGET.is_absolute():
    TARGET = ROOT / TARGET
CLI = TARGET / "release" / "jupwrap-cli"
EXPECTED = json.loads((JUDGE / "golden" / "e2e" / "expected.json").read_text())
RESULTS = JUDGE / "results"


def case(name):
    cs = E.get(name)
    assert E.input_hash(cs) == EXPECTED[name]["input_sha256"], \
        f"{name}: synthetic input drifted (numpy RNG changed?) - regenerate goldens"
    return cs


def run_cli(tmp: Path, igram, corr, nlooks, cost="smooth", init="mcf", mask=None, **kw):
    """Run jupwrap-cli; returns (unw, conncomp, stats, wall_s, maxrss_bytes)."""
    assert CLI.exists(), f"{CLI} missing: build with cargo build --offline --release -p jupwrap-cli"
    R, C = igram.shape
    tmp.mkdir(parents=True, exist_ok=True)
    np.ascontiguousarray(igram, np.complex64).tofile(tmp / "ig.c8")
    np.ascontiguousarray(corr, np.float32).tofile(tmp / "corr.f4")
    args = [str(CLI), "--igram", str(tmp / "ig.c8"), "--corr", str(tmp / "corr.f4"), "--nrow", str(R),
            "--ncol", str(C), "--nlooks", str(nlooks), "--cost", cost, "--init", init,
            "--unw", str(tmp / "unw.f4"), "--conncomp", str(tmp / "cc.u4"), "--stats", str(tmp / "stats.json")]
    if mask is not None:
        np.ascontiguousarray(mask, np.uint8).tofile(tmp / "mask.u1")
        args += ["--mask", str(tmp / "mask.u1")]
    for k, v in kw.items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, (tuple, list)):
            args += [flag] + [str(x) for x in v]
        elif isinstance(v, bool):
            args += [flag, "1" if v else "0"]
        else:
            args += [flag, str(v)]
    before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    t0 = time.time()
    p = subprocess.run(args, capture_output=True, text=True)
    wall = time.time() - t0
    rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    assert p.returncode == 0, f"jupwrap-cli failed ({p.returncode}): {p.stderr[-2000:]}"
    unw = np.fromfile(tmp / "unw.f4", np.float32).reshape(R, C)
    cc = np.fromfile(tmp / "cc.u4", np.uint32).reshape(R, C)
    stats = json.loads((tmp / "stats.json").read_text())
    # ru_maxrss is bytes on macOS, KiB on Linux; it is a high-water mark over all children
    scale = 1 if sys.platform == "darwin" else 1024
    return unw, cc, stats, wall, max(rss, before) * scale


def record(name, **vals):
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(vals, indent=1, default=float))


def score(cs, unw):
    reg = truth_regions(cs["rho"], cs["hole"], cs["mask"])
    fc, per = fraction_correct(unw, cs["phi"], reg)
    cong = congruence_error(unw, psi_of(cs["igram"]), cs["mask"])
    return fc, cong, reg


def assert_conncomp_contract(cc, mask=None):
    """0 = no component; labels 1..N with sizes non-increasing in label order."""
    assert cc.dtype == np.uint32
    if mask is not None:
        assert (cc[np.asarray(mask) == 0] == 0).all(), "masked pixels must have conncomp 0"
    n = int(cc.max())
    if n:
        sizes = np.bincount(cc.ravel(), minlength=n + 1)[1:]
        assert (sizes > 0).all(), "labels must be consecutive 1..N"
        assert (np.diff(sizes) <= 0).all(), "labels must be ordered by decreasing size"


@pytest.fixture
def tmpd(tmp_path):
    return tmp_path
