"""Named end-to-end cases (shared by gen/gen_e2e.py and the pytest end-to-end tests).
Inputs are regenerated deterministically from seeds (numpy pinned by uv.lock);
judge/golden/e2e/expected.json stores an input hash to detect drift."""
from __future__ import annotations

import hashlib

import numpy as np

from .synth import make_case

CASES = {
    # name: (kind, R, C, seed, nlooks, kwargs, cost)
    "clean_topo_256": ("clean_topo", 256, 256, 1, 9, {}, "smooth"),
    "topo_512_l4": ("topo", 512, 512, 11, 4, dict(base=(0.3, 0.8), max_grad=1.5), "smooth"),
    "topo_512_l3_harsh": ("topo", 512, 512, 12, 3, dict(base=(0.25, 0.7), max_grad=2.0), "smooth"),
    "masked_topo_384": ("masked_topo", 384, 384, 5, 9, {}, "smooth"),
    "defo_512": ("defo", 512, 512, 13, 4, dict(base=(0.35, 0.85), peak_cycles=25), "defo"),
    "topo_2048": ("topo", 2048, 2048, 21, 9, {}, "smooth"),
    "topo_1024_l4": ("topo", 1024, 1024, 31, 4, dict(base=(0.3, 0.8), max_grad=1.5), "smooth"),
    "defo_1024": ("defo", 1024, 1024, 32, 4, dict(base=(0.35, 0.85), peak_cycles=40), "defo"),
    "masked_topo_768": ("masked_topo", 768, 768, 33, 9, {}, "smooth"),
    "strip_33000x512": ("topo", 33000, 512, 41, 9, {}, "smooth"),
    "strip_4096x512": ("topo", 4096, 512, 41, 9, {}, "smooth"),
}


def get(name):
    kind, R, C, seed, L, kw, cost = CASES[name]
    cs = make_case(kind, R, C, seed, nlooks=L, **kw)
    cs["cost"] = cost
    cs["name"] = name
    return cs


def input_hash(cs):
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(cs["igram"]).tobytes())
    h.update(np.ascontiguousarray(cs["corr"]).tobytes())
    if cs["mask"] is not None:
        h.update(np.ascontiguousarray(cs["mask"]).tobytes())
    return h.hexdigest()


def readme_ramp(n=512):
    """snaphu-py README example: exp(i*pi*(x+y)) on a 512x512 grid over [-3,3]^2."""
    y, x = np.ogrid[-3:3:n * 1j, -3:3:n * 1j]
    phi = np.pi * (x + y)
    igram = np.exp(1j * phi).astype(np.complex64)
    corr = np.ones(igram.shape, np.float32)
    return phi, igram, corr
