"""Ablation runner.

    python -m lidcseg.experiments --config configs/improved.yaml --suite ablation
    python -m lidcseg.experiments --config configs/improved.yaml --suite cv5
    python -m lidcseg.experiments --config configs/improved.yaml --suite attention

Each variant runs in its own directory and the collected results are written to
`<out_dir>/<suite>_leaderboard.{json,md}`. One change per variant, so a
difference can be attributed to something.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import traceback
from pathlib import Path

from .config import Config, load_config

# Each entry is (suffix, {dotted-key: value}).
SUITES: dict[str, list[tuple[str, dict]]] = {
    # Isolates every departure from the notebook, one at a time.
    "ablation": [
        ("full", {}),
        ("no_attention", {"model.attention": "none"}),
        ("cab_attention", {"model.attention": "cab"}),
        ("no_deep_sup", {"model.deep_supervision": False}),
        ("no_aug", {"aug.enabled": False}),
        ("partial_skips", {"model.use_skips": False}),
        ("no_imagenet_scaling", {"model.imagenet_preprocess": False}),
        ("global_loss", {"loss.per_image": False}),
        ("dice_focal_loss", {"loss.name": "dice_focal"}),
        ("no_tta", {"eval.tta": False}),
        ("nodule0_only", {"data.nodules": ["nodule-0"]}),
    ],
    # The leakage measurement: same recipe, two protocols.
    "leakage": [
        ("patient_split", {"split.mode": "patient"}),
        ("slice_split", {"split.mode": "slice"}),
    ],
    "attention": [
        ("scanext", {"model.attention": "scanext"}),
        ("cab", {"model.attention": "cab"}),
        ("none", {"model.attention": "none"}),
    ],
    # Architecture comparison against external baselines, all under the same
    # split, loss, schedule and evaluation. This is the suite that says whether
    # the attention decoder earns its keep; `ablation` only compares the model
    # against versions of itself.
    #
    # Reading it: dplinknet_scanext - resnet50_unet isolates the LinkNet
    # decoder plus attention; resnet50_unet - unet isolates the ImageNet
    # encoder. The params column matters here, since the baselines are not
    # parameter-matched.
    "baselines": [
        ("dplinknet_scanext", {"model.arch": "dplinknet", "model.attention": "scanext"}),
        ("dplinknet_cab", {"model.arch": "dplinknet", "model.attention": "cab"}),
        ("dplinknet_noattn", {"model.arch": "dplinknet", "model.attention": "none"}),
        ("resnet50_unet", {"model.arch": "resnet50_unet", "model.attention": "none"}),
        (
            "unet_scratch",
            {
                "model.arch": "unet",
                "model.attention": "none",
                "model.pretrained": False,
                "model.imagenet_preprocess": False,
                "train.freeze_encoder_epochs": 0,
                "train.lr": 1.0e-3,
            },
        ),
    ],
    "loss": [
        ("tversky_g075", {"loss.name": "tversky_focal", "loss.tversky_gamma": 0.75}),
        ("tversky_g150", {"loss.name": "tversky_focal", "loss.tversky_gamma": 1.5}),
        ("tversky_b050", {"loss.name": "tversky_focal", "loss.tversky_beta": 0.5}),
        ("dice_focal", {"loss.name": "dice_focal"}),
        ("dice_bce", {"loss.name": "dice_bce"}),
    ],
    "cv5": [(f"fold{i}", {"split.folds": 5, "split.fold": i}) for i in range(5)],
}


def _apply(cfg: Config, dotted: dict) -> Config:
    """Return a copy of `cfg` with dotted keys replaced."""
    out = copy.deepcopy(cfg)
    for key, val in dotted.items():
        obj = out
        parts = key.split(".")
        for p in parts[:-1]:
            obj = getattr(obj, p)
        if not hasattr(obj, parts[-1]):
            raise AttributeError(f"no such config field: {key}")
        setattr(obj, parts[-1], val)
    return out


def run_suite(base: Config, suite: str, only: list[str] | None = None) -> list[dict]:
    from .train import run

    if suite not in SUITES:
        raise ValueError(f"unknown suite {suite!r}; choose from {sorted(SUITES)}")

    variants = SUITES[suite]
    if only:
        variants = [v for v in variants if v[0] in set(only)]
        if not variants:
            raise ValueError(f"no variants in '{suite}' matched {only}")

    rows: list[dict] = []
    for i, (suffix, overrides) in enumerate(variants, 1):
        cfg = _apply(base, overrides)
        cfg.name = f"{base.name}__{suite}__{suffix}"

        print("\n" + "=" * 72)
        print(f"[{i}/{len(variants)}] {cfg.name}")
        print(f"  overrides: {json.dumps(overrides, default=str)}")
        print("=" * 72)

        try:
            meta = run(cfg)
            test = meta["results"]["test"]
            rows.append({
                "variant": suffix,
                "arch": cfg.model.arch,
                "attention": cfg.model.attention,
                "overrides": overrides,
                "dice": test.get("dice"),
                "dice_ci": test.get("dice_ci"),
                "iou": test.get("iou"),
                "dice_global": test.get("dice_global"),
                "precision": test.get("precision"),
                "recall": test.get("recall"),
                "threshold": meta["results"]["threshold"],
                "params": meta["params"],
                "minutes": meta["minutes"],
                "status": "ok",
            })
        except Exception as e:
            # One failed variant must not lose the rest of an overnight sweep.
            traceback.print_exc()
            rows.append({
                "variant": suffix,
                "arch": cfg.model.arch,
                "attention": cfg.model.attention,
                "overrides": overrides,
                "status": "failed",
                "error": f"{type(e).__name__}: {e}",
            })

        _write_leaderboard(base, suite, rows)

    return rows


def _write_leaderboard(base: Config, suite: str, rows: list[dict]) -> None:
    """Rewritten after every variant, so a crashed sweep still leaves results."""
    out = Path(base.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    (out / f"{suite}_leaderboard.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    ok = [r for r in rows if r["status"] == "ok"]
    ok.sort(key=lambda r: r.get("dice") or 0, reverse=True)

    lines = [
        f"# {suite} leaderboard -- base `{base.name}`",
        "",
        f"Split: **{base.split.mode}-level**. Dice is per-image mean with a 95% "
        "bootstrap CI; `dice_global` pools all pixels.",
        ""
        + (
            "These architectures are **not parameter-matched** -- read the "
            "params column alongside the scores."
            if suite == "baselines"
            else ""
        ),
        "",
        "| variant | arch | Dice | 95% CI | IoU | Dice(global) | Prec | Rec | thr | params | min |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in ok:
        ci = r.get("dice_ci")
        ci_s = f"{ci[0]:.3f}-{ci[1]:.3f}" if ci else "--"
        arch = r.get("arch", "?")
        att = r.get("attention", "none")
        arch_s = f"{arch}+{att}" if arch == "dplinknet" and att != "none" else arch
        lines.append(
            f"| `{r['variant']}` | {arch_s} | **{r['dice']:.4f}** | {ci_s} | {r['iou']:.4f} | "
            f"{r['dice_global']:.4f} | {r['precision']:.3f} | {r['recall']:.3f} | "
            f"{r['threshold']:.2f} | {r['params'] / 1e6:.1f}M | {r['minutes']:.0f} |"
        )

    bad = [r for r in rows if r["status"] != "ok"]
    if bad:
        lines += ["", "## failed", ""]
        lines += [f"- `{r['variant']}`: {r['error']}" for r in bad]

    (out / f"{suite}_leaderboard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[leaderboard] {out / f'{suite}_leaderboard.md'}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run an ablation suite")
    ap.add_argument("--config", required=True)
    ap.add_argument("--suite", required=True, choices=sorted(SUITES))
    ap.add_argument("--set", dest="overrides", nargs="*", default=[], metavar="KEY=VALUE",
                    help="overrides applied to the base config before the suite")
    ap.add_argument("--only", nargs="*", default=None,
                    help="run just these variant names")
    ap.add_argument("--list", action="store_true", help="print the suite and exit")
    args = ap.parse_args(argv)

    if args.list:
        for name, variants in SUITES.items():
            print(f"\n{name}:")
            for suffix, ov in variants:
                print(f"  {suffix:<22} {json.dumps(ov, default=str)}")
        return 0

    base = load_config(args.config, args.overrides)
    rows = run_suite(base, args.suite, args.only)

    ok = [r for r in rows if r["status"] == "ok"]
    if ok:
        best = max(ok, key=lambda r: r["dice"])
        print(f"\n[best] {best['variant']}  Dice={best['dice']:.4f}  IoU={best['iou']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
