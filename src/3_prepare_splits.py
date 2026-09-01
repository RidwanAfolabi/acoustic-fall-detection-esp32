"""
STEP 3 - Leakage-free train / validation / test splits.

The corpus ships with a 10-fold assignment in the first filename field, and the
split is built from whole folds so that no clip -- and therefore none of a
clip's overlapping 0.975 s windows, and none of its augmented variants -- can
ever straddle the train/test boundary.

Why not a literal 70/15/15 of folds: 10 folds cannot be cut 7 / 1.5 / 1.5. Two
strategies are offered:

    fold  (default)  7 train / 2 val / 1 test whole folds  -> 70 / 20 / 10
                     Cleanest isolation, and the larger validation set makes
                     early stopping and LR-plateau decisions in Step 4 steadier.

    ratio            7 whole folds to train, then the remaining 3 folds are
                     divided clip-wise to hit 70 / 15 / 15 exactly. Train stays
                     fold-isolated; val and test draw from a shared fold pool
                     but never share a clip.

Fold assignment is not arbitrary. Fold 10 of the SAFE corpus is measurably less
class-balanced than the other nine (43/52 rather than ~48/47), so folds are
ranked by class skew and the most skewed are absorbed into train -- the largest
split -- leaving validation and test as balanced as the data allows.

Variant policy: training uses every augmentation variant, while validation and
test use **variant 0 only**, the canonical bathroom RIR. Evaluation is therefore
reverberant like the real deployment, reproducible run to run, and not inflated
by counting three variants of one clip as three independent test cases.

Outputs
    data/splits/split_manifest.csv     every window, tagged with its split
    data/splits/{train,val,test}_indices.npy
                                       row indices into chunks_int16.npy
    data/splits/splits_summary.json    counts, balance, class weights
    outputs/figures/splits_*.png

Usage
    python src/3_prepare_splits.py
    python src/3_prepare_splits.py --strategy ratio
    python src/3_prepare_splits.py --eval-variants all
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config

LOGGER = logging.getLogger("splits")

SPLITS = ("train", "val", "test")
CANONICAL_VARIANT = 0


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(console)
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
    LOGGER.addHandler(handler)


def section(title: str) -> None:
    LOGGER.info("\n" + "=" * 78)
    LOGGER.info(title)
    LOGGER.info("=" * 78)


def indent(text: str, prefix: str = "  ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


# ---------------------------------------------------------------------------
# Fold assignment
# ---------------------------------------------------------------------------


def rank_folds_by_skew(clips: pd.DataFrame) -> list[str]:
    """Folds ordered most class-skewed first.

    Skew is measured on distinct clips, not windows, so a fold is not judged by
    how many augmented copies it happens to have produced.
    """
    per_fold = clips.groupby("fold_id")["label"].apply(
        lambda s: abs((s == "fall").mean() - 0.5)
    )
    return list(per_fold.sort_values(ascending=False).index)


def assign_folds(
    clips: pd.DataFrame, n_train: int, n_val: int
) -> dict[str, list[str]]:
    """Choose which folds go to which split.

    The most class-skewed folds are pushed into train, where the larger sample
    absorbs the imbalance, so validation and test stay as balanced as possible.
    """
    ranked = rank_folds_by_skew(clips)
    if len(ranked) < n_train + n_val + 1:
        raise SystemExit(
            f"need at least {n_train + n_val + 1} folds, found {len(ranked)}"
        )
    train = sorted(ranked[:n_train])
    remainder = sorted(ranked[n_train:])
    return {"train": train, "val": remainder[:n_val], "test": remainder[n_val:]}


def split_clips_by_ratio(
    clips: pd.DataFrame, pool_folds: list[str], seed: int
) -> dict[str, set[str]]:
    """Divide a pool of folds clip-wise into equal val and test halves.

    Splitting is done on whole clips (by clip_uid) and stratified by label, so
    every window and variant of a clip lands on the same side.
    """
    rng = np.random.default_rng(seed)
    val: set[str] = set()
    test: set[str] = set()
    pool = clips[clips["fold_id"].isin(pool_folds)]
    for _, group in pool.groupby("label"):
        uids = group["clip_uid"].to_numpy()
        rng.shuffle(uids)
        half = len(uids) // 2
        val.update(uids[:half].tolist())
        test.update(uids[half:].tolist())
    return {"val": val, "test": test}


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify(manifest: pd.DataFrame, strategy: str) -> list[str]:
    """Assert the split really is leakage-free. Returns a list of problems."""
    section("LEAKAGE VERIFICATION")
    problems: list[str] = []

    assigned = manifest[manifest["split"].notna()]

    # 1. No source clip may appear in more than one split.
    spread = assigned.groupby("clip_uid")["split"].nunique()
    offenders = spread[spread > 1]
    LOGGER.info("Clips spanning >1 split ......... %d", len(offenders))
    if len(offenders):
        problems.append(f"{len(offenders)} clip(s) appear in multiple splits")

    # 2. Nor may a clip's augmented variants be separated.
    var_spread = assigned.groupby(["clip_uid", "variant_id"])["split"].nunique()
    if int((var_spread > 1).sum()):
        problems.append("some clip variants are split across sets")
    LOGGER.info(
        "Variants spanning >1 split ...... %d", int((var_spread > 1).sum())
    )

    # 3. Train folds must not reappear in evaluation.
    train_folds = set(assigned.loc[assigned["split"] == "train", "fold_id"])
    eval_folds = set(assigned.loc[assigned["split"].isin(("val", "test")), "fold_id"])
    overlap = train_folds & eval_folds
    LOGGER.info("Folds in both train and eval .... %d %s",
                len(overlap), sorted(overlap) if overlap else "")
    if overlap:
        problems.append(f"folds {sorted(overlap)} appear in train and evaluation")

    # Under 'ratio', val and test intentionally share a fold pool. That is not
    # training leakage, but it does make val a slightly optimistic proxy for
    # test, so it is stated rather than left implicit.
    val_folds = set(assigned.loc[assigned["split"] == "val", "fold_id"])
    test_folds = set(assigned.loc[assigned["split"] == "test", "fold_id"])
    shared = val_folds & test_folds
    if shared:
        note = f"val and test share fold(s) {sorted(shared)}"
        if strategy == "ratio":
            LOGGER.info("Val/test shared folds ........... %s (expected for "
                        "--strategy ratio)", sorted(shared))
        else:
            LOGGER.info("Val/test shared folds ........... %s", sorted(shared))
            problems.append(note)

    # 4. Row indices must be disjoint.
    counts = assigned["split"].value_counts()
    total = int(counts.sum())
    unique_rows = assigned["chunk_index"].nunique()
    LOGGER.info("Assigned windows ................ %d (unique rows %d)",
                total, unique_rows)
    if total != unique_rows:
        problems.append("a chunk index was assigned to more than one split")

    return problems


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def report(manifest: pd.DataFrame, clips: pd.DataFrame, folds: dict) -> dict:
    section("FOLD ASSIGNMENT")
    skew = clips.groupby("fold_id").agg(
        clips=("clip_uid", "nunique"),
        fall=("label", lambda s: int((s == "fall").sum())),
        non_fall=("label", lambda s: int((s == "non_fall").sum())),
    )
    skew["fall_pct"] = (100.0 * skew["fall"] / skew[["fall", "non_fall"]].sum(axis=1))
    skew["split"] = ""
    for name, ids in folds.items():
        skew.loc[skew.index.isin(ids), "split"] = name
    LOGGER.info(indent(skew.round(1).to_string()))
    LOGGER.info("\ntrain folds ... %s", ", ".join(folds["train"]))
    LOGGER.info("val   folds ... %s", ", ".join(folds["val"]))
    LOGGER.info("test  folds ... %s", ", ".join(folds["test"]))

    section("SPLIT COMPOSITION")
    assigned = manifest[manifest["split"].notna()]
    stats: dict[str, Any] = {}

    rows = []
    total_clips = assigned["clip_uid"].nunique()
    for name in SPLITS:
        sub = assigned[assigned["split"] == name]
        n_clips = sub["clip_uid"].nunique()
        fall = int((sub["label"] == "fall").sum())
        rows.append({
            "split": name,
            "clips": n_clips,
            "clips_pct": 100.0 * n_clips / max(total_clips, 1),
            "windows": len(sub),
            "windows_pct": 100.0 * len(sub) / max(len(assigned), 1),
            "fall": fall,
            "non_fall": len(sub) - fall,
            "fall_pct": 100.0 * fall / max(len(sub), 1),
        })
        stats[name] = rows[-1]
    LOGGER.info(indent(pd.DataFrame(rows).round(1).to_string(index=False)))
    LOGGER.info(
        "\nThe split ratio is the CLIPS column: %s.\n"
        "Window percentages look lopsided only because training keeps every\n"
        "augmentation variant while evaluation keeps one per clip. That is the\n"
        "intended asymmetry, not an unbalanced split.",
        " / ".join(f"{r['clips_pct']:.0f}%" for r in rows),
    )

    LOGGER.info("\nVariant composition per split:")
    LOGGER.info(indent(
        pd.crosstab(assigned["split"], assigned["variant_kind"]).to_string()
    ))

    dropped = manifest["split"].isna().sum()
    LOGGER.info(
        "\nWindows excluded from evaluation splits: %d", int(dropped)
    )
    LOGGER.info(
        "  (non-canonical variants of val/test clips - keeping them would count\n"
        "   several augmentations of one clip as independent test cases)"
    )

    if "negative_mixed" in assigned.columns and assigned["negative_mixed"].any():
        LOGGER.info("\nHard negatives per split:")
        LOGGER.info(indent(
            assigned.groupby("split")["negative_mixed"].agg(["sum", "mean"])
            .rename(columns={"sum": "windows", "mean": "fraction"})
            .round(3).to_string()
        ))
    return stats


def class_weights(manifest: pd.DataFrame) -> dict[str, float]:
    """Balanced class weights for the training split, for Step 4."""
    train = manifest[manifest["split"] == "train"]
    counts = train["label_index"].value_counts()
    total = int(counts.sum())
    n_classes = len(counts)
    return {
        str(int(idx)): float(total / (n_classes * count))
        for idx, count in counts.items()
    }


def write_figures(manifest: pd.DataFrame) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = config.FIGURES_DIR
    out.mkdir(parents=True, exist_ok=True)
    assigned = manifest[manifest["split"].notna()]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    pivot = pd.crosstab(assigned["split"], assigned["label"]).reindex(SPLITS)
    pivot.plot(kind="bar", stacked=True, ax=axes[0],
               color=["#d1495b", "#3b7dd8"], rot=0)
    axes[0].set_ylabel("windows")
    axes[0].set_title("Windows per split and class")
    axes[0].legend(title="")

    vpivot = pd.crosstab(assigned["split"], assigned["variant_kind"]).reindex(SPLITS)
    vpivot.plot(kind="bar", stacked=True, ax=axes[1], rot=0)
    axes[1].set_ylabel("windows")
    axes[1].set_title("Augmentation variants per split")
    axes[1].legend(title="")
    fig.tight_layout()
    path = out / "splits_composition.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return [path]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build leakage-free dataset splits.")
    p.add_argument("--processed-dir", type=Path, default=config.PROCESSED_DIR)
    p.add_argument("--out-dir", type=Path, default=config.SPLITS_DIR)
    p.add_argument("--strategy", choices=("fold", "ratio"), default="fold")
    p.add_argument("--train-folds", type=int, default=7)
    p.add_argument("--val-folds", type=int, default=2)
    p.add_argument("--eval-variants", choices=("canonical", "all"),
                   default="canonical",
                   help="which augmentation variants val/test may use")
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    p.add_argument("--no-figures", dest="figures", action="store_false")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config.ensure_dirs()
    setup_logging(config.LOGS_DIR / "prepare_splits.log")

    section("DATASET SPLITTING  -  Step 3")
    manifest_path = args.processed_dir / "manifest.csv"
    if not manifest_path.exists():
        LOGGER.info("No manifest at %s", manifest_path)
        LOGGER.info("Run Step 2 first:  python src/2_preprocess.py")
        return 2

    manifest = pd.read_csv(
        manifest_path,
        dtype={"clip_uid": str, "fold_id": str, "class_code": str, "event_code": str},
    )
    LOGGER.info("Manifest ........... %s (%d windows)", manifest_path, len(manifest))
    LOGGER.info("Strategy ........... %s", args.strategy)
    LOGGER.info("Eval variants ...... %s", args.eval_variants)

    # One row per source clip: fold assignment is a decision about clips, not
    # about the windows they happen to produce.
    clips = manifest.drop_duplicates("clip_uid")[
        ["clip_uid", "fold_id", "label"]
    ].reset_index(drop=True)
    LOGGER.info("Distinct clips ..... %d across %d folds",
                len(clips), clips["fold_id"].nunique())

    folds = assign_folds(clips, args.train_folds, args.val_folds)

    manifest["split"] = pd.Series(pd.NA, index=manifest.index, dtype="object")
    train_mask = manifest["fold_id"].isin(folds["train"])
    manifest.loc[train_mask, "split"] = "train"

    if args.strategy == "fold":
        manifest.loc[manifest["fold_id"].isin(folds["val"]), "split"] = "val"
        manifest.loc[manifest["fold_id"].isin(folds["test"]), "split"] = "test"
    else:
        pool = folds["val"] + folds["test"]
        halves = split_clips_by_ratio(clips, pool, args.seed)
        manifest.loc[manifest["clip_uid"].isin(halves["val"]), "split"] = "val"
        manifest.loc[manifest["clip_uid"].isin(halves["test"]), "split"] = "test"
        folds = {"train": folds["train"], "val": pool, "test": pool}

    # Evaluation sees one variant per clip unless told otherwise.
    if args.eval_variants == "canonical":
        drop = (
            manifest["split"].isin(("val", "test"))
            & (manifest["variant_id"] != CANONICAL_VARIANT)
        )
        manifest.loc[drop, "split"] = pd.NA

    stats = report(manifest, clips, folds)
    problems = verify(manifest, args.strategy)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.out_dir / "split_manifest.csv", index=False)
    for name in SPLITS:
        idx = manifest.loc[manifest["split"] == name, "chunk_index"].to_numpy(np.int64)
        np.save(args.out_dir / f"{name}_indices.npy", np.sort(idx))

    weights = class_weights(manifest)
    summary = {
        "strategy": args.strategy,
        "eval_variants": args.eval_variants,
        "folds": folds,
        "splits": stats,
        "class_weights": weights,
        "label_to_index": config.LABEL_TO_INDEX,
        "chunks_file": str(args.processed_dir / "chunks_int16.npy"),
        "window_samples": config.WINDOW_SAMPLES,
        "sample_rate": config.SAMPLE_RATE,
        "seed": args.seed,
        "problems": problems,
    }
    (args.out_dir / "splits_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    figures = write_figures(manifest) if args.figures else []

    section("ARTEFACTS")
    LOGGER.info("manifest ... %s", args.out_dir / "split_manifest.csv")
    for name in SPLITS:
        LOGGER.info("indices .... %s", args.out_dir / f"{name}_indices.npy")
    LOGGER.info("summary .... %s", args.out_dir / "splits_summary.json")
    for f in figures:
        LOGGER.info("figure ..... %s", f)

    LOGGER.info("\nClass weights for Step 4: %s", weights)

    section("RESULT")
    if problems:
        LOGGER.info("%d problem(s) found:", len(problems))
        for i, problem in enumerate(problems, start=1):
            LOGGER.info("  %d. %s", i, problem)
        return 1
    LOGGER.info("PASS - splits are leakage-free and ready for Step 4.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
