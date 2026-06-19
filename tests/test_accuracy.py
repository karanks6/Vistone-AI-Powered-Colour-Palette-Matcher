"""
tests/test_accuracy.py
======================
Benchmark runner for the Vistone skin-tone classifier.

Usage
-----
    python tests/test_accuracy.py
    python tests/test_accuracy.py --images tests/benchmark/images --gt tests/benchmark/ground_truth.json

Metrics reported
----------------
  • Monk Exact Accuracy      — predicted tone == ground truth
  • Monk Within-1 Accuracy   — |predicted - true| <= 1
  • Monk Within-2 Accuracy   — |predicted - true| <= 2
  • Undertone Accuracy        — predicted undertone == ground truth
  • Mean Monk Error           — average absolute tone deviation
  • Avg Tone Confidence       — mean reported confidence
  • Avg Undertone Confidence  — mean reported confidence
  • Per-tone breakdown table  — exact/within-1 per Monk level
"""

import os, sys, json, argparse
from collections import defaultdict

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from skin_tone import classify_monk_v10


# ─────────────────────────────────────────────────────────────────────
def run_benchmark(image_dir: str, gt_path: str, verbose: bool = False) -> dict:
    """
    Run classifier on every image in ground_truth.json and report metrics.

    Returns a summary dict with all computed metrics.
    """
    if not os.path.isfile(gt_path):
        print(f"[error] Ground-truth file not found: {gt_path}")
        return {}

    with open(gt_path, "r", encoding="utf-8") as f:
        gt_raw = json.load(f)

    # Filter out meta keys that start with underscore
    gt = {k: v for k, v in gt_raw.items() if not k.startswith("_")}

    if not gt:
        print("[warn] ground_truth.json has no labelled entries yet.")
        print("       Add entries following the _format schema and re-run.")
        return {}

    total          = 0
    exact_tone     = 0
    within1_tone   = 0
    within2_tone   = 0
    exact_ut       = 0
    abs_errors     = []
    tone_confs     = []
    ut_confs       = []
    errors         = []
    per_tone        = defaultdict(lambda: {"total": 0, "exact": 0, "within1": 0})

    for fname, labels in gt.items():
        true_tone = labels.get("monk")
        true_ut   = labels.get("undertone", "")
        if true_tone is None:
            continue

        img_path = os.path.join(image_dir, fname)
        if not os.path.isfile(img_path):
            print(f"  [skip] Image not found: {img_path}")
            continue

        try:
            result   = classify_monk_v10(img_path, debug=False)
        except Exception as e:
            print(f"  [error] {fname}: {e}")
            errors.append(fname)
            continue

        pred_tone = result["tone"]
        pred_ut   = result["undertone"]
        tc        = result.get("tone_confidence", 0.0)
        uc        = result.get("ut_confidence",   0.0)

        total += 1
        ae = abs(pred_tone - true_tone)
        abs_errors.append(ae)
        tone_confs.append(tc)
        ut_confs.append(uc)

        if ae == 0: exact_tone   += 1
        if ae <= 1: within1_tone += 1
        if ae <= 2: within2_tone += 1
        if pred_ut.lower() == true_ut.lower():
            exact_ut += 1

        per_tone[true_tone]["total"]   += 1
        per_tone[true_tone]["exact"]   += int(ae == 0)
        per_tone[true_tone]["within1"] += int(ae <= 1)

        if verbose:
            mark = "✓" if ae == 0 else ("~" if ae <= 1 else "✗")
            print(
                f"  {mark} {fname:<30s}"
                f"  tone: {pred_tone}/{true_tone} (err={ae})"
                f"  ut: {pred_ut}/{true_ut}"
                f"  conf: {tc:.0%}/{uc:.0%}"
            )

    if total == 0:
        print("[warn] No images were successfully processed.")
        return {}

    # ── Summary ─────────────────────────────────────────────────────
    print("\n" + "═" * 55)
    print("  VISTONE BENCHMARK RESULTS")
    print("═" * 55)
    print(f"  Images processed : {total} / {len(gt)}")
    print(f"  Errors           : {len(errors)}")
    print()
    print(f"  Monk Exact Acc.  : {exact_tone / total:.1%}  ({exact_tone}/{total})")
    print(f"  Monk Within-1    : {within1_tone / total:.1%}  ({within1_tone}/{total})")
    print(f"  Monk Within-2    : {within2_tone / total:.1%}  ({within2_tone}/{total})")
    print(f"  Mean Abs Error   : {sum(abs_errors) / total:.2f} tones")
    print()
    print(f"  Undertone Exact  : {exact_ut / total:.1%}  ({exact_ut}/{total})")
    print()
    print(f"  Avg Tone Conf    : {sum(tone_confs) / total:.1%}")
    print(f"  Avg Under. Conf  : {sum(ut_confs)   / total:.1%}")
    print()

    # ── Per-tone breakdown ───────────────────────────────────────────
    if per_tone:
        print("  Per-Monk-Tone Breakdown:")
        print("  " + "-" * 45)
        print(f"  {'Monk':<8} {'Total':>6} {'Exact':>8} {'Within-1':>10}")
        print("  " + "-" * 45)
        for tone_key in sorted(per_tone.keys()):
            d = per_tone[tone_key]
            n = d["total"]
            e = d["exact"]
            w = d["within1"]
            print(
                f"  Monk {tone_key:<4} "
                f"{n:>6} "
                f"{e:>5} ({e/n:.0%})"
                f"{w:>5} ({w/n:.0%})"
            )
        print("  " + "-" * 45)

    print("═" * 55)

    return {
        "total":           total,
        "exact_acc":       exact_tone / total,
        "within1_acc":     within1_tone / total,
        "within2_acc":     within2_tone / total,
        "mean_abs_error":  sum(abs_errors) / total,
        "undertone_acc":   exact_ut / total,
        "avg_tone_conf":   sum(tone_confs) / total,
        "avg_ut_conf":     sum(ut_confs) / total,
    }


# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vistone accuracy benchmark")
    parser.add_argument(
        "--images",
        default=os.path.join(os.path.dirname(__file__), "benchmark", "images"),
        help="Directory containing benchmark images",
    )
    parser.add_argument(
        "--gt",
        default=os.path.join(os.path.dirname(__file__), "benchmark", "ground_truth.json"),
        help="Path to ground_truth.json",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-image results",
    )
    args = parser.parse_args()

    run_benchmark(args.images, args.gt, verbose=args.verbose)
