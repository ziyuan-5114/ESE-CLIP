#!/usr/bin/env python3
import argparse
import glob
import json
import os
import re

import pandas as pd


def latest_csv_contains(csv_dir, *keywords):
    matches = []
    for path in glob.glob(os.path.join(csv_dir, "*.csv")):
        name = os.path.basename(path).lower()
        if all(k.lower() in name for k in keywords):
            matches.append(path)
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def load_final_compare(result_dir):
    path = latest_csv_contains(result_dir, "final_compare")
    if path is not None:
        df = pd.read_csv(path)
    else:
        parts = []
        for keyword in ["ese_full", "raw_full", "ese_prefix", "raw_prefix"]:
            part = latest_csv_contains(result_dir, keyword)
            if part is not None:
                parts.append(pd.read_csv(part))
        if not parts:
            raise FileNotFoundError(f"No result CSV found in {result_dir}")
        df = pd.concat(parts, ignore_index=True)
    df["Layer"] = df["Layer"].astype(int)
    return df


def load_json_if_exists(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_metric(df, model, eval_mode, layer, metric="MeanR"):
    hit = df[
        df["Model"].astype(str).eq(model)
        & df["EvalMode"].astype(str).eq(eval_mode)
        & df["Layer"].astype(int).eq(int(layer))
    ]
    if hit.empty:
        return None
    return float(hit.iloc[0][metric])


def infer_config_name(result_dir):
    name = os.path.basename(os.path.normpath(result_dir))
    match = re.search(r"tune_(.+?)_seed", name)
    if match:
        return match.group(1)
    if "shallow_weight" in name:
        return "current_pw_l6_b3"
    return name


def summarize_result(result_dir, baseline_df=None):
    df = load_final_compare(result_dir)
    run_args = load_json_if_exists(os.path.join(result_dir, "run_args.json"))
    weight_info = load_json_if_exists(os.path.join(result_dir, "express_layer_weights.json"))
    max_layer = int(df["Layer"].max())

    row = {
        "config": infer_config_name(result_dir),
        "result_dir": result_dir,
        "seed": run_args.get("seed"),
        "mode": weight_info.get("mode", run_args.get("express_weight_mode")),
        "shallow_layers": weight_info.get("shallow_layers", run_args.get("shallow_layers")),
        "shallow_boost": weight_info.get("shallow_boost", run_args.get("shallow_boost")),
        "shallow_alpha": weight_info.get("shallow_alpha", run_args.get("shallow_alpha")),
        "shallow_power": weight_info.get("shallow_power", run_args.get("shallow_power")),
    }

    for layer in [1, 2, 3, 4, 5, 6, 8, 10, 11, 12]:
        value = get_metric(df, "ESE-CLIP", "prefix_256", layer)
        if value is not None:
            row[f"L{layer}_prefix_MeanR"] = value

    l4_l6 = [row.get("L4_prefix_MeanR"), row.get("L5_prefix_MeanR"), row.get("L6_prefix_MeanR")]
    l4_l8 = [
        row.get("L4_prefix_MeanR"),
        row.get("L5_prefix_MeanR"),
        row.get("L6_prefix_MeanR"),
        get_metric(df, "ESE-CLIP", "prefix_256", 7),
        row.get("L8_prefix_MeanR"),
    ]
    row["L4_L6_prefix_MeanR_avg"] = sum(l4_l6) / len(l4_l6) if all(v is not None for v in l4_l6) else None
    row["L4_L8_prefix_MeanR_avg"] = sum(l4_l8) / len(l4_l8) if all(v is not None for v in l4_l8) else None
    row["L12_prefix_MeanR"] = get_metric(df, "ESE-CLIP", "prefix_256", max_layer)
    row["L12_full_MeanR"] = get_metric(df, "ESE-CLIP", "full_512", max_layer)

    shallow = df[
        df["Model"].astype(str).eq("ESE-CLIP")
        & df["EvalMode"].astype(str).eq("prefix_256")
        & (df["Layer"].astype(int) < max_layer)
    ].copy()
    if not shallow.empty:
        best = shallow.sort_values("MeanR", ascending=False).iloc[0]
        row["best_shallow_layer"] = int(best["Layer"])
        row["best_shallow_MeanR"] = float(best["MeanR"])

    if baseline_df is not None:
        base_l4_l6 = []
        for layer in [4, 5, 6]:
            base_l4_l6.append(get_metric(baseline_df, "ESE-CLIP", "prefix_256", layer))
        if all(v is not None for v in base_l4_l6) and row["L4_L6_prefix_MeanR_avg"] is not None:
            row["delta_vs_baseline_L4_L6"] = row["L4_L6_prefix_MeanR_avg"] - sum(base_l4_l6) / len(base_l4_l6)
        base_l12_prefix = get_metric(baseline_df, "ESE-CLIP", "prefix_256", max_layer)
        base_l12_full = get_metric(baseline_df, "ESE-CLIP", "full_512", max_layer)
        if base_l12_prefix is not None and row["L12_prefix_MeanR"] is not None:
            row["delta_vs_baseline_L12_prefix"] = row["L12_prefix_MeanR"] - base_l12_prefix
        if base_l12_full is not None and row["L12_full_MeanR"] is not None:
            row["delta_vs_baseline_L12_full"] = row["L12_full_MeanR"] - base_l12_full

    # A simple selection score: prioritize true early layers, keep L12 prefix from collapsing.
    row["selection_score"] = (
        0.70 * (row.get("L4_L6_prefix_MeanR_avg") or 0.0)
        + 0.20 * (row.get("L8_prefix_MeanR") or 0.0)
        + 0.10 * (row.get("L12_prefix_MeanR") or 0.0)
    )
    return row


def discover_tuning_dirs(root):
    pattern = os.path.join(root, "ese_clip_k256_tune_*")
    return sorted([p for p in glob.glob(pattern) if os.path.isdir(p)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs_root", type=str, default="/home/ziyuanmeng/prj_ziyuanmeng/outputs")
    parser.add_argument(
        "--baseline_dir",
        type=str,
        default="/home/ziyuanmeng/prj_ziyuanmeng/outputs/ese_clip_k256_shallow_weight",
        help="Existing current direction-1 result, usually piecewise L6 boost 3.",
    )
    parser.add_argument("--include_baseline", action="store_true")
    parser.add_argument("--result_dirs", type=str, nargs="*", default=None)
    parser.add_argument(
        "--output_csv",
        type=str,
        default="/home/ziyuanmeng/prj_ziyuanmeng/outputs/ese_clip_k256_weight_tuning_summary.csv",
    )
    args = parser.parse_args()

    baseline_df = None
    if args.baseline_dir and os.path.exists(args.baseline_dir):
        try:
            baseline_df = load_final_compare(args.baseline_dir)
        except Exception as exc:
            print(f"[Warn] Could not load baseline: {exc}")

    result_dirs = args.result_dirs if args.result_dirs else discover_tuning_dirs(args.outputs_root)
    if args.include_baseline and args.baseline_dir and os.path.exists(args.baseline_dir):
        result_dirs = [args.baseline_dir] + result_dirs

    rows = []
    for result_dir in result_dirs:
        if not os.path.exists(result_dir):
            print(f"[Skip] missing: {result_dir}")
            continue
        try:
            rows.append(summarize_result(result_dir, baseline_df=baseline_df))
        except Exception as exc:
            print(f"[Error] {result_dir}: {exc}")

    if not rows:
        raise RuntimeError("No tuning result directories were summarized.")

    summary = pd.DataFrame(rows)
    summary = summary.sort_values("selection_score", ascending=False).reset_index(drop=True)

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    summary.to_csv(args.output_csv, index=False)

    columns = [
        "config",
        "seed",
        "mode",
        "shallow_layers",
        "shallow_boost",
        "shallow_alpha",
        "L4_prefix_MeanR",
        "L5_prefix_MeanR",
        "L6_prefix_MeanR",
        "L4_L6_prefix_MeanR_avg",
        "L8_prefix_MeanR",
        "L12_prefix_MeanR",
        "L12_full_MeanR",
        "best_shallow_layer",
        "best_shallow_MeanR",
        "delta_vs_baseline_L4_L6",
        "delta_vs_baseline_L12_prefix",
        "selection_score",
    ]
    columns = [c for c in columns if c in summary.columns]

    print("\n========== Weight Tuning Summary ==========")
    print(summary[columns].to_string(index=False))
    print("\nSaved:", args.output_csv)
    print("\nRecommendation: pick the top config by selection_score, then run multi-seed validation on that config.")


if __name__ == "__main__":
    main()
