#!/usr/bin/env python3
import argparse
import glob
import os

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


def load_split_summary(result_dir):
    path = latest_csv_contains(result_dir, "split_summary_compare")
    if path is not None:
        df = pd.read_csv(path)
    else:
        parts = []
        for keyword in ["ese_prefix", "ese_info", "ese_task"]:
            part = latest_csv_contains(result_dir, keyword)
            if part is not None:
                parts.append(pd.read_csv(part))
        if not parts:
            raise FileNotFoundError(f"No split result CSV found in {result_dir}")
        df = pd.concat(parts, ignore_index=True)
    df["Layer"] = df["Layer"].astype(int)
    return df


def load_final_compare(result_dir):
    path = latest_csv_contains(result_dir, "final_compare")
    if path is None:
        return None
    df = pd.read_csv(path)
    df["Layer"] = df["Layer"].astype(int)
    return df


def infer_seed(result_dir):
    name = os.path.basename(os.path.normpath(result_dir))
    if "seed" in name:
        tail = name.split("seed", 1)[1]
        digits = "".join(ch for ch in tail if ch.isdigit())
        if digits:
            return int(digits)
    return None


def metric(df, eval_mode, layer, metric_name="MeanR"):
    hit = df[
        df["EvalMode"].astype(str).eq(eval_mode)
        & df["Layer"].astype(int).eq(int(layer))
    ]
    if hit.empty:
        return None
    return float(hit.iloc[0][metric_name])


def final_metric(final_df, eval_mode, layer=12):
    if final_df is None:
        return None
    hit = final_df[
        final_df["Model"].astype(str).eq("ESE-CLIP")
        & final_df["EvalMode"].astype(str).eq(eval_mode)
        & final_df["Layer"].astype(int).eq(int(layer))
    ]
    if hit.empty:
        return None
    return float(hit.iloc[0]["MeanR"])


def summarize_mode(df, eval_mode):
    row = {}
    for layer in [1, 4, 5, 6, 8, 10, 11, 12]:
        value = metric(df, eval_mode, layer)
        if value is not None:
            row[f"{eval_mode}_L{layer}"] = value

    l456 = [row.get(f"{eval_mode}_L4"), row.get(f"{eval_mode}_L5"), row.get(f"{eval_mode}_L6")]
    if all(v is not None for v in l456):
        row[f"{eval_mode}_L4_L6_avg"] = sum(l456) / 3.0

    sub = df[df["EvalMode"].astype(str).eq(eval_mode)]
    if not sub.empty:
        max_layer = int(sub["Layer"].max())
        shallow = sub[sub["Layer"] < max_layer].sort_values("MeanR", ascending=False)
        if not shallow.empty:
            best = shallow.iloc[0]
            row[f"{eval_mode}_best_shallow_layer"] = int(best["Layer"])
            row[f"{eval_mode}_best_shallow_MeanR"] = float(best["MeanR"])
    return row


def summarize_dir(result_dir):
    split_df = load_split_summary(result_dir)
    final_df = load_final_compare(result_dir)

    row = {
        "seed": infer_seed(result_dir),
        "result_dir": result_dir,
    }

    for eval_mode in sorted(split_df["EvalMode"].unique()):
        row.update(summarize_mode(split_df, eval_mode))

    row["full_512_L12"] = final_metric(final_df, "full_512", layer=12)
    row["prefix_256_L12_from_final_compare"] = final_metric(final_df, "prefix_256", layer=12)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_dirs",
        type=str,
        nargs="+",
        default=[
            "/home/ziyuanmeng/prj_ziyuanmeng/outputs/ese_clip_k256_split_v2_i64_t192",
            "/home/ziyuanmeng/prj_ziyuanmeng/outputs/ese_clip_k256_split_v2_i64_t192_seed123",
            "/home/ziyuanmeng/prj_ziyuanmeng/outputs/ese_clip_k256_split_v2_i64_t192_seed2026",
        ],
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="/home/ziyuanmeng/prj_ziyuanmeng/outputs/ese_clip_k256_split_v2_multiseed_summary.csv",
    )
    args = parser.parse_args()

    rows = []
    for result_dir in args.result_dirs:
        if not os.path.exists(result_dir):
            print(f"[Skip] missing: {result_dir}")
            continue
        try:
            rows.append(summarize_dir(result_dir))
        except Exception as exc:
            print(f"[Error] {result_dir}: {exc}")

    if not rows:
        raise RuntimeError("No result directories summarized.")

    summary = pd.DataFrame(rows)
    summary = summary.sort_values("seed", na_position="first").reset_index(drop=True)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    summary.to_csv(args.output_csv, index=False)

    display_cols = [
        "seed",
        "prefix_256_L4_L6_avg",
        "prefix_256_L8",
        "prefix_256_L11",
        "prefix_256_L12",
        "task_192_L4_L6_avg",
        "task_192_L8",
        "task_192_L12",
        "info_64_L4_L6_avg",
        "info_64_L8",
        "info_64_L12",
        "full_512_L12",
    ]
    display_cols = [c for c in display_cols if c in summary.columns]

    print("\n========== Split v2 Multi-Seed Summary ==========")
    print(summary[display_cols].to_string(index=False))
    print("\nSaved:", args.output_csv)

    if "prefix_256_L4_L6_avg" in summary.columns and summary["prefix_256_L4_L6_avg"].notna().sum() >= 2:
        mean = summary["prefix_256_L4_L6_avg"].mean()
        std = summary["prefix_256_L4_L6_avg"].std()
        print(f"\nprefix_256 L4-L6 avg: mean={mean:.4f}, std={std:.4f}")


if __name__ == "__main__":
    main()
