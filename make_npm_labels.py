#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从 npm_package_info.csv 生成“全量标签表”，不做任何拆分。
支持对 `unreviewed` 的多种策略：
  - drop      : (默认) 丢弃 unreviewed，不进入 0/1 标签
  - benign    : 将 unreviewed 当作 0
  - malicious : 将 unreviewed 当作 1
  - separate  : 单独输出 unreviewed_version.csv，不进入 0/1 标签

输出：
- labels_version.csv     (package, version, label)      # 版本级 0/1
- labels.csv             (package, label)               # 包级 0/1（对 labels_version 的 max 聚合）
- *unreviewed_version.csv (package, version)            # 当 --unreviewed-policy=separate 时才会写出

用法：
python make_npm_labels_version.py \
  --info /data_add/LiDi/MalwareBench/npm/npm_package_info.csv \
  --out-dir /data_add/LiDi/MalwareBench/npm \
  --unreviewed-policy drop
"""

import argparse
import os
import re
import pandas as pd

def normalize_tag(s: str) -> str:
    """标准化 threat_type：小写，去掉空格/连字符/下划线。"""
    s = str(s).strip().lower()
    return re.sub(r"[\s\-_]+", "", s)

# 基本集合（可按需扩展）
MALWARE_POS   = {"malware", "malicious", "confirmedmalware"}
BENIGN_POS    = {"falsepositive", "benign", "clean", "nonmalware", "legit", "whitelisted"}
UNREVIEWED_POS= {"unreviewed", "unknown", "pendingreview", "needsreview", "unverified"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--info", required=True, help="npm_package_info.csv 路径")
    ap.add_argument("--out-dir", required=True, help="输出目录")
    ap.add_argument("--unreviewed-policy", choices=["drop","benign","malicious","separate"], default="drop",
                    help="unreviewed 样本的处理方式（默认 drop）")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.info)
    # 兼容列名
    name_col   = next((c for c in ["name","package","package_name"]      if c in df.columns), None)
    ver_col    = next((c for c in ["version","ver","pkg_version"]        if c in df.columns), None)
    threat_col = next((c for c in ["threat_type","threat","label","is_malware","malware"] if c in df.columns), None)
    if not (name_col and ver_col and threat_col):
        raise ValueError(f"CSV 需要包含 name/version/threat_type（或同义）列，实际：{list(df.columns)}")

    df = df[[name_col, ver_col, threat_col]].dropna()
    df[name_col]   = df[name_col].astype(str).str.strip()
    df[ver_col]    = df[ver_col].astype(str).str.strip()
    df[threat_col] = df[threat_col].astype(str).map(normalize_tag)

    # 分类计数（原始）
    counts_all = df[threat_col].value_counts()

    # 标注与筛选
    labels = []
    unreviewed_rows = []
    for _, row in df.iterrows():
        tag = row[threat_col]
        pkg = row[name_col]
        ver = row[ver_col]

        if tag in MALWARE_POS:
            labels.append((pkg, ver, 1))
        elif tag in BENIGN_POS:
            labels.append((pkg, ver, 0))
        elif tag in UNREVIEWED_POS:
            if args.unreviewed_policy == "drop":
                # 不加入 labels
                continue
            elif args.unreviewed_policy == "benign":
                labels.append((pkg, ver, 0))
            elif args.unreviewed_policy == "malicious":
                labels.append((pkg, ver, 1))
            elif args.unreviewed_policy == "separate":
                unreviewed_rows.append((pkg, ver))
            else:
                continue
        else:
            # 未知类别：默认当作非恶意（或你可以选择 continue 丢弃）
            labels.append((pkg, ver, 0))

    # 版本级（去重）
    if labels:
        labels_ver = (
            pd.DataFrame(labels, columns=["package","version","label"])
              .drop_duplicates()
              .reset_index(drop=True)
        )
    else:
        labels_ver = pd.DataFrame(columns=["package","version","label"])

    out_ver = os.path.join(args.out_dir, "labels_version.csv")
    labels_ver.to_csv(out_ver, index=False)

    # 包级聚合（对 0/1 的版本表取 max）
    if not labels_ver.empty:
        labels_pkg = labels_ver.groupby("package", as_index=False)["label"].max()
    else:
        labels_pkg = pd.DataFrame(columns=["package","label"])
    out_pkg = os.path.join(args.out_dir, "labels.csv")
    labels_pkg.to_csv(out_pkg, index=False)

    # 可选输出：unreviewed_version.csv
    unrev_path = None
    if args.unreviewed_policy == "separate" and unreviewed_rows:
        unreviewed_ver = (
            pd.DataFrame(unreviewed_rows, columns=["package","version"])
              .drop_duplicates()
              .reset_index(drop=True)
        )
        unrev_path = os.path.join(args.out_dir, "unreviewed_version.csv")
        unreviewed_ver.to_csv(unrev_path, index=False)

    # 回显统计
    print("== Source threat_type distribution (normalized) ==")
    print(counts_all.to_string(), "\n")
    print(f"labels_version.csv  -> {out_ver}   rows: {len(labels_ver)}   pos_ratio: {labels_ver['label'].mean() if len(labels_ver)>0 else 'NA'}")
    print(f"labels.csv (pkg)    -> {out_pkg}    packages: {len(labels_pkg)} pos_ratio: {labels_pkg['label'].mean() if len(labels_pkg)>0 else 'NA'}")
    if unrev_path:
        print(f"unreviewed_version.csv -> {unrev_path} rows: {len(unreviewed_ver)}")
    print("Unreviewed policy:", args.unreviewed_policy)

if __name__ == "__main__":
    main()
