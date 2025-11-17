#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从 RubyGems 目录结构生成版本级标签表 labels_version.csv
目录假设：
  <mal_root>/<package>/<version>/(可能有 *.src 展开目录或源码树)
  <ben_root>/<package>/<version>/(可能有 *.src 展开目录或源码树)

输出：
  labels_version.csv    (package,version,label)  # label: 恶意=1, 良性=0
  （可选）labels.csv    (package,label)         # 包级聚合（对版本取 max）

示例：
python make_rubygems_labels_version.py \
  --mal-dir /data_add/LiDi/rubygems/rubygems_mal \
  --ben-dir /data_add/LiDi/rubygems/rubygems_ben \
  --out     /data_add/LiDi/rubygems/labels_version.csv \
  --require-ruby --require-gemspec
"""

import argparse
import os
from pathlib import Path
from typing import Iterator, List, Optional, Tuple
import pandas as pd

def pick_version_root(version_dir: Path) -> Path:
    """
    版本目录中如果存在展开目录（常见以 .src 结尾），优先进入；
    否则直接返回版本目录本身。
    """
    for sub in version_dir.iterdir():
        if sub.is_dir() and sub.name.endswith(".src"):
            return sub
    return version_dir

def has_gemspec(vroot: Path) -> bool:
    return any(vroot.rglob("*.gemspec"))

def has_ruby(vroot: Path) -> bool:
    return any(vroot.rglob("*.rb"))

def walk_pkg_versions(root: Path, label: int,
                      require_gemspec: bool = False,
                      require_ruby: bool = False) -> Iterator[Tuple[str, str]]:
    """
    遍历 <root>/<package>/<version>/... -> 产出 (package, version)
    根据选项过滤：必须包含 *.gemspec / *.rb 才算有效。
    """
    if not root.exists():
        return
    for pkg_dir in sorted([d for d in root.iterdir() if d.is_dir()]):
        pkg = pkg_dir.name
        for ver_dir in sorted([d for d in pkg_dir.iterdir() if d.is_dir()]):
            vroot = pick_version_root(ver_dir)
            if require_gemspec and not has_gemspec(vroot):
                continue
            if require_ruby and not has_ruby(vroot):
                continue
            yield (pkg, ver_dir.name)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mal-dir", required=True, help="恶意样本根目录（rubygems_mal）")
    ap.add_argument("--ben-dir", required=True, help="良性样本根目录（rubygems_ben）")
    ap.add_argument("--out", required=True, help="labels_version.csv 输出路径")
    ap.add_argument("--also-package-labels", action="store_true",
                    help="额外输出包级 labels.csv（版本取 max）")
    ap.add_argument("--require-gemspec", action="store_true",
                    help="仅保留包含 *.gemspec 的版本")
    ap.add_argument("--require-ruby", action="store_true",
                    help="仅保留包含 *.rb 的版本")
    args = ap.parse_args()

    mal_root = Path(args.mal_dir).expanduser().resolve()
    ben_root = Path(args.ben_dir).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Tuple[str, str, int]] = []

    # 恶意=1
    for pkg, ver in walk_pkg_versions(
        mal_root, 1,
        require_gemspec=args.require_gemspec,
        require_ruby=args.require_ruby
    ):
        rows.append((pkg, ver, 1))

    # 良性=0
    for pkg, ver in walk_pkg_versions(
        ben_root, 0,
        require_gemspec=args.require_gemspec,
        require_ruby=args.require_ruby
    ):
        rows.append((pkg, ver, 0))

    if not rows:
        print("警告：没有找到任何版本目录，检查 --mal-dir / --ben-dir 是否正确。")
        # 也写出空表，避免下游中断
        pd.DataFrame(columns=["package","version","label"]).to_csv(out_path, index=False)
        return

    labels_ver = (
        pd.DataFrame(rows, columns=["package","version","label"])
          .drop_duplicates()
          .reset_index(drop=True)
    )
    labels_ver.to_csv(out_path, index=False)

    print(f"labels_version.csv -> {out_path}  rows: {len(labels_ver)}")
    print("  正样本(1):", int((labels_ver["label"]==1).sum()))
    print("  负样本(0):", int((labels_ver["label"]==0).sum()))

    if args.also_package_labels:
        labels_pkg = labels_ver.groupby("package", as_index=False)["label"].max()
        pkg_out = out_dir / "labels.csv"
        labels_pkg.to_csv(pkg_out, index=False)
        print(f"labels.csv (package level) -> {pkg_out}  packages: {len(labels_pkg)}")

if __name__ == "__main__":
    main()
