#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author-style dataset generator for RubyGems (Tree-sitter version)

Directory layout (example):
  --mal-dir /rubygems_mal/<package>/<version>/(extracted src or .gem)
  --ben-dir /rubygems_ben/<package>/<version>/(extracted src or .gem)

Outputs:
  out_dir/train.csv, validation.csv, test.csv
  (columns: textual_description,label)  # label: 1=malicious, 0=benign
"""

import argparse, os, re, io, tarfile, gzip, zipfile, tempfile, shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
from sklearn.model_selection import train_test_split

# ---- Tree-sitter setup (Ruby) ----
try:
    from tree_sitter_languages import get_parser
except Exception as e:
    raise ImportError("Install: pip install -U tree_sitter tree_sitter_languages") from e

_RB_PARSER = None
def ruby_parser():
    global _RB_PARSER
    if _RB_PARSER is None:
        _RB_PARSER = get_parser("ruby")
    return _RB_PARSER

# ---- Phrase dictionaries (author-style, Ruby 映射) ----
REQUIRE_MAP = {
    # 网络/通信
    "net/http": "network module",
    "open-uri": "network module",
    "socket": "network module",
    "openssl": "network module",
    "uri": "network module",
    "rest-client": "network module",
    "httpclient": "network module",
    "net/ftp": "network module",
    # 进程/系统
    "open3": "process module",
    # 文件系统/工具
    "fileutils": "file system module",
    "tempfile": "file system module",
    # 编解码
    "base64": "base64 module",
    # 其他可按需扩展…
}

CALL_PHRASES = {
    # shell / 系统执行
    "system": "use operating system module call",
    "exec":   "use operating system module call",
    "spawn":  "use operating system module call",
    # 进程族
    "open3.popen": "use process module call",
    "io.popen":    "use process module call",
    # 运行期求值
    "eval":           None,  # 统一在 has_eval_exec 里描述
    "instance_eval":  None,
    "class_eval":     None,
    "module_eval":    None,
}

URL_RE = re.compile(r"(https?://|\burl\b)", re.I)
B64_RE = re.compile(r"\b[A-Za-z0-9+/]{24,}={0,2}\b")
BACKTICK_RE = re.compile(r"`[^`]+`|%x\{.*?\}|%x\([^\)]*\)|%x\[[^\]]*\]", re.S)

# ----------------- Ruby feature extraction -----------------
def ts_iter(node):
    yield node
    for c in node.children:
        yield from ts_iter(c)

def read_bytes(p: Path) -> Optional[bytes]:
    try:
        return p.read_bytes()
    except Exception:
        return None

def extract_ruby_features(rb_path: Path) -> Dict[str, object]:
    """
    结合 Tree-sitter（抓取字符串）+ 正则（抓 require/调用/反引号）
    返回:
      requires: set[str]   # 归一化后的 require 名称
      has_eval_exec: bool
      url_count: int
      b64_count: int
      calls: List[str]     # 归一化后的调用线索（system/exec/spawn/open3.popen/io.popen）
    """
    feats = {"requires": set(), "has_eval_exec": False, "url_count": 0, "b64_count": 0, "calls": []}
    src = read_bytes(rb_path)
    if not src:
        return feats

    text = src.decode("utf-8", "ignore")

    # 1) require / require_relative
    for m in re.finditer(r"""require(?:_relative)?\s*(?:\(?\s*['"]([^'"]+)['"]\s*\)?)""", text):
        req = m.group(1).strip().lower()
        feats["requires"].add(req)

    # 2) 关键调用（系统/进程/求值），以及反引号命令
    #    注意：这里做归一化，统一转小写匹配
    low = text.lower()
    if re.search(r"\b(kernel\s*\.\s*)?system\s*\(", low): feats["calls"].append("system")
    if re.search(r"\b(kernel\s*\.\s*)?exec\s*\(", low):   feats["calls"].append("exec")
    if re.search(r"\b(kernel\s*\.\s*)?spawn\s*\(", low):  feats["calls"].append("spawn")
    if re.search(r"\bopen3\s*\.\s*popen[23]?\b", low):    feats["calls"].append("open3.popen")
    if re.search(r"\bio\s*\.\s*popen\b", low):            feats["calls"].append("io.popen")
    if re.search(r"\b(eval|instance_eval|class_eval|module_eval)\s*\(", low):
        feats["has_eval_exec"] = True
    if BACKTICK_RE.search(text):
        # 反引号或 %x{} -> 也视为 shell 执行
        feats["calls"].append("system")

    # 3) Tree-sitter 抓字符串：统计 URL / base64
    try:
        tree = ruby_parser().parse(src)
        root = tree.root_node
        for n in ts_iter(root):
            if n.type in {"string", "string_content", "heredoc_body"}:
                s = src[n.start_byte:n.end_byte].decode("utf-8", "ignore")
                if URL_RE.search(s): feats["url_count"] += 1
                if B64_RE.search(s): feats["b64_count"] += 1
    except Exception:
        # 解析失败时，回退到全文统计（兜底）
        feats["url_count"] += len(URL_RE.findall(text))
        feats["b64_count"] += len(B64_RE.findall(text))

    return feats

# ----------------- File picking / archives -----------------
def choose_ruby_file(root: Path) -> Optional[Path]:
    # 优先 gemspec
    gs = list(root.rglob("*.gemspec"))
    if gs: return gs[0]
    # 再看 lib/**/*.rb
    lib_rb = list((root / "lib").rglob("*.rb")) if (root / "lib").exists() else []
    if lib_rb: return lib_rb[0]
    # 退化为任何 .rb
    any_rb = list(root.rglob("*.rb"))
    if any_rb: return any_rb[0]
    return None

def find_version_root(version_dir: Path) -> Path:
    """
    版本目录中如果存在解压好的源（常见以 .src 结尾或包含源码树），优先用它；
    否则直接用版本目录本身。若只有 .gem 且未解压，可尝试提取（简单兜底）。
    """
    # 1) 优先 <something>.src
    for sub in version_dir.iterdir():
        if sub.is_dir() and sub.name.endswith(".src"):
            return sub

    # 2) 若存在 .gem，尝试解包 data.tar.gz
    gem_files = [p for p in version_dir.glob("*.gem") if p.is_file()]
    if gem_files:
        tmp = Path(tempfile.mkdtemp(prefix="gem_extract_"))
        out = tmp / "gem_out"
        out.mkdir(parents=True, exist_ok=True)
        gem = gem_files[0]
        try:
            # .gem 通常内含 data.tar.gz 和 metadata.gz，这里只解 data.tar.gz
            with tarfile.open(gem, "r:*") as tf:
                data_member = next((m for m in tf.getmembers() if m.name.endswith("data.tar.gz")), None)
                if data_member:
                    data_bytes = tf.extractfile(data_member).read()
                    with tarfile.open(fileobj=io.BytesIO(data_bytes), mode="r:gz") as data_tar:
                        data_tar.extractall(out)
                    return out
        except Exception:
            pass  # 解包失败则继续
    return version_dir

# ----------------- Text formatting -----------------
def to_author_style_text(pkg: str, relpath: str, feats: Dict[str, object]) -> str:
    parts: List[str] = []

    # require -> phrase
    mapped = []
    for req in sorted(feats["requires"]):
        phrase = REQUIRE_MAP.get(req)
        if not phrase:
            # 归一化：把 'net/http/...' 视为 'net/http'
            base = req.split("/", 2)
            key = "/".join(base[:2]) if len(base) >= 2 else base[0]
            phrase = REQUIRE_MAP.get(key)
        if phrase and phrase not in mapped:
            mapped.append(phrase)
    for m in mapped:
        parts.append(f"import {m}")

    # calls -> phrase
    triggered = set()
    for k, phr in CALL_PHRASES.items():
        if phr and any(c.startswith(k) for c in feats["calls"]):
            triggered.add(phr)
    parts.extend(sorted(triggered))

    # eval
    if feats["has_eval_exec"]:
        parts.append("evaluate code at run-time")

    # URL / base64
    if feats["url_count"] > 0:
        parts.append("use URL")
    if feats["b64_count"] > 0:
        parts.extend(["use base64 string"] * min(4, feats["b64_count"]))

    middle = ", ".join(parts) if parts else "no notable behavior"
    return f"start entry {pkg}/{relpath}, {middle}, end of entry"

# ----------------- Dataset builder -----------------
def iter_labeled_versions(root_dir: Path, label: int):
    """
    遍历 root_dir/<package>/<version>/...
    生成 (package, version, version_root) 三元组
    """
    if not root_dir.exists():
        return
    for pkg_dir in sorted([d for d in root_dir.iterdir() if d.is_dir()]):
        pkg = pkg_dir.name
        for ver_dir in sorted([d for d in pkg_dir.iterdir() if d.is_dir()]):
            version_root = find_version_root(ver_dir)
            yield pkg, ver_dir.name, version_root, label

def build_dataset(mal_dir: Path, ben_dir: Path) -> pd.DataFrame:
    rows = []
    temp_dirs = []
    for pkg, ver, vroot, label in list(iter_labeled_versions(mal_dir, 1)) + list(iter_labeled_versions(ben_dir, 0)):
        rep = choose_ruby_file(vroot)
        if not rep:
            rows.append({"textual_description": f"start entry {pkg}/{ver}, no ruby file, end of entry",
                         "label": int(label)})
            continue
        feats = extract_ruby_features(rep)
        # relpath 相对 version_root
        try:
            relpath = str(rep.relative_to(vroot)).replace("\\", "/")
        except Exception:
            relpath = rep.name
        text = to_author_style_text(pkg, f"{ver}/{relpath}", feats)
        rows.append({"textual_description": text, "label": int(label)})
    return pd.DataFrame(rows)

def stratified_split_save(df: pd.DataFrame, out_dir: Path, val_size: float, test_size: float, random_state: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    if df.empty:
        # 仍然写出空文件，避免流程中断
        pd.DataFrame(columns=["textual_description","label"]).to_csv(out_dir / "train.csv", index=False)
        pd.DataFrame(columns=["textual_description","label"]).to_csv(out_dir / "validation.csv", index=False)
        pd.DataFrame(columns=["textual_description","label"]).to_csv(out_dir / "test.csv", index=False)
        return

    X = df["textual_description"].astype(str)
    y = df["label"].astype(int)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=random_state)

    pd.DataFrame({"textual_description": X_train, "label": y_train}).to_csv(out_dir / "train.csv", index=False)
    pd.DataFrame({"textual_description": X_val,   "label": y_val}).to_csv(out_dir / "validation.csv", index=False)
    pd.DataFrame({"textual_description": X_test,  "label": y_test}).to_csv(out_dir / "test.csv", index=False)

# ----------------- CLI -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mal-dir", required=True, help="恶意样本根目录（rubygems_mal）")
    ap.add_argument("--ben-dir", required=True, help="良性样本根目录（rubygems_ben）")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--val-size", type=float, default=0.10)
    ap.add_argument("--test-size", type=float, default=0.10)
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    mal_dir = Path(args.mal_dir).expanduser().resolve()
    ben_dir = Path(args.ben_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    df = build_dataset(mal_dir, ben_dir)
    stratified_split_save(df, out_dir, args.val_size, args.test_size, args.random_state)

if __name__ == "__main__":
    main()
