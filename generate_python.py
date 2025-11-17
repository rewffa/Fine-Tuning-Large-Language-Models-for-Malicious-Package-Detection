#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author-style dataset generator for PyPI (Tree-sitter, version-level)

输入：
  --packages-dir  目录通常 {packages_dir}/{package}/{version}/...
                  也兼容 {packages_dir}/{package}/{tar-gz|src|zip}/... 等中间层
  --labels-version  CSV: package,version,label  (label∈{0,1})

输出：
  out_dir/train.csv, validation.csv, test.csv  （列： textual_description,label）

依赖：
  pip install -U tree_sitter tree_sitter_languages pandas scikit-learn
"""

import argparse, os, re, tarfile, zipfile, tempfile, shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
from sklearn.model_selection import train_test_split

# ---- Tree-sitter setup ----
try:
    from tree_sitter_languages import get_parser
except Exception as e:
    raise ImportError("pip install -U tree_sitter tree_sitter_languages pandas scikit-learn") from e

_PY_PARSER = None
def py_parser():
    global _PY_PARSER
    if _PY_PARSER is None:
        _PY_PARSER = get_parser("python")
    return _PY_PARSER

# --- Phrase dict ---
IMPORT_MAP = {
    "os": "operating system module",
    "subprocess": "process module",
    "base64": "base64 module",
    "requests": "network module",
    "urllib": "network module",
    "urllib3": "network module",
    "socket": "network module",
    "http": "network module",
    "ftplib": "network module",
    "telnetlib": "network module",
    "paramiko": "network module",
    "operator": "operator module",
    "codecs": "code utility module",
    "code": "code utility module",
    "sys": "system module",
    "shutil": "file system module",
    "pathlib": "file system module",
    "os.path": "file system module",
    "ctypes": "native interface module",
}
CALL_PHRASES = {
    ("os", "system"): "use operating system module call",
    ("subprocess", "Popen"): "use process module call",
    ("subprocess", "run"): "use process module call",
    ("subprocess", "call"): "use process module call",
    ("subprocess", "check_call"): "use process module call",
    ("subprocess", "check_output"): "use process module call",
}
URL_RE = re.compile(r"(https?://|\burl\b)", re.I)
B64_RE = re.compile(r"\b[A-Za-z0-9+/]{24,}={0,2}\b")

# ---------- TS utils ----------
def _txt(src: bytes, n) -> str:
    return src[n.start_byte:n.end_byte].decode("utf-8", "ignore")
def _iter(n):
    yield n
    for c in n.children:
        yield from _iter(c)
def _left_ident(attr):
    cur = attr
    while cur and cur.type == "attribute":
        cur = cur.child_by_field_name("object")
    if cur and cur.type == "identifier":
        return cur.text.decode("utf-8","ignore")
    return ""
def _attr_name(attr):
    if attr.type == "attribute":
        t = attr.child_by_field_name("attribute")
        if t and t.type == "identifier":
            return t.text.decode("utf-8","ignore")
    return ""
def _call_name(src: bytes, call_node) -> Tuple[str,str]:
    fn = call_node.child_by_field_name("function")
    if not fn: return "",""
    if fn.type == "identifier": return "", _txt(src, fn)
    if fn.type == "attribute":  return _left_ident(fn), _attr_name(fn)
    return "", _txt(src, fn).strip()

def extract_features_py(file: Path) -> Dict[str,object]:
    feats = {"imports": set(), "calls": [], "has_eval_exec": False, "url_count": 0, "b64_count": 0}
    try:
        src = file.read_bytes()
    except Exception:
        return feats
    try:
        tree = py_parser().parse(src)
    except Exception:
        return feats
    root = tree.root_node
    for n in _iter(root):
        t = n.type
        if t == "import_statement":
            text = _txt(src, n)
            parts = re.split(r"[\s,]+", text.replace("import"," "))
            for p in parts:
                p = p.strip()
                if not p or p=="as": continue
                base = p.split(".")[0]
                if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", base):
                    feats["imports"].add(base)
        elif t == "import_from_statement":
            mod = n.child_by_field_name("module")
            if mod:
                base = _txt(src, mod).split(".")[0]
                if base: feats["imports"].add(base)
        elif t == "call":
            mod, func = _call_name(src, n)
            if mod or func: feats["calls"].append((mod, func))
        elif t in {"string", "string_content", "f_string"}:
            s = _txt(src, n)
            if URL_RE.search(s): feats["url_count"] += 1
            if B64_RE.search(s): feats["b64_count"] += 1
    if any(f in {"eval","exec"} for _,f in feats["calls"]):
        feats["has_eval_exec"] = True
    return feats

# ---------- file / archive ----------
def extract_if_archive(p: Path, tmp: Path) -> Path:
    name = p.name.lower()
    if p.is_file() and name.endswith((".tar.gz",".tgz",".tar",".zip",".whl")):
        out = tmp / p.stem.replace(".tar","")
        out.mkdir(parents=True, exist_ok=True)
        try:
            if name.endswith((".tar.gz",".tgz",".tar")):
                with tarfile.open(p, "r:*") as tf: tf.extractall(out)
            else:
                with zipfile.ZipFile(p,"r") as zf: zf.extractall(out)
            return out
        except Exception:
            return p.parent
    return p

def find_version_root(packages_dir: Path, pkg: str, ver: str) -> Optional[Path]:
    """
    目标目录优先：
      {packages_dir}/{pkg}/{ver}/...
    兼容情况：
      - {packages_dir}/{pkg}/{something包含ver}/...
      - 版本目录下存在中间层: tar-gz / zip / src / source / build / dist
    若仅有归档文件（tar.gz/zip/whl/tar），尝试解包后再用。
    """
    pkg_dir = packages_dir / pkg
    if not pkg_dir.exists():
        cand_pkg = [p for p in packages_dir.iterdir() if p.is_dir() and p.name.startswith(pkg)]
        if not cand_pkg:
            return None
        pkg_dir = cand_pkg[0]

    ver_dir = pkg_dir / ver
    if not ver_dir.exists():
        cands = [d for d in pkg_dir.iterdir() if d.is_dir() and ver in d.name]
        if cands:
            ver_dir = cands[0]
        else:
            return None

    subdirs = [d for d in ver_dir.iterdir() if d.is_dir()]
    preferred_names = {"tar-gz", "zip", "src", "source", "build", "dist"}
    if len(subdirs) == 1 and subdirs[0].name in preferred_names:
        inner = subdirs[0]
    else:
        named = [d for d in subdirs if d.name in preferred_names]
        inner = named[0] if named else ver_dir

    if not any(inner.rglob("*.py")):
        archives = (list(inner.glob("*.tar.gz")) + list(inner.glob("*.tgz")) +
                    list(inner.glob("*.zip")) + list(inner.glob("*.whl")) +
                    list(inner.glob("*.tar")))
        if archives:
            tmp = Path(tempfile.mkdtemp(prefix="pyver_"))
            try:
                root = extract_if_archive(archives[0], tmp)
                return root if root.exists() else inner
            except Exception:
                shutil.rmtree(tmp, ignore_errors=True)
                return inner
    return inner

def choose_rep_py(root: Path, package_hint: Optional[str] = None) -> Optional[Path]:
    """
    代表文件优先级：
      1) root/**/<package_hint>/__init__.py 或该目录下首个 .py
      2) root/src/** 下的 .py
      3) 非 tests/examples/docs/egg-info 目录下的 .py
      4) 兜底：任意 .py
    """
    deny_dirs = re.compile(r"(?:^|/)(tests?|examples?|docs?|.*egg\-info)(?:/|$)", re.I)

    if package_hint:
        norm = package_hint.replace("-", "_")
        hits = list(root.rglob(f"{norm}/__init__.py"))
        if hits: return hits[0]
        hits = [p for p in root.rglob("*.py") if f"/{norm}/" in str(p).replace("\\","/")]
        if hits: return hits[0]

    src_dir = root / "src"
    if src_dir.exists():
        cand = list(src_dir.rglob("*.py"))
        cand = [p for p in cand if not deny_dirs.search(str(p).replace("\\","/"))]
        if cand: return cand[0]

    cand = [p for p in root.rglob("*.py") if not deny_dirs.search(str(p).replace("\\","/"))]
    if cand: return cand[0]

    anyp = list(root.rglob("*.py"))
    return anyp[0] if anyp else None

def to_text(pkg: str, relpath: str, feats: Dict[str,object]) -> str:
    parts = []
    mapped = []
    for imp in sorted(feats["imports"]):
        ph = IMPORT_MAP.get(imp)
        if ph and ph not in mapped:
            mapped.append(ph)
    parts += [f"import {m}" for m in mapped]
    uniq = set(feats["calls"])
    for key, phr in CALL_PHRASES.items():
        if key in uniq: parts.append(phr)
    if feats["has_eval_exec"]: parts.append("evaluate code at run-time")
    if feats["url_count"]>0:  parts.append("use URL")
    if feats["b64_count"]>0:  parts += ["use base64 string"] * min(4, feats["b64_count"])
    middle = ", ".join(parts) if parts else "no notable behavior"
    return f"start entry {pkg}/{relpath}, {middle}, end of entry"

# ---------- dataset ----------
def build_dataset(packages_dir: Path, labels_version_csv: Path) -> pd.DataFrame:
    lv = pd.read_csv(labels_version_csv)
    if not {"package","version","label"} <= set(lv.columns):
        raise ValueError("labels_version.csv 需要列：package,version,label")
    lv["label"] = lv["label"].astype(int)

    rows = []
    for pkg, ver, lab in lv[["package","version","label"]].itertuples(index=False):
        vroot = find_version_root(packages_dir, pkg, str(ver))
        if not vroot:
            rows.append({"textual_description": f"start entry {pkg}/{ver}, no source found, end of entry", "label": int(lab)})
            continue
        rep = choose_rep_py(vroot, package_hint=pkg)
        if not rep:
            rows.append({"textual_description": f"start entry {pkg}/{ver}, no python file, end of entry", "label": int(lab)})
            continue
        feats = extract_features_py(rep)
        try:
            rel = f"{ver}/{rep.relative_to(vroot)}".replace("\\","/")
        except Exception:
            rel = f"{ver}/{rep.name}"
        rows.append({"textual_description": to_text(pkg, rel, feats), "label": int(lab)})
    return pd.DataFrame(rows)

def split_and_save(df: pd.DataFrame, out_dir: Path, val_size: float, test_size: float, seed: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    if df.empty:
        for n in ["train","validation","test"]:
            pd.DataFrame(columns=["textual_description","label"]).to_csv(out_dir/f"{n}.csv", index=False)
        return
    X, y = df["textual_description"].astype(str), df["label"].astype(int)
    Xtmp, Xte, ytmp, yte = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    vr = val_size/(1-test_size)
    Xtr, Xva, ytr, yva = train_test_split(Xtmp, ytmp, test_size=vr, stratify=ytmp, random_state=seed)
    pd.DataFrame({"textual_description":Xtr,"label":ytr}).to_csv(out_dir/"train.csv", index=False)
    pd.DataFrame({"textual_description":Xva,"label":yva}).to_csv(out_dir/"validation.csv", index=False)
    pd.DataFrame({"textual_description":Xte,"label":yte}).to_csv(out_dir/"test.csv", index=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packages-dir", required=True)
    ap.add_argument("--labels-version", required=True)  # CSV: package,version,label
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--val-size", type=float, default=0.10)
    ap.add_argument("--test-size", type=float, default=0.10)
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    df = build_dataset(Path(args.packages-dir if hasattr(args,'packages-dir') else args.packages_dir), Path(args.labels_version))
    split_and_save(df, Path(args.out_dir), args.val_size, args.test_size, args.random_state)

if __name__ == "__main__":
    main()
