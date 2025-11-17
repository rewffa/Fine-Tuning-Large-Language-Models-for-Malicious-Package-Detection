#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author-style dataset generator for npm (Tree-sitter, version-level, multi-file aggregation + alias resolution)
- 修正：改为迭代遍历语法树，避免 RecursionError
- 增强：增加 max-node-visits / max-file-bytes 保护，处理超大/深层 bundle

输入：
  --packages-dir   {packages_dir}/{package}/{version}/...（也兼容 {package}/{含version}/… 或 {package@version}/…）
  --labels-version CSV: package,version,label  (label∈{0,1})

输出：
  out_dir/{train,validation,test}.csv  （列： textual_description,label）

依赖：
  pip install -U tree_sitter tree_sitter_languages pandas scikit-learn
"""

import argparse, os, re, json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
from sklearn.model_selection import train_test_split

# ---- Tree-sitter setup ----
try:
    from tree_sitter_languages import get_parser
except Exception as e:
    raise ImportError("pip install -U tree_sitter tree_sitter_languages pandas scikit-learn") from e

_JS_PARSER = None
_TS_PARSER = None
def get_ts_parser_for_suffix(suffix: str):
    global _JS_PARSER, _TS_PARSER
    if suffix.lower().endswith(".ts"):
        if _TS_PARSER is None: _TS_PARSER = get_parser("typescript")
        return _TS_PARSER
    else:
        if _JS_PARSER is None: _JS_PARSER = get_parser("javascript")
        return _JS_PARSER

# ---- Phrase dict / regex ----
IMPORT_MAP = {
    # Node core
    "http":"network module","https":"network module","net":"network module",
    "dns":"network module","dgram":"network module","tls":"network module",
    "fs":"file system module","path":"file system module","os":"operating system module",
    "child_process":"process module","vm":"code utility module","crypto":"code utility module",
    # common 3rd-party
    "axios":"network module","node-fetch":"network module","got":"network module","request":"network module",
}
URL_RE = re.compile(r"(https?://|\burl\b)", re.I)
B64_RE = re.compile(r"\b[A-Za-z0-9+/]{24,}={0,2}\b")
DENY_DIRS_RE = re.compile(r"(?:^|/)(tests?|__tests__|examples?|docs?|node_modules)(?:/|$)", re.I)

# ---- Utility: TS traversal/text ----
def ts_text(src: bytes, node) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8","ignore")

def ts_iter_nodes(root, max_nodes: int = 500_000):
    """
    以显式栈遍历节点（先序），限制最大访问节点数，避免深度递归导致的 RecursionError。
    """
    stack = [root]
    visits = 0
    while stack:
        node = stack.pop()
        yield node
        visits += 1
        if visits >= max_nodes:
            break
        # 逆序压栈以保持从左到右遍历
        # 注意：children 可能很大，尽量不要 copy
        for i in range(node.child_count - 1, -1, -1):
            stack.append(node.children[i])

# ---- Parse aliases (require/import) from raw text ----
RE_REQUIRE_ALIAS = re.compile(
    r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)", re.M)
RE_REQUIRE_DESTRUCT = re.compile(
    r"\b(?:const|let|var)\s*\{\s*([^\}]+)\s*\}\s*=\s*require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)", re.M)
RE_IMPORT_DEFAULT = re.compile(
    r"\bimport\s+([A-Za-z_$][\w$]*)\s+from\s+['\"]([^'\"]+)['\"]", re.M)
RE_IMPORT_NAMESPACE = re.compile(
    r"\bimport\s+\*\s+as\s+([A-Za-z_$][\w$]*)\s+from\s+['\"]([^'\"]+)['\"]", re.M)
RE_IMPORT_NAMED = re.compile(
    r"\bimport\s*\{\s*([^\}]+)\s*\}\s*from\s*['\"]([^'\"]+)['\"]", re.M)
RE_IMPORT_BARE = re.compile(
    r"\bimport\s*['\"]([^'\"]+)['\"]", re.M)

def parse_js_import_aliases(text: str):
    """
    返回：
      imports: set[str]                 # 模块名（小写）
      alias_to_mod: dict[str,str]       # e.g. cp -> child_process
      name_to_full: dict[str,(mod,fn)]  # e.g. ex -> (child_process, exec)
    """
    imports = set()
    alias_to_mod: Dict[str,str] = {}
    name_to_full: Dict[str,Tuple[str,str]] = {}

    for alias, mod in RE_REQUIRE_ALIAS.findall(text):
        base = mod.strip().lower()
        imports.add(base)
        alias_to_mod[alias] = base

    for names, mod in RE_REQUIRE_DESTRUCT.findall(text):
        base = mod.strip().lower()
        imports.add(base)
        for item in names.split(","):
            item = item.strip()
            if not item: continue
            if ":" in item:
                orig, al = [t.strip() for t in item.split(":",1)]
                name_to_full[al] = (base, orig)
            else:
                name_to_full[item] = (base, item)

    for alias, mod in RE_IMPORT_DEFAULT.findall(text):
        base = mod.strip().lower()
        imports.add(base)
        alias_to_mod[alias] = base

    for alias, mod in RE_IMPORT_NAMESPACE.findall(text):
        base = mod.strip().lower()
        imports.add(base)
        alias_to_mod[alias] = base

    for names, mod in RE_IMPORT_NAMED.findall(text):
        base = mod.strip().lower()
        imports.add(base)
        for item in names.split(","):
            item = item.strip()
            if not item: continue
            if re.search(r"\sas\s", item):
                orig, al = [t.strip() for t in re.split(r"\sas\s", item, 1)]
                name_to_full[al] = (base, orig)
            else:
                name_to_full[item] = (base, item)

    for mod in RE_IMPORT_BARE.findall(text):
        imports.add(mod.strip().lower())

    return imports, alias_to_mod, name_to_full

# ---- Call extraction via TS ----
def extract_calls_ts(src: bytes, suffix: str,
                     alias_to_mod: Dict[str,str], name_to_full: Dict[str,Tuple[str,str]],
                     max_nodes: int):
    """
    返回：
      calls: list[(module, func)]
      has_eval_exec: bool
      calls_os: bool
      calls_proc: bool
    """
    calls: List[Tuple[str,str]] = []
    has_eval_exec = False
    calls_os = False
    calls_proc = False

    try:
        parser = get_ts_parser_for_suffix(suffix)
        tree = parser.parse(src)
        root = tree.root_node
    except Exception:
        text = src.decode("utf-8","ignore").lower()
        if re.search(r"\beval\s*\(", text): has_eval_exec = True
        if re.search(r"\bnew\s+function\s*\(", text): has_eval_exec = True
        if re.search(r"\bvm\s*\.\s*runinnewcontext\s*\(", text): has_eval_exec = True
        if re.search(r"\bchild_process\s*\.\s*exec(sync)?\s*\(", text): calls_os = True
        if re.search(r"\bchild_process\s*\.\s*(spawn|fork)\s*\(", text): calls_proc = True
        return calls, has_eval_exec, calls_os, calls_proc

    for n in ts_iter_nodes(root, max_nodes=max_nodes):
        if n.type != "call_expression":
            continue
        fn = n.child_by_field_name("function")
        if fn is None:
            continue

        mod, func = "", ""
        t = fn.type
        if t in ("identifier","private_identifier","shorthand_property_identifier"):
            func = ts_text(src, fn)
            if func in name_to_full:
                mod, func = name_to_full[func]
        elif t == "member_expression":
            obj = fn.child_by_field_name("object")
            prop = fn.child_by_field_name("property")
            # 找最左侧对象标识符
            base = None
            cur = obj
            while cur is not None and cur.type == "member_expression":
                cur = cur.child_by_field_name("object")
            if cur is not None and cur.type in ("identifier","private_identifier","shorthand_property_identifier"):
                base = ts_text(src, cur)
            if prop:
                func = ts_text(src, prop)
            if base:
                if base in alias_to_mod:
                    mod = alias_to_mod[base]
                else:
                    mod = base  # 直接把对象名当作模块名（如 http.request）
        else:
            # 其他复杂表达式忽略
            pass

        if func:
            if not mod and func in name_to_full:
                mod, func = name_to_full[func]
            calls.append((mod, func))

            lname = func.lower()
            if lname == "eval": has_eval_exec = True
            # new Function(...) 在 TS 里不是简单 identifier 判断，这里略过；正则降级可覆盖
            if (mod in {"child_process","cp"} or mod == "" and "child_process" in ts_text(src, n).lower()):
                if lname in {"exec","execsync"}: calls_os = True
                if lname in {"spawn","fork"}:    calls_proc = True

    return calls, has_eval_exec, calls_os, calls_proc

# ---- Strings (URL/Base64) via TS ----
def extract_strings_ts(src: bytes, suffix: str, max_nodes: int):
    url_cnt = b64_cnt = 0
    try:
        parser = get_ts_parser_for_suffix(suffix)
        tree = parser.parse(src)
        root = tree.root_node
        for n in ts_iter_nodes(root, max_nodes=max_nodes):
            if n.type in {"string","template_string"}:
                s = ts_text(src, n)
                if URL_RE.search(s): url_cnt += 1
                if B64_RE.search(s): b64_cnt += 1
    except Exception:
        txt = src.decode("utf-8","ignore")
        url_cnt += len(URL_RE.findall(txt))
        b64_cnt += len(B64_RE.findall(txt))
    return url_cnt, b64_cnt

# ---- Single-file features (enhanced with alias parsing) ----
def extract_features_js_enhanced(file: Path, max_file_bytes: int, max_nodes: int) -> Dict[str,object]:
    feats = {"imports": set(), "calls": [], "has_eval_exec": False,
             "url_count": 0, "b64_count": 0, "calls_os": False, "calls_proc": False}

    try:
        size = file.stat().st_size
    except Exception:
        size = 0

    try:
        src = file.read_bytes()
    except Exception:
        return feats

    # 超大 bundle：走轻量级通道（只做别名解析 + 字符串正则），避免极深语法树
    light_mode = (max_file_bytes > 0 and size > max_file_bytes)

    txt = src.decode("utf-8","ignore")
    imports, alias_to_mod, name_to_full = parse_js_import_aliases(txt)
    feats["imports"] |= imports

    if not light_mode:
        calls, has_eval_exec, calls_os, calls_proc = extract_calls_ts(
            src, file.suffix, alias_to_mod, name_to_full, max_nodes=max_nodes
        )
        feats["calls"].extend(calls)
        feats["has_eval_exec"] = feats["has_eval_exec"] or has_eval_exec
        feats["calls_os"] = feats["calls_os"] or calls_os
        feats["calls_proc"] = feats["calls_proc"] or calls_proc
    else:
        # 降级：简单模式补一些关键字
        low = txt.lower()
        if re.search(r"\beval\s*\(", low): feats["has_eval_exec"] = True
        if re.search(r"\bvm\s*\.\s*runinnewcontext\s*\(", low): feats["has_eval_exec"] = True
        if re.search(r"require\s*\(\s*['\"]child_process['\"]\s*\)\s*\.?\s*exec(sync)?\s*\(", low):
            feats["calls_os"] = True
        if re.search(r"require\s*\(\s*['\"]child_process['\"]\s*\)\s*\.?\s*(spawn|fork)\s*\(", low):
            feats["calls_proc"] = True

    u, b = extract_strings_ts(src, file.suffix, max_nodes=max_nodes) if not light_mode else (
        len(URL_RE.findall(txt)), len(B64_RE.findall(txt))
    )
    feats["url_count"] += u
    feats["b64_count"] += b

    return feats

# ---- Multi-file aggregation (top-K) ----
def extract_features_dir(root: Path, max_files: int = 20, include_tests: bool = False,
                         max_file_bytes: int = 2_000_000, max_nodes: int = 500_000):
    agg = {"imports": set(), "calls": [], "has_eval_exec": False,
           "url_count": 0, "b64_count": 0, "calls_os": False, "calls_proc": False}

    files = list(root.rglob("*.js")) + list(root.rglob("*.ts"))
    if not include_tests:
        files = [p for p in files if not DENY_DIRS_RE.search(str(p).replace("\\","/"))]

    def score(p: Path):
        s = str(p).replace("\\","/")
        pri = 0
        if "/dist/" in s: pri -= 30
        elif "/lib/" in s: pri -= 20
        elif "/src/" in s: pri -= 10
        if s.endswith("/index.js") or s.endswith("/index.ts"): pri -= 5
        if DENY_DIRS_RE.search(s): pri += 100
        try:
            size = p.stat().st_size
        except Exception:
            size = 0
        return (pri, -size)

    files.sort(key=score)
    for p in files[:max_files]:
        try:
            f = extract_features_js_enhanced(p, max_file_bytes=max_file_bytes, max_nodes=max_nodes)
        except Exception:
            # 再保险：任何单文件异常都不影响总体
            continue
        agg["imports"] |= f["imports"]
        agg["calls"].extend(f["calls"])
        agg["has_eval_exec"] = agg["has_eval_exec"] or f["has_eval_exec"]
        agg["url_count"] += f["url_count"]
        agg["b64_count"] += f["b64_count"]
        agg["calls_os"] = agg["calls_os"] or f["calls_os"]
        agg["calls_proc"] = agg["calls_proc"] or f["calls_proc"]

    return agg

# ---- package.json helpers ----
def read_package_json(root: Path) -> dict:
    pj = root / "package.json"
    if not pj.exists(): return {}
    try: return json.loads(pj.read_text("utf-8"))
    except Exception: return {}

def to_text(pkg: str, ver_rel: str, feats: Dict[str,object], pkg_json: dict) -> str:
    parts = []
    mapped = []
    for imp in sorted(feats["imports"]):
        base = imp.split("/",1)[0] if not imp.startswith("@") else imp.split("/",2)[0]
        ph = IMPORT_MAP.get(imp) or IMPORT_MAP.get(base)
        if ph and ph not in mapped: mapped.append(ph)
    parts += [f"import {m}" for m in mapped]
    if feats["calls_os"]:   parts.append("use operating system module call")
    if feats["calls_proc"]: parts.append("use process module call")
    if feats["has_eval_exec"]: parts.append("evaluate code at run-time")
    if feats["url_count"]>0: parts.append("use URL")
    if feats["b64_count"]>0: parts += ["use base64 string"] * min(4, feats["b64_count"])
    scripts = (pkg_json.get("scripts") or {}) if isinstance(pkg_json, dict) else {}
    if any(k in scripts for k in ["preinstall","install","postinstall"]):
        parts.append("modify startup behavior")
    middle = ", ".join(parts) if parts else "no notable behavior"
    return f"start entry {pkg}/{ver_rel}, {middle}, end of entry"

# ---- version root locating ----
def find_version_root(packages_dir: Path, pkg: str, ver: str) -> Optional[Path]:
    """
    常见布局：
      {packages_dir}/{pkg}/{ver}/...
      {packages_dir}/{pkg}/{包含ver的子目录}/...
      {packages_dir}/{pkg@ver}/...
    """
    pdir = packages_dir / pkg
    if not pdir.exists():
        cand = [d for d in packages_dir.iterdir()
                if d.is_dir() and (d.name == pkg or d.name.endswith(pkg.split("/")[-1]))]
        if not cand: return None
        pdir = cand[0]

    if (pdir / ver).exists():
        root = (pdir / ver)
    else:
        cands = [d for d in pdir.iterdir() if d.is_dir() and ver in d.name]
        if cands:
            root = cands[0]
        else:
            pv = packages_dir / f"{pkg}@{ver}"
            if pv.exists(): root = pv
            else: return None
    return root

# ---- dataset building / split ----
def build_dataset(packages_dir: Path, labels_version_csv: Path,
                  max_files: int, include_tests: bool,
                  max_file_bytes: int, max_nodes: int) -> pd.DataFrame:
    lv = pd.read_csv(labels_version_csv)
    if not {"package","version","label"} <= set(lv.columns):
        raise ValueError("labels_version.csv 需要列：package,version,label")
    lv["label"] = lv["label"].astype(int)

    rows = []
    for pkg, ver, lab in lv[["package","version","label"]].itertuples(index=False):
        vroot = find_version_root(packages_dir, pkg, str(ver))
        if not vroot:
            rows.append({"textual_description": f"start entry {pkg}/{ver}, no javascript file, end of entry", "label": int(lab)})
            continue

        pkg_json = read_package_json(vroot)
        feats = extract_features_dir(
            vroot,
            max_files=max_files,
            include_tests=include_tests,
            max_file_bytes=max_file_bytes,
            max_nodes=max_nodes,
        )
        rel = f"{ver}"
        rows.append({"textual_description": to_text(pkg, rel, feats, pkg_json), "label": int(lab)})
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
    ap.add_argument("--max-files", type=int, default=20, help="每个版本最多扫描的 .js/.ts 文件数")
    ap.add_argument("--include-tests", type=int, default=0, help="是否包含 tests/docs/examples/node_modules（1=是，0=否）")
    ap.add_argument("--max-node-visits", type=int, default=500_000, help="遍历语法树的最大节点访问数（防止超深语法树）")
    ap.add_argument("--max-file-bytes", type=int, default=2_000_000, help="超此大小的文件改走轻量提取（字节数）")
    args = ap.parse_args()

    df = build_dataset(
        Path(args.packages_dir), Path(args.labels_version),
        max_files=args.max_files,
        include_tests=bool(args.include_tests),
        max_file_bytes=args.max_file_bytes,
        max_nodes=args.max_node_visits,
    )
    split_and_save(df, Path(args.out_dir), args.val_size, args.test_size, args.random_state)

if __name__ == "__main__":
    main()
