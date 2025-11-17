# -*- coding: utf-8 -*-


import os
import re
import json
import argparse
from typing import Dict, Any, List

import torch
import pandas as pd
import datasets as hfds
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
    set_seed,
)

from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
)

# ---------------- Collator：只在 "<|assistant|>:" 之后计算 loss，并屏蔽 PAD ----------------
class SimpleCompletionOnlyCollator:
    def __init__(self, tokenizer, response_template="<|assistant|>:", max_length=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        ids = tokenizer(response_template, add_special_tokens=False, return_attention_mask=False)
        self.templ_ids = ids["input_ids"]

    def _find_template_start(self, ids):
        t = self.templ_ids
        if not t:
            return None
        L, T = len(ids), len(t)
        last = None
        for i in range(0, L - T + 1):
            if ids[i:i+T] == t:
                last = i
        return last

    def __call__(self, features):
        texts = [f["text"] if isinstance(f, dict) else f for f in features]
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        input_ids = batch["input_ids"]
        labels = input_ids.clone()

        for i in range(input_ids.size(0)):
            ids = input_ids[i].tolist()
            start = self._find_template_start(ids)
            # 屏蔽 "<|assistant|>:" 之前
            if start is None:
                labels[i].fill_(-100)
            else:
                if start > 0:
                    labels[i, :start] = -100

        # 屏蔽 padding
        if "attention_mask" in batch:
            labels[batch["attention_mask"] == 0] = -100

        batch["labels"] = labels
        return batch

# ---------------- 轻量版 QLoRA 预处理（避免大规模 FP32 拷贝触发 OOM） ----------------
def light_prepare_model_for_kbit_training(model):
    """
    - 冻结全部参量
    - 将少量 norm/bias 转 float32，提升稳定性，几乎不增显存
    - 开启 gradient checkpointing / input requires grad（如支持）
    """
    for p in model.parameters():
        p.requires_grad_(False)

    norm_keys = ("norm", "ln_f", "layernorm", "input_layernorm", "post_attention_layernorm")
    for name, module in model.named_modules():
        lname = name.lower()
        if any(k in lname for k in norm_keys):
            w = getattr(module, "weight", None)
            if w is not None and w.dtype != torch.float32:
                module.weight = torch.nn.Parameter(w.data.to(torch.float32), requires_grad=False)
            b = getattr(module, "bias", None)
            if b is not None and b.dtype != torch.float32:
                module.bias = torch.nn.Parameter(b.data.to(torch.float32), requires_grad=False)

    try: model.gradient_checkpointing_enable()
    except Exception: pass
    try: model.enable_input_require_grads()
    except Exception: pass
    return model

# ---------------- 列名探测 / CSV 读取 ----------------
TEXT_CANDS  = ["text", "prompt", "instruction", "content", "sentence", "textual_description"]
LABEL_CANDS = ["label", "y", "target", "gold"]

def detect_columns(df: pd.DataFrame, text_col=None, label_col=None):
    cols_lower = {c.lower(): c for c in df.columns}
    if text_col:
        if text_col in df.columns:
            tcol = text_col
        elif text_col.lower() in cols_lower:
            tcol = cols_lower[text_col.lower()]
        else:
            raise ValueError(f"--text_col='{text_col}' 不在列中：{list(df.columns)}")
    else:
        tcol = next((cols_lower[k] for k in TEXT_CANDS if k in cols_lower), None)

    if label_col:
        if label_col in df.columns:
            lcol = label_col
        elif label_col.lower() in cols_lower:
            lcol = cols_lower[label_col.lower()]
        else:
            raise ValueError(f"--label_col='{label_col}' 不在列中：{list(df.columns)}")
    else:
        lcol = next((cols_lower[k] for k in LABEL_CANDS if k in cols_lower), None)

    if tcol is None:
        non_label_cols = [c for c in df.columns if c.lower() not in LABEL_CANDS]
        if non_label_cols:
            tcol = non_label_cols[0]

    if tcol is None or lcol is None:
        raise ValueError(
            f"列名不符，未找到文本或标签列。实际列：{list(df.columns)}；"
            f"可用 --text_col 与 --label_col 指定。"
        )
    return tcol, lcol

def load_csv(path: str, text_col=None, label_col=None) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    tcol, lcol = detect_columns(df, text_col=text_col, label_col=label_col)
    df = df[[tcol, lcol]].rename(columns={tcol: "text", lcol: "label"}).copy()
    df["label"] = df["label"].astype(str).str.extract(r"([01])")[0]
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    return df

# ---------------- Prompt & Dataset（训练样本含标签；验证/测试不含） ----------------
def build_prompt(text: str) -> str:
    return (
        "<|system|>: 你是一个根据输入文本判断是否为恶意/风险的助手。输出严格为 0 或 1。\n"
        f"<|user|>: 文本：{text}\n请只输出 0 或 1（1=恶意/阳性；0=正常/阴性）。\n"
        "<|assistant|>: "
    )

def make_text_with_label(item: Dict[str, Any]) -> str:
    return build_prompt(item["text"]) + str(item["label"]).strip()

def make_text_without_label(item: Dict[str, Any]) -> str:
    return build_prompt(item["text"])

def df_to_dataset(df: pd.DataFrame, with_label: bool) -> hfds.Dataset:
    rows = df.to_dict(orient="records")
    if with_label:
        texts = [make_text_with_label(r) for r in rows]
    else:
        texts = [make_text_without_label(r) for r in rows]
    return hfds.Dataset.from_dict({"text": texts, "label": df["label"].astype(str).tolist()})

# ---------------- 简单评估（从生成里抓第一个 0/1） ----------------
DIGIT_RE = re.compile(r"[01]")

def first_digit01(s: str) -> str:
    m = DIGIT_RE.search(s)
    return m.group(0) if m else "0"

@torch.no_grad()
def evaluate_on_dataset(model, tokenizer, ds: hfds.Dataset, batch_size: int = 4, max_new_tokens: int = 4):
    model.eval()
    preds: List[str] = []
    gts: List[str] = [str(x) for x in ds["label"]]
    for i in range(0, len(ds), batch_size):
        batch = ds[i : i + batch_size]["text"]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        decoded = tokenizer.batch_decode(out, skip_special_tokens=False)
        for full in decoded:
            seg = full.split("<|assistant|>:")[-1]
            preds.append(first_digit01(seg))

    acc = accuracy_score(gts, preds)
    p, r, f1, _ = precision_recall_fscore_support(
        gts, preds, average="binary", pos_label="1", zero_division=0
    )
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}, preds

# ---------------- 设备内存预算（不含 "disk" 键） ----------------
def build_max_memory(reserve_gb: int, cpu_mem_gb: int):
    mem = {"cpu": f"{int(cpu_mem_gb)}GiB"}
    if torch.cuda.is_available():
        vis = os.environ.get("CUDA_VISIBLE_DEVICES")
        if vis:
            gpu_indices = list(range(len(vis.split(","))))
        else:
            gpu_indices = list(range(torch.cuda.device_count()))
        for i in gpu_indices:
            try:
                total = int(torch.cuda.get_device_properties(i).total_memory / (1024**3))
            except Exception:
                total = 32
            allow = max(1, total - int(reserve_gb))
            mem[i] = f"{allow}GiB"
    return mem

# ---------------- 加载 tokenizer + 量化模型 ----------------
def load_tokenizer_and_model(model_path, cache_dir, hf_token, is_local, reserve_gb, cpu_mem_gb, offload_dir):
    cfg = AutoConfig.from_pretrained(
        model_path, trust_remote_code=True, cache_dir=cache_dir,
        token=hf_token, local_files_only=is_local
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=True, trust_remote_code=True, cache_dir=cache_dir,
        token=hf_token, local_files_only=is_local
    )
    # ✅ decoder-only 用左填充，避免警告 & 生成错位
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    max_memory = build_max_memory(reserve_gb, cpu_mem_gb)
    if offload_dir:
        os.makedirs(offload_dir, exist_ok=True)

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        quantization_config=bnb_cfg,
        offload_folder=offload_dir,
        low_cpu_mem_usage=True,
        cache_dir=cache_dir,
        token=hf_token,
        local_files_only=is_local,
    )
    model.config.use_cache = False
    return tokenizer, model

# ---------------- 主流程 ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="模型名或本地 snapshot 目录")
    parser.add_argument("--hf_token", default=os.environ.get("HUGGINGFACE_TOKEN"))
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--offload_dir", default=None)
    parser.add_argument("--reserve_gb", type=int, default=4, help="每张 GPU 预留显存(GB)")
    parser.add_argument("--cpu_mem_gb", type=int, default=128, help="CPU 内存预算(GB)")

    parser.add_argument("--train_csv", default="data/train.csv")
    parser.add_argument("--val_csv",   default="data/validation.csv")
    parser.add_argument("--test_csv",  default="data/test.csv")
    parser.add_argument("--text_col",  default=None)
    parser.add_argument("--label_col", default=None)

    parser.add_argument("--outdir",    default="gptoss-20b-qlora-bincls")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--bsz", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)

    # LoRA（轻量默认，省显存）
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # 仅推理 / 评估
    parser.add_argument("--predict_only", action="store_true", help="只加载已训练 LoRA，在测试集上推理评估")
    parser.add_argument("--adapter_dir", default=None, help="LoRA 适配器目录（默认 <outdir>/adapter）")
    parser.add_argument("--save_preds_csv", default="preds_test.csv")
    parser.add_argument("--max_gen_new_tokens", type=int, default=4)

    args = parser.parse_args()
    set_seed(42)

    # 关闭“极速下载但未装hf-transfer”的硬依赖
    try:
        import hf_transfer  # noqa
    except Exception:
        if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") == "1":
            os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)

    # 数据
    train_df = load_csv(args.train_csv, text_col=args.text_col, label_col=args.label_col) if not args.predict_only else None
    val_df   = load_csv(args.val_csv,   text_col=args.text_col, label_col=args.label_col)
    test_df  = load_csv(args.test_csv,  text_col=args.text_col, label_col=args.label_col)

    # 构造数据集（训练: 带标签；验证/测试：不带标签）
    if not args.predict_only:
        train_ds = df_to_dataset(train_df, with_label=True)
    val_ds   = df_to_dataset(val_df,   with_label=False)
    test_ds  = df_to_dataset(test_df,  with_label=False)

    # 模型/分词器
    is_local = os.path.isdir(args.model)
    print(f">>> Loading from: {args.model} (local_only={is_local})")
    tokenizer, base_model = load_tokenizer_and_model(
        args.model, args.cache_dir, args.hf_token, is_local,
        args.reserve_gb, args.cpu_mem_gb, args.offload_dir
    )

    if args.predict_only:
        # 仅推理：加载 LoRA 适配器并评估测试集
        adapter_dir = args.adapter_dir or os.path.join(args.outdir, "adapter")
        print(f">>> Loading LoRA adapter from: {adapter_dir}")
        model = PeftModel.from_pretrained(base_model, adapter_dir)
        print(">>> Running evaluation on validation & test ...")
        val_metrics, _   = evaluate_on_dataset(model, tokenizer, val_ds,  batch_size=max(1, args.bsz), max_new_tokens=args.max_gen_new_tokens)
        test_metrics, pr = evaluate_on_dataset(model, tokenizer, test_ds, batch_size=max(1, args.bsz), max_new_tokens=args.max_gen_new_tokens)
        print("Val metrics:", json.dumps(val_metrics, ensure_ascii=False))
        print("Test metrics:", json.dumps(test_metrics, ensure_ascii=False))

        # 保存预测 CSV
        out_csv = args.save_preds_csv
        pd.DataFrame({
            "text": test_df["text"],
            "label": test_df["label"].astype(str),
            "pred": pr
        }).to_csv(out_csv, index=False)
        print(f"Saved predictions -> {out_csv}")
        return

    # 训练：注入 LoRA（轻量预处理）
    print(">>> Preparing model for k-bit training (QLoRA, light) & injecting LoRA ...")
    torch.cuda.empty_cache()
    model = light_prepare_model_for_kbit_training(base_model)
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    print("LoRA ready.")

    # Collator
    collator = SimpleCompletionOnlyCollator(
        tokenizer=tokenizer,
        response_template="<|assistant|>:",
        max_length=args.max_seq_len,
    )

    # 训练参数（用通用字段，兼容旧版）
    targs = TrainingArguments(
        output_dir=args.outdir,
        per_device_train_batch_size=args.bsz,
        per_device_eval_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        fp16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none",
    )

    from transformers import Trainer
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,           # 这里只是为了在日志中能看到 eval loss（不生成）
        data_collator=collator,
        tokenizer=tokenizer,           # 某些版本需要；新版本会有 deprecate 警告，可忽略
    )

    # 训练
    trainer.train()

    # 保存 LoRA 适配器与分词器
    adapter_dir = os.path.join(args.outdir, "adapter")
    os.makedirs(adapter_dir, exist_ok=True)
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(args.outdir)

    # 真实评估（基于生成）
    print(">>> Running final evaluation on validation & test ...")
    val_metrics, _   = evaluate_on_dataset(model, tokenizer, val_ds,  batch_size=max(1, args.bsz), max_new_tokens=args.max_gen_new_tokens)
    test_metrics, pr = evaluate_on_dataset(model, tokenizer, test_ds, batch_size=max(1, args.bsz), max_new_tokens=args.max_gen_new_tokens)
    with open(os.path.join(args.outdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"val": val_metrics, "test": test_metrics}, f, ensure_ascii=False, indent=2)
    print("Val metrics:", json.dumps(val_metrics, ensure_ascii=False))
    print("Test metrics:", json.dumps(test_metrics, ensure_ascii=False))

    # 同步导出测试集预测
    preds_csv = os.path.join(args.outdir, "preds_test.csv")
    pd.DataFrame({
        "text": test_df["text"],
        "label": test_df["label"].astype(str),
        "pred": pr
    }).to_csv(preds_csv, index=False)
    print(f"Saved predictions -> {preds_csv}")


if __name__ == "__main__":
    # 更稳：关闭 tokenizers 并行 + 降碎片化
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
    main()
