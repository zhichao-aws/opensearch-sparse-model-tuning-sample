import argparse
import json
import os

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=str, required=True)
args = parser.parse_args()

model_id = args.model_id

with open("overlaps.json", "r") as f:
    data = json.load(f)

all_overlap = data["all_overlap"]
additional_idxs = data["additional_idxs"]
target_idxs = data["target_idxs"]


def get_stats(model):
    norm = torch.norm(model.get_input_embeddings().weight, dim=1, keepdim=True).view(-1)
    bias = model.get_output_embeddings().bias
    return (
        float(norm[target_idxs].mean()),
        float(norm[additional_idxs].mean()),
        float(bias[target_idxs].mean()),
        float(bias[additional_idxs].mean()),
    )


# 基线模型
base_model = AutoModelForMaskedLM.from_pretrained(model_id)
base_stats = get_stats(base_model)

# 收集各变体在不同阶段的结果（含与基线差值）
rows = []

# 先把基线放入表格
rows.append(
    {
        "variant": model_id,
        "stage": "baseline",
        "norm_t": f"{base_stats[0]:.6f}",
        "norm_a": f"{base_stats[1]:.6f}",
        "bias_t": f"{base_stats[2]:.6f}",
        "bias_a": f"{base_stats[3]:.6f}",
        "d_norm_t": "+0.000000",
        "d_norm_a": "+0.000000",
        "d_bias_t": "+0.000000",
        "d_bias_a": "+0.000000",
        "p_tok_t": "",
        "p_tok_a": "",
    }
)

all_pts = sorted([x for x in os.listdir("pretrain") if x.startswith(model_id)])

for pt in all_pts:
    try:
        # MLM 预训练后
        mlm_model = AutoModelForMaskedLM.from_pretrained(f"pretrain/{pt}")
        stats_mlm = get_stats(mlm_model)
        rows.append(
            {
                "variant": pt,
                "stage": "MLM-PT",
                "norm_t": f"{stats_mlm[0]:.6f}",
                "norm_a": f"{stats_mlm[1]:.6f}",
                "bias_t": f"{stats_mlm[2]:.6f}",
                "bias_a": f"{stats_mlm[3]:.6f}",
                "d_norm_t": f"{(stats_mlm[0] - base_stats[0]):+.6f}",
                "d_norm_a": f"{(stats_mlm[1] - base_stats[1]):+.6f}",
                "d_bias_t": f"{(stats_mlm[2] - base_stats[2]):+.6f}",
                "d_bias_a": f"{(stats_mlm[3] - base_stats[3]):+.6f}",
                "p_tok_t": "",
                "p_tok_a": "",
            }
        )

        # 微调后
        finetuned_pt = [x for x in os.listdir("output/paper/bi") if pt in x][0]
        ft_model = AutoModelForMaskedLM.from_pretrained(
            f"output/paper/bi/{finetuned_pt}/final/checkpoint-150000"
        )
        stats_ft = get_stats(ft_model)

        # p_token 均值（仅微调后有）
        p_token = torch.load(
            f"output/paper/bi/{finetuned_pt}/final/evaluate_marco/msmarco.corpus.bin"
        )
        p_tok_t = float(p_token[target_idxs].mean())
        p_tok_a = float(p_token[additional_idxs].mean())

        rows.append(
            {
                "variant": pt,
                "stage": "Finetuned",
                "norm_t": f"{stats_ft[0]:.6f}",
                "norm_a": f"{stats_ft[1]:.6f}",
                "bias_t": f"{stats_ft[2]:.6f}",
                "bias_a": f"{stats_ft[3]:.6f}",
                "d_norm_t": f"{(stats_ft[0] - base_stats[0]):+.6f}",
                "d_norm_a": f"{(stats_ft[1] - base_stats[1]):+.6f}",
                "d_bias_t": f"{(stats_ft[2] - base_stats[2]):+.6f}",
                "d_bias_a": f"{(stats_ft[3] - base_stats[3]):+.6f}",
                "p_tok_t": f"{p_tok_t:.6f}",
                "p_tok_a": f"{p_tok_a:.6f}",
            }
        )
    except Exception:
        pass

# 打印对齐表格
headers = [
    "Variant",
    "Stage",
    "Norm[T]",
    "Norm[A]",
    "Bias[T]",
    "Bias[A]",
    "ΔNorm[T]",
    "ΔNorm[A]",
    "ΔBias[T]",
    "ΔBias[A]",
    "pTok[T]",
    "pTok[A]",
]

# 将行转为二维数组（字符串）
mat = []
for r in rows:
    mat.append(
        [
            r["variant"],
            r["stage"],
            r["norm_t"],
            r["norm_a"],
            r["bias_t"],
            r["bias_a"],
            r["d_norm_t"],
            r["d_norm_a"],
            r["d_bias_t"],
            r["d_bias_a"],
            r["p_tok_t"],
            r["p_tok_a"],
        ]
    )

# 计算每列最大宽度
col_widths = [len(h) for h in headers]
for row in mat:
    for i, cell in enumerate(row):
        if len(cell) > col_widths[i]:
            col_widths[i] = len(cell)


# 格式化打印
def print_row(row_vals):
    print("  ".join(val.ljust(col_widths[i]) for i, val in enumerate(row_vals)))


print_row(headers)
print_row(["-" * w for w in col_widths])
for row in mat:
    print_row(row)


for finetuned_pt in sorted(os.listdir("output/paper/bi")):
    try:
        p_token = torch.load(
            f"output/paper/bi/{finetuned_pt}/final/evaluate_marco/msmarco.corpus.bin"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            f"output/paper/bi/{finetuned_pt}/final/checkpoint-150000"
        )
        if len(tokenizer.vocab) != 30522:
            continue
        model = AutoModelForMaskedLM.from_pretrained(
            f"output/paper/bi/{finetuned_pt}/final/checkpoint-150000"
        )
        stats = get_stats(model)
        print(
            f"{p_token[target_idxs].mean():.6f}, {p_token[additional_idxs].mean():.6f}, {stats[0]:.6f}, {stats[1]:.6f}, {finetuned_pt}"
        )
    except:
        pass
