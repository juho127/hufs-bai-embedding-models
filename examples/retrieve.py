#!/usr/bin/env python3
"""Rank two fully synthetic Korean passages with a local embedding checkpoint."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from model_loading import load_sentence_transformer


PASSAGES = [
    "[가상 문서 A] 모의 납품계약 기록에는 물품 수량, 배송 예정일, "
    "검수 담당자와 일정 변경 내역이 정리되어 있다.",
    "[가상 문서 B] 모의 회의실 이용 기록에는 예약 시간, 사용 인원, "
    "프로젝터 대여 여부와 회의 준비 사항이 정리되어 있다.",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/hufs-bai-legal-embed-1b-v0.1",
                        help="Local base-model or trained SentenceTransformer checkpoint directory")
    parser.add_argument("--query", default="물품 배송 일정이 변경된 기록을 찾고 싶어요.")
    parser.add_argument("--device", default=None, help="For example: cuda, cuda:0, or cpu")
    parser.add_argument("--max-seq-length", type=int, default=1536)
    args = parser.parse_args()

    if not Path(args.model).is_dir():
        parser.error("Local model directory not found. Run scripts/download_release.py first, "
                     "or pass --model with a trained checkpoint directory.")
    if not args.query.strip() or args.max_seq_length < 8:
        parser.error("Provide a non-empty query and --max-seq-length of at least 8.")

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cuda = torch.device(device).type == "cuda"
    if torch.device(device).type not in ("cuda", "cpu"):
        parser.error("This example supports CPU or CUDA.")
    if cuda and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; install the matching PyTorch wheel or use --device cpu.")
    if cuda:
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                parser.error("This example requires BF16 support on CUDA; alternatively use --device cpu.")

    # Keep FP32 parameters, including full-fine-tuning updates, and use BF16
    # only for CUDA compute.
    model, _ = load_sentence_transformer(args.model, device=device, dtype=torch.float32)
    model.max_seq_length = args.max_seq_length
    for name, expected in (("query", "query: "), ("document", "passage: ")):
        if model.prompts.get(name) != expected:
            raise RuntimeError(f"Unexpected {name} prompt in checkpoint: {model.prompts.get(name)!r}")
    model.eval()
    options = {"normalize_embeddings": True, "convert_to_tensor": True,
               "show_progress_bar": False, "batch_size": 2}
    precision = torch.autocast("cuda", dtype=torch.bfloat16) if cuda else nullcontext()
    with torch.inference_mode(), precision:
        # The checkpoint supplies pooling and prompt configuration. Do not
        # prepend query/passages manually or prefixes would appear twice.
        documents = model.encode_document(PASSAGES, **options)
        query = model.encode_query([args.query], **options)
    # Normalize again in FP32 before cosine ranking, as in the evaluator.
    documents = torch.nn.functional.normalize(documents.float(), p=2, dim=1)
    query = torch.nn.functional.normalize(query.float(), p=2, dim=1)
    scores = (query @ documents.T)[0].cpu().tolist()
    print(f"Query: {args.query}")
    for rank, index in enumerate(sorted(range(len(scores)), key=lambda i: (-scores[i], i)), 1):
        print(f"{rank}. cosine={scores[index]:.4f}  {PASSAGES[index]}")
    print("Synthetic example only; scores are not a legal retrieval benchmark.")


if __name__ == "__main__":
    main()
