#!/usr/bin/env python3
"""Rank two synthetic Korean defense administration passages with a local model."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from model_loading import load_sentence_transformer


# These invented passages are not examples from the training or evaluation data.
PASSAGES = [
    "[가상 국방행정 문서 A] 교육지원 부대의 모의 차량 정비계획에는 "
    "정비 입고 일정, 안전 점검 담당자와 부품 신청 절차가 정리되어 있다.",
    "[가상 국방행정 문서 B] 군 관사 공용시설의 모의 관리계획에는 "
    "공용 세탁실 운영 시간, 시설 청소 일정과 주민 의견 접수 방법이 정리되어 있다.",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/hufs-bai-defense-embed-1b-v0.2",
                        help="Local base-model or trained SentenceTransformer checkpoint directory")
    parser.add_argument("--query", default="교육용 차량의 정비 일정과 부품 신청 절차는 어디서 확인하나요?")
    parser.add_argument("--device", default=None, help="For example: cuda, cuda:0, or cpu")
    parser.add_argument("--max-seq-length", type=int, default=1536)
    args = parser.parse_args()

    if not Path(args.model).is_dir():
        parser.error("Local model directory not found. Download the defense release with "
                     "scripts/download_release.py --manifest releases/defense-v0.2.0/manifest.json "
                     "--output models/hufs-bai-defense-embed-1b-v0.2 first, or pass --model.")
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

    # Preserve FP32 parameters and use BF16 only for CUDA compute.
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
        # Encode methods apply the checkpoint prompts; do not prepend them twice.
        documents = model.encode_document(PASSAGES, **options)
        query = model.encode_query([args.query], **options)
    documents = torch.nn.functional.normalize(documents.float(), p=2, dim=1)
    query = torch.nn.functional.normalize(query.float(), p=2, dim=1)
    scores = (query @ documents.T)[0].cpu().tolist()
    print(f"Query: {args.query}")
    for rank, index in enumerate(sorted(range(len(scores)), key=lambda i: (-scores[i], i)), 1):
        print(f"{rank}. cosine={scores[index]:.4f}  {PASSAGES[index]}")
    print("Synthetic example only; scores are not a defense retrieval benchmark.")


if __name__ == "__main__":
    main()
