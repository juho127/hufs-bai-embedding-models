"""Create artificial fixtures for software checks, NEVER a legal benchmark."""
import argparse
import json
from pathlib import Path

TOPICS = [
    ("계약", "납품 일정과 계약 문구 확인"),
    ("임대차", "보증금 관련 서류의 작성 경위"),
    ("노동", "근무시간 기록과 임금 내역 대조"),
    ("손해배상", "사고 경위와 증빙 자료의 연결"),
    ("행정", "처분 통지서와 신청 기록 확인"),
    ("형사", "진술서와 사건 기록의 대조"),
    ("소비자", "상품 설명과 구매 계약 확인"),
    ("상속", "재산 목록과 가족관계 서류 확인"),
    ("개인정보", "정보 제공 내역과 동의서 확인"),
    ("저작권", "이용 계약과 저작물 목록 확인"),
    ("회사", "회의록과 의사결정 경위 확인"),
    ("보험", "약관과 사고 접수 기록 확인"),
    ("부동산", "등기 기록과 거래 서류 대조"),
    ("조세", "신고 내역과 증빙 서류 확인"),
    ("가사", "합의서와 신청서의 내용 확인"),
    ("절차", "제출 문서와 접수 일자 확인"),
]


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/smoke")
    parser.add_argument("--count", type=int, default=256)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(args.count):
        case = f"SYNTHETIC-{i:04d}"
        domain, issue = TOPICS[i % len(TOPICS)]
        query = f"가상 사건 {case}의 {domain} 분야에서 {issue}에 관한 문서를 찾고 싶습니다."
        positive = (
            f"[실행 점검용 가상 문서] 문서 식별자 {case}. 분야: {domain}. 검토 주제: {issue}. "
            f"이 문서는 가상 사건 {case}의 서류 목록과 사실관계 정리 과정을 설명합니다. "
            "신청인이 제출한 기록, 당사자의 설명, 문서 작성일과 수정 이력을 서로 대조합니다. "
            "해당 기록이 질문의 대상 사건과 연결되는지 확인하고 관련 부분을 찾아 제시합니다. "
            "이 내용은 소프트웨어 실행을 점검하기 위해 만든 예시이며 실제 조문, 판결 또는 법률 의견이 아닙니다."
        )
        rows.append({"query": query, "positive": positive,
                     "query_id": f"q-{case}", "positive_id": f"p-{case}",
                     "group_id": case})
    write_jsonl(out / "pairs.jsonl", rows)
    write_jsonl(out / "corpus.jsonl", [{"id": r["positive_id"], "text": r["positive"], "group_id": r["group_id"]} for r in rows])
    # Overlap with training here is intentional: this is only an end-to-end runtime fixture.
    write_jsonl(out / "queries.jsonl", [{"id": r["query_id"], "text": r["query"], "group_id": r["group_id"]} for r in rows[:16]])
    write_jsonl(out / "qrels.jsonl", [{"query_id": r["query_id"], "corpus_id": r["positive_id"], "relevance": 1} for r in rows[:16]])
    (out / "README.txt").write_text(
        "SYNTHETIC RUNTIME FIXTURE ONLY. No legal facts or measured research results.\n"
        "Queries intentionally overlap the training fixture; do not report these metrics as held-out quality.\n",
        encoding="utf-8")
    print(f"Wrote {len(rows)} synthetic fixture pairs to {out}.")


if __name__ == "__main__":
    main()
