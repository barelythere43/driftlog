"""
Run RAG evaluation against the golden dataset. Requires API and DB to be available.

  python -m eval.run_eval
  python -m eval.run_eval --dry-run
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import httpx
from datasets import Dataset
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    ResponseRelevancy,
)

try:
    import asyncpg
except ImportError:
    asyncpg = None  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

QUERY_URL = "http://localhost:8000/api/v1/query"
GOLDEN_PATH = PROJECT_ROOT / "eval" / "golden_dataset.json"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"

DATABASE_URL = os.environ.get("DATABASE_URL", "")


def _db_url_for_asyncpg() -> str:
    url = DATABASE_URL.strip()
    if url.startswith("postgresql+psycopg://"):
        url = url.replace("postgresql+psycopg://", "postgresql://", 1)
    if "@postgres:" in url:
        url = url.replace("@postgres:", "@localhost:")
    return url


async def fetch_chunk_contents(chunk_ids: list[str]) -> list[str]:
    """Query the database for chunk content by IDs. Returns list of content strings in order of chunk_ids."""
    if not chunk_ids or asyncpg is None:
        return []
    url = _db_url_for_asyncpg()
    if not url:
        return []
    try:
        conn = await asyncpg.connect(url)
        try:
            rows = await conn.fetch(
                "SELECT id::text, content FROM chunks WHERE id = ANY($1::uuid[])",
                chunk_ids,
            )
            by_id = {r["id"]: r["content"] for r in rows}
            return [by_id.get(cid, "") for cid in chunk_ids]
        finally:
            await conn.close()
    except Exception:
        return []


async def run_queries_and_collect(
    golden: list[dict],
    dry_run: bool,
) -> tuple[list[dict], list[dict]]:
    """POST each question to the query API; collect answer, citations, etc. Returns (rows_for_ragas, raw_responses)."""
    rows_for_ragas = []
    raw_responses = []

    async with httpx.AsyncClient(timeout=60.0) as client:
        for item in golden:
            q = item["question"]
            filters = item.get("filters")
            payload = {"question": q, "filters": filters}
            resp = await client.post(QUERY_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()

            answer = data.get("answer", "")
            citations = data.get("citations") or []
            confidence = data.get("confidence", 0.0)
            chunks_retrieved = data.get("chunks_retrieved", 0)
            chunks_after_rerank = data.get("chunks_after_rerank", 0)

            raw_responses.append({
                "id": item["id"],
                "category": item["category"],
                "question": q,
                "answer": answer,
                "citations": citations,
                "confidence": confidence,
                "chunks_retrieved": chunks_retrieved,
                "chunks_after_rerank": chunks_after_rerank,
            })

            if dry_run:
                print(f"[{item['id']}] {item['category']}: {q[:60]}...")
                print(f"  answer: {(answer or '')[:200]}...")
                print(f"  confidence={confidence}, chunks_retrieved={chunks_retrieved}, chunks_after_rerank={chunks_after_rerank}")
                print()
                rows_for_ragas.append({
                    "user_input": q,
                    "response": answer,
                    "retrieved_contexts": [],
                    "reference": item.get("expected_answer", ""),
                })
                continue

            chunk_ids = [c.get("chunk_id") for c in citations if c.get("chunk_id")]
            contexts = await fetch_chunk_contents(chunk_ids)
            if not contexts:
                contexts = ["No context retrieved"]
            reference = item.get("expected_answer", "")

            rows_for_ragas.append({
                "user_input": q,
                "response": answer,
                "retrieved_contexts": contexts,
                "reference": reference,
            })

    return rows_for_ragas, raw_responses


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG evaluation on golden dataset")
    parser.add_argument("--dry-run", action="store_true", help="Hit API and print responses only, no Ragas scoring")
    args = parser.parse_args()

    if not GOLDEN_PATH.exists():
        raise SystemExit(f"Golden dataset not found: {GOLDEN_PATH}")

    with open(GOLDEN_PATH) as f:
        golden = json.load(f)
    if not golden:
        print("Golden dataset is empty.")
        return

    rows, raw_responses = asyncio.run(run_queries_and_collect(golden, args.dry_run))

    if args.dry_run:
        print("Dry run complete. Exiting without Ragas evaluation.")
        return

    # Split in-scope (Ragas) vs out-of-scope (refusal check only)
    in_scope_indices = [i for i, g in enumerate(golden) if g["category"] != "out_of_scope"]
    out_of_scope_indices = [i for i, g in enumerate(golden) if g["category"] == "out_of_scope"]

    # Out-of-scope: evaluate by refusal language
    REFUSAL_PHRASES = ("don't have enough information", "no information", "can't answer", "outside")
    oos_results = []
    for i in out_of_scope_indices:
        meta = golden[i]
        raw = raw_responses[i]
        answer = (raw.get("answer") or "").lower()
        has_refusal = any(phrase in answer for phrase in REFUSAL_PHRASES)
        passed = "pass" if has_refusal else "fail"
        oos_results.append({"id": meta["id"], "category": "out_of_scope", "pass": passed})

    def safe_avg(items: list[dict], key: str) -> float:
        nums = [x[key] for x in items if x.get(key) is not None]
        return sum(nums) / len(nums) if nums else 0.0

    if oos_results:
        print("\n--- Out-of-scope (refusal check) ---")
        for r in oos_results:
            print(f"  {r['id']:<20} {r['pass']}")
        print()

    # Build Ragas dataset only from in-scope items
    if not in_scope_indices:
        print("No in-scope questions for Ragas evaluation.")
        per_question = [{"id": r["id"], "category": r["category"], "pass": r["pass"]} for r in oos_results]
        by_cat = defaultdict(list)
        for pq in per_question:
            by_cat[pq["category"]].append(pq)
    else:
        rows_in_scope = [rows[i] for i in in_scope_indices]
        golden_in_scope = [golden[i] for i in in_scope_indices]

        # Ensure no empty retrieved_contexts for Ragas
        ragas_data = {
            "user_input": [r["user_input"] for r in rows_in_scope],
            "response": [r["response"] for r in rows_in_scope],
            "retrieved_contexts": [
                r["retrieved_contexts"] if r["retrieved_contexts"] else ["No context retrieved"]
                for r in rows_in_scope
            ],
            "reference": [r["reference"] for r in rows_in_scope],
        }
        hf_dataset = Dataset.from_dict(ragas_data)
        eval_dataset = EvaluationDataset.from_hf_dataset(hf_dataset)

        # Evaluator LLM with higher max_tokens so metrics (e.g. ResponseRelevancy) can return multiple generations
        evaluator_llm = LangchainLLMWrapper(
            ChatOpenAI(model="gpt-4o-mini", max_tokens=1024),
        )
        result = evaluate(
            eval_dataset,
            metrics=[Faithfulness(), ResponseRelevancy(), ContextPrecision(), ContextRecall()],
            llm=evaluator_llm,
            embeddings=LangchainEmbeddingsWrapper(OpenAIEmbeddings()),
        )

        scores = result.scores
        PASS_THRESHOLD = 0.5

        print("\n--- Per-question results (in-scope) ---")
        print(f"{'id':<20} {'category':<14} {'faithfulness':<14} {'relevancy':<12} {'precision':<12} {'recall':<10} pass/fail")
        print("-" * 95)

        per_question = []
        for i, meta in enumerate(golden_in_scope):
            s = scores[i] if i < len(scores) else {}
            f = s.get("faithfulness", float("nan"))
            r = s.get("response_relevancy", s.get("answer_relevancy", float("nan")))
            p = s.get("context_precision", float("nan"))
            c = s.get("context_recall", float("nan"))
            vals = [f, r, p, c]
            avg = sum(v for v in vals if v == v) / 4.0 if sum(1 for v in vals if v == v) == 4 else float("nan")
            passed = "pass" if (avg >= PASS_THRESHOLD and avg == avg) else "fail"
            print(f"{meta['id']:<20} {meta['category']:<14} {f:<14.4f} {r:<12.4f} {p:<12.4f} {c:<10.4f} {passed}")
            per_question.append({
                "id": meta["id"],
                "category": meta["category"],
                "faithfulness": f if f == f else None,
                "response_relevancy": r if r == r else None,
                "context_precision": p if p == p else None,
                "context_recall": c if c == c else None,
                "pass": passed,
            })

        # Add out-of-scope to per_question for aggregates / JSON
        for r in oos_results:
            per_question.append({"id": r["id"], "category": r["category"], "pass": r["pass"]})

        # Averages per category (in-scope categories only for Ragas metrics)
        by_cat = defaultdict(list)
        for pq in per_question:
            by_cat[pq["category"]].append(pq)

        in_scope_cats = [c for c in sorted(by_cat.keys()) if c != "out_of_scope"]
        if in_scope_cats:
            print("\n--- Averages by category ---")
            for cat in in_scope_cats:
                items = by_cat[cat]
                n = len(items)
                f = safe_avg(items, "faithfulness")
                r = safe_avg(items, "response_relevancy")
                p = safe_avg(items, "context_precision")
                c = safe_avg(items, "context_recall")
                print(f"  {cat}: faithfulness={f:.4f}, response_relevancy={r:.4f}, context_precision={p:.4f}, context_recall={c:.4f} (n={n})")

        in_scope_only = [pq for pq in per_question if pq["category"] != "out_of_scope"]
        if in_scope_only:
            print("\n--- Overall (in-scope) ---")
            f = safe_avg(in_scope_only, "faithfulness")
            r = safe_avg(in_scope_only, "response_relevancy")
            p = safe_avg(in_scope_only, "context_precision")
            c = safe_avg(in_scope_only, "context_recall")
            print(f"  faithfulness={f:.4f}, response_relevancy={r:.4f}, context_precision={p:.4f}, context_recall={c:.4f}")

    # Save results (include oos in per_question and by_cat)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"eval_run_{ts}.json"
    by_cat_agg = {}
    for cat, items in by_cat.items():
        if cat == "out_of_scope":
            by_cat_agg[cat] = {"pass_count": sum(1 for x in items if x.get("pass") == "pass"), "total": len(items)}
        else:
            by_cat_agg[cat] = {
                "faithfulness": safe_avg(items, "faithfulness"),
                "response_relevancy": safe_avg(items, "response_relevancy"),
                "context_precision": safe_avg(items, "context_precision"),
                "context_recall": safe_avg(items, "context_recall"),
            }
    agg_f = safe_avg([pq for pq in per_question if pq.get("category") != "out_of_scope"], "faithfulness") if per_question else 0.0
    agg_r = safe_avg([pq for pq in per_question if pq.get("category") != "out_of_scope"], "response_relevancy") if per_question else 0.0
    agg_p = safe_avg([pq for pq in per_question if pq.get("category") != "out_of_scope"], "context_precision") if per_question else 0.0
    agg_c = safe_avg([pq for pq in per_question if pq.get("category") != "out_of_scope"], "context_recall") if per_question else 0.0
    out = {
        "timestamp": ts,
        "aggregate": {"faithfulness": agg_f, "response_relevancy": agg_r, "context_precision": agg_p, "context_recall": agg_c},
        "by_category": by_cat_agg,
        "per_question": per_question,
        "out_of_scope_results": oos_results,
        "raw_responses": raw_responses,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
