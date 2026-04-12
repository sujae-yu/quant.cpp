"""Stage 5: RE-SEARCH.

If Stage 4 verification fails (verdict != CONFIDENT), the researcher
asks the locator to try a different region. Capped at MAX_RETRIES
attempts. After that, return calibrated uncertainty.
"""
from dataclasses import dataclass
from typing import List

from . import locator, lookup, verifier
from .gist import Gist
from .lookup import LookupResult
from .verifier import VerifyResult


MAX_RETRIES = 2


@dataclass
class ResearchResult:
    final_answer: str
    final_verdict: str    # CONFIDENT | UNSURE | CONTRADICTED | EXHAUSTED
    final_chunk: int
    attempts: List[dict]  # one per attempt: {chunk, answer, verdict, reason}
    n_retries: int


def research(
    question: str,
    initial_lookup: LookupResult,
    initial_verify: VerifyResult,
    gist: Gist,
    doc_text: str,
    *,
    max_retries: int = MAX_RETRIES,
    verbose: bool = False,
) -> ResearchResult:
    """If verification didn't pass, retry with different regions."""
    attempts = [{
        "chunk": initial_lookup.chunk_id,
        "answer": initial_lookup.answer,
        "verdict": initial_verify.verdict,
        "reason": initial_verify.reason,
    }]

    if initial_verify.verdict == "CONFIDENT":
        return ResearchResult(
            final_answer=initial_lookup.answer,
            final_verdict="CONFIDENT",
            final_chunk=initial_lookup.chunk_id,
            attempts=attempts,
            n_retries=0,
        )

    # Phase A-2 insight: when the locator finds the RIGHT article but
    # the WRONG chunk within it, the answer is usually in an adjacent
    # chunk (±1-2). Human pattern: "it's not on this page — let me
    # check the next page" before going to a completely different section.
    #
    # Strategy: retry 1 = try adjacent chunks first (same article region),
    # then retry 2+ = locator picks a different article entirely.
    initial_cid = initial_lookup.chunk_id
    n_chunks = len(gist.chunks)

    # Build neighbor list: [cid-1, cid+1, cid-2, cid+2] (if they exist)
    neighbors = []
    for offset in [1, -1, 2, -2]:
        ncid = initial_cid + offset
        if 0 <= ncid < n_chunks:
            neighbors.append(ncid)

    excluded = [initial_cid]
    neighbor_idx = 0

    for retry in range(max_retries):
        if verbose:
            print(f"[researcher] retry {retry+1}/{max_retries}, excluding chunks {excluded}")

        # First retries: try adjacent chunks (same article neighborhood)
        if neighbor_idx < len(neighbors):
            ncid = neighbors[neighbor_idx]
            neighbor_idx += 1
            if ncid in excluded:
                continue
            chunk = gist.chunks[ncid]
            new_region = locator.RegionPointer(
                chunk_id=ncid, confidence="medium",
                candidates=[ncid], char_start=chunk.char_start,
                char_end=chunk.char_end, score=0.0, method="neighbor",
            )
            if verbose:
                print(f"[researcher] trying neighbor chunk {ncid}")
        else:
            # Later retries: locator picks a different region entirely
            new_region = locator.locate(question, gist, excluded_chunks=excluded, verbose=verbose)
        # If locator picked a chunk we already excluded (parser failure or only-one-chunk doc), bail
        if new_region.chunk_id in excluded:
            if verbose:
                print(f"[researcher] locator returned excluded chunk {new_region.chunk_id}, stopping")
            break

        try:
            new_lookup = lookup.lookup(question, new_region, doc_text, verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"[researcher] lookup exception on chunk {new_region.chunk_id}: {e}")
            excluded.append(new_region.chunk_id)
            attempts.append({
                "chunk": new_region.chunk_id, "answer": f"[EXCEPTION: {e}]",
                "verdict": "ERROR", "reason": str(e),
            })
            continue

        # Skip verification if lookup returned an error (server crash/timeout)
        if new_lookup.method == "error":
            if verbose:
                print(f"[researcher] lookup error on chunk {new_region.chunk_id}, skipping")
            excluded.append(new_region.chunk_id)
            attempts.append({
                "chunk": new_region.chunk_id,
                "answer": new_lookup.answer,
                "verdict": "ERROR",
                "reason": "lookup error",
            })
            continue

        new_verify = verifier.verify(
            question, new_lookup.answer, gist,
            region_text=new_lookup.region_text,
            chunk_id=new_lookup.chunk_id,
            verbose=verbose,
        )

        attempts.append({
            "chunk": new_lookup.chunk_id,
            "answer": new_lookup.answer,
            "verdict": new_verify.verdict,
            "reason": new_verify.reason,
        })

        if new_verify.verdict == "CONFIDENT":
            return ResearchResult(
                final_answer=new_lookup.answer,
                final_verdict="CONFIDENT",
                final_chunk=new_lookup.chunk_id,
                attempts=attempts,
                n_retries=retry + 1,
            )

        excluded.append(new_lookup.chunk_id)

    # All retries exhausted (A13: explicit logging when all chunks tried)
    if verbose:
        n_available = len(gist.chunks)
        n_tried = len(excluded)
        print(f"[researcher] exhausted: tried {n_tried}/{n_available} chunks, "
              f"no CONFIDENT answer found")

    # Return the best uncertain answer. Prefer non-error, non-refusal answers.
    best = attempts[-1]
    for a in attempts:
        if a["verdict"] not in ("ERROR", "CONTRADICTED"):
            best = a
            break
    last = best
    return ResearchResult(
        final_answer=last["answer"],
        final_verdict="EXHAUSTED",
        final_chunk=last["chunk"],
        attempts=attempts,
        n_retries=len(attempts) - 1,
    )
