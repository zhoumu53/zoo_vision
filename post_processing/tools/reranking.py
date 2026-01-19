from itertools import permutations
from typing import Dict, List, Tuple, Any, Optional


def _known_scores_from_label_counts(
    label_counts: Dict[str, int],
    known_labels: List[str],
    score_mode: str = "ratio",
) -> Dict[str, float]:
    """
    label_counts: already filtered counts for this stitched_id (typically only known labels remain)
    score_mode:
      - "ratio": count(label) / sum(counts over ALL labels in label_counts)
      - "count": raw count(label)
    """
    total = sum(label_counts.values()) if label_counts else 0
    scores = {}
    for lab in known_labels:
        c = float(label_counts.get(lab, 0))
        if score_mode == "count":
            scores[lab] = c
        else:
            scores[lab] = (c / total) if total > 0 else 0.0
    return scores


def _rank_labels(scores: Dict[str, float]) -> List[Tuple[str, float]]:
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def assign_two_identities_from_trackid2label_counts(
    trackid2label_counts: Dict[int, Dict[str, int]],
    known_labels: List[str],
    score_mode: str = "ratio",
    eps: float = 1e-12,
) -> Dict[int, Dict[str, Any]]:
    """
    Solve the "two individuals in group" assignment problem from your native structure:
      trackid2label_counts: stitched_id -> {label: count}

    Returns a dict per stitched_id with:
      - known_scores
      - known_ranking
      - assigned_label (global optimum 1-1 mapping)
      - assignment_total_score
      - assignment_debug (how it decided)

    Assumptions:
      - len(known_labels) == 2
      - There are exactly 2 stitched IDs in the group (or you call this per-room/per-group).
      - trackid2label_counts has already been filtered so non-known labels are removed
        (or you still pass known_labels; we only score those labels anyway).
    """
    if len(known_labels) != 2:
        raise ValueError(f"Expected exactly 2 known_labels, got {len(known_labels)}: {known_labels}")

    stitched_ids = sorted(trackid2label_counts.keys())
    if len(stitched_ids) != 2:
        raise ValueError(f"Expected exactly 2 stitched IDs, got {len(stitched_ids)}: {stitched_ids}")

    # Build score matrix
    score_matrix: Dict[int, Dict[str, float]] = {}
    for sid in stitched_ids:
        score_matrix[sid] = _known_scores_from_label_counts(
            label_counts=trackid2label_counts.get(sid, {}),
            known_labels=known_labels,
            score_mode=score_mode,
        )

    # Evaluate both possible 1-1 assignments
    # Use total score, then tie-break by "margin" (confidence separation) if totals equal.
    best = None  # (total, margin_sum, assignment_dict)
    for perm in permutations(known_labels, 2):
        a0, a1 = perm[0], perm[1]
        sid0, sid1 = stitched_ids[0], stitched_ids[1]

        s0 = score_matrix[sid0][a0]
        s1 = score_matrix[sid1][a1]
        total = s0 + s1

        # Margin: how much better is the chosen label vs the alternative label for each stitched id
        alt0 = [l for l in known_labels if l != a0][0]
        alt1 = [l for l in known_labels if l != a1][0]
        m0 = s0 - score_matrix[sid0][alt0]
        m1 = s1 - score_matrix[sid1][alt1]
        margin_sum = m0 + m1

        candidate = (total, margin_sum, {sid0: a0, sid1: a1})
        if best is None:
            best = candidate
        else:
            # primary: higher total
            if candidate[0] > best[0] + eps:
                best = candidate
            # secondary: higher margin (more confident global assignment)
            elif abs(candidate[0] - best[0]) <= eps and candidate[1] > best[1] + eps:
                best = candidate
            # tertiary: deterministic ordering (lexicographic labels) to avoid randomness
            elif abs(candidate[0] - best[0]) <= eps and abs(candidate[1] - best[1]) <= eps:
                if tuple(candidate[2][sid] for sid in stitched_ids) < tuple(best[2][sid] for sid in stitched_ids):
                    best = candidate

    best_total, best_margin, best_assignment = best

    # Enrich output
    out: Dict[int, Dict[str, Any]] = {}
    for sid in stitched_ids:
        ks = score_matrix[sid]
        out[sid] = {
            "known_scores": ks,
            "known_ranking": _rank_labels(ks),
            "assigned_label": best_assignment[sid],
            "assignment_total_score": best_total,
            "assignment_debug": {
                "score_mode": score_mode,
                "best_margin_sum": best_margin,
                "chosen_pair": {stitched_ids[0]: best_assignment[stitched_ids[0]],
                                stitched_ids[1]: best_assignment[stitched_ids[1]]},
            },
        }
    return out



if __name__ == "__main__":
    case1_trackid2label_counts = {
        0: {"Indi": 8, "Chandra": 7},     # (after filtering Panang/Thai/Fahra out)
        1: {"Indi": 10},                 # no Chandra votes remain
    }
    known = ["Chandra", "Indi"]

    case2_trackid2label_counts = {
        0: {"Indi": 20, "Chandra": 11},
        1: {"Indi": 20, "Chandra": 6},
    }


    print("CASE 1:")
    r1 = assign_two_identities_from_trackid2label_counts(case1_trackid2label_counts, known, score_mode="ratio")
    for sid, info in r1.items():
        print(sid, info["known_ranking"], "->", info["assigned_label"])

    print("\nCASE 2:")
    r2 = assign_two_identities_from_trackid2label_counts(case2_trackid2label_counts, known, score_mode="ratio")
    for sid, info in r2.items():
        print(sid, info["known_ranking"], "->", info["assigned_label"])
