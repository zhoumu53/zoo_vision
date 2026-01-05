from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
from datetime import datetime

from post_processing.core.tracklet_manager import Tracklet
from post_processing.utils import load_embedding
from sklearn.cluster import KMeans

import logging
logger = logging.getLogger(__name__)    

def _l2norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x)
    if norm < eps:
        return x
    return x / norm



def _head_tail_proto(
    feats: np.ndarray,
    head_k: int = 5,
    tail_k: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    feats: (T, D) or (D,)
    Returns (head_proto, tail_proto), both L2-normalized.
    """
    feats = np.asarray(feats)
    if feats.ndim == 1:
        feats = feats[None, :]  # (D,) -> (1, D)

    T = feats.shape[0]
    k_h = min(head_k, T)
    k_t = min(tail_k, T)

    head = _l2norm(feats[:k_h].mean(axis=0))
    tail = _l2norm(feats[-k_t:].mean(axis=0))
    return head, tail


### TODO: num_identities parameter -- how to set it? dynamic?

def stitch_tracklets_bidirectional_gallery(
    tracklets: List[Tracklet],
    num_identities: int = 4,
    max_gap_frames: int = 600,
    local_sim_th: float = 0.5,
    gallery_sim_th: float = 0.45,
    head_k: int = 5,
    tail_k: int = 5,
    gallery_k: int = 10,
    w_local: float = 0.6,
    w_gallery: float = 0.4,
    logger_: Optional[logging.Logger] = None,
) -> None:
    """
    Bidirectional temporal stitching with identity galleries for long-term elephant tracking.
    (with temporal non-overlap constraint per identity)
    """
    log = logger_ or logger
    
    if not tracklets:
        log.info("stitch_tracklets_bidirectional_gallery: no tracklets.")
        return
    
    if num_identities <= 1:
        for t in tracklets:
            t.stitched_id = 0
        log.info("bidirectional_gallery: num_identities=1 -> all stitched_id = 0")
        return
    
    # -------------------------------
    # 1) Extract features for all tracklets
    # -------------------------------
    head_vecs: Dict[int, np.ndarray] = {}
    tail_vecs: Dict[int, np.ndarray] = {}
    mean_vecs: Dict[int, np.ndarray] = {}
    start_timestamps: Dict[int, datetime] = {}
    end_timestamps: Dict[int, datetime] = {}
    valid_indices: List[int] = []
    
    for i, t in enumerate(tracklets):
        if t.invalid_flag:
            continue
        
        if t.feature_path is None:
            log.warning(f"[bidirectional] Tracklet {i} has no feature_path; mark invalid.")
            t.invalid_flag = True
            continue
        
        try:
            feats, frame_ids, _, _ = load_embedding(t.feature_path)
        except Exception as e:
            log.exception(
                f"[bidirectional] Failed to load embedding from {t.feature_path} for tracklet {i}: {e}"
            )
            t.invalid_flag = True
            continue
        
        feats = np.asarray(feats)
        if feats.size == 0:
            log.warning(f"[bidirectional] Tracklet {i} has empty features; mark invalid.")
            t.invalid_flag = True
            continue
        
        head, tail = _head_tail_proto(feats, head_k=head_k, tail_k=tail_k)
        mean = _l2norm(feats.mean(axis=0))
        
        head_vecs[i] = head
        tail_vecs[i] = tail
        mean_vecs[i] = mean
        
        if t.start_timestamp is None or t.end_timestamp is None:
            raise ValueError(f"Tracklet {i} has no start_timestamp/end_timestamp")
        
        sf = t.start_timestamp
        ef = t.end_timestamp
        start_timestamps[i] = sf
        end_timestamps[i] = ef
        valid_indices.append(i)
    
    if not valid_indices:
        sid = 0
        for t in tracklets:
            t.stitched_id = sid
            sid += 1
        log.warning(
            "stitch_tracklets_bidirectional_gallery: all tracklets invalid; "
            "assigned unique stitched_id."
        )
        return
    
    # -------------------------------
    # 2) Sort by temporal order
    # -------------------------------
    order = sorted(valid_indices, key=lambda i: start_timestamps[i])
    log.info(
        f"bidirectional_gallery: N_valid={len(valid_indices)}, "
        f"num_identities={num_identities}, max_gap_frames={max_gap_frames}, "
        f"local_sim_th={local_sim_th}, gallery_sim_th={gallery_sim_th}"
    )
    
    # -------------------------------
    # 3) Identity management
    # -------------------------------
    # Each identity: {
    #   "id": int,
    #   "gallery": List[np.ndarray],
    #   "last_indices": List[int],
    #   "last_end_time": datetime
    # }
    identities: List[Dict] = []
    next_id = 0
    assigned_ids: Dict[int, int] = {}
    processed: List[int] = []
    
    def _create_new_identity(idx: int) -> int:
        nonlocal next_id
        new_id = next_id
        identities.append({
            "id": new_id,
            "gallery": [mean_vecs[idx]],
            "last_indices": [idx],
            "last_end_time": end_timestamps[idx],  # track time range
        })
        assigned_ids[idx] = new_id
        tracklets[idx].stitched_id = new_id
        next_id += 1
        return new_id
    
    def _add_to_identity(idx: int, identity_idx: int):
        identity = identities[identity_idx]
        identity["gallery"].append(mean_vecs[idx])
        identity["last_indices"].append(idx)
        
        # update last_end_time (monotonic in time because of sorted order)
        identity["last_end_time"] = max(identity["last_end_time"], end_timestamps[idx])
        
        # Keep only top-K most recent in gallery
        if len(identity["gallery"]) > gallery_k:
            identity["gallery"] = identity["gallery"][-gallery_k:]
            identity["last_indices"] = identity["last_indices"][-gallery_k:]
        
        assigned_ids[idx] = identity["id"]
        tracklets[idx].stitched_id = identity["id"]
    
    fps = 25  # TODO - check online code
    
    # -------------------------------
    # 4) Process tracklets in temporal order
    # -------------------------------
    for i in order:
        sf_i = start_timestamps[i]
        ef_i = end_timestamps[i]
        head_i = head_vecs[i]
        mean_i = mean_vecs[i]
        
        if not identities:
            _create_new_identity(i)
            processed.append(i)
            log.debug(f"[bidirectional] tracklet {i}: created first identity 0")
            continue
        
        # -------------------------------
        # a) Local matching: head(i) vs tail(prev_tracks)
        # -------------------------------
        best_local_idx = None
        best_local_sim = -1.0
        best_local_identity_idx = None
        
        for j in reversed(processed):
            ef_j = end_timestamps[j]
            gap_frames = int((sf_i - ef_j).total_seconds() * fps)
            
            if gap_frames < 0:
                continue
            if gap_frames > max_gap_frames:
                break
            
            local_sim = float(np.dot(tail_vecs[j], head_i))
            if local_sim > best_local_sim:
                best_local_sim = local_sim
                best_local_idx = j
                for idx, ident in enumerate(identities):
                    if assigned_ids[j] == ident["id"]:
                        best_local_identity_idx = idx
                        break
        
        # -------------------------------
        # b) Gallery matching: mean(i) vs each identity's gallery
        # -------------------------------
        gallery_scores: List[float] = []
        for identity in identities:
            gallery = identity["gallery"]
            if not gallery:
                gallery_scores.append(0.0)
                continue
            sims = [float(np.dot(mean_i, g)) for g in gallery]
            gallery_scores.append(float(np.mean(sims)))
        
        # -------------------------------
        # c) Combined scoring with temporal conflict check
        # -------------------------------
        combined_scores: List[float] = []
        no_conflict: List[bool] = []
        
        for idx, identity in enumerate(identities):
            last_end = identity.get("last_end_time", None)
            
            # temporal conflict: this identity already has a track that ends after this one starts
            conflict = last_end is not None and sf_i < last_end
            no_conflict.append(not conflict)
            
            if conflict:
                # strong penalty so this identity is not chosen if any non-conflicting identity exists
                combined_scores.append(-1e6)
                continue
            
            local_contrib = 0.0
            if best_local_identity_idx == idx and best_local_sim > 0:
                local_contrib = best_local_sim
            
            gallery_contrib = gallery_scores[idx]
            combined = w_local * local_contrib + w_gallery * gallery_contrib
            combined_scores.append(combined)
        
        # pick best identity among non-conflicting ones; if none, fallback to global best (rare case)
        valid_candidates = [k for k, ok in enumerate(no_conflict) if ok]
        if valid_candidates:
            best_identity_idx = max(valid_candidates, key=lambda k: combined_scores[k])
        else:
            best_identity_idx = int(np.argmax(combined_scores))
        
        best_score = combined_scores[best_identity_idx]
        
        # -------------------------------
        # Decision logic
        # -------------------------------
        decision_made = False
        
        # strong local continuity (also inherently non-conflicting via gap_frames)
        if best_local_sim >= local_sim_th and best_local_identity_idx is not None:
            _add_to_identity(i, best_local_identity_idx)
            log.debug(
                f"[bidirectional] tracklet {i} -> identity {identities[best_local_identity_idx]['id']} "
                f"via strong local (sim={best_local_sim:.3f})"
            )
            decision_made = True
        
        # combined score strong enough
        elif best_score >= (w_local * local_sim_th + w_gallery * gallery_sim_th):
            _add_to_identity(i, best_identity_idx)
            log.debug(
                f"[bidirectional] tracklet {i} -> identity {identities[best_identity_idx]['id']} "
                f"via combined (score={best_score:.3f}, local={best_local_sim:.3f}, "
                f"gallery={gallery_scores[best_identity_idx]:.3f})"
            )
            decision_made = True
        
        # new identity if we still have capacity
        elif len(identities) < num_identities:
            new_id = _create_new_identity(i)
            log.debug(
                f"[bidirectional] tracklet {i}: created new identity {new_id} "
                f"(best_score={best_score:.3f} < threshold)"
            )
            decision_made = True
        
        # forced assignment (all identities already exist, thresholds not met)
        if not decision_made:
            _add_to_identity(i, best_identity_idx)
            log.debug(
                f"[bidirectional] tracklet {i} -> identity {identities[best_identity_idx]['id']} "
                f"(forced, max identities reached, score={best_score:.3f})"
            )
        
        processed.append(i)
    
    # -------------------------------
    # 5) Handle invalid tracklets
    # -------------------------------
    for idx, t in enumerate(tracklets):
        if idx not in valid_indices:
            t.stitched_id = next_id
            next_id += 1
    
    num_final_ids = len(set(assigned_ids.values()))
    log.info(
        f"bidirectional_gallery: stitched {len(valid_indices)} valid tracklets "
        f"into {num_final_ids} identities (used IDs: {sorted(set(assigned_ids.values()))})"
    )


    ### update track id - change to stitched_id
    for t in tracklets:
        t.track_id = t.stitched_id

