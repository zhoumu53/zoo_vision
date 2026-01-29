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
### TODO: stitching - with cosine sim thresholding - assign new id if no good match

def stitch_tracklets_bidirectional_gallery_dynamic(
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
    short_tracklet_th: int = 50,  # threshold for short tracklets
    long_tracklet_th: int = 100,  # threshold for long tracklets
    logger_: Optional[logging.Logger] = None,
) -> None:
    """
    Bidirectional temporal stitching with identity galleries for long-term elephant tracking.
    (with temporal non-overlap constraint per identity)
    
    Dynamic weight adjustment based on tracklet length:
    - Short tracklets (< short_tracklet_th frames): Focus on local matching
    - Medium tracklets: Use default weights
    - Long tracklets (> long_tracklet_th frames): Focus more on gallery matching
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
    tracklet_lengths: Dict[int, int] = {}  # store feature lengths
    start_timestamps: Dict[int, datetime] = {}
    end_timestamps: Dict[int, datetime] = {}
    valid_indices: List[int] = []
    
    for i, t in enumerate(tracklets):
        if t.invalid_flag:
            continue
        
        if t.feature_path is None:
            log.warning(f"[bidirectional] Tracklet {t.track_id} has no feature_path; mark invalid.")
            t.invalid_flag = True
            continue
        
        if not t.feature_path.exists():
            log.error(
                f"[bidirectional] Feature path {t.feature_path} does not exist; mark invalid."
            )
            t.invalid_flag = True
            continue
        
        try:
            feats, frame_ids, _, _ = load_embedding(t.feature_path)
        except Exception as e:
            log.error(
                f"[bidirectional] Failed to load embedding from {t.feature_path} for tracklet {i}, skipping."
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
        tracklet_lengths[i] = len(feats)  # store length
        
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
            "last_end_time": end_timestamps[idx],
        })
        assigned_ids[idx] = new_id
        tracklets[idx].stitched_id = new_id
        next_id += 1
        return new_id
    
    def _add_to_identity(idx: int, identity_idx: int):
        identity = identities[identity_idx]
        identity["gallery"].append(mean_vecs[idx])
        identity["last_indices"].append(idx)
        
        identity["last_end_time"] = max(identity["last_end_time"], end_timestamps[idx])
        
        if len(identity["gallery"]) > gallery_k:
            identity["gallery"] = identity["gallery"][-gallery_k:]
            identity["last_indices"] = identity["last_indices"][-gallery_k:]
        
        assigned_ids[idx] = identity["id"]
        tracklets[idx].stitched_id = identity["id"]
    
    def _get_dynamic_weights(feat_length: int) -> Tuple[float, float]:
        """
        Compute dynamic weights based on tracklet length.
        
        Short tracklets: More local weight (temporal continuity is more reliable)
        Long tracklets: More gallery weight (appearance is more reliable)
        """
        if feat_length < short_tracklet_th:
            # Short: 80% local, 20% gallery
            return 0.8, 0.2
        elif feat_length > long_tracklet_th:
            # Long: 40% local, 60% gallery
            return 0.4, 0.6
        else:
            # Medium: interpolate linearly
            ratio = (feat_length - short_tracklet_th) / (long_tracklet_th - short_tracklet_th)
            w_local_dynamic = 0.8 - (0.4 * ratio)  # 0.8 -> 0.4
            w_gallery_dynamic = 0.2 + (0.4 * ratio)  # 0.2 -> 0.6
            return w_local_dynamic, w_gallery_dynamic
    
    fps = 25
    
    # -------------------------------
    # 4) Process tracklets in temporal order
    # -------------------------------
    for i in order:
        sf_i = start_timestamps[i]
        ef_i = end_timestamps[i]
        head_i = head_vecs[i]
        mean_i = mean_vecs[i]
        feat_len_i = tracklet_lengths[i]
        
        # Get dynamic weights based on current tracklet length
        w_local_dynamic, w_gallery_dynamic = _get_dynamic_weights(feat_len_i)
        
        if not identities:
            _create_new_identity(i)
            processed.append(i)
            log.debug(f"[bidirectional] tracklet {i} (len={feat_len_i}): created first identity 0")
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
            
            conflict = last_end is not None and sf_i < last_end
            no_conflict.append(not conflict)
            
            if conflict:
                combined_scores.append(-1e6)
                continue
            
            local_contrib = 0.0
            if best_local_identity_idx == idx and best_local_sim > 0:
                local_contrib = best_local_sim
            
            gallery_contrib = gallery_scores[idx]
            # Use dynamic weights
            combined = w_local_dynamic * local_contrib + w_gallery_dynamic * gallery_contrib
            combined_scores.append(combined)
        
        valid_candidates = [k for k, ok in enumerate(no_conflict) if ok]
        if valid_candidates:
            best_identity_idx = max(valid_candidates, key=lambda k: combined_scores[k])
        else:
            best_identity_idx = int(np.argmax(combined_scores))
        
        best_score = combined_scores[best_identity_idx]
        
        # -------------------------------
        # Decision logic (with dynamic thresholds)
        # -------------------------------
        decision_made = False
        
        # For short tracklets, prioritize local matching more
        local_th_adjusted = local_sim_th * (1.2 if feat_len_i < short_tracklet_th else 1.0)
        
        # For stitch_tracklets_bidirectional_gallery_dynamic (line ~326)
        if best_local_sim >= local_th_adjusted and best_local_identity_idx is not None:
            # CRITICAL: Check for temporal conflict before accepting strong local match
            identity = identities[best_local_identity_idx]
            last_end = identity.get("last_end_time", None)
            conflict = last_end is not None and sf_i < last_end
            
            if not conflict:
                _add_to_identity(i, best_local_identity_idx)
                log.debug(
                    f"[bidirectional] tracklet {i} (len={feat_len_i}, w_local={w_local_dynamic:.2f}) "
                    f"-> identity {identities[best_local_identity_idx]['id']} "
                    f"via strong local (sim={best_local_sim:.3f})"
                )
                decision_made = True
            else:
                log.debug(
                    f"[bidirectional] tracklet {i} (len={feat_len_i}): strong local match "
                    f"(sim={best_local_sim:.3f}) rejected due to temporal conflict "
                    f"(sf_i={sf_i}, identity_last_end={last_end})"
                )
        
        elif best_score >= (w_local_dynamic * local_sim_th + w_gallery_dynamic * gallery_sim_th):
            _add_to_identity(i, best_identity_idx)
            log.debug(
                f"[bidirectional] tracklet {i} (len={feat_len_i}, w_local={w_local_dynamic:.2f}) "
                f"-> identity {identities[best_identity_idx]['id']} "
                f"via combined (score={best_score:.3f}, local={best_local_sim:.3f}, "
                f"gallery={gallery_scores[best_identity_idx]:.3f})"
            )
            decision_made = True
        
        elif len(identities) < num_identities:
            new_id = _create_new_identity(i)
            log.debug(
                f"[bidirectional] tracklet {i} (len={feat_len_i}): created new identity {new_id} "
                f"(best_score={best_score:.3f} < threshold)"
            )
            decision_made = True
        
        if not decision_made:
            _add_to_identity(i, best_identity_idx)
            log.debug(
                f"[bidirectional] tracklet {i} (len={feat_len_i}) "
                f"-> identity {identities[best_identity_idx]['id']} "
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
            log.warning(f"[bidirectional] Tracklet {t.track_id} has no feature_path; mark invalid.")
            t.invalid_flag = True
            continue
        
        if not t.feature_path.exists():
            log.error(
                f"[bidirectional] Feature path {t.feature_path} does not exist; mark invalid."
            )
            t.invalid_flag = True
            continue
        
        try:
            feats, frame_ids, _, _ = load_embedding(t.feature_path)
        except Exception as e:
            log.error(
                f"[bidirectional] Failed to load embedding from {t.feature_path} for tracklet {i}, skipping."
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
        
        
        if best_local_sim >= local_sim_th and best_local_identity_idx is not None:
            # CRITICAL: Check for temporal conflict before accepting strong local match
            identity = identities[best_local_identity_idx]
            last_end = identity.get("last_end_time", None)
            conflict = last_end is not None and sf_i < last_end
            
            if not conflict:
                _add_to_identity(i, best_local_identity_idx)
                log.debug(
                    f"[bidirectional] tracklet {i} "
                    f"-> identity {identities[best_local_identity_idx]['id']} "
                    f"via strong local (sim={best_local_sim:.3f})"
                )
                decision_made = True
            else:
                log.debug(
                    f"[bidirectional] tracklet {i}: strong local match "
                    f"(sim={best_local_sim:.3f}) rejected due to temporal conflict "
                    f"(sf_i={sf_i}, identity_last_end={last_end})"
                )
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



def stitch_tracklets_bidirectional_gallery_robust(
    tracklets: List[Tracklet],
    num_identities: int = 4,
    max_gap_frames: int = 600,  # 24 seconds
    max_identity_gap_frames: int = 45000,  # 30 minutes - NEW
    local_sim_th: float = 0.5,
    gallery_sim_th: float = 0.45,
    head_k: int = 5,
    tail_k: int = 5,
    gallery_k: int = 10,
    w_local: float = 0.6,
    w_gallery: float = 0.4,
    short_tracklet_th: int = 50,
    long_tracklet_th: int = 100,
    min_gallery_consistency: float = 0.35,  # NEW: reject if too dissimilar
    logger_: Optional[logging.Logger] = None,
) -> None:
    """
    Robust bidirectional stitching that handles imperfect classifiers.
    
    Key improvements:
    1. Strict temporal overlap prevention (checks ALL tracklets in identity)
    2. Maximum gap enforcement (prevents stitching across hours)
    3. Gallery consistency check (prevents pollution from wrong classifications)
    4. Gap-aware penalties
    """
    log = logger_ or logger
    
    if not tracklets:
        log.info("stitch_tracklets: no tracklets.")
        return
    
    if num_identities <= 1:
        for t in tracklets:
            t.stitched_id = 0
        log.info("stitch_tracklets: num_identities=1 -> all stitched_id = 0")
        return
    
    # -------------------------------
    # 1) Extract features for all tracklets
    # -------------------------------
    head_vecs: Dict[int, np.ndarray] = {}
    tail_vecs: Dict[int, np.ndarray] = {}
    mean_vecs: Dict[int, np.ndarray] = {}
    tracklet_lengths: Dict[int, int] = {}
    start_timestamps: Dict[int, datetime] = {}
    end_timestamps: Dict[int, datetime] = {}
    valid_indices: List[int] = []
    
    for i, t in enumerate(tracklets):
        if t.invalid_flag:
            continue
        
        if t.feature_path is None or not t.feature_path.exists():
            log.warning(f"Tracklet {t.track_id} has no valid feature_path; mark invalid.")
            t.invalid_flag = True
            continue
        
        try:
            feats, frame_ids, _, _ = load_embedding(t.feature_path)
        except Exception as e:
            log.error(f"Failed to load embedding from {t.feature_path} for tracklet {i}")
            t.invalid_flag = True
            continue
        
        feats = np.asarray(feats)
        if feats.size == 0:
            log.warning(f"Tracklet {i} has empty features; mark invalid.")
            t.invalid_flag = True
            continue
        
        head, tail = _head_tail_proto(feats, head_k=head_k, tail_k=tail_k)
        mean = _l2norm(feats.mean(axis=0))
        
        head_vecs[i] = head
        tail_vecs[i] = tail
        mean_vecs[i] = mean
        tracklet_lengths[i] = len(feats)
        
        if t.start_timestamp is None or t.end_timestamp is None:
            raise ValueError(f"Tracklet {i} has no start_timestamp/end_timestamp")
        
        start_timestamps[i] = t.start_timestamp
        end_timestamps[i] = t.end_timestamp
        valid_indices.append(i)
    
    if not valid_indices:
        sid = 0
        for t in tracklets:
            t.stitched_id = sid
            sid += 1
        log.warning("All tracklets invalid; assigned unique stitched_id.")
        return
    
    # -------------------------------
    # 2) Sort by temporal order
    # -------------------------------
    order = sorted(valid_indices, key=lambda i: start_timestamps[i])
    log.info(
        f"Stitching: N_valid={len(valid_indices)}, num_identities={num_identities}, "
        f"max_gap={max_gap_frames}, max_identity_gap={max_identity_gap_frames}"
    )
    
    # -------------------------------
    # 3) Identity management with helper functions
    # -------------------------------
    identities: List[Dict] = []
    next_id = 0
    assigned_ids: Dict[int, int] = {}
    processed: List[int] = []
    
    fps = 25
    
    def _has_temporal_overlap(idx: int, identity_idx: int) -> bool:
        """Check if tracklet idx overlaps with ANY tracklet in the identity"""
        identity = identities[identity_idx]
        sf_i = start_timestamps[idx]
        ef_i = end_timestamps[idx]
        
        for prev_idx in identity["last_indices"]:
            sf_prev = start_timestamps[prev_idx]
            ef_prev = end_timestamps[prev_idx]
            # Overlap if NOT (i ends before prev starts OR i starts after prev ends)
            if not (ef_i <= sf_prev or sf_i >= ef_prev):
                return True
        return False
    
    def _min_gap_to_identity(idx: int, identity_idx: int) -> float:
        """Find minimum temporal gap (in frames) from tracklet idx to any tracklet in identity"""
        identity = identities[identity_idx]
        sf_i = start_timestamps[idx]
        ef_i = end_timestamps[idx]
        
        min_gap = float('inf')
        for prev_idx in identity["last_indices"]:
            sf_prev = start_timestamps[prev_idx]
            ef_prev = end_timestamps[prev_idx]
            
            # Gap = time between the two tracklets
            if ef_i <= sf_prev:  # i ends before prev starts
                gap = int((sf_prev - ef_i).total_seconds() * fps)
            elif sf_i >= ef_prev:  # i starts after prev ends
                gap = int((sf_i - ef_prev).total_seconds() * fps)
            else:  # Overlap
                return -1  # Negative indicates overlap
            
            min_gap = min(min_gap, gap)
        
        return min_gap if min_gap != float('inf') else -1
    
    def _gallery_consistency(idx: int, identity_idx: int) -> float:
        """Check how consistent tracklet idx is with identity's gallery"""
        identity = identities[identity_idx]
        gallery = identity.get("gallery", [])
        
        if not gallery:
            return 0.0
        
        mean_vec = mean_vecs[idx]
        sims = [float(np.dot(mean_vec, g)) for g in gallery]
        
        # Use median to be robust to outliers
        return float(np.median(sims))
    
    def _create_new_identity(idx: int) -> int:
        nonlocal next_id
        new_id = next_id
        identities.append({
            "id": new_id,
            "gallery": [mean_vecs[idx]],
            "last_indices": [idx],
            "last_end_time": end_timestamps[idx],
        })
        assigned_ids[idx] = new_id
        tracklets[idx].stitched_id = new_id
        next_id += 1
        log.debug(f"Created identity {new_id} with tracklet {idx}")
        return new_id
    
    def _add_to_identity(idx: int, identity_idx: int) -> bool:
        """Add tracklet to identity with safety checks. Returns True if successful."""
        identity = identities[identity_idx]
        
        # Safety check: verify no overlap
        if _has_temporal_overlap(idx, identity_idx):
            log.error(f"SAFETY CHECK FAILED: tracklet {idx} overlaps with identity {identity['id']}")
            return False
        
        # Safety check: verify gap not too large
        min_gap = _min_gap_to_identity(idx, identity_idx)
        if min_gap > max_identity_gap_frames:
            log.debug(f"Gap too large ({min_gap} frames) for tracklet {idx} to identity {identity['id']}")
            return False
        
        # Safety check: verify gallery consistency
        consistency = _gallery_consistency(idx, identity_idx)
        if len(identity["gallery"]) >= 3 and consistency < min_gallery_consistency:
            log.debug(
                f"Gallery consistency too low ({consistency:.3f}) for tracklet {idx} "
                f"to identity {identity['id']}"
            )
            return False
        
        # All checks passed - add to identity
        identity["gallery"].append(mean_vecs[idx])
        identity["last_indices"].append(idx)
        identity["last_end_time"] = max(identity["last_end_time"], end_timestamps[idx])
        
        if len(identity["gallery"]) > gallery_k:
            identity["gallery"] = identity["gallery"][-gallery_k:]
            identity["last_indices"] = identity["last_indices"][-gallery_k:]
        
        assigned_ids[idx] = identity["id"]
        tracklets[idx].stitched_id = identity["id"]
        return True
    
    def _get_dynamic_weights(feat_length: int) -> Tuple[float, float]:
        """Dynamic weights based on tracklet length"""
        if feat_length < short_tracklet_th:
            return 0.8, 0.2  # Short: trust local more
        elif feat_length > long_tracklet_th:
            return 0.4, 0.6  # Long: trust gallery more
        else:
            ratio = (feat_length - short_tracklet_th) / (long_tracklet_th - short_tracklet_th)
            w_local_dynamic = 0.8 - (0.4 * ratio)
            w_gallery_dynamic = 0.2 + (0.4 * ratio)
            return w_local_dynamic, w_gallery_dynamic
    
    # -------------------------------
    # 4) Process tracklets in temporal order
    # -------------------------------
    for i in order:
        sf_i = start_timestamps[i]
        ef_i = end_timestamps[i]
        head_i = head_vecs[i]
        mean_i = mean_vecs[i]
        feat_len_i = tracklet_lengths[i]
        
        w_local_dynamic, w_gallery_dynamic = _get_dynamic_weights(feat_len_i)
        
        if not identities:
            _create_new_identity(i)
            processed.append(i)
            continue
        
        # -------------------------------
        # a) Local matching: head(i) vs tail(prev_tracks)
        # -------------------------------
        best_local_idx = None
        best_local_sim = -1.0
        best_local_identity_idx = None
        best_local_gap = float('inf')
        
        for j in reversed(processed):
            ef_j = end_timestamps[j]
            gap_frames = int((sf_i - ef_j).total_seconds() * fps)
            
            # Skip overlaps
            if gap_frames < 0:
                continue
            
            # Stop if gap too large
            if gap_frames > max_gap_frames:
                break
            
            # Compute local similarity with gap penalty
            gap_penalty = 1.0 - (gap_frames / max_gap_frames) * 0.3  # 0-30% penalty
            local_sim = float(np.dot(tail_vecs[j], head_i)) * gap_penalty
            
            if local_sim > best_local_sim:
                best_local_sim = local_sim
                best_local_idx = j
                best_local_gap = gap_frames
                # Find which identity j belongs to
                for idx, ident in enumerate(identities):
                    if assigned_ids.get(j) == ident["id"]:
                        best_local_identity_idx = idx
                        break
        
        # -------------------------------
        # b) Gallery matching with constraints
        # -------------------------------
        gallery_scores: List[float] = []
        valid_for_gallery: List[bool] = []
        
        for idx, identity in enumerate(identities):
            # Check temporal constraints first
            has_overlap = _has_temporal_overlap(i, idx)
            min_gap = _min_gap_to_identity(i, idx)
            
            if has_overlap or min_gap > max_identity_gap_frames:
                gallery_scores.append(-1e6)
                valid_for_gallery.append(False)
                continue
            
            # Compute gallery similarity
            gallery = identity["gallery"]
            if not gallery:
                gallery_scores.append(0.0)
                valid_for_gallery.append(True)
                continue
            
            # Weighted by recency
            sims = [float(np.dot(mean_i, g)) for g in gallery]
            n = len(sims)
            weights = np.exp(np.linspace(-0.5, 0, n))
            weights /= weights.sum()
            weighted_sim = float(np.dot(sims, weights))
            
            # Apply gap penalty for gallery too
            if min_gap > 0:
                gap_penalty = 1.0 - (min_gap / max_identity_gap_frames) * 0.4  # 0-40% penalty
                weighted_sim *= gap_penalty
            
            gallery_scores.append(weighted_sim)
            valid_for_gallery.append(True)
        
        # -------------------------------
        # c) Combined scoring
        # -------------------------------
        combined_scores: List[float] = []
        
        for idx in range(len(identities)):
            if not valid_for_gallery[idx]:
                combined_scores.append(-1e6)
                continue
            
            local_contrib = 0.0
            if best_local_identity_idx == idx and best_local_sim > 0:
                local_contrib = best_local_sim
            
            gallery_contrib = gallery_scores[idx]
            combined = w_local_dynamic * local_contrib + w_gallery_dynamic * gallery_contrib
            combined_scores.append(combined)
        
        # Find best valid identity
        valid_candidates = [k for k in range(len(identities)) if valid_for_gallery[k]]
        
        if not valid_candidates:
            # No valid candidates - create new identity
            _create_new_identity(i)
            processed.append(i)
            log.debug(f"Tracklet {i}: no valid candidates, created new identity")
            continue
        
        best_identity_idx = max(valid_candidates, key=lambda k: combined_scores[k])
        best_score = combined_scores[best_identity_idx]
        
        # -------------------------------
        # Decision logic
        # -------------------------------
        decision_made = False
        
        # Strong local match (with all safety checks in _add_to_identity)
        local_th_adjusted = local_sim_th * (1.2 if feat_len_i < short_tracklet_th else 1.0)
        
        if best_local_sim >= local_th_adjusted and best_local_identity_idx is not None:
            if _add_to_identity(i, best_local_identity_idx):
                log.debug(
                    f"Tracklet {i} -> identity {identities[best_local_identity_idx]['id']} "
                    f"via local (sim={best_local_sim:.3f}, gap={best_local_gap})"
                )
                decision_made = True
        
        # Combined score threshold
        if not decision_made and best_score >= (w_local_dynamic * local_sim_th + w_gallery_dynamic * gallery_sim_th):
            if _add_to_identity(i, best_identity_idx):
                log.debug(
                    f"Tracklet {i} -> identity {identities[best_identity_idx]['id']} "
                    f"via combined (score={best_score:.3f})"
                )
                decision_made = True
        
        # Create new identity if capacity allows
        if not decision_made and len(identities) < num_identities:
            _create_new_identity(i)
            log.debug(f"Tracklet {i}: created new identity (score={best_score:.3f} below threshold)")
            decision_made = True
        
        # Forced assignment (last resort)
        if not decision_made:
            if _add_to_identity(i, best_identity_idx):
                log.debug(
                    f"Tracklet {i} -> identity {identities[best_identity_idx]['id']} "
                    f"(forced, max identities reached)"
                )
            else:
                # Even forced assignment failed safety checks - create overflow identity
                overflow_id = _create_new_identity(i)
                log.warning(
                    f"Tracklet {i}: forced assignment failed safety checks, "
                    f"created overflow identity {overflow_id}"
                )
        
        processed.append(i)
    
    # -------------------------------
    # 5) Post-processing: Validate and log
    # -------------------------------
    for identity in identities:
        timeline = [(start_timestamps[idx], end_timestamps[idx], idx) 
                    for idx in identity["last_indices"]]
        timeline.sort()
        
        # Check for overlaps in final result
        for k in range(len(timeline) - 1):
            if timeline[k][1] > timeline[k + 1][0]:
                log.error(
                    f"FINAL OVERLAP in identity {identity['id']}: "
                    f"tracklet {timeline[k][2]} ends at {timeline[k][1]}, "
                    f"tracklet {timeline[k+1][2]} starts at {timeline[k+1][0]}"
                )
    
    # Handle invalid tracklets
    for idx, t in enumerate(tracklets):
        if idx not in valid_indices:
            t.stitched_id = next_id
            next_id += 1
    
    num_final_ids = len(set(assigned_ids.values()))
    log.info(
        f"Stitching complete: {len(valid_indices)} tracklets -> {num_final_ids} identities "
        f"(IDs: {sorted(set(assigned_ids.values()))})"
    )
    
    # Update track_id to stitched_id
    for t in tracklets:
        t.track_id = t.stitched_id



def stitch_tracklets_bidirectional_gallery_strict(
    tracklets: List[Tracklet],
    num_identities: int = 4,
    max_gap_frames: int = 600,  # 24 seconds for local matching
    max_identity_gap_frames: int = 45000,  # 30 minutes max gap within identity
    local_sim_th: float = 0.5,
    gallery_sim_th: float = 0.45,
    head_k: int = 5,
    tail_k: int = 5,
    gallery_k: int = 10,
    w_local: float = 0.6,
    w_gallery: float = 0.4,
    short_tracklet_th: int = 50,
    long_tracklet_th: int = 100,
    min_gallery_consistency: float = 0.35,
    logger_: Optional[logging.Logger] = None,
) -> None:
    """
    Strict stitching that enforces hard temporal constraints.
    
    Key improvements over robust:
    1. Validates against ALL tracklets in identity (not just last_end_time)
    2. Prevents any temporal overlap within an identity
    3. Enforces maximum gap between any tracklet in identity
    4. Better logging for debugging
    """
    log = logger_ or logger
    
    if not tracklets:
        log.info("stitch_tracklets: no tracklets.")
        return
    
    if num_identities <= 1:
        for t in tracklets:
            t.stitched_id = 0
        log.info("stitch_tracklets: num_identities=1 -> all stitched_id = 0")
        return
    
    # -------------------------------
    # 1) Extract features and timestamps
    # -------------------------------
    head_vecs: Dict[int, np.ndarray] = {}
    tail_vecs: Dict[int, np.ndarray] = {}
    mean_vecs: Dict[int, np.ndarray] = {}
    tracklet_lengths: Dict[int, int] = {}
    start_timestamps: Dict[int, datetime] = {}
    end_timestamps: Dict[int, datetime] = {}
    valid_indices: List[int] = []
    
    for i, t in enumerate(tracklets):
        if t.invalid_flag:
            continue
        
        if t.feature_path is None or not t.feature_path.exists():
            log.warning(f"Tracklet {t.track_id} has no valid feature_path; mark invalid.")
            t.invalid_flag = True
            continue
        
        try:
            feats, frame_ids, _, _ = load_embedding(t.feature_path)
        except Exception as e:
            log.error(f"Failed to load embedding from {t.feature_path} for tracklet {i}")
            t.invalid_flag = True
            continue
        
        feats = np.asarray(feats)
        if feats.size == 0:
            log.warning(f"Tracklet {i} has empty features; mark invalid.")
            t.invalid_flag = True
            continue
        
        head, tail = _head_tail_proto(feats, head_k=head_k, tail_k=tail_k)
        mean = _l2norm(feats.mean(axis=0))
        
        head_vecs[i] = head
        tail_vecs[i] = tail
        mean_vecs[i] = mean
        tracklet_lengths[i] = len(feats)
        
        if t.start_timestamp is None or t.end_timestamp is None:
            raise ValueError(f"Tracklet {i} has no start_timestamp/end_timestamp")
        
        start_timestamps[i] = t.start_timestamp
        end_timestamps[i] = t.end_timestamp
        valid_indices.append(i)
    
    if not valid_indices:
        sid = 0
        for t in tracklets:
            t.stitched_id = sid
            sid += 1
        log.warning("All tracklets invalid; assigned unique stitched_id.")
        return
    
    # -------------------------------
    # 2) Sort by temporal order
    # -------------------------------
    order = sorted(valid_indices, key=lambda i: start_timestamps[i])
    log.info(
        f"Strict stitching: N_valid={len(valid_indices)}, num_identities={num_identities}, "
        f"max_gap={max_gap_frames}, max_identity_gap={max_identity_gap_frames}"
    )
    
    # -------------------------------
    # 3) Identity management
    # -------------------------------
    identities: List[Dict] = []
    next_id = 0
    assigned_ids: Dict[int, int] = {}
    processed: List[int] = []
    
    fps = 25
    
    def _validate_temporal_compatibility(idx: int, identity_idx: int) -> Tuple[bool, str]:
        """
        Strict validation: check if tracklet idx is temporally compatible with ALL 
        tracklets in the identity.
        
        Returns: (is_valid, reason)
        """
        identity = identities[identity_idx]
        sf_i = start_timestamps[idx]
        ef_i = end_timestamps[idx]
        
        for prev_idx in identity["tracklet_indices"]:
            sf_prev = start_timestamps[prev_idx]
            ef_prev = end_timestamps[prev_idx]
            
            # Check for overlap
            if not (ef_i <= sf_prev or sf_i >= ef_prev):
                return False, f"overlaps with tracklet {prev_idx} [{sf_prev} - {ef_prev}]"
            
            # Check gap size
            if ef_i <= sf_prev:
                gap_frames = int((sf_prev - ef_i).total_seconds() * fps)
            else:
                gap_frames = int((sf_i - ef_prev).total_seconds() * fps)
            
            if gap_frames > max_identity_gap_frames:
                return False, f"gap too large ({gap_frames} frames) to tracklet {prev_idx}"
        
        return True, "valid"
    
    def _compute_gallery_similarity(idx: int, identity_idx: int) -> float:
        """Compute similarity to identity's gallery"""
        identity = identities[identity_idx]
        gallery = identity.get("gallery", [])
        
        if not gallery:
            return 0.0
        
        mean_vec = mean_vecs[idx]
        sims = [float(np.dot(mean_vec, g)) for g in gallery]
        
        # Weighted by recency
        n = len(sims)
        weights = np.exp(np.linspace(-0.5, 0, n))
        weights /= weights.sum()
        weighted_sim = float(np.dot(sims, weights))
        
        return weighted_sim
    
    def _get_dynamic_weights(feat_length: int) -> Tuple[float, float]:
        """Dynamic weights based on tracklet length"""
        if feat_length < short_tracklet_th:
            return 0.8, 0.2
        elif feat_length > long_tracklet_th:
            return 0.4, 0.6
        else:
            ratio = (feat_length - short_tracklet_th) / (long_tracklet_th - short_tracklet_th)
            w_local_dynamic = 0.8 - (0.4 * ratio)
            w_gallery_dynamic = 0.2 + (0.4 * ratio)
            return w_local_dynamic, w_gallery_dynamic
    
    def _create_new_identity(idx: int) -> int:
        nonlocal next_id
        new_id = next_id
        identities.append({
            "id": new_id,
            "gallery": [mean_vecs[idx]],
            "tracklet_indices": [idx],  # Track ALL indices
        })
        assigned_ids[idx] = new_id
        tracklets[idx].stitched_id = new_id
        next_id += 1
        log.debug(f"Created identity {new_id} with tracklet {idx} [{start_timestamps[idx]} - {end_timestamps[idx]}]")
        return new_id
    
    def _add_to_identity(idx: int, identity_idx: int, reason: str = "") -> bool:
        """Add tracklet to identity with strict validation"""
        # Strict validation
        is_valid, validation_msg = _validate_temporal_compatibility(idx, identity_idx)
        
        if not is_valid:
            log.debug(f"Cannot add tracklet {idx} to identity {identities[identity_idx]['id']}: {validation_msg}")
            return False
        
        # Gallery consistency check
        if len(identities[identity_idx]["gallery"]) >= 3:
            consistency = _compute_gallery_similarity(idx, identity_idx)
            if consistency < min_gallery_consistency:
                log.debug(
                    f"Gallery consistency too low ({consistency:.3f}) for tracklet {idx} "
                    f"to identity {identities[identity_idx]['id']}"
                )
                return False
        
        # All checks passed - add to identity
        identity = identities[identity_idx]
        identity["gallery"].append(mean_vecs[idx])
        identity["tracklet_indices"].append(idx)
        
        # Keep gallery size limited
        if len(identity["gallery"]) > gallery_k:
            identity["gallery"] = identity["gallery"][-gallery_k:]
            # Keep ALL tracklet indices for validation, not just gallery
        
        assigned_ids[idx] = identity["id"]
        tracklets[idx].stitched_id = identity["id"]
        
        log.debug(
            f"Added tracklet {idx} [{start_timestamps[idx]} - {end_timestamps[idx]}] "
            f"to identity {identity['id']} {reason}"
        )
        return True
    
    # -------------------------------
    # 4) Process tracklets in temporal order
    # -------------------------------
    for i in order:
        sf_i = start_timestamps[i]
        ef_i = end_timestamps[i]
        head_i = head_vecs[i]
        mean_i = mean_vecs[i]
        feat_len_i = tracklet_lengths[i]
        
        w_local_dynamic, w_gallery_dynamic = _get_dynamic_weights(feat_len_i)
        
        if not identities:
            _create_new_identity(i)
            processed.append(i)
            continue
        
        # -------------------------------
        # a) Local matching: head(i) vs tail(prev_tracks)
        # -------------------------------
        best_local_idx = None
        best_local_sim = -1.0
        best_local_identity_idx = None
        best_local_gap = float('inf')
        
        for j in reversed(processed):
            ef_j = end_timestamps[j]
            gap_frames = int((sf_i - ef_j).total_seconds() * fps)
            
            # Skip overlaps or negative gaps
            if gap_frames < 0:
                continue
            
            # Stop if gap too large for local matching
            if gap_frames > max_gap_frames:
                break
            
            # Compute local similarity with gap penalty
            gap_penalty = 1.0 - (gap_frames / max_gap_frames) * 0.3
            local_sim = float(np.dot(tail_vecs[j], head_i)) * gap_penalty
            
            if local_sim > best_local_sim:
                best_local_sim = local_sim
                best_local_idx = j
                best_local_gap = gap_frames
                # Find which identity j belongs to
                for idx, ident in enumerate(identities):
                    if assigned_ids.get(j) == ident["id"]:
                        best_local_identity_idx = idx
                        break
        
        # -------------------------------
        # b) Gallery matching with temporal pre-filtering
        # -------------------------------
        gallery_scores: List[float] = []
        valid_for_gallery: List[bool] = []
        
        for idx, identity in enumerate(identities):
            # Pre-check temporal compatibility
            is_valid, _ = _validate_temporal_compatibility(i, idx)
            
            if not is_valid:
                gallery_scores.append(-1e6)
                valid_for_gallery.append(False)
                continue
            
            # Compute gallery similarity
            gallery = identity.get("gallery", [])
            if not gallery:
                gallery_scores.append(0.0)
                valid_for_gallery.append(True)
                continue
            
            weighted_sim = _compute_gallery_similarity(i, idx)
            
            gallery_scores.append(weighted_sim)
            valid_for_gallery.append(True)
        
        # -------------------------------
        # c) Combined scoring
        # -------------------------------
        combined_scores: List[float] = []
        
        for idx in range(len(identities)):
            if not valid_for_gallery[idx]:
                combined_scores.append(-1e6)
                continue
            
            local_contrib = 0.0
            if best_local_identity_idx == idx and best_local_sim > 0:
                local_contrib = best_local_sim
            
            gallery_contrib = gallery_scores[idx]
            combined = w_local_dynamic * local_contrib + w_gallery_dynamic * gallery_contrib
            combined_scores.append(combined)
        
        # Find best valid identity
        valid_candidates = [k for k in range(len(identities)) if valid_for_gallery[k]]
        
        if not valid_candidates:
            # No valid candidates - create new identity
            new_id = _create_new_identity(i)
            processed.append(i)
            log.debug(f"Tracklet {i}: no temporally valid candidates, created new identity {new_id}")
            continue
        
        best_identity_idx = max(valid_candidates, key=lambda k: combined_scores[k])
        best_score = combined_scores[best_identity_idx]
        
        # -------------------------------
        # Decision logic with strict validation
        # -------------------------------
        decision_made = False
        
        # Strong local match
        local_th_adjusted = local_sim_th * (1.2 if feat_len_i < short_tracklet_th else 1.0)
        
        if best_local_sim >= local_th_adjusted and best_local_identity_idx is not None:
            if _add_to_identity(i, best_local_identity_idx, f"(local sim={best_local_sim:.3f}, gap={best_local_gap})"):
                decision_made = True
        
        # Combined score threshold
        if not decision_made and best_score >= (w_local_dynamic * local_sim_th + w_gallery_dynamic * gallery_sim_th):
            if _add_to_identity(i, best_identity_idx, f"(combined score={best_score:.3f})"):
                decision_made = True
        
        # Create new identity if capacity allows
        if not decision_made and len(identities) < num_identities:
            new_id = _create_new_identity(i)
            log.debug(f"Tracklet {i}: created new identity {new_id} (score={best_score:.3f} below threshold)")
            decision_made = True
        
        # Forced assignment to best candidate
        if not decision_made:
            if _add_to_identity(i, best_identity_idx, "(forced - max identities reached)"):
                decision_made = True
            else:
                # Even forced assignment failed - create overflow identity
                overflow_id = _create_new_identity(i)
                log.warning(
                    f"Tracklet {i}: ALL identities have temporal conflicts, "
                    f"created overflow identity {overflow_id}"
                )
        
        processed.append(i)
    
    # -------------------------------
    # 5) Final validation and logging
    # -------------------------------
    for identity in identities:
        timeline = [(start_timestamps[idx], end_timestamps[idx], idx) 
                    for idx in identity["tracklet_indices"]]
        timeline.sort()
        
        log.info(f"Identity {identity['id']}: {len(timeline)} tracklets")
        
        # Verify no overlaps
        for k in range(len(timeline) - 1):
            if timeline[k][1] > timeline[k + 1][0]:
                log.error(
                    f"VALIDATION FAILED - Identity {identity['id']} has overlap: "
                    f"tracklet {timeline[k][2]} ends {timeline[k][1]}, "
                    f"tracklet {timeline[k+1][2]} starts {timeline[k+1][0]}"
                )
    
    # Handle invalid tracklets
    for idx, t in enumerate(tracklets):
        if idx not in valid_indices:
            t.stitched_id = next_id
            next_id += 1
    
    num_final_ids = len(set(assigned_ids.values()))
    log.info(
        f"Strict stitching complete: {len(valid_indices)} tracklets -> {num_final_ids} identities "
        f"(IDs: {sorted(set(assigned_ids.values()))})"
    )
    
    # Update track_id to stitched_id
    for t in tracklets:
        t.track_id = t.stitched_id


def stitch_tracklets_temporal_window(
    tracklets: List[Tracklet],
    num_identities: int = 4,
    time_window_size: int = 3600,  # 1 hour in seconds
    max_gap_frames: int = 600,  # 24 seconds for local matching
    local_sim_th: float = 0.5,
    gallery_sim_th: float = 0.45,
    head_k: int = 5,
    tail_k: int = 5,
    gallery_k: int = 10,
    w_local: float = 0.6,
    w_gallery: float = 0.4,
    short_tracklet_th: int = 50,
    long_tracklet_th: int = 100,
    min_gallery_consistency: float = 0.35,
    logger_: Optional[logging.Logger] = None,
) -> None:
    """
    Time-window-based stitching with strict temporal constraints.
    
    Algorithm:
    1. Divide entire time range into non-overlapping windows
    2. Process each window:
       - Assign IDs to tracklets that START in this window
       - Ensure no temporal overlap within same ID
       - Use local + gallery matching to link to previous windows
    3. Forward propagate gallery representations
    
    Key advantages:
    - Guarantees no overlaps (checked within each window)
    - Better handling of simultaneous appearances
    - More interpretable (can visualize per window)
    """
    log = logger_ or logger
    
    if not tracklets:
        log.info("stitch_tracklets: no tracklets.")
        return
    
    if num_identities <= 1:
        for t in tracklets:
            t.stitched_id = 0
        log.info("stitch_tracklets: num_identities=1 -> all stitched_id = 0")
        return
    
    # -------------------------------
    # 1) Extract features and timestamps
    # -------------------------------
    head_vecs: Dict[int, np.ndarray] = {}
    tail_vecs: Dict[int, np.ndarray] = {}
    mean_vecs: Dict[int, np.ndarray] = {}
    tracklet_lengths: Dict[int, int] = {}
    start_timestamps: Dict[int, datetime] = {}
    end_timestamps: Dict[int, datetime] = {}
    valid_indices: List[int] = []
    
    for i, t in enumerate(tracklets):
        if t.invalid_flag:
            continue
        
        if t.feature_path is None or not t.feature_path.exists():
            log.warning(f"Tracklet {t.track_id} has no valid feature_path; mark invalid.")
            t.invalid_flag = True
            continue
        
        try:
            feats, frame_ids, _, _ = load_embedding(t.feature_path)
        except Exception as e:
            log.error(f"Failed to load embedding from {t.feature_path} for tracklet {i}")
            t.invalid_flag = True
            continue
        
        feats = np.asarray(feats)
        if feats.size == 0:
            log.warning(f"Tracklet {i} has empty features; mark invalid.")
            t.invalid_flag = True
            continue
        
        head, tail = _head_tail_proto(feats, head_k=head_k, tail_k=tail_k)
        mean = _l2norm(feats.mean(axis=0))
        
        head_vecs[i] = head
        tail_vecs[i] = tail
        mean_vecs[i] = mean
        tracklet_lengths[i] = len(feats)
        
        if t.start_timestamp is None or t.end_timestamp is None:
            raise ValueError(f"Tracklet {i} has no start_timestamp/end_timestamp")
        
        start_timestamps[i] = t.start_timestamp
        end_timestamps[i] = t.end_timestamp
        valid_indices.append(i)
    
    if not valid_indices:
        sid = 0
        for t in tracklets:
            t.stitched_id = sid
            sid += 1
        log.warning("All tracklets invalid; assigned unique stitched_id.")
        return
    
    # -------------------------------
    # 2) Create time windows
    # -------------------------------
    all_starts = [start_timestamps[i] for i in valid_indices]
    all_ends = [end_timestamps[i] for i in valid_indices]
    global_start = min(all_starts)
    global_end = max(all_ends)
    
    log.info(f"Global time range: {global_start} to {global_end}")
    
    # Create time windows
    from datetime import timedelta
    windows = []
    current_time = global_start
    while current_time < global_end:
        window_end = current_time + timedelta(seconds=time_window_size)
        windows.append((current_time, window_end))
        current_time = window_end
    
    log.info(f"Created {len(windows)} time windows of size {time_window_size}s")
    
    # -------------------------------
    # 3) Identity state management
    # -------------------------------
    identities: List[Dict] = []
    next_id = 0
    assigned_ids: Dict[int, int] = {}
    fps = 25
    
    def _get_dynamic_weights(feat_length: int) -> Tuple[float, float]:
        """Dynamic weights based on tracklet length"""
        if feat_length < short_tracklet_th:
            return 0.8, 0.2
        elif feat_length > long_tracklet_th:
            return 0.4, 0.6
        else:
            ratio = (feat_length - short_tracklet_th) / (long_tracklet_th - short_tracklet_th)
            w_local_dynamic = 0.8 - (0.4 * ratio)
            w_gallery_dynamic = 0.2 + (0.4 * ratio)
            return w_local_dynamic, w_gallery_dynamic
    
    def _create_new_identity(idx: int) -> int:
        """Create a new identity"""
        nonlocal next_id
        new_id = next_id
        identities.append({
            "id": new_id,
            "gallery": [mean_vecs[idx]],
            "tracklet_indices": [idx],
            "last_end_time": end_timestamps[idx],
        })
        assigned_ids[idx] = new_id
        tracklets[idx].stitched_id = new_id
        next_id += 1
        log.debug(f"Created identity {new_id} with tracklet {idx}")
        return new_id
    
    def _get_active_identities_in_window(window_start: datetime, window_end: datetime) -> List[int]:
        """Get identities that have tracklets active (overlapping) in this window"""
        active_ids = set()
        for identity_idx, identity in enumerate(identities):
            for track_idx in identity["tracklet_indices"]:
                track_start = start_timestamps[track_idx]
                track_end = end_timestamps[track_idx]
                # Check if tracklet overlaps with window
                if not (track_end <= window_start or track_start >= window_end):
                    active_ids.add(identity_idx)
                    break
        return list(active_ids)
    
    def _check_temporal_conflict(idx: int, identity_idx: int) -> bool:
        """Check if tracklet idx conflicts with ANY tracklet in identity"""
        identity = identities[identity_idx]
        sf_i = start_timestamps[idx]
        ef_i = end_timestamps[idx]
        
        for prev_idx in identity["tracklet_indices"]:
            sf_prev = start_timestamps[prev_idx]
            ef_prev = end_timestamps[prev_idx]
            # Overlap if NOT (i ends before prev starts OR i starts after prev ends)
            if not (ef_i <= sf_prev or sf_i >= ef_prev):
                return True  # Conflict!
        return False
    
    def _compute_local_score(idx: int, identity_idx: int) -> float:
        """Compute local matching score (head of idx vs tail of recent tracklets in identity)"""
        identity = identities[identity_idx]
        sf_i = start_timestamps[idx]
        head_i = head_vecs[idx]
        
        best_sim = 0.0
        
        # Find most recent tracklet in this identity
        for track_idx in reversed(identity["tracklet_indices"]):
            ef_track = end_timestamps[track_idx]
            gap_frames = int((sf_i - ef_track).total_seconds() * fps)
            
            if gap_frames < 0:  # Overlap
                continue
            
            if gap_frames > max_gap_frames:  # Too far
                continue
            
            # Compute similarity with gap penalty
            gap_penalty = 1.0 - (gap_frames / max_gap_frames) * 0.3
            local_sim = float(np.dot(tail_vecs[track_idx], head_i)) * gap_penalty
            
            if local_sim > best_sim:
                best_sim = local_sim
                break  # Found the most recent one
        
        return best_sim
    
    def _compute_gallery_score(idx: int, identity_idx: int) -> float:
        """Compute gallery matching score"""
        identity = identities[identity_idx]
        gallery = identity.get("gallery", [])
        
        if not gallery:
            return 0.0
        
        mean_vec = mean_vecs[idx]
        sims = [float(np.dot(mean_vec, g)) for g in gallery]
        
        # Weighted by recency
        n = len(sims)
        weights = np.exp(np.linspace(-0.5, 0, n))
        weights /= weights.sum()
        weighted_sim = float(np.dot(sims, weights))
        
        return weighted_sim
    
    def _add_to_identity(idx: int, identity_idx: int) -> bool:
        """Add tracklet to identity with validation"""
        # Check for conflicts
        if _check_temporal_conflict(idx, identity_idx):
            log.debug(f"Tracklet {idx} conflicts with identity {identities[identity_idx]['id']}")
            return False
        
        # Gallery consistency check
        if len(identities[identity_idx]["gallery"]) >= 3:
            consistency = _compute_gallery_score(idx, identity_idx)
            if consistency < min_gallery_consistency:
                log.debug(
                    f"Gallery consistency too low ({consistency:.3f}) for tracklet {idx} "
                    f"to identity {identities[identity_idx]['id']}"
                )
                return False
        
        # All checks passed - add
        identity = identities[identity_idx]
        identity["gallery"].append(mean_vecs[idx])
        identity["tracklet_indices"].append(idx)
        identity["last_end_time"] = max(identity["last_end_time"], end_timestamps[idx])
        
        # Keep gallery limited
        if len(identity["gallery"]) > gallery_k:
            identity["gallery"] = identity["gallery"][-gallery_k:]
        
        assigned_ids[idx] = identity["id"]
        tracklets[idx].stitched_id = identity["id"]
        
        log.debug(
            f"Added tracklet {idx} [{start_timestamps[idx]} - {end_timestamps[idx]}] "
            f"to identity {identity['id']}"
        )
        return True
    
    # -------------------------------
    # 4) Process each time window
    # -------------------------------
    for window_idx, (window_start, window_end) in enumerate(windows):
        log.info(f"\n{'='*60}")
        log.info(f"Processing window {window_idx+1}/{len(windows)}: {window_start} - {window_end}")
        
        # Get tracklets that START in this window
        tracklets_in_window = [
            i for i in valid_indices
            if window_start <= start_timestamps[i] < window_end
        ]
        
        # Sort by start time within window
        tracklets_in_window.sort(key=lambda i: start_timestamps[i])
        
        log.info(f"Found {len(tracklets_in_window)} tracklets starting in this window")
        
        if not tracklets_in_window:
            continue
        
        # Get identities that are active in this window
        active_identity_indices = _get_active_identities_in_window(window_start, window_end)
        log.info(f"Active identities in window: {[identities[i]['id'] for i in active_identity_indices]}")
        
        # Process each tracklet in this window
        for idx in tracklets_in_window:
            sf_i = start_timestamps[idx]
            ef_i = end_timestamps[idx]
            feat_len_i = tracklet_lengths[idx]
            
            w_local_dynamic, w_gallery_dynamic = _get_dynamic_weights(feat_len_i)
            
            # First tracklet ever
            if not identities:
                _create_new_identity(idx)
                continue
            
            # -------------------------------
            # a) Identify candidate identities
            # -------------------------------
            # Candidates are identities that:
            # 1. Are NOT currently active in this window (to prevent overlap)
            # 2. OR are active but end before this tracklet starts
            
            candidate_identity_indices = []
            for identity_idx in range(len(identities)):
                # Check if this identity has any tracklet overlapping with current tracklet
                if not _check_temporal_conflict(idx, identity_idx):
                    candidate_identity_indices.append(identity_idx)
            
            log.debug(
                f"Tracklet {idx} ({sf_i} - {ef_i}): "
                f"{len(candidate_identity_indices)} candidates out of {len(identities)} identities"
            )
            
            if not candidate_identity_indices:
                # No compatible identities - create new one
                new_id = _create_new_identity(idx)
                log.debug(f"Tracklet {idx}: no compatible identities, created new ID {new_id}")
                continue
            
            # -------------------------------
            # b) Score each candidate
            # -------------------------------
            scores = []
            for identity_idx in candidate_identity_indices:
                local_score = _compute_local_score(idx, identity_idx)
                gallery_score = _compute_gallery_score(idx, identity_idx)
                
                combined = w_local_dynamic * local_score + w_gallery_dynamic * gallery_score
                scores.append((identity_idx, combined, local_score, gallery_score))
            
            # Sort by combined score
            scores.sort(key=lambda x: x[1], reverse=True)
            best_identity_idx, best_score, best_local, best_gallery = scores[0]
            
            log.debug(
                f"Tracklet {idx}: best candidate = identity {identities[best_identity_idx]['id']}, "
                f"score={best_score:.3f} (local={best_local:.3f}, gallery={best_gallery:.3f})"
            )
            
            # -------------------------------
            # c) Decision logic
            # -------------------------------
            decision_made = False
            
            # Threshold for assignment
            threshold = w_local_dynamic * local_sim_th + w_gallery_dynamic * gallery_sim_th
            
            # Strong local match
            if best_local >= local_sim_th * (1.2 if feat_len_i < short_tracklet_th else 1.0):
                if _add_to_identity(idx, best_identity_idx):
                    log.debug(f"Tracklet {idx} -> identity {identities[best_identity_idx]['id']} (strong local)")
                    decision_made = True
            
            # Combined score threshold
            if not decision_made and best_score >= threshold:
                if _add_to_identity(idx, best_identity_idx):
                    log.debug(f"Tracklet {idx} -> identity {identities[best_identity_idx]['id']} (combined)")
                    decision_made = True
            
            # Create new identity if capacity allows
            if not decision_made and len(identities) < num_identities:
                new_id = _create_new_identity(idx)
                log.debug(f"Tracklet {idx}: created new identity {new_id} (score below threshold)")
                decision_made = True
            
            # Forced assignment
            if not decision_made:
                if _add_to_identity(idx, best_identity_idx):
                    log.debug(f"Tracklet {idx} -> identity {identities[best_identity_idx]['id']} (forced)")
                else:
                    # Even forced fails - overflow identity
                    overflow_id = _create_new_identity(idx)
                    log.warning(f"Tracklet {idx}: created overflow identity {overflow_id}")
    
    # -------------------------------
    # 5) Final validation
    # -------------------------------
    log.info(f"\n{'='*60}")
    log.info("Final validation:")
    
    for identity in identities:
        timeline = [(start_timestamps[idx], end_timestamps[idx], idx) 
                    for idx in identity["tracklet_indices"]]
        timeline.sort()
        
        log.info(f"Identity {identity['id']}: {len(timeline)} tracklets")
        
        # Check for overlaps
        for k in range(len(timeline) - 1):
            if timeline[k][1] > timeline[k + 1][0]:
                log.error(
                    f"OVERLAP DETECTED in identity {identity['id']}: "
                    f"tracklet {timeline[k][2]} ends {timeline[k][1]}, "
                    f"tracklet {timeline[k+1][2]} starts {timeline[k+1][0]}"
                )
    
    # Handle invalid tracklets
    for idx, t in enumerate(tracklets):
        if idx not in valid_indices:
            t.stitched_id = next_id
            next_id += 1
    
    num_final_ids = len(set(assigned_ids.values()))
    log.info(
        f"Temporal window stitching complete: {len(valid_indices)} tracklets -> {num_final_ids} identities"
    )
    
    # Update track_id to stitched_id
    for t in tracklets:
        t.track_id = t.stitched_id