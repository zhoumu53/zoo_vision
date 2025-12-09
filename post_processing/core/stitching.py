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


def _make_prototypes(
    feats: np.ndarray, head_k: int = 5, tail_k: int = 5
) -> Dict[str, np.ndarray]:
    """
    Compute head / tail / mean prototypes from per-frame features.
    feats: (L, D)
    """
    assert feats.ndim == 2, f"feats must be 2D, got {feats.shape}"
    L = feats.shape[0]
    k_head = min(head_k, L)
    k_tail = min(tail_k, L)

    head = _l2norm(feats[:k_head].mean(axis=0))
    tail = _l2norm(feats[-k_tail:].mean(axis=0))
    mean = _l2norm(feats.mean(axis=0))
    return {"head": head, "tail": tail, "mean": mean}


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))



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


def stitch_tracklets(
    tracklets: List[Tracklet],
    num_identities: int = 2,
    head_k: int = 5,
    tail_k: int = 5,
) -> None:
    """
    Cluster-based stitching using head & tail prototypes.

    - If num_identities <= 1: assign stitched_id = 0 to all tracklets.
    - Else:
        * For each valid tracklet:
            - load features from feature_path
            - compute head & tail prototypes (using up to head_k / tail_k frames)
            - build per-tracklet embedding = concat(head, tail), then L2-normalize
        * Stack embeddings and run KMeans with k = min(num_identities, N_valid)
        * Use cluster label as stitched_id
        * Invalid tracklets get unique stitched_id after the clusters
    """

    if not tracklets:
        print("stitch_tracklets: no tracklets, skipping.")
        return

    # ---------- trivial case: only 1 identity ----------
    if num_identities <= 1:
        for t in tracklets:
            t.stitched_id = 0
        print("stitch_tracklets: num_identities=1 -> all stitched_id = 0")
        return

    emb_list: list[np.ndarray] = []
    idx_list: list[int] = []

    # ---------- build per-tracklet head+tail embeddings ----------
    for i, t in enumerate(tracklets):
        if t.invalid_flag:
            continue

        if t.feature_path is None:
            print(f"Tracklet {i} has no feature_path; mark invalid.")
            t.invalid_flag = True
            continue

        try:
            feats, frame_ids, _ = load_embedding(t.feature_path)
        except Exception as e:
            logger.exception(
                f"Failed to load embedding from {t.feature_path} for tracklet {i}: {e}"
            )
            t.invalid_flag = True
            continue

        if feats is None or np.asarray(feats).size == 0:
            print(f"Tracklet {i} has empty features; mark invalid.")
            t.invalid_flag = True
            continue

        head, tail = _head_tail_proto(feats, head_k=head_k, tail_k=tail_k)
        print(f"Tracklet {i}: head and tail prototypes computed.", head.shape, tail.shape)

        # embed = concat(head, tail) → shape (2D,)
        embed = np.concatenate([head, tail], axis=0)
        embed = _l2norm(embed)

        emb_list.append(embed)
        idx_list.append(i)

    if not idx_list:
        # everything invalid → assign unique IDs just to not break downstream
        sid = 0
        for t in tracklets:
            t.stitched_id = sid
            sid += 1
        print(
            "stitch_tracklets: all tracklets invalid; assigned unique stitched_id."
        )
        return

    X = np.stack(emb_list, axis=0)  # (N_valid, 2D)
    N_valid = X.shape[0]

    # ---------- choose k ----------
    k = min(num_identities, N_valid)
    print(
        f"stitch_tracklets (head+tail clustering): N_valid={N_valid}, num_identities={num_identities}, k={k}"
    )

    # ---------- clustering ----------
    if k <= 1:
        labels = np.zeros(N_valid, dtype=int)
    else:
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
        labels = kmeans.fit_predict(X)

    # ---------- assign stitched_id to valid ----------
    for local_idx, tracklet_idx in enumerate(idx_list):
        lab = int(labels[local_idx])
        tracklets[tracklet_idx].stitched_id = lab

    # ---------- invalid ones get unique IDs after ----------
    next_id = int(labels.max()) + 1 if labels.size > 0 else 0
    for i, t in enumerate(tracklets):
        if i not in idx_list:
            t.stitched_id = next_id
            next_id += 1

    print(
        f"stitch_tracklets: assigned stitched_id via head+tail clustering, "
        f"used IDs 0..{next_id - 1}."
    )


def stitch_tracklets_global_frames(
    tracklets: List[Tracklet],
    num_identities: int = 2,
    head_k: int = 5,
    tail_k: int = 5,
    # 相似度权重 / 阈值：
    w_global: float = 0.7,
    w_local: float = 0.3,
    assign_th: float = 0.6,
    ema_alpha: float = 0.7,
    logger_: Optional[logging.Logger] = None,
) -> None:
    """
    Global Frames + Head/Tail stitching.

    思路：
      1. 对每个 tracklet 计算 head / tail / embed：
         - head: 前 head_k 帧平均
         - tail: 后 tail_k 帧平均
         - embed: (head + tail) / 2 再 L2 norm 作为 identity 表征
      2. 按 start_frame_id 对所有 tracklet 排序（同一 camera + 同一视频）。
      3. 依次处理每个 tracklet：
         - 对每个已有 identity 计算：
             global_sim = cos(embed_i, identity_proto)
             local_sim  = cos(head_i, tail_last_track_of_identity)
             score      = w_global * global_sim + w_local * local_sim
         - 决策：
             若 identities 为空 → 新建 identity 0
             否则选 score 最大的 identity:
               if best_score >= assign_th:
                   assign 给这个 identity
               elif 当前 identity 数 < num_identities:
                   开新 identity
               else:
                   强制 assign 给得分最高的 identity
         - 更新 identity:
             proto = l2norm( ema_alpha * old_proto + (1-ema_alpha) * embed_i )
             last_track_idx = i

    结果：
      - 函数在原地设置 tracklet.stitched_id
      - invalid_flag=True 的 tracklet 会被跳过；最后统一给它们独立 ID
    """

    log = logger_ or logger

    if not tracklets:
        print("stitch_tracklets_global_frames: no tracklets.")
        return

    # trivial: 只有 1 个 identity 的情况，全部设成 0
    if num_identities <= 1:
        for t in tracklets:
            t.stitched_id = 0
        print("global_frames: num_identities=1 -> all stitched_id = 0")
        return

    # -------------------------------
    # 1) 预计算每个 tracklet 的 head / tail / embed
    # -------------------------------
    protos: Dict[int, Dict[str, np.ndarray]] = {}
    valid_indices: List[int] = []

    for i, t in enumerate(tracklets):
        if t.invalid_flag:
            continue

        if t.feature_path is None:
            print(f"[global] Tracklet {i} has no feature_path; mark invalid.")
            t.invalid_flag = True
            continue

        try:
            feats, frame_ids, _ = load_embedding(t.feature_path)
        except Exception as e:
            log.exception(
                f"[global] Failed to load embedding from {t.feature_path} for tracklet {i}: {e}"
            )
            t.invalid_flag = True
            continue

        feats = np.asarray(feats)
        if feats.size == 0:
            print(f"[global] Tracklet {i} has empty features; mark invalid.")
            t.invalid_flag = True
            continue

        head, tail = _head_tail_proto(feats, head_k=head_k, tail_k=tail_k)
        embed = _l2norm((head + tail) / 2.0)

        protos[i] = {
            "head": head,
            "tail": tail,
            "embed": embed,
        }
        valid_indices.append(i)

    if not valid_indices:
        # 全部 invalid，就给每个一个唯一 ID，防止 downstream 崩
        sid = 0
        for t in tracklets:
            t.stitched_id = sid
            sid += 1
        print(
            "stitch_tracklets_global_frames: all tracklets invalid; assigned unique IDs."
        )
        return

    # -------------------------------
    # 2) 按全局 start_frame_id 排序
    # -------------------------------
    def _start_frame(idx: int) -> int:
        t = tracklets[idx]
        if t.start_frame_id is not None:
            return int(t.start_frame_id)
        # fallback：没有 start_frame_id 就当 0
        return 0

    order = sorted(valid_indices, key=_start_frame)

    print(
        f"global_frames: N_valid={len(valid_indices)}, num_identities={num_identities}, "
        f"w_global={w_global}, w_local={w_local}, assign_th={assign_th}, ema_alpha={ema_alpha}"
    )

    # -------------------------------
    # 3) identity 管理结构
    # -------------------------------
    # 每个 identity: {"id": int, "proto": np.ndarray, "last_idx": Optional[int]}
    identities: List[Dict[str, object]] = []
    next_id = 0

    def _assign_new_identity(idx: int, embed_vec: np.ndarray):
        nonlocal next_id
        identities.append(
            {
                "id": next_id,
                "proto": embed_vec.copy(),
                "last_idx": idx,
            }
        )
        tracklets[idx].stitched_id = next_id
        next_id += 1

    # -------------------------------
    # 4) 按时间顺序，逐 tracklet 赋 identity
    # -------------------------------
    for i in order:
        emb_i = protos[i]["embed"]
        head_i = protos[i]["head"]

        if not identities:
            # 第一个 tracklet：新建 identity 0
            _assign_new_identity(i, emb_i)
            continue

        # 计算对每个 identity 的 score
        scores: List[float] = []
        for ident in identities:
            proto = ident["proto"]  # type: ignore
            global_sim = float(np.dot(emb_i, proto))  # cos，因为都 L2 norm 了

            last_idx = ident["last_idx"]  # type: ignore
            if last_idx is not None and last_idx in protos:
                tail_prev = protos[last_idx]["tail"]
                local_sim = float(np.dot(head_i, tail_prev))
            else:
                local_sim = 0.0

            score = w_global * global_sim + w_local * local_sim
            scores.append(score)

        # 找到 score 最大的 identity
        best_idx = int(np.argmax(scores))
        best_score = scores[best_idx]
        best_ident = identities[best_idx]

        # 决策
        if best_score >= assign_th:
            # assign 到 best_ident
            tracklets[i].stitched_id = best_ident["id"]  # type: ignore
            # 更新 identity proto（EMA）
            old_proto = best_ident["proto"]  # type: ignore
            best_ident["proto"] = _l2norm(
                ema_alpha * old_proto + (1.0 - ema_alpha) * emb_i
            )
            best_ident["last_idx"] = i
        elif len(identities) < num_identities:
            # score 不高，但 identity 还没满 -> 开一个新的
            _assign_new_identity(i, emb_i)
        else:
            # identity 已满（例如 2 个），但没有任何一个超过阈值
            # 还是 assign 到 best_ident，防止孤立 ID 爆炸
            tracklets[i].stitched_id = best_ident["id"]  # type: ignore
            old_proto = best_ident["proto"]  # type: ignore
            best_ident["proto"] = _l2norm(
                ema_alpha * old_proto + (1.0 - ema_alpha) * emb_i
            )
            best_ident["last_idx"] = i

    # -------------------------------
    # 5) 对于 invalid 的 tracklet，给单独 ID
    # -------------------------------
    for idx, t in enumerate(tracklets):
        if idx not in valid_indices:
            t.stitched_id = next_id
            next_id += 1

    print(
        f"global_frames: used {len(identities)} identities, "
        f"ids={[ident['id'] for ident in identities]}, "
        f"total stitched_id used={next_id}."
    )