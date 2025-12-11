


from datetime import datetime
from typing import List, Dict, Optional
from collections import defaultdict
import logging
from post_processing.core.tracklet_manager import Tracklet

def _fmt_time(dt: Optional[datetime]) -> str:
    if dt is None:
        return "None"
    return dt.strftime("%H:%M:%S")


def validate_stitched_timelines(
    tracklets: List[Tracklet],
    camera_id: Optional[str] = None,
    logger_: Optional[logging.Logger] = None,
) -> None:
    """
    For each stitched_id, print the stitched chain in temporal order:

        ID 0: file1(start-end) -> file2(start-end) -> ...

    Also checks:
        - that start times are non-decreasing within the chain
        - warns if a track starts before the previous one ends (temporal overlap).
    """
    log = logger_ or logger

    # filter by camera and stitched_id
    filtered: List[Tracklet] = []
    for t in tracklets:
        if t.stitched_id is None:
            continue
        if camera_id is not None and t.camera_id != camera_id:
            continue
        if t.invalid_flag:
            continue
        filtered.append(t)

    if not filtered:
        log.warning("validate_stitched_timelines: no valid tracklets found.")
        return

    chains: Dict[int, List[Tracklet]] = defaultdict(list)
    for t in filtered:
        chains[int(t.stitched_id)].append(t)

    log.info(
        f"validate_stitched_timelines: found {len(chains)} stitched IDs "
        f"(camera={camera_id if camera_id else 'ALL'})."
    )

    for sid, chain in sorted(chains.items(), key=lambda kv: kv[0]):
        # sort chain by start_timestamp
        chain_sorted = sorted(
            chain,
            key=lambda t: t.start_timestamp or datetime.min,
        )

        line_parts = []
        prev_end: Optional[datetime] = None
        has_issue = False

        for t in chain_sorted:
            name = t.track_filename if getattr(t, "track_filename", "") else t.track_id
            st = t.start_timestamp
            et = t.end_timestamp

            part = f"{name}({_fmt_time(st)}-{_fmt_time(et)})"
            line_parts.append(part)

            if prev_end is not None and st is not None:
                if st < prev_end:
                    # temporal overlap
                    overlap_sec = (prev_end - st).total_seconds()
                    log.warning(
                        f"[validate] ID {sid}: temporal overlap: "
                        f"{name} starts {_fmt_time(st)} before previous ended {_fmt_time(prev_end)} "
                        f"(overlap={overlap_sec:.2f}s)"
                    )
                    has_issue = True

            if et is not None:
                prev_end = et

        line = " -> ".join(line_parts)
        if has_issue:
            print(f"[ID {sid}] (ISSUES) {line}")
        else:
            print(f"[ID {sid}] {line}")
