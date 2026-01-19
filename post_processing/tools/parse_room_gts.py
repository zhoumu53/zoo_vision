from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import pandas as pd


# ============================================================
# Canonical individuals and social groups
# ============================================================
INDIVIDUALS: Tuple[str, ...] = ("Chandra", "Indi", "Panang", "Farha", "Thai")

SOCIAL_GROUPS: Dict[int, List[str]] = {
    1: ["Chandra", "Indi"],
    2: ["Panang", "Farha", "Zali"],
    3: ["Thai"],
}

# IMPORTANT:
# Keepers may write Farha as "Fahra". Canonical is "Farha".
INDIV_ALIASES: Dict[str, List[str]] = {
    "Chandra": ["chandra"],
    "Indi": ["indi"],
    "Panang": ["panang", "panange"],
    "Farha": ["farha", "fahra"],
    "Zali": ["zali"],
    "Thai": ["thai"],
}

# Shorthands that imply multiple individuals
MULTI_ALIASES: Dict[str, List[str]] = {
    "indis": ["Indi", "Chandra"],
    # You told earlier: "Panangs" => Panang & Farha (NOT Zali unless you confirm)
    "panangs": ["Panang", "Farha"],

}

### typos
MULTI_ALIASES.update({
    "panangs": ["Panang", "Farha"],
    "pananags": ["Panang", "Farha"],  # typo
})

INDIV_ALIASES["Farha"] = ["farha", "fahra"]  # keep both spellings

# ============================================================
# Sandbox room detection (mit/ohne)
# ============================================================

BOTH_RE = re.compile(r"(?i)\b(Sandboxen\s*mit|sandboxen\s*\+|sb\s*mit)\b")
MIT_RE = re.compile(r"(?i)\b(sandbox\s*mit|sandbox\s*\+|sb\s*mit)\b")

# Add 'ohni' as a tolerated misspelling (and optionally 'ohne' variants)
OHNE_RE = re.compile(
    r"(?i)\b("
    r"sandbox\s*ohne|"
    r"sandbox\s*ohni|"     # typo
    r"sandbox\s*-|"
    r"sb\s*ohne|"
    r"sb\s*ohni|"          # typo
    r"waldbox|"
    r"waldbodenbox|"
    r"\bwb\b"
    r")\b"
)


def normalize_typos(text: str) -> str:
    """
    Normalize common keeper typos / variants into canonical tokens
    BEFORE running alias detection and room regex.

    Keep this conservative: only normalize patterns you have observed.
    """
    if not isinstance(text, str) or not text:
        return ""

    t = text

    # --- room typos ---
    # "ohni" -> "ohne"
    t = re.sub(r"(?i)\bohni\b", "ohne", t)

    # --- name typos ---
    # "Pananags" -> "Panangs" (and we already map panangs -> Panang+Farha)
    t = re.sub(r"(?i)\bpananags\b", "panangs", t)

    # "Fahra" -> "Farha" (optional; if your canonical is Farha)
    t = re.sub(r"(?i)\bfahra\b", "farha", t)

    # Normalize Sandbox+ / Sandbox- explicitly
    t = re.sub(r"(?i)sandbox\s*\+", "sandbox mit", t)
    t = re.sub(r"(?i)sandbox\s*-", "sandbox ohne", t)

    return t

def detect_room(text: str) -> Optional[str]:
    """
    Detect sandbox room label:
      - 'mit' / 'ohne' if detectable
      - 'both' if both appear
      - None otherwise
    """
    if not isinstance(text, str) or not text.strip():
        return None
    t = text.lower()
    has_mit = bool(MIT_RE.search(t))
    has_ohne = bool(OHNE_RE.search(t))
    has_both = bool(BOTH_RE.search(t))

    if has_mit and not has_ohne:
        return "mit"
    if has_ohne and not has_mit:
        return "ohne"
    if has_both or (has_mit and has_ohne):
        return "both"
    return None


# ============================================================
# Parsing "names first, then location"
#   Example:
#     "Farha und Panang: Bullenanlage + SB ohne"
#     "Panang& Farha& Zali:SB ohne+ Bullen AA"
# ============================================================

# Matches "HEAD: REST"
HEAD_RE = re.compile(r"^\s*(?P<head>[^:]+?)\s*:\s*(?P<rest>.*)\s*$", flags=re.UNICODE)


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def find_individuals_in_text(text: str) -> List[str]:
    """
    Find individuals mentioned in a string. This is used on the "head" portion first.
    Supports:
      - MULTI_ALIASES (e.g., 'indis', 'panangs')
      - individual aliases (e.g., 'fahra' -> Farha)
    """
    if not isinstance(text, str) or not text.strip():
        return []

    t = norm(text)
    found: Set[str] = set()

    # Multi-alias hits
    for key, members in MULTI_ALIASES.items():
        if re.search(rf"\b{re.escape(key)}\b", t, flags=re.IGNORECASE):
            found.update(members)

    # Individual alias hits
    for indiv, aliases in INDIV_ALIASES.items():
        for a in aliases:
            if re.search(rf"\b{re.escape(a)}\b", t, flags=re.IGNORECASE):
                found.add(indiv)
                break

    # Keep stable ordering
    return [i for i in INDIVIDUALS if i in found]


def split_lines(raw: str) -> List[str]:
    """
    Split keeper text into rows. Support:
      - newline
      - semicolon
      - our aggregation delimiter '||' (must be treated as a line break)
    """
    if not isinstance(raw, str) or not raw.strip():
        return []
    
    raw = normalize_typos(raw)

    # First split on newlines
    chunks = re.split(r"[\r\n]+", raw)

    parts: List[str] = []
    for chunk in chunks:
        # split on our join delimiter (with optional spaces)
        for piece in re.split(r"\s*\|\|\s*", chunk):
            # split on semicolons too
            parts.extend([p.strip() for p in piece.split(";") if p.strip()])

    return parts



def parse_line_names_then_room(line: str) -> Tuple[List[str], Optional[str]]:
    """
    For one keeper line, do:
      1) Parse names from HEAD (before ':') if present; else from full line.
      2) Parse room from REST (after ':') if present; else from full line.
    Returns (individuals, room_detected_or_None_or_conflict).
    """
    if not isinstance(line, str) or not line.strip():
        return [], None

    m = HEAD_RE.match(line)
    if m:
        head = m.group("head")
        rest = m.group("rest")
        inds = find_individuals_in_text(head)
        room = detect_room(rest)
        # If head has no names, fallback to full-line name detection
        if not inds:
            inds = find_individuals_in_text(line)
        # If rest has no room label, fallback to full line
        if room is None:
            room = detect_room(line)
        return inds, room

    # No ":" -> parse from entire line
    inds = find_individuals_in_text(line)
    room = detect_room(line)
    return inds, room


def group_id_for_individual(ind: str) -> Optional[int]:
    for gid, members in SOCIAL_GROUPS.items():
        if ind in members:
            return gid
    return None


def unique_join(values: List[str], sep: str = " | ") -> str:
    seen = set()
    out = []
    for v in values:
        if not isinstance(v, str):
            continue
        vv = v.strip()
        if not vv:
            continue
        if vv not in seen:
            seen.add(vv)
            out.append(vv)
    return sep.join(out)


# ============================================================
# XLSX parsing
# ============================================================
@dataclass
class ParseConfig:
    xlsx_path: str | Path
    sheet_name: str | int = 0
    date_col: str = "date"
    location_col: str = "location"
    dayfirst: bool = True
    output_csv: Optional[str | Path] = None
    output_xlsx: Optional[str | Path] = None


def parse_keeper_xlsx(cfg: ParseConfig) -> pd.DataFrame:
    """
    Output: one row per (date_norm, individual). Always 6 rows per date_norm.

    Keeps ALL original columns (aggregated per date) + adds:
      - individual
      - social_group_id
      - raw_all (all keeper text for that day)
      - raw_individual (snippets assigned to the individual)
      - room (mit/ohne/others) derived from assigned snippets using "names then room" logic
    """
    df = pd.read_excel(cfg.xlsx_path, sheet_name=cfg.sheet_name, dtype=str, engine="openpyxl").fillna("")

    if cfg.date_col not in df.columns:
        raise KeyError(f"Missing date column {cfg.date_col!r}. Found: {list(df.columns)}")
    if cfg.location_col not in df.columns:
        raise KeyError(f"Missing location column {cfg.location_col!r}. Found: {list(df.columns)}")

    # Forward-fill dates
    df[cfg.date_col] = df[cfg.date_col].replace("", pd.NA).ffill()

    dt = pd.to_datetime(df[cfg.date_col], errors="coerce", dayfirst=cfg.dayfirst)
    df["_date_norm"] = dt.dt.date.astype(str)  # NaT -> 'NaT'

    source_cols = list(df.columns)
    out_rows: List[Dict[str, str]] = []

    for date_norm, gdf in df.groupby("_date_norm", dropna=False):
        # Aggregate all columns for the day (information-preserving)
        day_agg: Dict[str, str] = {}
        for c in source_cols:
            # IMPORTANT: keep location rows as separate "lines" so parsing does not mix individuals
            if c == cfg.location_col:
                day_agg[c] = unique_join(gdf[c].tolist(), sep="\n")
            else:
                day_agg[c] = unique_join(gdf[c].tolist(), sep=" || ")

        raw_all = day_agg.get(cfg.location_col, "")


        # Evidence store per individual
        snips_per_ind: Dict[str, List[str]] = {i: [] for i in INDIVIDUALS}
        rooms_per_ind: Dict[str, List[str]] = {i: [] for i in INDIVIDUALS}

        for line in split_lines(raw_all):
            line = normalize_typos(line)
            inds, room = parse_line_names_then_room(line)
            if not inds:
                continue  # cannot attribute safely
            for ind in inds:
                snips_per_ind[ind].append(line)
                if room in ("mit", "ohne"):
                    rooms_per_ind[ind].append(room)
                elif room == "both":
                    rooms_per_ind[ind].append("both")

        for ind in INDIVIDUALS:
            room_votes = rooms_per_ind[ind]
            room_set = set(room_votes)


            if "both" in room_set:
                final_room = "both"
            elif "mit" in room_set and "ohne" in room_set:
                final_room = "both"
            elif "mit" in room_set:
                final_room = "mit"
            elif "ohne" in room_set:
                final_room = "ohne"
            else:
                final_room = "others"

            row = dict(day_agg)
            row["date_norm"] = date_norm
            row["individual"] = ind
            row["social_group_id"] = group_id_for_individual(ind)
            row["raw_all"] = raw_all
            row["raw_individual"] = unique_join(snips_per_ind[ind], sep=" | ")
            row["room"] = final_room

            out_rows.append(row)

    out = pd.DataFrame(out_rows)

    # Reorder columns (key first, keep all)
    preferred = [
        cfg.date_col, "social_group_id", "individual",
        "room", "raw_individual", "raw_all",
    ]
    out = out[preferred]

    out = out.sort_values([cfg.date_col, "social_group_id", "individual"]).reset_index(drop=True)

    if cfg.output_csv:
        out.to_csv(cfg.output_csv, index=False)
    if cfg.output_xlsx:
        out.to_excel(cfg.output_xlsx, index=False, engine="openpyxl")

    return out


# ============================================================
# Minimal validation
# ============================================================
def validate(parsed: pd.DataFrame) -> None:
    expected = len(INDIVIDUALS)
    counts = parsed.groupby("date")["individual"].count()
    bad = counts[counts != expected]
    if not bad.empty:
        print(f"[ERROR] Some dates do not have {expected} individuals:")
        for d, c in bad.items():
            print(f"  - {d}: {c}")
    else:
        print(f"[OK] Every date has {expected} individual rows.")

    inv = parsed[parsed["room"] == "others"]
    print(f"[INFO] other room rows: {len(inv)}")


def validate_counts_by_room_per_date(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    id_col: str = "individual",
    room_col: str = "room",
) -> pd.DataFrame:
    """
    For each unique date, count how many unique individuals are in each room.

    Always outputs columns:
      - date
      - mit
      - ohne
      - others

    Missing combinations are filled with 0.
    """

    # --- Basic checks ---
    for c in (date_col, id_col, room_col):
        if c not in df.columns:
            raise KeyError(f"Missing required column: {c}")

    # Defensive: one row per (date, individual)
    tmp = df[[date_col, id_col, room_col]].drop_duplicates(
        subset=[date_col, id_col],
        keep="last",
    )

    # Force room categories so zeros are kept
    room_categories = ["mit", "ohne", "others", "both"]
    tmp[room_col] = pd.Categorical(tmp[room_col], categories=room_categories)

    # Count unique individuals per (date, room)
    counts = (
        tmp.groupby([date_col, room_col], dropna=False)[id_col]
        .nunique()
        .unstack(fill_value=0)
        .reset_index()
    )

    # Ensure all room columns exist (even if never seen)
    for r in room_categories:
        if r not in counts.columns:
            counts[r] = 0

    # Final column order
    counts = counts[[date_col, "mit", "ohne", "others", "both"]]

    # Print for inspection
    # print("\n[Room counts per date]")
    # print(counts.to_string(index=False))

    return counts


def assert_zero_in_mit_or_ohne(
    counts_df: pd.DataFrame,
    *,
    date_col: str = "date",
    mit_col: str = "mit",
    ohne_col: str = "ohne",
    raise_on_fail: bool = False,
) -> None:
    """
    Assert / warn if for any date:
      - mit == 0 OR
      - ohne == 0

    Expects counts_df to already contain per-date counts
    with columns: date, mit, ohne, others.
    """

    # Sanity checks
    for c in (date_col, mit_col, ohne_col):
        if c not in counts_df.columns:
            raise KeyError(f"Missing required column: {c}")

    # Find problematic dates
    bad = counts_df[
        (counts_df[mit_col] == 0) | (counts_df[ohne_col] == 0)
    ].copy()

    if bad.empty:
        print("[OK] All dates have at least one ID in BOTH 'mit' and 'ohne'.")
        return

    # Print detailed report
    print("\n[WARNING] Dates with zero count in 'mit' or 'ohne':")
    print(bad[[date_col, mit_col, ohne_col, "others"]].to_string(index=False))

    # Optional hard failure
    if raise_on_fail:
        raise AssertionError(
            f"{len(bad)} date(s) have mit==0 or ohne==0. See printed table."
        )
    
# ============================================================
if __name__ == "__main__":
    cfg = ParseConfig(
        xlsx_path="/home/mu/Downloads/keeper_info.xlsx",
        sheet_name=0,
        date_col="date",
        location_col="location",
        dayfirst=True,
        output_csv="/media/mu/zoo_vision/data/elephants_sandbox.csv",
        output_xlsx="/media/mu/zoo_vision/data/elephants_sandbox.xlsx",
    )

    parsed = parse_keeper_xlsx(cfg)
    # print(parsed.head(30).to_string(index=False))
    validate(parsed)

    parsed_df = parsed.rename(columns={"date_norm": "date"})
    counts_df = validate_counts_by_room_per_date(
        parsed_df,
        date_col="date",
        id_col="individual",
        room_col="room",
    )

    # Step 2: assert conditions
    assert_zero_in_mit_or_ohne(
        counts_df,
        date_col="date",
        raise_on_fail=False,  # set True if you want pipeline to stop
    )
    print("\nWrote:", cfg.output_csv, "and", cfg.output_xlsx)
