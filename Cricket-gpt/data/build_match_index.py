import os
import glob
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

RAW_DIR = "data/raw/t20_json"
OUT_INDEX = "data/processed/match_index.json"

# Wicket kinds that do NOT count as a bowler wicket (run out isn't credited to bowler, etc.)
NON_BOWLER_WICKETS = {
    "run out",
    "retired hurt",
    "retired out",
    "obstructing the field",
    "hit the ball twice",
    "handled the ball",
    "timed out",
}

def parse_date(info: dict):
    dates = info.get("dates") or []
    if not dates:
        return None
    # Cricsheet dates are usually ["YYYY-MM-DD"]
    try:
        return max(datetime.fromisoformat(d) for d in dates).date().isoformat()
    except Exception:
        return str(dates[0])

def norm_team(name: str) -> str:
    # You can expand this mapping over time if needed
    n = (name or "").strip()
    aliases = {
        "IND": "India",
        "AUS": "Australia",
        "ENG": "England",
        "PAK": "Pakistan",
        "NZ": "New Zealand",
        "SA": "South Africa",
        "WI": "West Indies",
        "SL": "Sri Lanka",
        "BAN": "Bangladesh",
        "AFG": "Afghanistan",
    }
    return aliases.get(n, n)

def is_legal_delivery(delivery: dict) -> bool:
    # Wide does NOT count as a legal ball; no-ball DOES count as a ball faced in many ball-by-ball feeds,
    # but for simplicity we'll treat wides as illegal and everything else as legal.
    extras = delivery.get("extras") or {}
    return not ("wides" in extras)

def compute_match_stats(match_json: dict):
    info = match_json.get("info", {}) if isinstance(match_json, dict) else {}
    teams = [norm_team(t) for t in (info.get("teams") or [])]
    date = parse_date(info)

    winner = None
    outcome = info.get("outcome") or {}
    if isinstance(outcome, dict):
        winner = outcome.get("winner")

    venue = info.get("venue")
    city = info.get("city")
    match_type = info.get("match_type")

    innings = match_json.get("innings") or []

    innings_summaries = []
    batter_runs_total = defaultdict(int)  # across match
    bowler_wickets_total = defaultdict(int)  # across match

    for inn in innings:
        team = norm_team(inn.get("team", "Unknown"))
        runs_total = 0
        wickets_total = 0
        legal_balls = 0

        batter_runs = defaultdict(int)
        bowler_wickets = defaultdict(int)

        overs = inn.get("overs") or []
        for ov in overs:
            deliveries = ov.get("deliveries") or []
            for d in deliveries:
                runs = d.get("runs") or {}
                runs_total += int(runs.get("total", 0) or 0)

                # balls
                if is_legal_delivery(d):
                    legal_balls += 1

                # batter runs
                batter = d.get("batter") or d.get("striker") or "Unknown"
                batter_runs[batter] += int(runs.get("batter", 0) or 0)

                # wickets
                wlist = d.get("wickets") or ([] if not d.get("wicket") else [d.get("wicket")])
                if wlist:
                    for w in wlist:
                        wickets_total += 1
                        kind = (w.get("kind") or "").lower().strip()
                        bowler = d.get("bowler") or "Unknown"
                        if kind and kind not in NON_BOWLER_WICKETS:
                            bowler_wickets[bowler] += 1

        overs_float = round(legal_balls / 6.0, 1)

        # inning top batter/bowler
        top_batter = None
        if batter_runs:
            top_batter = max(batter_runs.items(), key=lambda x: x[1])

        top_bowler = None
        if bowler_wickets:
            top_bowler = max(bowler_wickets.items(), key=lambda x: x[1])

        for k, v in batter_runs.items():
            batter_runs_total[k] += v
        for k, v in bowler_wickets.items():
            bowler_wickets_total[k] += v

        innings_summaries.append({
            "team": team,
            "runs": runs_total,
            "wickets": wickets_total,
            "overs": overs_float,
            "top_batter": {"name": top_batter[0], "runs": top_batter[1]} if top_batter else None,
            "top_bowler": {"name": top_bowler[0], "wkts": top_bowler[1]} if top_bowler else None,
        })

    match_top_scorer = None
    if batter_runs_total:
        n, r = max(batter_runs_total.items(), key=lambda x: x[1])
        match_top_scorer = {"name": n, "runs": r}

    match_top_wickets = None
    if bowler_wickets_total:
        n, w = max(bowler_wickets_total.items(), key=lambda x: x[1])
        match_top_wickets = {"name": n, "wkts": w}

    # build a stable ID (date + teams + match_type_number if exists)
    mt_num = info.get("match_type_number")
    id_parts = [date or "unknown-date", "-vs-".join(teams or ["Unknown"]), str(mt_num or "na")]
    match_id = "__".join(id_parts)

    return {
        "match_id": match_id,
        "date": date,
        "match_type": match_type,
        "teams": teams,
        "winner": winner,
        "venue": venue,
        "city": city,
        "innings": innings_summaries,
        "top_scorer": match_top_scorer,
        "top_wickets": match_top_wickets,
    }

def main():
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.json")))
    if not files:
        raise FileNotFoundError(f"No JSON files found in {RAW_DIR}")

    out = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            stats = compute_match_stats(data)
            stats["source_file"] = os.path.basename(fp)
            out.append(stats)
        except Exception as e:
            print(f"Skipping {fp} due to error: {e}")

    # sort newest first if date exists
    def sort_key(x):
        d = x.get("date")
        return d or "0000-00-00"
    out.sort(key=sort_key, reverse=True)

    Path(os.path.dirname(OUT_INDEX)).mkdir(parents=True, exist_ok=True)
    with open(OUT_INDEX, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote {len(out)} matches to {OUT_INDEX}")

if __name__ == "__main__":
    main()
