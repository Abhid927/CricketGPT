import json
import re
from pathlib import Path

INDEX_PATH = Path("data/processed/match_index.json")

# Common aliases people type
TEAM_ALIASES = {
    "pak": "Pakistan",
    "nz": "New Zealand",
    "newzealand": "New Zealand",
    "new zealand": "New Zealand",

    "ind": "India",
    "aus": "Australia",
    "eng": "England",
    "sa": "South Africa",
    "rsa": "South Africa",
    "wi": "West Indies",
    "ban": "Bangladesh",
    "sl": "Sri Lanka",
    "afg": "Afghanistan",
    "ire": "Ireland",
    "zim": "Zimbabwe",
    "ned": "Netherlands",
    "sco": "Scotland",
    "uae": "United Arab Emirates",
    "usa": "United States of America",
}

def load_index():
    if not INDEX_PATH.exists():
        return []
    return json.loads(INDEX_PATH.read_text(encoding="utf-8"))

def norm(s: str) -> str:
    s = (s or "").strip().lower()
    # normalize punctuation/hyphens to spaces so "new-zealand" works
    s = re.sub(r"[-_/]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def wants_most_recent(text: str) -> bool:
    t = norm(text)
    return any(k in t for k in ["most recent", "latest", "last match", "recent match", "most recently"])

def wants_oldest(text: str) -> bool:
    t = norm(text)
    return any(k in t for k in ["oldest", "earliest", "first match", "first ever", "initial match"])

def wants_score(text: str) -> bool:
    t = norm(text)
    return any(k in t for k in ["score", "scores", "runs", "target", "chased", "won by", "who won", "winner"])

def wants_top_scorer(text: str) -> bool:
    t = norm(text)
    return any(k in t for k in ["top score", "top-scored", "top scored", "highest scorer", "most runs"])

def wants_top_wickets(text: str) -> bool:
    t = norm(text)
    return any(k in t for k in ["most wickets", "top wickets", "best bowling", "most wkts", "most wickets"])

def _find_team_mentions(user_text: str, canonical_teams: set[str]):
    """
    Returns list of (start_index, canonical_team) in order of appearance in the user's text.
    Uses:
      - exact canonical team names
      - common aliases (pak, nz, etc.)
    """
    t = norm(user_text)

    mentions = []

    # 1) match full canonical team names (word-boundary safe)
    # Sort longer first so "United Arab Emirates" beats "Emirates" etc.
    for team in sorted(canonical_teams, key=lambda x: len(x), reverse=True):
        pat = r"\b" + re.escape(norm(team)) + r"\b"
        m = re.search(pat, t)
        if m:
            mentions.append((m.start(), team))

    # 2) match aliases
    for alias, canonical in TEAM_ALIASES.items():
        if canonical not in canonical_teams:
            continue
        pat = r"\b" + re.escape(norm(alias)) + r"\b"
        m = re.search(pat, t)
        if m:
            mentions.append((m.start(), canonical))

    # order by appearance in the message
    mentions.sort(key=lambda x: x[0])

    # de-duplicate while preserving order
    out = []
    seen = set()
    for _, team in mentions:
        if team in seen:
            continue
        seen.add(team)
        out.append(team)

    return out

def extract_teams(user_text: str, canonical_teams: set[str]):
    """
    Try to pull out 2 teams from the user's message.
    Priority:
      1) direct mentions / aliases by appearance order
      2) parsing "X vs Y" style if still ambiguous
    """
    teams = _find_team_mentions(user_text, canonical_teams)
    if len(teams) >= 2:
        return teams[:2]

    # Fallback: parse "x vs y" or "x v y" or "x versus y"
    t = norm(user_text)
    parts = re.split(r"\bvs\b|\bv\b|\bversus\b", t)
    if len(parts) >= 2:
        left = parts[0].strip()
        right = parts[1].strip()

        # find best matching canonical team for each side
        def best_match(side: str):
            # exact alias first
            if side in TEAM_ALIASES and TEAM_ALIASES[side] in canonical_teams:
                return TEAM_ALIASES[side]
            # direct canonical contained
            for team in sorted(canonical_teams, key=lambda x: len(x), reverse=True):
                if norm(team) in side:
                    return team
            return None

        a = best_match(left)
        b = best_match(right)
        if a and b and a != b:
            return [a, b]

    return teams[:2]  # could be 0/1

def find_matches(index, team_a: str, team_b: str):
    res = []
    for m in index:
        teams = m.get("teams") or []
        if team_a in teams and team_b in teams:
            res.append(m)
    # Assuming index is newest-first
    return res

def format_match_summary(m: dict):
    date = m.get("date", "Unknown date")
    teams = m.get("teams", ["?", "?"])
    winner = m.get("winner", "Unknown")
    innings = m.get("innings") or []

    lines = []
    lines.append(f"{teams[0]} vs {teams[1]} — {date}")
    if innings:
        for inn in innings:
            team = inn.get("team")
            runs = inn.get("runs")
            wkts = inn.get("wickets")
            ovs = inn.get("overs")
            lines.append(f"{team}: {runs}/{wkts} ({ovs} ov)")
    lines.append(f"Winner: {winner}")

    ts = m.get("top_scorer")
    if ts:
        lines.append(f"Top scorer: {ts['name']} ({ts['runs']})")

    tw = m.get("top_wickets")
    if tw:
        lines.append(f"Most wickets: {tw['name']} ({tw['wkts']})")

    return "\n".join(lines)

def answer_if_factual(user_text: str):
    index = load_index()
    if not index:
        return None

    # Build set of canonical teams from index
    canonical_teams = set()
    for m in index:
        for t in (m.get("teams") or []):
            canonical_teams.add(t)

    teams = extract_teams(user_text, canonical_teams)
    if len(teams) < 2:
        return None

    team_a, team_b = teams[0], teams[1]
    matches = find_matches(index, team_a, team_b)
    if not matches:
        return f"I couldn’t find a {team_a} vs {team_b} match in the local Cricsheet index."

    # choose oldest/most recent
    if wants_oldest(user_text):
        m = matches[-1]
    else:
        m = matches[0]

    # If user didn't ask for stats-ish keywords, let LLM handle it
    if not (wants_score(user_text) or wants_top_scorer(user_text) or wants_top_wickets(user_text) or wants_most_recent(user_text)):
        return None

    return format_match_summary(m)
