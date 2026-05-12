# preprocess_cricsheet.py
"""
Creates instruction-style training data from Cricsheet T20 JSON.

Outputs examples like:
<|match|>
Match: A vs B
<|user|> Continue this over: Over 16.3: Bowler to Batter,
<|assistant|> ...

<|match|>
Match: A vs B
<|user|> Summarize the last 3 overs in 2-3 sentences:
Over 17.1: ...
...
<|assistant|> ...

<|match|>
Match: A vs B
<|user|> Explain why this wicket was important:
Over 14.2: ... OUT! ...
<|assistant|> ...

This dramatically improves the "chatbot feel" versus raw ball-by-ball streams.
"""

import os
import glob
import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple

RAW_DIR = "data/raw/t20_json"
OUT_TRAIN = "data/processed/train.txt"
OUT_VAL = "data/processed/val.txt"

VAL_SPLIT = 0.02
SEED = 42
MAX_FILES = None  # set to int (e.g., 200) for quick testing

# If True, we wrap each example with <|match|> markers (recommended)
ADD_MATCH_TOKEN = True
LINE_END = "\n"

# Special tokens (must match what you used when training tokenizer)
MATCH_TOKEN = "<|match|>"
USER = "<|user|>"
ASSISTANT = "<|assistant|>"

# -----------------------
# Commentary templates
# -----------------------
DOT_BALL = [
    "no run.",
    "solid defence, no run.",
    "tight line, can't get it away — dot ball.",
]
ONE_RUN = [
    "worked away for a single.",
    "pushes into the gap and takes one.",
    "nudged for a quick single.",
]
TWO_RUNS = [
    "back for two.",
    "excellent running — two taken.",
    "finds the gap, they come back for a couple.",
]
THREE_RUNS = [
    "they scamper back for three.",
    "good running — three runs taken.",
    "worked into space and they take three.",
]
FOUR = [
    "and that's FOUR! finds the fence.",
    "crunched away — FOUR.",
    "beautifully timed, races to the boundary for FOUR!",
]
SIX = [
    "SIX! launched into the stands.",
    "goes downtown — that's SIX!",
    "clean strike and it's SIX all the way!",
]
WICKET = [
    "OUT! the batter has to go.",
    "WICKET! big moment in the innings.",
    "Gone! that's a huge breakthrough.",
]

# “Assistant” targets for instruction tasks (we generate these labels ourselves)
SUMMARY_STARTERS = [
    "Big momentum swing here. ",
    "That passage tells a clear story. ",
    "A tense little phase. ",
    "Plenty happening in those overs. ",
    "Key moments packed into that stretch. ",
]
SUMMARY_MIDDLES = [
    "The bowling kept it tight with dots and smart lengths, forcing risk. ",
    "There were a couple of boundaries to release pressure, but control stayed with the fielding side. ",
    "Rotating strike was crucial, with ones and twos keeping the scoreboard ticking. ",
    "A wicket (or a near-miss) changed the feel of the innings and raised the stakes. ",
    "The batters tried to find gaps, but the bowler hit good areas consistently. ",
]
SUMMARY_ENDERS = [
    "Overall: pressure, then release, then pressure again.",
    "It set up a nervy finish.",
    "It shifted the momentum toward the bowling side.",
    "It nudged the game toward the batting side.",
    "It left the next over as the big turning point.",
]

WICKET_EXPLAIN_OPENERS = [
    "That wicket is massive. ",
    "Huge moment in the innings. ",
    "That’s a real game-changer. ",
    "Big breakthrough at a crucial time. ",
]
WICKET_EXPLAIN_BODY = [
    "It breaks a partnership and forces a new batter to start against a set bowler. ",
    "It slows momentum and can trigger a mini-collapse if the next batter feels pressure. ",
    "It changes match-ups, making the next few balls all about survival and rotating strike. ",
    "It often flips field settings and plans—suddenly the bowling side can attack. ",
]
WICKET_EXPLAIN_ENDERS = [
    "From here, the batting side has to rebuild quickly.",
    "Now the bowling side can squeeze hard with dots.",
    "This is where the innings can wobble.",
    "It puts the chase / total right on the edge.",
]

# -----------------------
# Helpers
# -----------------------
def pick(lst):
    return random.choice(lst)

def safe_get(d, path, default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur

def clamp_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default

def format_ball(over_num, ball_num, bowler, batter, runs_batter, runs_extras, wicket_fell, wicket_kind=None):
    total_runs = (runs_batter or 0) + (runs_extras or 0)

    if wicket_fell:
        phrase = pick(WICKET)
        if wicket_kind:
            phrase = f"{phrase} ({wicket_kind})."
    else:
        if total_runs == 0:
            phrase = pick(DOT_BALL)
        elif total_runs == 1:
            phrase = pick(ONE_RUN)
        elif total_runs == 2:
            phrase = pick(TWO_RUNS)
        elif total_runs == 3:
            phrase = pick(THREE_RUNS)
        elif total_runs == 4:
            phrase = pick(FOUR)
        elif total_runs == 6:
            phrase = pick(SIX)
        else:
            phrase = f"they pick up {total_runs} runs."

    over_num_int = clamp_int(over_num, 0)
    ball_num_int = clamp_int(ball_num, 0)

    return f"Over {over_num_int}.{ball_num_int}: {bowler} to {batter}, {phrase}"

def iter_innings_overs_deliveries(match_json):
    """
    Yields tuples: (inning_team, over_number, deliveries_list)
    Supports common Cricsheet JSON patterns.
    """
    innings = match_json.get("innings", [])
    for inn in innings:
        team = inn.get("team", "Unknown")

        if isinstance(inn, dict) and "overs" in inn and isinstance(inn["overs"], list):
            for ov in inn["overs"]:
                over_num = ov.get("over", 0)
                deliveries = ov.get("deliveries", [])
                if isinstance(deliveries, list) and deliveries:
                    yield team, over_num, deliveries

        elif isinstance(inn, dict) and "deliveries" in inn and isinstance(inn["deliveries"], list):
            buckets = {}
            for d in inn["deliveries"]:
                o = d.get("over")
                if o is None:
                    o = d.get("over_number", 0)
                buckets.setdefault(o, []).append(d)

            for over_num in sorted(buckets.keys(), key=lambda x: (x is None, x)):
                deliveries = buckets[over_num]
                if deliveries:
                    yield team, (over_num if over_num is not None else 0), deliveries

def get_match_header(match_json):
    info = match_json.get("info", {}) if isinstance(match_json, dict) else {}
    teams = info.get("teams", [])
    if isinstance(teams, list) and len(teams) >= 2:
        return f"Match: {teams[0]} vs {teams[1]}"
    if isinstance(teams, list) and len(teams) == 1:
        return f"Match: {teams[0]} vs Unknown"
    return "Match: Unknown vs Unknown"

# -----------------------
# Instruction example builders
# -----------------------
def make_example(match_header: str, user_text: str, assistant_text: str) -> List[str]:
    lines = []
    if ADD_MATCH_TOKEN:
        lines.append(MATCH_TOKEN)
    lines.append(match_header)
    lines.append(f"{USER} {user_text}".strip())
    lines.append(f"{ASSISTANT} {assistant_text}".strip())
    if ADD_MATCH_TOKEN:
        lines.append(MATCH_TOKEN)
    lines.append("")  # blank separator between examples
    return lines

def normalize_joined_over_lines(over_lines: List[str]) -> str:
    # Keep it as one paragraph; the model trained as a next-token predictor likes consistent formatting
    return " ".join([ln.strip() for ln in over_lines if ln.strip()])

def synth_summary(over_lines: List[str]) -> str:
    # Lightweight synthetic “label” — not perfect cricket logic but teaches the assistant to summarize
    # You can make this richer later.
    out = pick(SUMMARY_STARTERS) + pick(SUMMARY_MIDDLES) + pick(SUMMARY_ENDERS)
    return out.strip()

def synth_wicket_explain(wicket_line: str) -> str:
    out = pick(WICKET_EXPLAIN_OPENERS) + pick(WICKET_EXPLAIN_BODY) + pick(WICKET_EXPLAIN_ENDERS)
    return out.strip()

def extract_overs_as_lines(match_json) -> List[str]:
    """
    Flattens the match into a list of ball commentary lines in order:
    Match header is NOT included here (handled separately).
    """
    lines = []
    for team, over_num, deliveries in iter_innings_overs_deliveries(match_json):
        for i, delivery in enumerate(deliveries, start=1):
            batter = delivery.get("batter") or delivery.get("striker") or "Unknown Batter"
            bowler = delivery.get("bowler") or "Unknown Bowler"

            runs_batter = safe_get(delivery, ["runs", "batter"], 0) or 0
            runs_extras = safe_get(delivery, ["runs", "extras"], 0) or 0

            wicket_fell = False
            wicket_kind = None
            if "wickets" in delivery and delivery["wickets"]:
                wicket_fell = True
                wicket_kind = delivery["wickets"][0].get("kind")
            elif "wicket" in delivery and delivery["wicket"]:
                wicket_fell = True
                wicket_kind = delivery["wicket"].get("kind")

            line = format_ball(
                over_num=over_num,
                ball_num=i,
                bowler=bowler,
                batter=batter,
                runs_batter=runs_batter,
                runs_extras=runs_extras,
                wicket_fell=wicket_fell,
                wicket_kind=wicket_kind
            )
            lines.append(line)
    return lines

def group_by_over(lines: List[str]) -> List[List[str]]:
    """
    Groups consecutive ball lines by the "Over X.Y:" prefix's X part.
    Returns list of overs, where each over is a list of lines.
    """
    overs = []
    cur = []
    cur_over = None
    for ln in lines:
        # parse "Over 16.3:"
        try:
            prefix = ln.split(":")[0]  # "Over 16.3"
            over_part = prefix.replace("Over", "").strip()  # "16.3"
            over_int = over_part.split(".")[0]  # "16"
        except Exception:
            over_int = "NA"

        if cur_over is None:
            cur_over = over_int
            cur = [ln]
        elif over_int == cur_over:
            cur.append(ln)
        else:
            overs.append(cur)
            cur_over = over_int
            cur = [ln]
    if cur:
        overs.append(cur)
    return overs

def build_instruction_examples(match_header: str, ball_lines: List[str]) -> List[str]:
    """
    Creates a mixture of:
    - Continue-this-over examples
    - Summarize-last-3-overs examples
    - Explain-wicket examples
    """
    out_lines = []

    overs = group_by_over(ball_lines)

    # A) Continue-this-over: sample some overs and create prefix->continuation pairs
    # We provide a partial prompt and the assistant continues the same over (or next few balls).
    for ov in overs:
        if len(ov) < 2:
            continue
        # sample ~15% of overs for continue tasks (tune as you like)
        if random.random() > 0.15:
            continue

        # choose a starting ball inside the over to continue from
        start_idx = random.randint(0, max(0, len(ov) - 2))
        prompt_ball = ov[start_idx]

        # user asks to continue from that ball
        user_text = f"Continue this over: {prompt_ball}"

        # assistant target is remaining balls in this over (and optionally spill into next over)
        cont = ov[start_idx + 1:]
        assistant_text = normalize_joined_over_lines(cont)

        if assistant_text:
            out_lines.extend(make_example(match_header, user_text, assistant_text))

    # B) Summarize last 3 overs: slide a window of 3 overs and sample some
    if len(overs) >= 3:
        for i in range(2, len(overs)):
            if random.random() > 0.08:  # ~8% windows; adjust for dataset size
                continue
            window = overs[i-2] + overs[i-1] + overs[i]
            window_text = "\n".join(window)
            user_text = "Summarize the last 3 overs in 2-3 sentences:\n" + window_text
            assistant_text = synth_summary(window)
            out_lines.extend(make_example(match_header, user_text, assistant_text))

    # C) Explain wicket: find wicket lines and sample some
    wicket_lines = [ln for ln in ball_lines if "OUT!" in ln or "WICKET!" in ln or "Gone!" in ln]
    for wl in wicket_lines:
        if random.random() > 0.25:
            continue
        user_text = f"Explain why this wicket was important:\n{wl}"
        assistant_text = synth_wicket_explain(wl)
        out_lines.extend(make_example(match_header, user_text, assistant_text))

    return out_lines

def process_file(fp: str) -> List[str]:
    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)

    match_header = get_match_header(data)
    ball_lines = extract_overs_as_lines(data)

    # If file is empty / weird, skip
    if not ball_lines:
        return []

    # Instruction examples
    return build_instruction_examples(match_header, ball_lines)

def main():
    random.seed(SEED)

    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.json")))
    if not files:
        raise FileNotFoundError(f"No JSON files found in {RAW_DIR}")

    if MAX_FILES is not None:
        files = files[:MAX_FILES]

    random.shuffle(files)

    n_val = max(1, int(len(files) * VAL_SPLIT))
    val_files = set(files[:n_val])
    train_files = files[n_val:]

    Path(os.path.dirname(OUT_TRAIN)).mkdir(parents=True, exist_ok=True)

    # Write train
    train_count = 0
    with open(OUT_TRAIN, "w", encoding="utf-8") as f_train:
        for fp in train_files:
            lines = process_file(fp)
            if not lines:
                continue
            train_count += 1
            f_train.write(LINE_END.join(lines) + LINE_END)

    # Write val
    val_count = 0
    with open(OUT_VAL, "w", encoding="utf-8") as f_val:
        for fp in val_files:
            lines = process_file(fp)
            if not lines:
                continue
            val_count += 1
            f_val.write(LINE_END.join(lines) + LINE_END)

    print(f"Processed {train_count} train files and {val_count} val files (files with usable examples).")
    print(f"Saved: {OUT_TRAIN}")
    print(f"Saved: {OUT_VAL}")
    print(f"ADD_MATCH_TOKEN={ADD_MATCH_TOKEN}")
    print("NOTE: This outputs instruction-style examples (continue/summarize/explain).")

if __name__ == "__main__":
    main()
