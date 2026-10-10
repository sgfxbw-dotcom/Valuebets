#!/usr/bin/env python3
"""
Betway value-bet finder.

Pulls Betway 1X2 odds from The Odds API, estimates match probabilities with a
Poisson goals model fitted on historical results, and lists bets where the
model's probability beats the odds by a safe margin. Includes a bet tracker
that measures closing-line value (CLV), the best check on whether your edge
is real.

This does NOT produce guaranteed winners. Expect long losing runs.
"""
import argparse
import csv
import difflib
import math
import os
import sqlite3
import sys
from datetime import datetime

API_URL = "https://api.the-odds-api.com/v4/sports/{sport}/odds"
DB_PATH = os.environ.get("BETS_DB", "bets.db")

MIN_EV = 0.03          # only flag bets with >= 3% expected value
KELLY_SCALE = 0.25     # quarter Kelly
MAX_STAKE_PCT = 0.02   # never stake more than 2% of bankroll on one bet
SANITY_GAP = 0.15      # flag picks where model and market differ by 15+ points

# Odds API team name -> football-data.co.uk team name (add your own as needed)
ALIASES = {
    "Manchester United": "Man United",
    "Manchester City": "Man City",
    "Wolverhampton Wanderers": "Wolves",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Brighton and Hove Albion": "Brighton",
    "Leeds United": "Leeds",
    "Leicester City": "Leicester",
}


# ---------------------------------------------------------------- model ----

def poisson(k, lam):
    return math.exp(-lam) * lam ** k / math.factorial(k)


def outcome_probs(lam_home, lam_away, max_goals=10):
    """Home/draw/away probabilities from independent Poisson goal counts."""
    home = draw = away = 0.0
    for i in range(max_goals + 1):
        pi = poisson(i, lam_home)
        for j in range(max_goals + 1):
            p = pi * poisson(j, lam_away)
            if i > j:
                home += p
            elif i == j:
                draw += p
            else:
                away += p
    total = home + draw + away
    return {"home": home / total, "draw": draw / total, "away": away / total}


class Model:
    """Home/away attack and defence strengths with shrinkage toward average."""

    def __init__(self, rows, shrink=5):
        n = len(rows)
        self.avg_h = sum(r[2] for r in rows) / n
        self.avg_a = sum(r[3] for r in rows) / n
        t = {}
        for h, a, hg, ag in rows:
            x = t.setdefault(h, dict(hf=0, ha=0, hn=0, af=0, aa=0, an=0))
            x["hf"] += hg; x["ha"] += ag; x["hn"] += 1
            y = t.setdefault(a, dict(hf=0, ha=0, hn=0, af=0, aa=0, an=0))
            y["af"] += ag; y["aa"] += hg; y["an"] += 1
        k = shrink
        for s in t.values():
            s["att_h"] = (s["hf"] + k * self.avg_h) / (s["hn"] + k) / self.avg_h
            s["def_h"] = (s["ha"] + k * self.avg_a) / (s["hn"] + k) / self.avg_a
            s["att_a"] = (s["af"] + k * self.avg_a) / (s["an"] + k) / self.avg_a
            s["def_a"] = (s["aa"] + k * self.avg_h) / (s["an"] + k) / self.avg_h
        self.teams = t

    def expected_goals(self, home, away):
        h, a = self.teams[home], self.teams[away]
        return (self.avg_h * h["att_h"] * a["def_a"],
                self.avg_a * a["att_a"] * h["def_h"])

    def match_team(self, name):
        name = ALIASES.get(name, name)
        if name in self.teams:
            return name
        close = difflib.get_close_matches(name, list(self.teams), n=1, cutoff=0.6)
        return close[0] if close else None


def load_model(csv_path):
    """Load a football-data.co.uk style CSV (HomeTeam, AwayTeam, FTHG, FTAG)."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                rows.append((r["HomeTeam"], r["AwayTeam"], int(r["FTHG"]), int(r["FTAG"])))
            except (KeyError, ValueError, TypeError):
                continue
    if len(rows) < 50:
        sys.exit(f"Need at least 50 completed matches in {csv_path}, found {len(rows)}.")
    return Model(rows)


# ---------------------------------------------------------------- odds -----

DEMO_EVENTS = [
    {"home_team": "Arsenal", "away_team": "Chelsea", "commence_time": "2026-10-17T14:00:00Z",
     "bookmakers": [{"key": "betway", "markets": [{"key": "h2h", "outcomes": [
         {"name": "Arsenal", "price": 2.10}, {"name": "Draw", "price": 3.50},
         {"name": "Chelsea", "price": 3.60}]}]}]},
    {"home_team": "Liverpool", "away_team": "Brentford", "commence_time": "2026-10-17T16:30:00Z",
     "bookmakers": [{"key": "betway", "markets": [{"key": "h2h", "outcomes": [
         {"name": "Liverpool", "price": 1.60}, {"name": "Draw", "price": 4.20},
         {"name": "Brentford", "price": 5.50}]}]}]},
]


def fetch_events(sport):
    import requests
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        sys.exit("Set ODDS_API_KEY first (free key at https://the-odds-api.com), or use --demo.")
    r = requests.get(API_URL.format(sport=sport), timeout=20, params={
        "apiKey": key, "bookmakers": "betway", "markets": "h2h", "oddsFormat": "decimal"})
    r.raise_for_status()
    print(f"(API requests remaining: {r.headers.get('x-requests-remaining', '?')})")
    return r.json()


# ------------------------------------------------------------ betting ------

def expected_value(p, odds):
    return p * odds - 1


def kelly_fraction(p, odds, scale=KELLY_SCALE):
    b = odds - 1
    if b <= 0:
        return 0.0
    return max(0.0, (p * b - (1 - p)) / b) * scale


def find_value(events, model, bankroll, min_ev=MIN_EV):
    bets, skipped = [], []
    for ev in events:
        home, away = ev["home_team"], ev["away_team"]
        bw = next((b for b in ev.get("bookmakers", []) if b["key"] == "betway"), None)
        mk = next((m for m in (bw or {}).get("markets", []) if m["key"] == "h2h"), None)
        if not mk:
            continue
        prices = {o["name"]: o["price"] for o in mk["outcomes"]}
        if "Draw" not in prices or home not in prices or away not in prices:
            continue
        mh, ma = model.match_team(home), model.match_team(away)
        if not mh or not ma or mh == ma:
            skipped.append(f"{home} vs {away}")
            continue
        probs = outcome_probs(*model.expected_goals(mh, ma))
        picks = (("home", home), ("draw", "Draw"), ("away", away))
        inv = {k: 1 / prices[n] for k, n in picks}
        overround = sum(inv.values())
        for key, name in picks:
            odds, p = prices[name], probs[key]
            ev_ = expected_value(p, odds)
            if ev_ < min_ev:
                continue
            stake = min(kelly_fraction(p, odds) * bankroll, MAX_STAKE_PCT * bankroll)
            market_p = inv[key] / overround
            bets.append({
                "match": f"{home} vs {away}", "kickoff": ev.get("commence_time", ""),
                "pick": name, "odds": odds, "prob": p, "market_prob": market_p,
                "ev": ev_, "stake": round(stake, 2),
                "check": abs(p - market_p) > SANITY_GAP})
    return sorted(bets, key=lambda b: -b["ev"]), skipped


def cmd_scan(args):
    model = load_model(args.results)
    events = DEMO_EVENTS if args.demo else fetch_events(args.sport)
    bets, skipped = find_value(events, model, args.bankroll, args.min_ev)
    if skipped:
        print("Skipped (team name not found in results data; add to ALIASES):")
        for s in skipped:
            print(f"  - {s}")
        print()
    if not bets:
        print("No value bets found right now. That is normal; most matches have no edge.")
        return
    print(f"{len(bets)} candidate bet(s), bankroll {args.bankroll:.2f}:\n")
    for b in bets:
        print(f"{b['match']}  ({b['kickoff']})")
        print(f"  Pick {b['pick']} @ {b['odds']:.2f} | model {b['prob']:.1%} vs "
              f"Betway no-vig {b['market_prob']:.1%} | EV {b['ev']:+.1%} | stake {b['stake']:.2f}")
        if b["check"]:
            print("  !! Model and market disagree a lot. Usually the model is wrong "
                  "(injuries, lineups, form). Check news before betting.")
        print(f"  Log it: python value_finder.py log --match \"{b['match']}\" "
              f"--pick \"{b['pick']}\" --odds {b['odds']} --prob {b['prob']:.4f} "
              f"--stake {b['stake']}\n")


# ------------------------------------------------------------ tracker ------

def db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS bets(
        id INTEGER PRIMARY KEY, placed_at TEXT, match TEXT, pick TEXT,
        odds REAL, prob REAL, stake REAL, closing_odds REAL, result TEXT)""")
    return con


def cmd_log(args):
    con = db()
    cur = con.execute(
        "INSERT INTO bets(placed_at, match, pick, odds, prob, stake) VALUES(?,?,?,?,?,?)",
        (datetime.now().isoformat(timespec="seconds"), args.match, args.pick,
         args.odds, args.prob, args.stake))
    con.commit()
    print(f"Logged bet #{cur.lastrowid}. After the market closes, run: "
          f"python value_finder.py close {cur.lastrowid} --odds <closing odds>")


def cmd_close(args):
    con = db()
    con.execute("UPDATE bets SET closing_odds=? WHERE id=?", (args.odds, args.id))
    con.commit()
    print(f"Closing odds saved for bet #{args.id}.")


def cmd_settle(args):
    con = db()
    con.execute("UPDATE bets SET result=? WHERE id=?", (args.result, args.id))
    con.commit()
    print(f"Bet #{args.id} marked {args.result}.")


def cmd_report(args):
    rows = db().execute(
        "SELECT odds, prob, stake, closing_odds, result FROM bets").fetchall()
    if not rows:
        print("No bets logged yet.")
        return
    settled = [r for r in rows if r[4] in ("won", "lost")]
    staked = sum(r[2] for r in settled)
    profit = sum(r[2] * (r[0] - 1) if r[4] == "won" else -r[2] for r in settled)
    clv = [r[0] / r[3] - 1 for r in rows if r[3]]
    print(f"Bets logged: {len(rows)} | settled: {len(settled)}")
    if settled:
        wins = sum(1 for r in settled if r[4] == "won")
        print(f"Win rate: {wins / len(settled):.1%} "
              f"(model expected {sum(r[1] for r in settled) / len(settled):.1%})")
        print(f"Staked {staked:.2f} | profit {profit:+.2f} | ROI {profit / staked:+.1%}" if staked else "")
    if clv:
        print(f"Average CLV: {sum(clv) / len(clv):+.2%} over {len(clv)} bets "
              "(consistently positive = real edge)")
    if len(settled) < 300:
        print(f"\nOnly {len(settled)} settled bets. Results are mostly luck below ~300; "
              "trust CLV over profit until then.")


# --------------------------------------------------------------- main ------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan", help="find value bets")
    s.add_argument("--sport", default="soccer_epl", help="Odds API sport key")
    s.add_argument("--results", required=True, help="CSV of past results")
    s.add_argument("--bankroll", type=float, default=1000.0)
    s.add_argument("--min-ev", type=float, default=MIN_EV)
    s.add_argument("--demo", action="store_true", help="use built-in sample odds")
    s.set_defaults(fn=cmd_scan)

    l = sub.add_parser("log", help="record a bet you placed")
    l.add_argument("--match", required=True)
    l.add_argument("--pick", required=True)
    l.add_argument("--odds", type=float, required=True)
    l.add_argument("--prob", type=float, required=True)
    l.add_argument("--stake", type=float, required=True)
    l.set_defaults(fn=cmd_log)

    c = sub.add_parser("close", help="save closing odds for a bet")
    c.add_argument("id", type=int)
    c.add_argument("--odds", type=float, required=True)
    c.set_defaults(fn=cmd_close)

    r = sub.add_parser("settle", help="record the result")
    r.add_argument("id", type=int)
    r.add_argument("result", choices=["won", "lost", "void"])
    r.set_defaults(fn=cmd_settle)

    sub.add_parser("report", help="ROI, win rate, CLV").set_defaults(fn=cmd_report)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
