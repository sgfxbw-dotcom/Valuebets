"""Streamlit front end for the Betway value-bet finder.

Run:  streamlit run app.py
"""
import os
import tempfile
from datetime import datetime

import pandas as pd
import streamlit as st

import value_finder as vf

RESULTS_FILE = "results.csv"
SPORTS = {
    "England - Premier League": "soccer_epl",
    "Spain - La Liga": "soccer_spain_la_liga",
    "Germany - Bundesliga": "soccer_germany_bundesliga",
    "Italy - Serie A": "soccer_italy_serie_a",
    "France - Ligue 1": "soccer_france_ligue_one",
}

st.set_page_config(page_title="Value Bets", page_icon="⚽", layout="centered")
st.title("Betway Value Finder")
st.caption("Finds bets where the model beats the odds. It cannot guarantee wins; "
           "expect long losing runs. Stake only what you can afford to lose.")

# API key from Streamlit secrets (hosted) or environment (local)
try:
    if "ODDS_API_KEY" in st.secrets:
        os.environ["ODDS_API_KEY"] = st.secrets["ODDS_API_KEY"]
except Exception:
    pass


@st.cache_resource
def get_model(path, mtime):
    return vf.load_model(path)


def log_bet(b):
    con = vf.db()
    con.execute(
        "INSERT INTO bets(placed_at, match, pick, odds, prob, stake) VALUES(?,?,?,?,?,?)",
        (datetime.now().isoformat(timespec="seconds"), b["match"], b["pick"],
         b["odds"], b["prob"], b["stake"]))
    con.commit()
    con.close()


# ------------------------------------------------------------- sidebar -----
with st.sidebar:
    st.header("Settings")
    has_key = bool(os.environ.get("ODDS_API_KEY"))
    demo = st.checkbox("Demo mode (sample odds)", value=not has_key)
    if not has_key and not demo:
        st.warning("No ODDS_API_KEY set. Use demo mode or add your key.")
    league = st.selectbox("League", list(SPORTS))
    bankroll = st.number_input("Bankroll", min_value=10.0, value=1000.0, step=50.0)
    min_ev = st.slider("Minimum edge (EV)", 0.0, 0.15, vf.MIN_EV, 0.005,
                       help="Higher = fewer, safer-looking bets. Below 3% is mostly noise.")
    upload = st.file_uploader("Past results CSV (football-data.co.uk)", type="csv")
    if upload is not None:
        with open(RESULTS_FILE, "wb") as f:
            f.write(upload.getvalue())
        st.success("Results saved.")

model = None
if os.path.exists(RESULTS_FILE):
    try:
        model = get_model(RESULTS_FILE, os.path.getmtime(RESULTS_FILE))
    except SystemExit as e:
        st.error(str(e))

scan_tab, bets_tab, report_tab = st.tabs(["Scan", "My bets", "Report"])

# ---------------------------------------------------------------- scan -----
with scan_tab:
    if model is None:
        st.info("Upload a past-results CSV in the sidebar to get started "
                "(columns: HomeTeam, AwayTeam, FTHG, FTAG).")
    elif st.button("Scan now", type="primary"):
        try:
            events = vf.DEMO_EVENTS if demo else vf.fetch_events(SPORTS[league])
            st.session_state["scan"] = vf.find_value(events, model, bankroll, min_ev)
        except SystemExit as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Could not fetch odds: {e}")

    scan = st.session_state.get("scan")
    if scan:
        bets, skipped = scan
        if skipped:
            st.warning("Skipped, team names not matched: " + ", ".join(skipped) +
                       ". Add them to ALIASES in value_finder.py.")
        if not bets:
            st.info("No value bets right now. That's normal; most matches have no edge.")
        for i, b in enumerate(bets):
            with st.container(border=True):
                st.subheader(f"{b['pick']} @ {b['odds']:.2f}")
                st.write(b["match"])
                st.caption(b["kickoff"])
                c1, c2, c3 = st.columns(3)
                c1.metric("Model", f"{b['prob']:.1%}")
                c2.metric("Betway", f"{b['market_prob']:.1%}")
                c3.metric("EV", f"{b['ev']:+.1%}")
                st.write(f"Suggested stake: **{b['stake']:.2f}**")
                if b["check"]:
                    st.warning("Model and market disagree a lot. The model is usually "
                               "the one that's wrong. Check team news first.")
                if st.button("I placed this bet", key=f"log{i}"):
                    log_bet(b)
                    st.success("Logged. Add closing odds later in My bets.")

# ------------------------------------------------------------- my bets -----
with bets_tab:
    con = vf.db()
    df = pd.read_sql_query(
        "SELECT id, placed_at, match, pick, odds, prob, stake, closing_odds, result "
        "FROM bets ORDER BY id DESC", con)
    if df.empty:
        st.info("No bets logged yet.")
    else:
        st.dataframe(df.drop(columns=["prob"]), hide_index=True, use_container_width=True)
        with st.form("update_bet"):
            st.write("Update a bet")
            bid = st.number_input("Bet ID", min_value=1, step=1, value=int(df["id"].max()))
            closing = st.number_input("Closing odds (0 = leave as is)", min_value=0.0, step=0.01)
            result = st.selectbox("Result", ["(leave as is)", "won", "lost", "void"])
            if st.form_submit_button("Save"):
                if closing > 0:
                    con.execute("UPDATE bets SET closing_odds=? WHERE id=?", (closing, int(bid)))
                if result != "(leave as is)":
                    con.execute("UPDATE bets SET result=? WHERE id=?", (result, int(bid)))
                con.commit()
                st.success(f"Bet #{int(bid)} updated.")
                st.rerun()
    con.close()

# -------------------------------------------------------------- report -----
with report_tab:
    con = vf.db()
    df = pd.read_sql_query("SELECT odds, prob, stake, closing_odds, result FROM bets", con)
    con.close()
    if df.empty:
        st.info("Log some bets to see your report.")
    else:
        settled = df[df["result"].isin(["won", "lost"])]
        profit = (settled.loc[settled.result == "won", "stake"]
                  * (settled.loc[settled.result == "won", "odds"] - 1)).sum() \
            - settled.loc[settled.result == "lost", "stake"].sum()
        staked = settled["stake"].sum()
        clv = (df["odds"] / df["closing_odds"] - 1).dropna()
        c1, c2 = st.columns(2)
        c1.metric("Bets logged", len(df))
        c2.metric("Settled", len(settled))
        if len(settled):
            c1.metric("Win rate", f"{(settled.result == 'won').mean():.1%}",
                      help=f"Model expected {settled.prob.mean():.1%}")
            c2.metric("ROI", f"{profit / staked:+.1%}" if staked else "n/a",
                      delta=f"{profit:+.2f}")
        if len(clv):
            st.metric("Average closing-line value", f"{clv.mean():+.2%}",
                      help="Consistently positive over 100+ bets suggests a real edge.")
        if len(settled) < 300:
            st.caption(f"Only {len(settled)} settled bets. Profit is mostly luck below "
                       "~300 bets; trust closing-line value over profit until then.")
