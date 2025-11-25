# t20_strategy_app_final.py
import streamlit as st
import pandas as pd
import joblib
from phase_fielding import generate_field_setting, plot_fielding_positions
from pulp import LpProblem, LpMaximize, LpVariable, lpSum

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(layout="wide", page_title="Sri Lanka T20 Strategy Assistant")

# ---------------------------
# Load models (wrapped safely)
# ---------------------------
def safe_load(path, name):
    try:
        m = joblib.load(path)
        return m
    except Exception as e:
        st.session_state.setdefault("_model_errors", []).append(f"Failed to load {name}: {e}")
        return None

# Win & toss
win_model = safe_load("win_predictor.pkl", "win_model")
toss_model = safe_load("toss_strategy.pkl", "toss_model")

# Phase models (batting runs)
powerplay_model = safe_load("powerplay_model.pkl", "powerplay_model")
middle_model = safe_load("middle_model.pkl", "middle_model")
death_model = safe_load("death_model.pkl", "death_model")

pp_runs_model = safe_load("pp_runs_model.pkl", "pp_runs_model")
mo_runs_model = safe_load("mo_runs_model.pkl", "mo_runs_model")
do_runs_model = safe_load("do_runs_model.pkl", "do_runs_model")

# Phase conceded runs models (bowling conceded)
pp_conc_model = safe_load("pp_conc_model.pkl", "pp_conc_model")
mo_conc_model = safe_load("mo_conc_model.pkl", "mo_conc_model")
do_conc_model = safe_load("do_conc_model.pkl", "do_conc_model")

# Wickets TAKEN models (bowling wickets predicted)
pp_wkts_model = safe_load("pp_wkts_model.pkl", "pp_wkts_model")
mo_wkts_model = safe_load("mo_wkts_model.pkl", "mo_wkts_model")
do_wkts_model = safe_load("do_wkts_model.pkl", "do_wkts_model")

# Wickets LOST models (batting expected wickets lost)
pp_wkts_lost_model = safe_load("pp_wkts_lost_model.pkl", "pp_wkts_lost_model")
mo_wkts_lost_model = safe_load("mo_wkts_lost_model.pkl", "mo_wkts_lost_model")
do_wkts_lost_model = safe_load("do_wkts_lost_model.pkl", "do_wkts_lost_model")

# Role/selection models
batters_model = safe_load("batters_model.pkl", "batters_model")
fast_bowlers_model = safe_load("fast_bowlers_model.pkl", "fast_bowlers_model")
spinners_model = safe_load("spinners_model.pkl", "spinners_model")
allrounders_model = safe_load("allrounders_model.pkl", "allrounders_model")

# Show any model load errors at top
if "_model_errors" in st.session_state:
    for err in st.session_state["_model_errors"]:
        st.error(err)

# ---------------------------
# Load dataset (for dropdowns & players)
# ---------------------------
DATA_PATH = "Cleaned_DataT20.xlsx"
PLAYERS_PATH = "ProcessedPlayers.xlsx"  # used for LP Best XI
PLAYER_FILL = "filled_player.csv"       # used earlier for player scoring table

try:
    df = pd.read_excel(DATA_PATH)
except Exception as e:
    st.error(f"Failed to load dataset {DATA_PATH}: {e}")
    df = pd.DataFrame()

# Dropdown options
opponent_teams = sorted(df["opponent_team"].drop_duplicates()) if "opponent_team" in df.columns else []
venues = sorted(df["venue"].drop_duplicates()) if "venue" in df.columns else []
pitch_options = sorted(df["pitch_type"].astype(str).str.strip().str.title().drop_duplicates()) if "pitch_type" in df.columns else []
ground_options = sorted(df["ground_dimension"].astype(str).str.strip().str.title().drop_duplicates()) if "ground_dimension" in df.columns else []
weather_types = sorted(df["weather_type"].drop_duplicates()) if "weather_type" in df.columns else []

team_tiers = {}
if {"opponent_team","opponent_tier"}.issubset(df.columns):
    team_tiers = df[["opponent_team","opponent_tier"]].drop_duplicates().set_index("opponent_team").to_dict()["opponent_tier"]

# ---------------------------
# Title & sidebar
# ---------------------------
st.markdown("<h1 style='text-align:center;'>🇱🇰🏏 Sri Lanka T20 Match Strategy Assistant</h1>", unsafe_allow_html=True)

st.sidebar.header("Match Conditions")
opponent_team = st.sidebar.selectbox("Opponent Team", opponent_teams)
venue = st.sidebar.selectbox("Venue", venues)
pitch_type = st.sidebar.selectbox("Pitch Type", pitch_options)
ground_size = st.sidebar.selectbox("Ground Dimension", ground_options)
weather = st.sidebar.selectbox("Weather Type", weather_types)

opponent_tier = team_tiers.get(opponent_team, "Unknown")
st.sidebar.markdown(f"**Opponent Tier:** <span style='color:green'>{opponent_tier}</span>", unsafe_allow_html=True)

opponent_score = st.sidebar.number_input("Opponent's 1st Innings Score (leave 0 if not applicable)", min_value=0, max_value=400, value=0, step=1)

# Base input df used for most models
base_input = {
    "opponent_team": opponent_team,
    "venue": venue,
    "pitch_type": pitch_type,
    "ground_dimension": ground_size,
    "weather_type": weather,
    "opponent_tier": opponent_tier
}
base_df = pd.DataFrame([base_input])

# Toss selection
st.subheader("Toss Result")
toss_result = st.radio("Select Toss Result", ["Won", "Lost"], horizontal=True)

# Determine toss_pred variable (None if not predicted)
toss_pred = None
if toss_result == "Won" and toss_model is not None:
    try:
        toss_pred = toss_model.predict(base_df)[0]  # 1=Bat, 0=Bowl
    except Exception as e:
        st.warning(f"Could not run toss model prediction: {e}")
        toss_pred = None

# ---------------------------
# Main generate button
# ---------------------------
if st.button("🚀 Generate Match Strategy"):

    # Build win_input (include toss info)
    if toss_result == "Won":
        toss_decision_val = "bat" if toss_pred == 1 else "bowl" if toss_pred == 0 else "unknown"
        toss_won_val = 1
    else:
        toss_decision_val = "none"
        toss_won_val = 0

    win_input = pd.DataFrame([{
        **base_input,
        "toss_won": toss_won_val,
        "toss_decision": toss_decision_val
    }])

    # ---------------------------
    # Match Outcome Prediction
    # ---------------------------
    st.subheader("🔮 Match Outcome Predictions")
    if win_model is not None:
        try:
            win_proba = win_model.predict_proba(win_input)[0][1] * 100
            st.metric("Win Probability (SL)", f"{win_proba:.1f}%")
        except Exception as e:
            st.error(f"Win model prediction failed: {e}")
    else:
        st.info("Win model not loaded — cannot predict probability.")

    # Show toss strategy (if toss won show predicted decision)
    if toss_result == "Won":
        if toss_pred is not None:
            toss_str = "Bat First" if toss_pred == 1 else "Bowl First"
            st.markdown(f"**🧭 Toss Strategy (model):** <span style='color:blue'>{toss_str}</span>", unsafe_allow_html=True)
        else:
            st.markdown("**🧭 Toss Strategy:** Model not available/prediction failed.", unsafe_allow_html=True)
    else:
        st.markdown("**🧭 Toss Strategy:** Decision by opponent (lost).", unsafe_allow_html=True)

    # ---------------------------
    # Phase-wise Game Plan header
    # ---------------------------
    st.subheader("📊 Phase-wise Game Plan")

    # ---------------------------
    # BATTING STRATEGY (showing logic as you requested)
    # ---------------------------
    # show_batting True when: (toss won & toss_pred==1) OR (toss lost)
    show_batting = False
    if toss_result == "Won" and toss_pred == 1:
        show_batting = True
    if toss_result == "Lost":
        show_batting = True

    # Define show_phase for batting (outside conditional)
    def show_batting_phase(name, model, run_model, wkts_lost_model):
        # safe predictions
        strat = "N/A"
        runs_pred = 0
        wkts_lost_pred = 0
        try:
            strat = model.predict(base_df)[0] if model is not None else "Strategy model missing"
        except Exception:
            strat = "Strategy model failed"
        try:
            runs_pred = run_model.predict(base_df)[0] if run_model is not None else 0
        except Exception:
            runs_pred = 0
        try:
            wkts_lost_pred = wkts_lost_model.predict(base_df)[0] if wkts_lost_model is not None else 0
        except Exception:
            wkts_lost_pred = 0

        st.markdown(f"### {name}")
        st.markdown(f"- **Expected Runs to Score**: `{int(runs_pred)} runs`")
        st.markdown(f"- **Expected Wickets Lost**: `{int(wkts_lost_pred)} wickets`")
        st.markdown(f"- **Strategy**: <span style='color:green'>{strat}</span>", unsafe_allow_html=True)

    if show_batting:
        st.subheader("🏏 Phase-wise Batting Strategy")
        show_batting_phase("Powerplay (1–6)", powerplay_model, pp_runs_model, pp_wkts_lost_model)
        show_batting_phase("Middle Overs (7–15)", middle_model, mo_runs_model, mo_wkts_lost_model)
        show_batting_phase("Death Overs (16–20)", death_model, do_runs_model, do_wkts_lost_model)

    # ---------------------------
    # Advanced Chase Mode (if opponent batted first)
    # ---------------------------
    if opponent_score > 0:
        target_score = opponent_score + 1
        required_rr = target_score / 20
        req_pp = round(required_rr * 6)
        req_mo = round(required_rr * 9)
        req_do = target_score - (req_pp + req_mo)

        st.subheader("🔥 Advanced Chasing Strategy")
        st.markdown(f"- Target Score: **{target_score}**")
        st.markdown(f"- Required Run Rate: **{required_rr:.2f} runs/over**")
        st.markdown(f"- Powerplay Target: **{req_pp} runs**")
        st.markdown(f"- Middle Overs Target: **{req_mo} runs**")
        st.markdown(f"- Death Overs Target: **{req_do} runs**")

        if target_score < 140:
            st.success("🟢 Low Score Scenario (<140)")
            st.markdown("- Powerplay: Keep wickets, take only calculated risks.")
        elif 140 <= target_score <= 160:
            st.info("🔵 Moderate Chase (140–160)")
            st.markdown("- Middle Overs: Anchor with set batters, keep RRR in check.")
        elif 160 < target_score <= 180:
            st.warning("🟠 Challenging Target (160–180)")
            st.markdown("- Death Overs: Need set finisher, boundary hitting is key.")
        else:
            st.error("🔴 High-pressure Chase (>180)")
            st.markdown("- All out aggression required in death overs.")

    # ---------------------------
    # BOWLING STRATEGY (phase-wise) — display same style as batting
    # show_bowling True when: (toss won & toss_pred==0) OR (toss lost)
    # ---------------------------
    show_bowling = False
    if toss_result == "Won" and toss_pred == 0:
        show_bowling = True
    if toss_result == "Lost":
        show_bowling = True

    def show_bowling_phase(name, wkts_model, conc_model, strategy_model=None):
        # wkts_model -> wickets expected to TAKE in that phase
        # conc_model -> expected conceded runs in that phase
        wkts_take = 0
        conc_runs = 0
        strat = "Standard bowling plan"
        try:
            wkts_take = wkts_model.predict(base_df)[0] if wkts_model is not None else 0
        except Exception:
            wkts_take = 0
        try:
            conc_runs = conc_model.predict(base_df)[0] if conc_model is not None else 0
        except Exception:
            conc_runs = 0
        try:
            strat = strategy_model.predict(base_df)[0] if strategy_model is not None else strat
        except Exception:
            pass

        st.markdown(f"### {name}")
        st.markdown(f"- **Expected Wickets (to TAKE)**: `{int(wkts_take)} wickets`")
        st.markdown(f"- **Expected Conceded Runs**: `{int(conc_runs)} runs`")
        st.markdown(f"- **Strategy**: <span style='color:green'>{strat}</span>", unsafe_allow_html=True)

    if show_bowling:
        st.subheader("🛡️ Phase-wise Bowling Strategy (SL Bowling Innings)")
        # Powerplay
        show_bowling_phase("Powerplay (1–6)", pp_wkts_model, pp_conc_model, powerplay_model)
        # Middle
        show_bowling_phase("Middle Overs (7–15)", mo_wkts_model, mo_conc_model, middle_model)
        # Death
        show_bowling_phase("Death Overs (16–20)", do_wkts_model, do_conc_model, death_model)

        # Summary expected conceded runs
        try:
            total_conceded = (int(pp_conc_model.predict(base_df)[0]) if pp_conc_model else 0) + \
                             (int(mo_conc_model.predict(base_df)[0]) if mo_conc_model else 0) + \
                             (int(do_conc_model.predict(base_df)[0]) if do_conc_model else 0)
            st.markdown(f"### 🔢 Expected Total to Restrict Opponent: **{total_conceded} runs**")
        except Exception:
            pass

    # ---------------------------
    # Team composition (unchanged, cleaned)
    # ---------------------------
    st.subheader("🧩 Team Composition")

    if batters_model and fast_bowlers_model and spinners_model and allrounders_model:
        try:
            raw_batters = batters_model.predict(base_df)[0]
            raw_fasts = fast_bowlers_model.predict(base_df)[0]
            raw_spins = spinners_model.predict(base_df)[0]
            raw_alls = allrounders_model.predict(base_df)[0]
        except Exception:
            raw_batters = raw_fasts = raw_spins = raw_alls = 0
    else:
        raw_batters = raw_fasts = raw_spins = raw_alls = 0

    role_preds = {
        "Batters": float(raw_batters),
        "Fast Bowlers": float(raw_fasts),
        "Spinners": float(raw_spins),
        "All-Rounders": float(raw_alls)
    }

    total_predicted = sum(role_preds.values())
    if total_predicted == 0:
        final_roles = {"Batters": 4, "Fast Bowlers": 3, "Spinners": 2, "All-Rounders": 2}
    else:
        normalized_roles = {role: round((count / total_predicted) * 11) for role, count in role_preds.items()}
        # adjust
        while sum(normalized_roles.values()) != 11:
            diff = 11 - sum(normalized_roles.values())
            if diff > 0:
                normalized_roles[max(normalized_roles, key=normalized_roles.get)] += 1
            else:
                m = min(normalized_roles, key=normalized_roles.get)
                if normalized_roles[m] > 0:
                    normalized_roles[m] -= 1
        final_roles = normalized_roles

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"- 🏏 Batters: <b>{final_roles['Batters']}</b>", unsafe_allow_html=True)
        st.markdown(f"- 🌀 Spinners: <b>{final_roles['Spinners']}</b>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"- ⚡ Fast Bowlers: <b>{final_roles['Fast Bowlers']}</b>", unsafe_allow_html=True)
        st.markdown(f"- ♟️ All-Rounders: <b>{final_roles['All-Rounders']}</b>", unsafe_allow_html=True)

    # ---------------------------
    # Best XI selection (LP) — unchanged but safe
    # ---------------------------
    try:
        players_df = pd.read_excel(PLAYERS_PATH)
        # numeric safe conversion
        for col in ["runs","sr","wkts","econ"]:
            if col in players_df.columns:
                players_df[col] = pd.to_numeric(players_df[col], errors='coerce').fillna(0)
        players_df = players_df[players_df['role'].isin(['batter','fast bowler','spinner','allrounder'])]
        players_df['score'] = players_df.get('runs',0)*0.5 + players_df.get('sr',0)*0.3 + players_df.get('wkts',0)*10 - players_df.get('econ',0)*2
        prob = LpProblem("Best_XI_Selection", LpMaximize)
        player_vars = {row['player']: LpVariable(row['player'], cat='Binary') for _, row in players_df.iterrows()}
        prob += lpSum(player_vars[player] * players_df.loc[players_df['player'] == player, 'score'].values[0] for player in player_vars)
        prob += lpSum(player_vars.values()) == 11
        # wicketkeeper constraint if available
        wk_batters = players_df[(players_df.get('is_wk',0)==1) & (players_df['role']=='batter')]['player'].tolist()
        if wk_batters:
            prob += lpSum([player_vars[p] for p in wk_batters]) == 1
        # role constraints (if possible)
        # (We skip if role counts unknown - maintain flexibility)
        prob.solve()
        selected_players = [p for p in player_vars if player_vars[p].varValue == 1.0]
        if selected_players:
            selected_df = players_df[players_df['player'].isin(selected_players)].copy()
            selected_df['role_display'] = selected_df.apply(lambda x: f"{x['role']} (WK)" if int(x.get('is_wk',0))==1 else x['role'], axis=1)
            display_df = selected_df[['player','role_display']].reset_index(drop=True)
            display_df.index = display_df.index + 1
            display_df.index.name = "No."
            st.subheader("🏅 Best XI")
            st.table(display_df)
    except Exception as e:
        st.info("Best XI selection skipped (players file missing or error).")
        st.write(f"Details: {e}")

    # ---------------------------
    # Fielding Strategy (unchanged)
    # ---------------------------
    st.header("🏏 Phase-wise Fielding Strategy")
    for phase in ["powerplay","middle","death"]:
        try:
            strategy, field_positions = generate_field_setting(base_df, phase=phase)
            st.subheader(f"{phase.capitalize()} Fielding Strategy")
            fig = plot_fielding_positions(field_positions, title=f"{phase.capitalize()} Setup")
            st.pyplot(fig)
        except Exception as e:
            st.info(f"Fielding plot skipped for {phase}: {e}")

# Footer
st.markdown("---")
st.caption("Model-based strategy suggestions. Use as advisory — real match decisions require human judgment and live info.")
