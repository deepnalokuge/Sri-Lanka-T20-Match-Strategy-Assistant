# app_updated.py
import streamlit as st
import pandas as pd
import joblib
from phase_fielding import generate_field_setting, plot_fielding_positions
from pulp import LpProblem, LpMaximize, LpVariable, lpSum
import numpy as np

# ------------------------------
# Load models (ensure these exist)
# ------------------------------
win_model = joblib.load("win_predictor.pkl")
toss_model = joblib.load("toss_strategy.pkl")

powerplay_model = joblib.load("powerplay_model.pkl")
middle_model = joblib.load("middle_model.pkl")
death_model = joblib.load("death_model.pkl")

pp_runs_model = joblib.load("pp_runs_model.pkl")
mo_runs_model = joblib.load("mo_runs_model.pkl")
do_runs_model = joblib.load("do_runs_model.pkl")

pp_wkts_model = joblib.load("pp_wkts_model.pkl")   # trained on SL bowling data (wickets SL took)
mo_wkts_model = joblib.load("mo_wkts_model.pkl")
do_wkts_model = joblib.load("do_wkts_model.pkl")

pp_wkts_model_op = pp_wkts_model  # alias (same models used for bowling predictions)
mo_wkts_model_op = mo_wkts_model
do_wkts_model_op = do_wkts_model

batters_model = joblib.load("batters_model.pkl")
fast_bowlers_model = joblib.load("fast_bowlers_model.pkl")
spinners_model = joblib.load("spinners_model.pkl")
allrounders_model = joblib.load("allrounders_model.pkl")

# ------------------------------
# Load dataset (use your uploaded file path)
# ------------------------------
DATA_PATH = "/mnt/data/Cleaned_DataT20.xlsx"  # your uploaded file
df = pd.read_excel(DATA_PATH)

# produce some safe defaults or aggregated stats for defend-mode if detailed columns missing
# We'll look for historical phase columns (names may differ). We'll attempt several common names.
def try_get_series(df, candidates):
    for c in candidates:
        if c in df.columns:
            return df[c]
    return None

# candidate names you might have used; adjust if your file uses different names
pp_candidates = ['opp_pp_runs', 'pp_runs_opponent', 'opp_powerplay_runs', 'pp_runs']
mo_candidates = ['opp_mo_runs', 'mo_runs_opponent', 'opp_middle_runs', 'mo_runs']
do_candidates = ['opp_do_runs', 'do_runs_opponent', 'opp_death_runs', 'do_runs']

pp_hist = try_get_series(df, pp_candidates)
mo_hist = try_get_series(df, mo_candidates)
do_hist = try_get_series(df, do_candidates)

# fallback: if none exist, try to infer from first_innings_total or similar
fallback_total_candidates = ['first_innings_total', 'opp_first_innings_score', 'opp_innings_score', 'first_innings_score']
fallback_total = try_get_series(df, fallback_total_candidates)

# ------------------------------
# Prepare UI
# ------------------------------
st.set_page_config(layout="wide", page_title="Sri Lanka T20 Strategy Assistant")
st.markdown("<h1 style='text-align:center;'>🇱🇰🏏 Sri Lanka T20 Match Strategy Assistant (Updated)</h1>", unsafe_allow_html=True)

# Sidebar inputs
st.sidebar.header("Match Conditions")
opponent_team = st.sidebar.selectbox("Opponent Team", sorted(df["opponent_team"].drop_duplicates()))
venue = st.sidebar.selectbox("Venue", sorted(df["venue"].drop_duplicates()))
pitch_type = st.sidebar.selectbox("Pitch Type", sorted(df["pitch_type"].str.strip().str.title().drop_duplicates()))
ground_size = st.sidebar.selectbox("Ground Dimension", sorted(df["ground_dimension"].str.strip().str.title().drop_duplicates()))
weather = st.sidebar.selectbox("Weather Type", sorted(df["weather_type"].drop_duplicates()))

# map tier if available
team_tiers = df[["opponent_team", "opponent_tier"]].drop_duplicates().set_index("opponent_team").to_dict().get("opponent_tier", {})
opponent_tier = team_tiers.get(opponent_team, "Unknown")
st.sidebar.markdown(f"**Opponent Tier:** <span style='color:green'>{opponent_tier}</span>", unsafe_allow_html=True)

opponent_score = st.sidebar.number_input("Opponent's 1st Innings Score", min_value=0, max_value=300, step=1, value=0)

# Base input (without toss features)
base_input = {
    "opponent_team": opponent_team,
    "venue": venue,
    "pitch_type": pitch_type,
    "ground_dimension": ground_size,
    "weather_type": weather,
    "opponent_tier": opponent_tier
}
base_df = pd.DataFrame([base_input])

# Toss radio (user chooses Won/Lost)
st.subheader("Toss Result")
toss_result = st.radio("Select Toss Result", ["Won", "Lost"], horizontal=True)

# Optional: let user override toss decision (if they want)
override_toss_choice = st.sidebar.selectbox("If SL wins toss, prefer to:", ["Auto (model)", "Bat First", "Bowl First"])

# Main generate button
if st.button("🚀 Generate Match Strategy"):

    # ------------------------------
    # Determine toss features (for win predictor)
    # ------------------------------
    if toss_result == "Won":
        # predict toss decision with toss_model unless overridden by user
        if override_toss_choice == "Auto (model)":
            toss_pred = toss_model.predict(base_df)[0]  # 1 = bat, 0 = bowl
            toss_decision_val = "bat" if toss_pred == 1 else "bowl"
            toss_str = "Bat First" if toss_pred == 1 else "Bowl First"
        else:
            toss_decision_val = "bat" if override_toss_choice == "Bat First" else "bowl"
            toss_str = override_toss_choice
        toss_won_val = 1
    else:
        toss_won_val = 0
        toss_decision_val = "none"
        toss_str = "Decision by Opponent (Lost)"

    # Build input for win model (includes toss features)
    win_input = pd.DataFrame([{
        **base_input,
        "toss_won": toss_won_val,
        "toss_decision": toss_decision_val
    }])

    # ------------------------------
    # Win probability (now depends on toss as feature)
    # ------------------------------
    st.subheader("Match Outcome Predictions")
    try:
        win_prob = win_model.predict_proba(win_input)[0][1] * 100
        st.metric("Win Probability (SL)", f"{win_prob:.1f}%")
    except Exception as e:
        st.error(f"Error predicting win probability: {e}")
        st.write("Make sure the win model was trained with toss_won and toss_decision features.")

    # Show toss strategy
    if toss_result == "Won":
        st.markdown(f"**Toss Strategy:** <span style='color:blue'>{toss_str}</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"**Toss Strategy:** <span style='color:red'>{toss_str}</span>", unsafe_allow_html=True)

    # ------------------------------
    # If SL bowls first (i.e., toss_result Won and decision is 'bowl', OR user lost toss but will bowl)
    # We need two cases:
    #  A) SL bowls first (SL is bowling innings 1)
    #  B) SL defends (SL batted first earlier, now bowling to defend)
    # ------------------------------
    sl_bowling_first = (toss_result == "Won" and toss_decision_val == "bowl")
    # OR if user lost toss but opponent chose to bat (we can't know opponent choice without data). We'll keep simple: only use model predicted toss_decision when toss won.
    # For user control, we could add a radio to set "SL will bowl first" — skipped unless requested.

    # ------------------------------
    # Bowling-first logic (SL bowling innings 1)
    # Use pp_wkts_model_op/mo/do to predict wickets SL can take in each phase
    # Also give tactical recommendations (bowler usage, field settings)
    # ------------------------------
    if sl_bowling_first:
        st.subheader("🟦 SL Bowling First — Phase-wise Bowling Plan")

        try:
            # For bowling predictions we expect the models to accept the same base_df features (or similar)
            # We pass base_df (without toss) as model training likely used same features
            pp_wkts_pred = pp_wkts_model_op.predict(base_df)[0]
            mo_wkts_pred = mo_wkts_model_op.predict(base_df)[0]
            do_wkts_pred = do_wkts_model_op.predict(base_df)[0]
        except Exception as e:
            st.error(f"Error predicting bowling wickets per phase: {e}")
            pp_wkts_pred = mo_wkts_pred = do_wkts_pred = None

        # Estimate opponent phase par/runs (best-effort)
        if pp_hist is not None and mo_hist is not None and do_hist is not None:
            pp_par = int(pp_hist.median())
            mo_par = int(mo_hist.median())
            do_par = int(do_hist.median())
        elif fallback_total is not None:
            median_total = int(fallback_total.median())
            # distribute as typical first innings proportions (approx): PP ~28%, MO ~45%, DO ~27%
            pp_par = int(round(median_total * 0.28))
            mo_par = int(round(median_total * 0.45))
            do_par = median_total - (pp_par + mo_par)
        else:
            # hard defaults (reasonable for T20)
            pp_par, mo_par, do_par = 45, 80, 45

        # Show predictions & targets for bowling first
        st.markdown(f"- **Estimated wickets SL can take**: Powerplay: **{pp_wkts_pred}**, Middle: **{mo_wkts_pred}**, Death: **{do_wkts_pred}**")
        st.markdown(f"- **Par runs to restrict opponent to (historical median)**: Powerplay: **{pp_par}**, Middle: **{mo_par}**, Death: **{do_par}**")
        st.markdown("**Suggested Bowling Goals (per phase)**:")

        # Powerplay guidance
        if pp_wkts_pred is not None:
            st.markdown(f"- **Powerplay (1-6):** Aim to take **{int(pp_wkts_pred)}** wickets and limit to **≤{pp_par}** runs. Use 2 aggressive pacers, one swinger early. Field: two slips + gully if swing.")
        else:
            st.markdown(f"- **Powerplay (1-6):** Limit to ≤{pp_par} runs. Use opening pace bowlers to attack the stumps.")

        # Middle overs guidance
        if mo_wkts_pred is not None:
            st.markdown(f"- **Middle Overs (7-15):** Aim to take **{int(mo_wkts_pred)}** wickets. Use spin partnership, attack with shorter boundary ropes where possible.")
        else:
            st.markdown("- **Middle Overs (7-15):** Use spinners and attacking fields to choke runs and take wickets.")

        # Death overs guidance
        if do_wkts_pred is not None:
            st.markdown(f"- **Death Overs (16-20):** Aim for **{int(do_wkts_pred)}** wickets; bowl yorkers and slower balls to prevent boundaries.")
        else:
            st.markdown("- **Death Overs (16-20):** Yorkers + slower balls; protect straight boundaries.")

        # Fielding visualization for bowling-first
        for phase in ["powerplay", "middle", "death"]:
            strategy, field_positions = generate_field_setting(base_df, phase=phase, bowling=True)
            st.subheader(f" {phase.capitalize()} Fielding Strategy (Bowling First)")
            fig = plot_fielding_positions(field_positions, title=f"{phase.capitalize()} Bowling Setup")
            st.pyplot(fig)

    # ------------------------------
    # Defend-the-target logic (SL batted first and now bowling)
    # When SL batted first (i.e., opponent_score > 0 but SL batted first)
    # We'll detect this when opponent_score > 0 and toss decision indicates SL batted first OR user lost toss but SL batted first - here we assume the UI only collects opponent_score.
    # We'll present recommended bowling strategy to defend target_score (which is opponent_score + 1 only for chasing; here SL batted first so target = SL_score, but user provided opponent_score= opponent first-innings)
    # For defend case we need to detect: user earlier entered SL first-innings score (not available)
    # Instead, provide defend guidance when opponent_score == 0 but user can input SL first innings in a separate input. To keep simple we provide defend guidance when user indicates "SL batted first" via a checkbox.
    # ------------------------------
    sl_batted_first = st.sidebar.checkbox("SL batted first / want defend plan", value=False)
    if sl_batted_first:
        st.subheader("🟩 Defend-the-target — Bowling Strategy to defend your total")
        # We need SL total to compute par. Ask user for SL total if they selected defend plan
        sl_total = st.sidebar.number_input("Enter SL's first-innings total (if defending)", min_value=0, max_value=300, step=1, value=0)
        if sl_total <= 0:
            st.info("Enter SL first-innings total to get defend-phase targets.")
        else:
            # Opponent target = sl_total + 1
            opp_target = sl_total + 1
            # Suggest per-phase maximum allowed to defend
            # If historical medians available, scale them to SL_total proportionally
            if pp_hist is not None and mo_hist is not None and do_hist is not None:
                total_med = (pp_hist.median() + mo_hist.median() + do_hist.median())
                if total_med > 0:
                    pp_allow = int(round((pp_hist.median() / total_med) * opp_target))
                    mo_allow = int(round((mo_hist.median() / total_med) * opp_target))
                    do_allow = opp_target - (pp_allow + mo_allow)
                else:
                    pp_allow, mo_allow, do_allow = int(opp_target*0.28), int(opp_target*0.45), int(opp_target*0.27)
            else:
                pp_allow, mo_allow, do_allow = int(opp_target*0.28), int(opp_target*0.45), int(opp_target*0.27)

            st.markdown(f"- **You must restrict opponent to total ≤ {opp_target-1}** (target to defend: {sl_total})")
            st.markdown(f"- **Phase caps to defend**: Powerplay ≤ **{pp_allow}**, Middle ≤ **{mo_allow}**, Death ≤ **{do_allow}**")
            st.markdown("**Bowling Plan to Defend:**")
            st.markdown("- Powerplay: Attack with new ball, aim to take 1-2 early wickets. Use slips/gully if swing.")
            st.markdown("- Middle Overs: Use spinners to control runs; target weak middle-order batsmen.")
            st.markdown("- Death Overs: Use your best death-overs bowlers, minimize boundary balls; vary pace and bowl yorkers.")

            # fielding visuals
            for phase in ["powerplay", "middle", "death"]:
                strategy, field_positions = generate_field_setting(base_df, phase=phase, bowling=True)
                st.subheader(f" {phase.capitalize()} Fielding Strategy (Defend Plan)")
                fig = plot_fielding_positions(field_positions, title=f"{phase.capitalize()} Defend Setup")
                st.pyplot(fig)

    # ------------------------------
    # Existing batting-first / chase logic (unchanged)
    # ------------------------------
    st.subheader("📊 Batting / Phase-wise Game Plan (if SL batting)")

    # Helper to show ML predicted batting phase targets (batting first)
    def show_batting_phase(name, model, run_model, wkt_model):
        try:
            strategy = model.predict(base_df)[0]
            runs = run_model.predict(base_df)[0]
            wkts = wkt_model.predict(base_df)[0]
        except Exception as e:
            st.error(f"Error predicting {name}: {e}")
            strategy, runs, wkts = "N/A", 0, 0
        st.markdown(f"### {name}")
        st.markdown(f"- **Target**: `{int(runs)} runs`, `{int(wkts)} wickets`")
        st.markdown(f"- **Strategy**: <span style='color:green'>{strategy}</span>", unsafe_allow_html=True)

    # If opponent_score > 0 then we are in chase mode (opponent batted first)
    if opponent_score > 0:
        target_score = opponent_score + 1
        required_rr = target_score / 20
        req_pp = round(required_rr * 6)
        req_mo = round(required_rr * 9)
        req_do = target_score - (req_pp + req_mo)

        st.subheader(" Advanced Chasing Strategy")
        st.markdown(f"- Target Score: **{target_score}**")
        st.markdown(f"- Required Run Rate: **{required_rr:.2f} runs/over**")
        st.markdown(f"- Powerplay Target: **{req_pp} runs**")
        st.markdown(f"- Middle Overs Target: **{req_mo} runs**")
        st.markdown(f"- Death Overs Target: **{req_do} runs**")

        # brief tactical hints
        if target_score < 140:
            st.success("🟢 Low Score Scenario (<140)")
        elif 140 <= target_score <= 160:
            st.info("🔵 Moderate Chase (140–160)")
        elif 160 < target_score <= 180:
            st.warning("🟠 Challenging Target (160–180)")
        else:
            st.error("🔴 High-pressure Chase (>180)")

    else:
        show_batting_phase("Powerplay", powerplay_model, pp_runs_model, pp_wkts_model)
        show_batting_phase("Middle Overs", middle_model, mo_runs_model, mo_wkts_model)
        show_batting_phase("Death Overs", death_model, do_runs_model, do_wkts_model)

    # ------------------------------
    # Team Composition & Best XI (existing logic, kept minimal)
    # ------------------------------
    st.subheader(" Team Composition")

    try:
        raw_batters = batters_model.predict(base_df)[0]
        raw_fasts = fast_bowlers_model.predict(base_df)[0]
        raw_spins = spinners_model.predict(base_df)[0]
        raw_alls = allrounders_model.predict(base_df)[0]
    except Exception as e:
        st.error(f"Error predicting role counts: {e}")
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
        normalized_roles = {
            role: round((count / total_predicted) * 11) for role, count in role_preds.items()
        }
        while sum(normalized_roles.values()) != 11:
            diff = 11 - sum(normalized_roles.values())
            if diff > 0:
                normalized_roles[max(normalized_roles, key=normalized_roles.get)] += 1
            else:
                r = min(normalized_roles, key=normalized_roles.get)
                if normalized_roles[r] > 0:
                    normalized_roles[r] -= 1
        final_roles = normalized_roles

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"- Batters: <b>{final_roles['Batters']}</b>", unsafe_allow_html=True)
        st.markdown(f"- Spinners: <b>{final_roles['Spinners']}</b>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"- Fast Bowlers: <b>{final_roles['Fast Bowlers']}</b>", unsafe_allow_html=True)
        st.markdown(f"- All-Rounders: <b>{final_roles['All-Rounders']}</b>", unsafe_allow_html=True)

    # Best XI LP section (kept as before if players file exists)
    try:
        players_df = pd.read_excel("ProcessedPlayers.xlsx")
        players_df['runs'] = pd.to_numeric(players_df['runs'], errors='coerce').fillna(0)
        players_df['sr'] = pd.to_numeric(players_df['sr'], errors='coerce').fillna(0)
        players_df['wkts'] = pd.to_numeric(players_df['wkts'], errors='coerce').fillna(0)
        players_df['econ'] = pd.to_numeric(players_df['econ'], errors='coerce').fillna(0)
        players_df = players_df[players_df['role'].isin(['batter', 'fast bowler', 'spinner', 'allrounder'])]
        players_df['score'] = players_df['runs']*0.5 + players_df['sr']*0.3 + players_df['wkts']*10 - players_df['econ']*2

        prob = LpProblem("Best_XI_Selection", LpMaximize)
        player_vars = {row['player']: LpVariable(row['player'], cat='Binary') for _, row in players_df.iterrows()}
        prob += lpSum(player_vars[player] * players_df.loc[players_df['player'] == player, 'score'].values[0] for player in player_vars)
        prob += lpSum(player_vars.values()) == 11

        wicketkeepers = players_df[(players_df.get('is_wk', 0) == 1) & (players_df['role'] == 'batter')]['player'].tolist()
        if wicketkeepers:
            prob += lpSum([player_vars[p] for p in wicketkeepers]) == 1
        else:
            st.warning("No wicketkeepers flagged in the players file; LP will select best available batters as fallback.")

        # role constraints
        role_map = {'Batters': 'batter', 'Fast Bowlers': 'fast bowler', 'Spinners': 'spinner', 'All-Rounders': 'allrounder'}
        for role_name, count in final_roles.items():
            role_players = [p for p in player_vars if players_df.loc[players_df['player'] == p, 'role'].values[0] == role_map[role_name]]
            if role_players:
                prob += lpSum([player_vars[p] for p in role_players]) == count

        prob.solve()
        selected_players = [p for p in player_vars if player_vars[p].varValue == 1.0]
        st.subheader(" Best XI")
        if selected_players:
            selected_df = players_df[players_df['player'].isin(selected_players)].copy()
            selected_df['role_display'] = selected_df.apply(lambda x: f"{x['role']} (WK)" if int(x.get('is_wk', 0)) == 1 else x['role'], axis=1)
            display_df = selected_df[['player', 'role_display']].reset_index(drop=True)
            display_df.index = display_df.index + 1
            display_df.index.name = "No."
            st.table(display_df)
        else:
            st.info("LP solver did not select any players (check ProcessedPlayers.xlsx and role constraints).")
    except Exception as e:
        st.error(f"Error building Best XI: {e}")

    # Fielding strategy visuals (batting and bowling contexts)
    st.header("🏏 Phase-wise Fielding Strategy")
    for phase in ["powerplay", "middle", "death"]:
        # If bowling context use bowling=True else default
        bowling_flag = sl_bowling_first or sl_batted_first
        strategy, field_positions = generate_field_setting(base_df, phase=phase, bowling=bowling_flag)
        st.subheader(f" {phase.capitalize()} Fielding Strategy")
        fig = plot_fielding_positions(field_positions, title=f"{phase.capitalize()} Setup")
        st.pyplot(fig)

# Footer
st.markdown("---")
st.caption("""
Note: Bowling-first and defend strategies are produced using the wicket prediction models (trained on SL bowling data) plus historical phase medians where available. If something looks off, retrain the phase models with explicit bowling-target columns or provide per-phase opponent-run columns in the dataset.
""")
