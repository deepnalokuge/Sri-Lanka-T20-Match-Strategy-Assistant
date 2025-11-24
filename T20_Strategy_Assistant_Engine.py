import streamlit as st
import pandas as pd
import joblib
from phase_fielding import generate_field_setting, plot_fielding_positions
from pulp import LpProblem, LpMaximize, LpVariable, lpSum
import numpy as np
from math import ceil

# ================================
# Load Models
# ================================
win_model = joblib.load("win_predictor.pkl")
toss_model = joblib.load("toss_strategy.pkl")

powerplay_model = joblib.load("powerplay_model.pkl")
middle_model = joblib.load("middle_model.pkl")
death_model = joblib.load("death_model.pkl")

pp_runs_model = joblib.load("pp_runs_model.pkl")
mo_runs_model = joblib.load("mo_runs_model.pkl")
do_runs_model = joblib.load("do_runs_model.pkl")

pp_wkts_model = joblib.load("pp_wkts_model.pkl")
mo_wkts_model = joblib.load("mo_wkts_model.pkl")
do_wkts_model = joblib.load("do_wkts_model.pkl")

batters_model = joblib.load("batters_model.pkl")
fast_bowlers_model = joblib.load("fast_bowlers_model.pkl")
spinners_model = joblib.load("spinners_model.pkl")
allrounders_model = joblib.load("allrounders_model.pkl")


# ================================
# Load Data
# ================================
df = pd.read_excel("Cleaned_DataT20.xlsx")

opponent_teams = sorted(df["opponent_team"].drop_duplicates())
venues = sorted(df["venue"].drop_duplicates())
pitch_options = sorted(df["pitch_type"].str.strip().str.title().drop_duplicates())
ground_options = sorted(df["ground_dimension"].str.strip().str.title().drop_duplicates())
weather_types = sorted(df["weather_type"].drop_duplicates())

team_tiers = (
    df[["opponent_team", "opponent_tier"]]
    .drop_duplicates()
    .set_index("opponent_team")
    .to_dict()["opponent_tier"]
)

# -----------------------
# Load player pool early (moved up so bowling assignments can be suggested)
# -----------------------
players_df = pd.read_excel("ProcessedPlayers.xlsx")
players_df['runs'] = pd.to_numeric(players_df['runs'], errors='coerce')
players_df['sr'] = pd.to_numeric(players_df['sr'], errors='coerce')
players_df['wkts'] = pd.to_numeric(players_df['wkts'], errors='coerce')
players_df['econ'] = pd.to_numeric(players_df['econ'], errors='coerce')
players_df.dropna(subset=['runs', 'sr', 'wkts', 'econ', 'role'], inplace=True)

# compute simple skill score (same as original)
players_df['score'] = (
    players_df['runs'] * 0.5 + players_df['sr'] * 0.3 +
    players_df['wkts'] * 10 + players_df['econ'] * -2
)
players_df = players_df[players_df['role'].isin(['batter', 'fast bowler', 'spinner', 'allrounder'])]

# ================================
# Title
# ================================
st.markdown("<h1 style='text-align: center;'>🇱🇰🏏 Sri Lanka T20 Match Strategy Assistant</h1>", unsafe_allow_html=True)


# ================================
# Sidebar Inputs
# ================================
st.sidebar.header("Match Conditions")

opponent_team = st.sidebar.selectbox("Opponent Team", opponent_teams)
venue = st.sidebar.selectbox("Venue", venues)
pitch_type = st.sidebar.selectbox("Pitch Type", pitch_options)
ground_size = st.sidebar.selectbox("Ground Dimension", ground_options)
weather = st.sidebar.selectbox("Weather Type", weather_types)

opponent_tier = team_tiers.get(opponent_team, "Unknown")
st.sidebar.markdown(
    f"**Opponent Tier:** <span style='color: green;'>{opponent_tier}</span>",
    unsafe_allow_html=True
)

opponent_score = st.sidebar.number_input(
    "Opponent's 1st Innings Score", min_value=0, max_value=300, step=1, value=0
)


# ================================
# Base Input (WITHOUT toss details)
# ================================
base_input = {
    "opponent_team": opponent_team,
    "venue": venue,
    "pitch_type": pitch_type,
    "ground_dimension": ground_size,
    "weather_type": weather,
    "opponent_tier": opponent_tier
}

base_df = pd.DataFrame([base_input])


# ================================
# Toss Selection
# ================================
st.subheader("Toss Result")
toss_result = st.radio("Select Toss Result", ["Won", "Lost"], horizontal=True)


# -----------------------------
# --- Bowling Strategy Helpers
# -----------------------------
def predict_phase_stats_for_opponent(input_df):
    """Predict opponent phase runs and predicted wickets SL will take (models assumed to predict that)."""
    pp_runs = float(pp_runs_model.predict(input_df)[0])
    pp_wkts = float(pp_wkts_model.predict(input_df)[0])
    mo_runs = float(mo_runs_model.predict(input_df)[0])
    mo_wkts = float(mo_wkts_model.predict(input_df)[0])
    do_runs = float(do_runs_model.predict(input_df)[0])
    do_wkts = float(do_wkts_model.predict(input_df)[0])

    return {
        'PP': {'runs': pp_runs, 'wkts_taken_by_sl': pp_wkts},
        'MO': {'runs': mo_runs, 'wkts_taken_by_sl': mo_wkts},
        'DO': {'runs': do_runs, 'wkts_taken_by_sl': do_wkts}
    }


def select_phase_bowling_tactics(phase: str, wicket_goal: int, target_econ: float):
    if phase == 'PP':
        return (f"Attacking in the Powerplay: use 2 strike bowlers up front, set catching sweeper on the off-side and wide third-man. "
                f"Aim for {wicket_goal} wicket(s) and keep econ ≤ {target_econ}/over.")
    if phase == 'MO':
        return (f"Middle overs: mix spinners and change-of-pace seamers; look to choke partnerships with attacking fields for spinners. "
                f"Aim for {wicket_goal} wicket(s).")
    return (f"Death overs: use yorkers, slower full deliveries and boundary protection. Rotate your death specialists; "
            f"Aim for {wicket_goal} wicket(s) and keep death econ ≤ {target_econ}/over.")


def allocate_overs(players: list, overs: list):
    out = []
    for p, o in zip(players, overs):
        out.append({'name': p.get('player'), 'overs': o, 'bowling_type': p.get('role')})
    return out


def suggest_bowler_usage_from_pool(players_df_local, plan):
    """Heuristic assignment using players_df: prefer fast bowlers for PP/DO and spinners for MO if available."""
    bowlers = players_df_local[players_df_local['role'].isin(['fast bowler', 'spinner', 'allrounder'])].copy()
    # rank by score
    bowlers = bowlers.sort_values('score', ascending=False)
    pacers = bowlers[bowlers['role'] == 'fast bowler'].to_dict('records')
    spinners = bowlers[bowlers['role'] == 'spinner'].to_dict('records')
    allrs = bowlers[bowlers['role'] == 'allrounder'].to_dict('records')

    assignment = {'PP': [], 'MO': [], 'DO': []}

    # PP: 6 overs -> allocate 2,2,2
    pp_list = []
    if len(pacers) >= 2:
        pp_list = allocate_overs(pacers[:2], [2, 2])
        if spinners:
            pp_list += allocate_overs(spinners[:1], [2])
    else:
        # fallback to top three bowlers
        top = bowlers.head(3).to_dict('records')
        pp_overs = [2] * len(top)
        pp_list = allocate_overs(top, pp_overs)
    assignment['PP'] = pp_list

    # MO: 8 overs -> prefer spinner + change bowler
    mo_list = []
    if spinners:
        mo_list = allocate_overs(spinners[:1], [4])
    if len(pacers) >= 1:
        mo_list += allocate_overs(pacers[:1], [4])
    if not mo_list:
        top = bowlers.head(2).to_dict('records')
        mo_list = allocate_overs(top, [4, 4])
    assignment['MO'] = mo_list

    # DO: 6 overs -> death specialists
    do_list = []
    if len(pacers) >= 2:
        do_list = allocate_overs(pacers[:2], [3, 3])
    elif pacers:
        do_list = allocate_overs(pacers[:1], [4])
    else:
        top = bowlers.head(2).to_dict('records')
        do_list = allocate_overs(top, [3, 3])
    assignment['DO'] = do_list

    return assignment


def compute_bowling_targets(phase_preds, margin_pct=0.10):
    plan = {}
    for phase, v in phase_preds.items():
        runs_pred = v['runs']
        wkts_pred = v['wkts_taken_by_sl']
        target_runs = max(0, int(round(runs_pred * (1 - margin_pct))))
        wicket_goal = max(1, int(ceil(wkts_pred)))
        overs = 6 if phase == 'PP' else (8 if phase == 'MO' else 6)
        target_econ = round(target_runs / overs, 2)
        plan[phase] = {
            'pred_runs': round(runs_pred, 1),
            'pred_wkts_taken_by_sl': round(wkts_pred, 1),
            'target_runs': target_runs,
            'wicket_goal': wicket_goal,
            'target_economy': target_econ,
            'tactic': select_phase_bowling_tactics(phase, wicket_goal, target_econ)
        }
    return plan


def render_bowling_plan_ui(plan, bowler_usage):
    st.subheader('Bowling Strategy — If SL BOWLS First')
    for phase in ['PP', 'MO', 'DO']:
        with st.expander(f"{phase} — Target: {plan[phase]['target_runs']} runs, Wickets: {plan[phase]['wicket_goal']}"):
            st.write(f"Predicted opponent {phase} runs: {plan[phase]['pred_runs']} | Predicted wickets SL would take: {plan[phase]['pred_wkts_taken_by_sl']}")
            st.markdown(f"**Goal:** Keep opponent ≤ **{plan[phase]['target_runs']}** runs in {phase} (≈ econ {plan[phase]['target_economy']}/over).")
            st.markdown(f"**Wicket target:** Take **{plan[phase]['wicket_goal']}** wicket(s) in this phase.")
            st.markdown(f"**Tactics:** {plan[phase]['tactic']}")
            st.markdown("**Suggested bowlers & overs for phase:**")
            for b in bowler_usage[phase]:
                st.write(f"- {b['name']} — {b['overs']} over(s) — {b.get('bowling_type','')}")


def compute_defend_strategy(target_score, phase_preds):
    """Rescale predicted phase runs to target and compute defend goals."""
    total_pred = phase_preds['PP']['runs'] + phase_preds['MO']['runs'] + phase_preds['DO']['runs']
    if total_pred <= 0:
        total_pred = 1
    scale = target_score / total_pred
    rescaled = {}
    for phase in ['PP', 'MO', 'DO']:
        pred_runs = phase_preds[phase]['runs'] * scale
        goal_runs = max(0, int(np.floor(pred_runs * 0.9)))  # aim 10% below the chase path
        wicket_goal = max(1, int(ceil(phase_preds[phase]['wkts_taken_by_sl'])))
        overs = 6 if phase == 'PP' else (8 if phase == 'MO' else 6)
        rescaled[phase] = {
            'required_by_chasing_team': round(pred_runs, 1),
            'target_runs_to_allow': goal_runs,
            'wicket_goal': wicket_goal,
            'target_econ': round(goal_runs / overs, 2)
        }
    return rescaled


def render_defend_ui(rescaled_plan, bowler_usage, target_score):
    st.subheader(f'Bowling Strategy — Defend {target_score}')
    for phase in ['PP', 'MO', 'DO']:
        with st.expander(f"{phase} — Allow ≤ {rescaled_plan[phase]['target_runs_to_allow']} (Goal wickets: {rescaled_plan[phase]['wicket_goal']})"):
            st.write(f"Opposition likely to score this phase (chase path): {rescaled_plan[phase]['required_by_chasing_team']}")
            st.markdown(f"**Goal:** Concede ≤ **{rescaled_plan[phase]['target_runs_to_allow']}** in {phase}.")
            st.markdown(f"**Wicket target:** Take **{rescaled_plan[phase]['wicket_goal']}** wicket(s) here to apply pressure.")
            st.markdown("**Suggested bowlers:**")
            for b in bowler_usage[phase]:
                st.write(f"- {b['name']} — {b['overs']} over(s) — {b.get('bowling_type','')}")


# ================================
# Main Button
# ================================
if st.button("🚀 Generate Match Strategy"):

    # ========================================
    # TOSS-DEPENDENT WIN INPUT (FIXED ✔)
    # ========================================
    if toss_result == "Won":
        toss_pred = toss_model.predict(base_df)[0]  # 1 = Bat, 0 = Bowl
        toss_decision_val = "bat" if toss_pred == 1 else "bowl"
        toss_str = "Bat First" if toss_pred == 1 else "Bowl First"
        toss_won_val = 1
    else:
        toss_won_val = 0
        toss_decision_val = "none"
        toss_str = "Decision by Opponent (Lost)"

    # Build final win input
    win_input = pd.DataFrame([{
        **base_input,
        "toss_won": toss_won_val,
        "toss_decision": toss_decision_val
    }])

    # ================================
    # Match Outcome Predictions
    # ================================
    st.subheader(" Match Outcome Predictions")

    win_prob = win_model.predict_proba(win_input)[0][1] * 100
    st.metric("Win Probability (SL)", f"{win_prob:.1f}%")

    if toss_result == "Won":
        st.markdown(f" Toss Strategy: <span style='color:blue'>{toss_str}</span>", unsafe_allow_html=True)
    else:
        st.markdown(f" Toss Strategy: <span style='color:red'>{toss_str}</span>", unsafe_allow_html=True)

    st.subheader("📊 Phase-wise Game Plan")


    # Helper function
    def show_phase(name, model, run_model, wkt_model):
        strategy = model.predict(base_df)[0]
        runs = run_model.predict(base_df)[0]
        wkts = wkt_model.predict(base_df)[0]

        st.markdown(f"### {name}")
        st.markdown(f"- **Target**: `{int(runs)} runs`, `{int(wkts)} wickets`")
        st.markdown(f"- **Strategy**: <span style='color:green'>{strategy}</span>", unsafe_allow_html=True)


    # ========================================
    # Advanced Chase Mode (FIXED +1)
    # ========================================
    if opponent_score > 0:

        target_score = opponent_score + 1  # FIXED ✔
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

        # Chase difficulty
        if target_score < 140:
            st.success("🟢 Low Score Scenario (<140)")
            st.markdown("- Powerplay: Keep wickets, take only calculated risks.")
            st.markdown("- Middle Overs: Rotate strike, build partnerships.")
            st.markdown("- Death Overs: Finish calmly, avoid risky shots.")
        elif 140 <= target_score <= 160:
            st.info("🔵 Moderate Chase (140–160)")
            st.markdown("- Powerplay: Utilize field restrictions, brisk start.")
            st.markdown("- Middle Overs: Anchor with set batters, keep RRR in check.")
            st.markdown("- Death Overs: Accelerate if needed, controlled aggression.")
        elif 160 < target_score <= 180:
            st.warning("🟠 Challenging Target (160–180)")
            st.markdown("- Powerplay: Positive intent, target weaker bowlers.")
            st.markdown("- Middle Overs: Attack spinners/part-timers, maintain 7–9 RPO.")
            st.markdown("- Death Overs: Need set finisher, boundary hitting is key.")
        else:
            st.error("🔴 High-pressure Chase (>180)")
            st.markdown("- Powerplay: Go hard early, 50+ ideal.")
            st.markdown("- Middle Overs: Maintain 8–10 RPO.")
            st.markdown("- Death Overs: Full aggression, strong finisher needed.")

    else:
        show_phase("Powerplay", powerplay_model, pp_runs_model, pp_wkts_model)
        show_phase("Middle Overs", middle_model, mo_runs_model, mo_wkts_model)
        show_phase("Death Overs", death_model, do_runs_model, do_wkts_model)


    # ================================
    # NEW: Bowling Strategy Section (Inserted RIGHT AFTER phase-wise game plan)
    # ================================
    st.markdown("---")
    st.header("🎯 Bowling Strategy (automatically generated)")

    # Predict opponent phase stats (runs + expected wickets SL will take)
    phase_preds = predict_phase_stats_for_opponent(base_df)

    # Aggressiveness slider
    margin = st.slider('Aggressiveness (aim % below predicted opponent runs)', min_value=0.0, max_value=0.30, value=0.10, step=0.01)

    # If toss decision or user wants manual mode: determine whether we should show bowling-first plan or defend plan
    # Show both options depending on game flow:
    # - If toss_result == "Won" and toss_pred == 0 (bowl), then SL bowls first -> show Bowling-first plan
    # - If opponent_score > 0 -> SL batted first and we must defend -> show defend plan
    # - Also allow user to force view via a selectbox
    bowling_view = st.selectbox("Select bowling scenario to view", ["Auto (recommended)", "SL bowls first", "SL defends a target"])

    # Auto decision
    if bowling_view == "Auto (recommended)":
        if toss_result == "Won" and toss_decision_val == "bowl":
            chosen_view = "SL bowls first"
        elif opponent_score > 0:
            chosen_view = "SL defends a target"
        else:
            # default to bowls first view so we present proactive bowling plan
            chosen_view = "SL bowls first"
    else:
        chosen_view = bowling_view

    # Compute and render chosen bowling plan
    if chosen_view == "SL bowls first":
        plan = compute_bowling_targets(phase_preds, margin_pct=margin)
        # suggest bowlers from players pool (players_df loaded earlier)
        bowler_usage = suggest_bowler_usage_from_pool(players_df, plan)
        render_bowling_plan_ui(plan, bowler_usage)

    else:  # SL defends a target
        if opponent_score == 0:
            st.info("Set 'Opponent's 1st Innings Score' in the sidebar to generate an exact defend plan.")
            # still show a guideline plan using predicted total scaled to an assumed target (150)
            assumed_target = 150
            st.caption(f"Showing example defend plan against an assumed target of {assumed_target}.")
            rescaled_plan = compute_defend_strategy(assumed_target, phase_preds)
            bowler_usage = suggest_bowler_usage_from_pool(players_df, rescaled_plan)
            render_defend_ui(rescaled_plan, bowler_usage, assumed_target)
        else:
            target_score = opponent_score + 1
            rescaled_plan = compute_defend_strategy(target_score, phase_preds)
            bowler_usage = suggest_bowler_usage_from_pool(players_df, rescaled_plan)
            render_defend_ui(rescaled_plan, bowler_usage, target_score)

    st.markdown("---")
    st.info("Bowling strategy computed from your phase run/wicket models. Tweak aggressiveness slider to adjust targets.")

    # ================================
    # TEAM COMPOSITION
    # ================================
    st.subheader(" Team Composition")

    raw_batters = batters_model.predict(base_df)[0]
    raw_fasts = fast_bowlers_model.predict(base_df)[0]
    raw_spins = spinners_model.predict(base_df)[0]
    raw_alls = allrounders_model.predict(base_df)[0]

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

    # Display
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"- Batters: <b>{final_roles['Batters']}</b>", unsafe_allow_html=True)
        st.markdown(f"- Spinners: <b>{final_roles['Spinners']}</b>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"- Fast Bowlers: <b>{final_roles['Fast Bowlers']}</b>", unsafe_allow_html=True)
        st.markdown(f"- All-Rounders: <b>{final_roles['All-Rounders']}</b>", unsafe_allow_html=True)


    # ================================
    # Team Selection (LP Optimization)
    # ================================
    prob = LpProblem("Best_XI_Selection", LpMaximize)
    player_vars = {row['player']: LpVariable(row['player'], cat='Binary') for _, row in players_df.iterrows()}

    prob += lpSum(player_vars[player] * players_df.loc[players_df['player'] == player, 'score'].values[0]
                  for player in player_vars)
    prob += lpSum(player_vars.values()) == 11

    wk_batters = players_df[(players_df['is_wk'] == 1) & (players_df['role'] == 'batter')]['player'].tolist()
    if wk_batters:
        prob += lpSum([player_vars[p] for p in wk_batters]) == 1
    else:
        st.error(" No wicketkeepers available in the batter list!")

    for role_name, count in final_roles.items():
        role_map = {
            'Batters': 'batter',
            'Fast Bowlers': 'fast bowler',
            'Spinners': 'spinner',
            'All-Rounders': 'allrounder'
        }
        role_players = [player for player in player_vars
                        if players_df.loc[players_df['player'] == player, 'role'].values[0] == role_map[role_name]]
        if role_players:
            prob += lpSum([player_vars[player] for player in role_players]) == count

    prob.solve()

    selected_players = [player for player in player_vars if player_vars[player].varValue == 1.0]

    st.subheader(" Best XI")
    if selected_players:
        selected_df = players_df[players_df['player'].isin(selected_players)].copy()
        selected_df['role_display'] = selected_df.apply(
            lambda x: f"{x['role']} (WK)" if int(x.get('is_wk', 0)) == 1 else x['role'],
            axis=1
        )
        display_df = selected_df[['player', 'role_display']].reset_index(drop=True)
        display_df.index = display_df.index + 1
        display_df.index.name = "No."
        st.table(display_df)


    # ================================
    # FIELDING STRATEGY
    # ================================
    st.header("🏏 Phase-wise Fielding Strategy")

    for phase in ["powerplay", "middle", "death"]:
        strategy, field_positions = generate_field_setting(base_df, phase)
        st.subheader(f" {phase.capitalize()} Fielding Strategy")
        fig = plot_fielding_positions(field_positions, title=f"{phase.capitalize()} Setup")
        st.pyplot(fig)


# ================================
# Footer
# ================================
st.markdown("---")
st.caption("""
**Note**: This predictive model is based on historical T20 match data.
Actual match outcomes may vary due to real-time conditions.
""")
