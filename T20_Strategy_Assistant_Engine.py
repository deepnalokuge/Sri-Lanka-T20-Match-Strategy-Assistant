import streamlit as st
import pandas as pd
import joblib
from phase_fielding import generate_field_setting, plot_fielding_positions
from pulp import LpProblem, LpMaximize, LpVariable, lpSum


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

pp_conc_model = joblib.load("pp_conc_model.pkl")
mo_conc_model = joblib.load("mo_conc_model.pkl")
do_conc_model = joblib.load("do_conc_model.pkl")

pp_wkts_model = joblib.load("pp_wkts_model.pkl")
mo_wkts_model = joblib.load("mo_wkts_model.pkl")
do_wkts_model = joblib.load("do_wkts_model.pkl")

pp_wkts_lost_model = joblib.load("pp_wkts_lost_model.pkl")
mo_wkts_lost_model = joblib.load("mo_wkts_lost_model.pkl")
do_wkts_lost_model = joblib.load("do_wkts_lost_model.pkl")

pp_bowl_strategy_model = joblib.load("pp_bowl_strategy.pkl")
mo_bowl_strategy_model = joblib.load("mo_bowl_strategy.pkl")
do_bowl_strategy_model = joblib.load("do_bowl_strategy.pkl")

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
    "Opponent's 1st Innings Score** (leave 0 if not applicable)", min_value=0, max_value=300, step=1, value=0
)


# ================================
# Base Input
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


# ================================
# Main Button
# ================================
if st.button("🚀 Generate Match Strategy"):

    # ========================================
    # TOSS INPUT
    # ========================================
    if toss_result == "Won":
        toss_pred = toss_model.predict(base_df)[0]  # 1 Bat, 0 Bowl
        toss_decision_val = "bat" if toss_pred == 1 else "bowl"
        toss_str = "Bat First" if toss_pred == 1 else "Bowl First"
        toss_won_val = 1
    else:
        toss_won_val = 0
        toss_decision_val = "none"
        toss_str = "Decision by Opponent (Lost)"

    win_input = pd.DataFrame([{**base_input, "toss_won": toss_won_val, "toss_decision": toss_decision_val}])


    # ================================
    # Win Probability
    # ================================
    st.subheader(" Match Outcome Predictions")

    win_prob = win_model.predict_proba(win_input)[0][1] * 100
    st.metric("Win Probability (SL)", f"{win_prob:.1f}%")

    if toss_result == "Won":
        st.markdown(f" Toss Strategy: <span style='color:blue'>{toss_str}</span>", unsafe_allow_html=True)
    else:
        st.markdown(f" Toss Strategy: <span style='color:red'>{toss_str}</span>", unsafe_allow_html=True)


    # ================================
    # Phase-wise Game Plan
    # ================================
    st.subheader(" Phase-wise Game Plan")

    # 🔥 CONDITION ADDED: If opponent_score > 0 → DO NOT SHOW batting/bowling strategy
    if opponent_score == 0:

        # ----------------------------
        # Batting Strategy
        # ----------------------------
        show_batting = False
        if toss_result == "Won" and toss_pred == 1:
            show_batting = True
        if toss_result == "Lost":
            show_batting = True

        def show_phase(name, model, run_model, wkts_lost_model):
            strategy = model.predict(base_df)[0]
            runs = run_model.predict(base_df)[0]
            wkts_lost = wkts_lost_model.predict(base_df)[0]

            st.markdown(f"### {name}")
            st.markdown(f"- **Expected Runs to Score**: `{int(runs)} runs`")
            st.markdown(f"- **Expected Wickets Lost**: `{int(wkts_lost)} wickets`")
            st.markdown(f"- **Strategy:** <span style='color:green'>{strategy}</span>", unsafe_allow_html=True)

        if show_batting:
            st.subheader("🏏 Phase-wise Batting Strategy")
            show_phase("Powerplay", powerplay_model, pp_runs_model, pp_wkts_lost_model)
            show_phase("Middle Overs", middle_model, mo_runs_model, mo_wkts_lost_model)
            show_phase("Death Overs", death_model, do_runs_model, do_wkts_lost_model)

        # ----------------------------
        # Bowling Strategy
        # ----------------------------
        if (toss_result == "Won" and toss_decision_val == "bowl") or (toss_result == "Lost"):

            st.subheader("⚾ Phase-wise Bowling Strategy")

            try:
                pp_wkts_b = pp_wkts_model.predict(base_df)[0]
                mo_wkts_b = mo_wkts_model.predict(base_df)[0]
                do_wkts_b = do_wkts_model.predict(base_df)[0]

                conc_pp = pp_conc_model.predict(base_df)[0]
                conc_mo = mo_conc_model.predict(base_df)[0]
                conc_do = do_conc_model.predict(base_df)[0]

                pp_bowl_str = pp_bowl_strategy_model.predict(base_df)[0]
                mo_bowl_str = mo_bowl_strategy_model.predict(base_df)[0]
                do_bowl_str = do_bowl_strategy_model.predict(base_df)[0]

            except Exception as e:
                st.error(f"❌ Bowling prediction failed: {e}")
                pp_wkts_b = mo_wkts_b = do_wkts_b = 0
                conc_pp = conc_mo = conc_do = 0
                pp_bowl_str = mo_bowl_str = do_bowl_str = "Unavailable"

            st.markdown("### Powerplay")
            st.markdown(f"- **Expected Wickets Taken:** `{int(pp_wkts_b)}`")
            st.markdown(f"- **Expected Runs Conceded:** `{int(conc_pp)} runs`")
            st.markdown(f"- **Strategy:** <span style='color:green'>{pp_bowl_str}</span>", unsafe_allow_html=True)

            st.markdown("### Middle Overs")
            st.markdown(f"- **Expected Wickets Taken:** `{int(mo_wkts_b)}`")
            st.markdown(f"- **Expected Runs Conceded:** `{int(conc_mo)} runs`")
            st.markdown(f"- **Strategy:** <span style='color:green'>{mo_bowl_str}</span>", unsafe_allow_html=True)

            st.markdown("### Death Overs")
            st.markdown(f"- **Expected Wickets Taken:** `{int(do_wkts_b)}`")
            st.markdown(f"- **Expected Runs Conceded:** `{int(conc_do)} runs`")
            st.markdown(f"- **Strategy:** <span style='color:green'>{do_bowl_str}</span>", unsafe_allow_html=True)


    # =====================================================
    # ADVANCED CHASING STRATEGY  (Shown ONLY IF score > 0)
    # =====================================================
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
            st.markdown("- Middle Overs: Rotate strike, build partnerships.")
            st.markdown("- Death Overs: Finish calmly, avoid risk.")
        elif 140 <= target_score <= 160:
            st.info("🔵 Moderate Chase (140–160)")
            st.markdown("- Powerplay: Utilize field restrictions.")
            st.markdown("- Middle Overs: Anchor with set batters.")
            st.markdown("- Death Overs: Controlled acceleration.")
        elif 160 < target_score <= 180:
            st.warning("🟠 Challenging Target (160–180)")
            st.markdown("- Powerplay: Positive intent.")
            st.markdown("- Middle Overs: Attack weaker bowlers.")
            st.markdown("- Death Overs: Power hitting needed.")
        else:
            st.error("🔴 High-pressure Chase (>180)")
            st.markdown("- Powerplay: Go hard early (50+ ideal).")
            st.markdown("- Middle Overs: Maintain 8–10 RPO.")
            st.markdown("- Death Overs: All-out aggression.")


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

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"- Batters: <b>{final_roles['Batters']}</b>", unsafe_allow_html=True)
        st.markdown(f"- Spinners: <b>{final_roles['Spinners']}</b>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"- Fast Bowlers: <b>{final_roles['Fast Bowlers']}</b>", unsafe_allow_html=True)
        st.markdown(f"- All-Rounders: <b>{final_roles['All-Rounders']}</b>", unsafe_allow_html=True)


    # ================================
    # Best XI Selection
    # ================================
    players_df = pd.read_excel("ProcessedPlayers.xlsx")

    players_df['runs'] = pd.to_numeric(players_df['runs'], errors='coerce')
    players_df['sr'] = pd.to_numeric(players_df['sr'], errors='coerce')
    players_df['wkts'] = pd.to_numeric(players_df['wkts'], errors='coerce')
    players_df['econ'] = pd.to_numeric(players_df['econ'], errors='coerce')
    players_df.dropna(subset=['runs', 'sr', 'wkts', 'econ'], inplace=True)

    players_df['score'] = (
        players_df['runs'] * 0.5 + players_df['sr'] * 0.3 +
        players_df['wkts'] * 10 + players_df['econ'] * -2
    )
    players_df = players_df[players_df['role'].isin(['batter', 'fast bowler', 'spinner', 'allrounder'])]

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
