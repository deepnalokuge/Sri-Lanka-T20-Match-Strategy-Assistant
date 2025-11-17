import numpy as np
import matplotlib.pyplot as plt

def generate_field_setting(input_df, phase='powerplay'):
    pitch = input_df['pitch_type'].iloc[0].lower()
    weather = input_df['weather_type'].iloc[0].lower()
    opponent = input_df['opponent_team'].iloc[0].lower()
    strategy = 'balanced'

    if phase == 'powerplay':
        if 'pace' in pitch:
            strategy = 'aggressive'
            positions = [
                "slip", "slip", "gully", "point", "cover", "mid-off", "mid-on", "fine leg", "third man", "bowler", "keeper"
            ]
        elif 'spin' in pitch:
            strategy = 'contain_spin'
            positions = [
                "slip", "point", "cover", "midwicket", "mid-on", "mid-off", "deep square leg", "short fine leg", "deep point", "bowler", "keeper"
            ]
        else:
            strategy = 'balanced'
            positions = [
                "slip", "point", "cover", "mid-off", "mid-on", "square leg", "fine leg", "third man", "deep point", "bowler", "keeper"
            ]

    elif phase == 'middle':
        if 'spin' in pitch:
            strategy = 'rotate_attack'
            positions = [
                "slip", "cover", "extra cover", "midwicket", "mid-off", "mid-on", "deep square leg", "deep midwicket", "deep point", "bowler", "keeper"
            ]
        else:
            strategy = 'contain'
            positions = [
                "point", "cover", "mid-off", "mid-on", "deep extra cover", "deep midwicket", "deep square leg", "third man", "fine leg", "bowler", "keeper"
            ]

    elif phase == 'death':
        if 'batting' in pitch:
            strategy = 'deep_defense'
            positions = [
                "third man", "fine leg", "deep midwicket", "deep extra cover", "long off", "long on", "deep square leg", "point", "cover", "bowler", "keeper"
            ]
        else:
            strategy = 'controlled'
            positions = [
                "long off", "long on", "deep midwicket", "deep square leg", "third man", "fine leg", "cover", "point", "mid-off", "bowler", "keeper"
            ]
    else:
        strategy = 'balanced'
        positions = ["point", "cover", "mid-off", "mid-on", "deep square leg", "fine leg", "third man", "deep point", "midwicket", "bowler", "keeper"]

    return strategy, positions

def plot_fielding_positions(positions, title="Fielding Positions"):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-60, 60)
    ax.set_ylim(-60, 60)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("lightgreen")
    circle = plt.Circle((0, 0), 55, color='white', fill=False, linestyle='--')
    ax.add_artist(circle)

    angles = [i * (360 / len(positions)) for i in range(len(positions))]
    for i, pos in enumerate(positions):
        angle = angles[i]
        x = 40 * round(np.cos(np.radians(angle)), 2)
        y = 40 * round(np.sin(np.radians(angle)), 2)
        ax.text(x, y, pos, ha='center', va='center', fontsize=8, bbox=dict(facecolor='white', alpha=0.6, edgecolor='black'))

    return fig
