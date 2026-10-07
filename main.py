import random as rnd
import time, msvcrt, re
import math as mth
import numpy as np
import math_matrix as mth_mtx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import multiprocessing as mp

def plot_tournament_results(hasil_urut, strategies_category, strategies_creator,
                             tournament_divider, time_divider, top_n=30, show_worst=False):
    """
    hasil_urut          : list hasil sorted [(score, name, win, draw, lose), ...]
    strategies_category  : dict nama -> 'NICE'/'NASTY'
    strategies_creator    : dict nama -> nama pencipta
    tournament_divider   : pembagi untuk skor rata-rata
    time_divider         : pembagi untuk win/draw/lose rata-rata
    top_n             : jumlah strategi teratas yang ditampilkan
    show_worst        : True -> ambil dari bawah (strategi terburuk) alih-alih teratas
    """
    data = hasil_urut[-top_n:][::-1] if show_worst else hasil_urut[:top_n]

    names  = [item[1] for item in data]
    scores = [item[0] / tournament_divider for item in data]
    wins   = [item[2] / time_divider for item in data]
    draws  = [item[3] / time_divider for item in data]
    loses  = [item[4] / time_divider for item in data]
    colors = ['#2ecc71' if strategies_category[n] == 'NICE' else '#e74c3c' for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(names) * 0.32)))
    y_pos = np.arange(len(names))
    title_prefix = 'Terburuk' if show_worst else 'Teratas'

    bars_left = axes[0].barh(y_pos, scores, color=colors)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels([])
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Rata-rata skor per turnamen')
    axes[0].set_title(f'{title_prefix} {len(names)} Strategi berdasarkan Skor')

    max_score = max(scores) if scores else 1
    for i, (bar, v, name) in enumerate(zip(bars_left, scores, names)):

        axes[0].text(max_score * 0.01, bar.get_y() + bar.get_height() / 2,
                     name, va='center', ha='left', fontsize=7, color='white',
                     fontweight='bold', clip_on=True)

        axes[0].text(v, bar.get_y() + bar.get_height() / 2, f' {v:.1f}',
                     va='center', ha='left', fontsize=7, color='black')

    nice_patch = mpatches.Patch(color='#2ecc71', label='NICE')
    nasty_patch = mpatches.Patch(color='#e74c3c', label='NASTY')
    axes[0].legend(handles=[nice_patch, nasty_patch], loc='lower right', fontsize=8)

    # === Panel kanan: stacked win/draw/loss ===
    bars_win  = axes[1].barh(y_pos, wins, color='#3498db', label='Menang')
    bars_draw = axes[1].barh(y_pos, draws, left=wins, color='#95a5a6', label='Seri')
    bars_lose = axes[1].barh(y_pos, loses, left=[w + d for w, d in zip(wins, draws)],
                              color='#c0392b', label='Kalah')
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels([])
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Rata-rata jumlah pertandingan per turnamen')
    axes[1].set_title('Win / Draw / Loss')
    axes[1].legend(loc='lower right', fontsize=8)

    annot_left = axes[0].annotate("", xy=(0, 0), xytext=(15, 15), textcoords="offset points",
                                   bbox=dict(boxstyle="round", fc="w", ec="black"),
                                   fontsize=8, visible=False)
    annot_right = axes[1].annotate("", xy=(0, 0), xytext=(15, 15), textcoords="offset points",
                                    bbox=dict(boxstyle="round", fc="w", ec="black"),
                                    fontsize=8, visible=False)

    all_bar_groups = {
        axes[0]: (bars_left, annot_left),
        axes[1]: (list(bars_win) + list(bars_draw) + list(bars_lose), annot_right),
    }

    def on_move(event):
        redraw = False
        for ax, (bars, annot) in all_bar_groups.items():
            if event.inaxes != ax:
                if annot.get_visible():
                    annot.set_visible(False)
                    redraw = True
                continue
            found = False
            for idx in range(len(names)):
                # cek batang yang sesuai baris ini (untuk panel kanan, bar index perlu di-mod)
                bar = bars[idx % len(names)] if ax == axes[1] else bars[idx]
                contains, _ = bar.contains(event)
                if contains:
                    creator = strategies_creator.get(names[idx], 'Unknown')
                    annot.xy = (event.xdata, event.ydata)
                    annot.set_text(f"{names[idx]}\nPencipta: {creator}")
                    annot.set_visible(True)
                    found = True
                    redraw = True
                    break
            if not found and annot.get_visible():
                annot.set_visible(False)
                redraw = True
        if redraw:
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)

    plt.tight_layout()
    plt.savefig('tournament_result.png', dpi=150)
    plt.show()

def timeout_input_instant(detik):
    start = time.time()
    while time.time() - start < detik:
        if msvcrt.kbhit():
            char = msvcrt.getch().decode()
            return char
    return None

def many_name_strategy_function(strategy):
    strategy_code_name = strategy
    if strategy_code_name not in strategies:
        for i in range(len(strategys_many_name)):
            if strategy_code_name in strategys_many_name[i].split(' / '):
                return strategys_many_name[i]
    return strategy

def update_score(x, y):
    global score
    score[0] += x

    score[1] += y

def reset_think_memory():
    return [
            0, # 0
            0, # 1
            ["Normal",0,0,None,False,0.25,np.zeros([4, 2]),0,3,False,0,[0,0],0,5,0,0], # 2
            [0,0], # 3
            0, # 4
            [0,0,0], # 5
            [0.5,0], # 6
            0, # 7
            [0,0], # 8
            [0,0.5], # 9
            0, # 10
            [0,0,0], # 11
            [0,0,0], # 12
            [0.5,0], # 13
            {"mode": "TFT", "matrix": np.zeros((2, 2))}, # 14
            [0, 0, 0, 0, 0], # 15
            [0,0,0,0], # 16
            0, # 17
            0, # 18
            0, # 19
            [[0,0,0,0,1,0],[0,0,0,0,0,0],0], # 20
            [[30,0],[30,0],[0,30],[0,30],0], # 21
            [0,0,0,0], # 22
            [[0,0,0,0],[0,0,0,0],[0,0,0,0]], # 23
            [0.1,0,0,0], # 24
            0, # 25
            [0,0], # 26
            [0,0], # 27
            '2', # 28
            0, # 29
            0, # 30
            [
                {
                    "n": 155,
                    "min_n": 44,
                    "max_n": 302,
                    
                    # Matriks & Vector NumPy
                    "adj": np.random.uniform(-0.15, 0.15, (155, 155)),
                    "volt": np.zeros(155, dtype=np.float64),
                    "th": np.random.uniform(0.9, 1.1, 155),
                    "ref": np.zeros(155, dtype=np.int32),
                    "a": 0.5,
                    "energy": 1000.0,
                    "lactate": 0.0,

                    # Hormon & Neurotransmitter Biologi
                    "DA": 0.5, "CORT": 0.2, "SER": 0.5, "OXY": 0.5,
                    "ADR": 0.2, "NOR": 0.2, "TES": 0.3, "GAB": 0.5,
                    "END": 0.5, "ACH": 0.5, "VAS": 0.3, "ADE": 0.0,
                    "MEL": 0.0, "GLU": 0.7,

                    "day_tick": 0,
                    "is_sleeping": False,
                    "last_spike_time":np.full(155, -np.inf),

                    "decay factor": [0.75 ** i for i in range(10)],
                    "lead": 0
                }
            ], # 31
            0, # 32
            [0,0,0], # 33
            [
                {
                    ("CCCCC","CCCCC"):["CCCCC",100]
                },
                [0,("CCCCC","CCCCC"),0]
            ], # 34
            [False, 0, 0, [0] * 12, 0], # 35
            [[1,1,1,1,1],[0,0,0,0,0]], # 36
            0, # 37
            [
                {
                    ("CCCCC", "CCCCC"):1
                },
            [None,0]
            ], # 38
            [0,0,0.5,0,0,0], # 39
            [0,0.5,0], # 40
            [0,0,0.5], # 41
            0, # 42
            [
                [
                    {
                        ("C","C"):"C"
                    } for _ in range(64)
                ],
            [rnd.randint(0,8) for _ in range(64)],
            [rnd.randint(0,8) for _ in range(64)],
            [0 for _ in range(64)],
            [0 for _ in range(64)],
            [0 for _ in range(64)]
            ], # 43
            [
                {
                    ("C","C"):"C"
                },
                [
                    {
                        ("C","C"):"C"
                    },
                    {
                        ("C","C"):"C"
                    }
                ],
            [None,None,None]
            ], # 44
            [0,0,0,0], # 45
            [
                {
                    ("C","C"):[1,0]
                },
            [[None, 0.5],0.1]
            ], # 46
            0, # 47
            0, # 48
            0, # 49
            0, # 50
            [0,0], # 51
            0, # 52
            [0,0], # 53
            [0,0], # 54
            [0,0], # 55
            0, # 56
            [0,0,0,0,0], # 57
            [0,0,0,0,0], # 58
            [0,0,0,0,0], # 59
            [0,0,0,0,0], # 60
            [0,0,0,0,0], # 61
            [0,0,0,0,0], # 62
            0, # 63
            0, # 64
            [
                {
                    "n":24,
                    "input old val":[0, 0, 0, 0],
                    "input weight":[[rnd.uniform(-1,1) for _ in range(4)] for _ in range(24)],
                    "recurrent weight":[rnd.uniform(-1,1) for _ in range(24)],
                    "bias":[rnd.uniform(-1,1) for _ in range(24)],
                    "n val":[0 for _ in range(24)],
                    "n real val":[0 for _ in range(24)],
                    "n old val":[0 for _ in range(24)],
                    "n old real val":[0 for _ in range(24)],
                    "output weight":[[rnd.uniform(-1,1) for _ in range(24)], [rnd.uniform(-1,1) for _ in range(24)]],
                    "output bias":[rnd.uniform(-1,1), rnd.uniform(-1,1)],
                    "output val":[0.5, 0.5],
                    "output old val":[0.5, 0.5],
                    'scalar trace wi':[[0 for _ in range(4)] for _ in range(24)],
                    'scalar trace wr':[0 for _ in range(24)],
                    'scalar trace wo':[[0 for _ in range(24)], [0 for _ in range(24)]],
                    'old scalar trace wi':[[0 for _ in range(4)] for _ in range(24)],
                    'old scalar trace wr':[0 for _ in range(24)],
                    'old scalar trace wo':[[0 for _ in range(24)], [0 for _ in range(24)]],
                    'G':[0 for _ in range(24)]
                }
            ], # 65
            [0,1], # 66
            0, # 67
            0, # 68
            0, # 69
            [
                {
                    ("C", "C"):1
                },
                None
            ], # 70
            [2,0], # 71
            [0,0], # 72
            [0,0], # 73
            0, # 74
            0, # 75
            0, # 76
            [0,0], # 77
            [0,0.5], # 78
            1, # 79
            [0,0,0], # 80
            0, # 81
            [False,0,0,0,False,0], # 82
            0, # 83
            [2,[1] * 4,0,21,2,float("inf"),2,2], # 84
            [0,0], # 85
            0, # 86
            0, # 87
            0, # 88
            [0,0,5,0,1,1,{'C':0,'D':1},0], # 89
            0, # 90
            [{'CC':[0,0], 'CD':[0,0], 'DC':[0,0], 'DD':[0,0]}, 'CC', 0], # 91
            0, # 92
            [0 for _ in range(7)], # 93
            0, # 94
            0, # 95
            0, # 96
            0, # 97
            0, # 98
            0, # 99
            [
                {
                    "CCCCC":["TTT",100],
                    "CCCCD":["TTT",100],
                    "CCCDC":["TTT",100],
                    "CCDCC":["TTT",100],
                    "CDCCC":["TTT",100],
                    "DCCCC":["TTT",100],
                },
                [0,"CCCCC",0]
            ], # 100
            [0,0,0], # 101
            [0,0,0], # 102
            [0,0,0,0], # 103
            0.5, # 104
            0, # 105
            [0,0], # 106
            0, # 107
            0, # 108
            [0,0], # 109
            0, # 110
            0, # 111
            0, # 112
            [0,0], # 113
            [1,0,0], # 114
            0, # 115
            [0,0,0], # 116
            0.5, # 117
            [0,0,0], # 118
            [
                {
                    "C":[1,0],
                    "D":[1,1]
                },
                "C"
            ], # 119
            1, # 120
            [0,0,0,0,False,"Normal"], # 121
            0, # 122
            0, # 123
            [0,0], # 124
            [
                {
                    ("C", "C", "C"):0,
                    ("C", "C", "D"):0,
                    ("C", "D", "C"):0,
                    ("C", "D", "D"):0,
                    ("D", "C", "C"):0,
                    ("D", "C", "D"):0,
                    ("D", "D", "C"):0,
                    ("D", "D", "D"):0
                },
                [],
                0
            ], # 125
            [0,0,0,0,0,0,0,0,{'CC':[0,0],'CD':[0,0],'DC':[0,0],'DD':[0,0]},0], # 126
            [0,0,0], # 127
            [1, False, 1, 0], # 128
            0, # 129
            0, # 130
            [False,[]], # 131
            [0,0], # 132
            0, # 133
            [0,0,0,0], # 134
            [0,0,0,0], # 135
            [0,0,0,0,0,0], # 136
            [
                [
                    {
                        # opp, self, my_sc > opp_sc, opp_sc > my_sc : prob to choose C
                        ("C", "C", "T", "F"):1,
                        ("D", "C", "T", "F"):1,
                        ("C", "D", "T", "F"):1,
                        ("D", "D", "T", "F"):1,

                        ("C", "C", "F", "F"):1,
                        ("D", "C", "F", "F"):0,
                        ("C", "D", "F", "F"):1,
                        ("D", "D", "F", "F"):0,

                        ("C", "C", "F", "T"):0,
                        ("D", "C", "F", "T"):0,
                        ("C", "D", "F", "T"):0,
                        ("D", "D", "F", "T"):0
                    }
                ],
                ["C"]
            ], # 137
            0, # 138
            [
                {
                    (("C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C"),("C","C", "C", "C", "C", "C", "C", "C", "C", "C", "C")):[0,0]
                },
            [[None, 0.5],0.1]
            ], # 139
            [
                {
                    (("C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C"),("C","C", "C", "C", "C", "C", "C", "C", "C", "C", "C")):[0,0]
                },
            [[None, 0.5],0.1]
            ], # 140
            [
                {
                    (("C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C"),("C","C", "C", "C", "C", "C", "C", "C", "C", "C", "C")):[0,0]
                },
            [[None, 0.5],0.1]
            ], # 141
            False, # 142
            ['0',1], # 143
            ["D",0,1,0], # 144
            [3,-10], # 145
            0, # 146
            [0,0], # 147
            0, # 148
            [0,0,0,0], # 149
            0, # 150
            0, # 151
            0, # 152
            [0,0], # 153
            [7,0], # 154
            ["Normal",0], # 155
            [{'C': 0, 'D': 1}, {'C': 0, 'D': 1}, False], # 156
            True, # 157
            0, # 158
            0, # 159
            [0,0], # 160
            0, # 161
            0, # 162
            [0,0], # 163
            [0, 'unknown'], # 164
            [0,0,0,0], # 165
            0, # 166
            [0,0], # 167
            [[0 for _ in range(15)], 0], # 168
            [[0 for _ in range(10)], 0], # 169
            0, # 170
            [0 for _ in range(40)], # 171
            0, # 172
            False, # 173
            0, # 174
            [{}], # 175
            False, # 176
            [
                0,
                {
                    # keyword : [list reaksi]
                    'wondering': [
                        'Why is [REPLACE] on your mind?',
                        'Hmm, [REPLACE] keeps crossing your mind, doesn\'t it?',
                        'Since when have you been thinking about [REPLACE]?'
                    ],
                    'hate': [
                        'Why do you hate [REPLACE]?',
                        'What makes you hate [REPLACE] so much? >:-(',
                        'Hate is a heavy thing, why [REPLACE]?'
                    ],
                    'love': [
                        'So, you love [REPLACE]?',
                        'Wow, in love with [REPLACE]? Tell me all about it! :-)',
                        'Love is beautiful, what does it feel like to love [REPLACE]?'
                    ],
                    
                    'like': [
                        'Okay, so you like [REPLACE]?',
                        'Cool! Liking [REPLACE] is fun!',
                        'Why do you like [REPLACE]?'
                    ],
                    'miss': [
                        'You miss [REPLACE], don\'t you? :-(',
                        'What is it like missing [REPLACE]?',
                        'How long have you been missing [REPLACE]?'
                    ],
                    'fear': [
                        'What makes you afraid of [REPLACE]?',
                        'It\'s perfectly normal to be afraid of [REPLACE]—want to tell me about it?',
                        'If [REPLACE] weren\'t scary, what would you do?'
                    ],
                    'sad': [
                        'Why are you sad about [REPLACE]? :-(',
                        'I\'m sorry to hear about [REPLACE]',
                        'When you\'re sad about [REPLACE], what usually makes you feel better?'
                    ],
                    'happy': [
                        'Wow, you\'re happy because of [REPLACE], right? I\'m happy for you! :-)',
                        'What makes you happiest about [REPLACE]?',
                        'Glad to hear you\'re happy about [REPLACE]!'
                    ],
                    'angry': [
                        'Are you angry because of [REPLACE]? >:-(',
                        'Being angry about [REPLACE] is exhausting, you know—how do you want to let it out?',
                        'Okay, take a deep breath first. Tell me slowly about being angry over [REPLACE].'
                    ],
                    'tired': [
                        'Tired because of [REPLACE], huh? You should take a break :-(',
                        'If you\'re tired because of [REPLACE], what makes you want to give up?',
                        'Being tired is a sign that you\'ve been fighting hard regarding [REPLACE]'
                    ],
                    'Alone': [
                        'Why do you feel alone about [REPLACE]?',
                        'Sometimes it\'s not good to be alone, especially about [REPLACE] :-(',
                        'When you feel like you\'re [REPLACE], what do you usually do?'
                    ],
                    'we are': [
                        'Yes, we are [REPLACE]',
                        'We are [REPLACE]? Interesting, tell me more.'
                    ],
                    'dreaming': [
                        'Dreaming about [REPLACE]? A nightmare or a pleasant dream?',
                        'What if the dream about [REPLACE] came true?',
                        'Interesting, tell me more about that dream involving [REPLACE].'
                    ],
                    'friend': [
                        'What about the friend who [REPLACE]?',
                        'Is that friend who [REPLACE] a close friend?',
                        'What makes you think about the friend who [REPLACE]?'
                    ],
                    'because': [
                        'Okay, so you think that\'s because [REPLACE]?',
                        'So the main reason is because [REPLACE], right?',
                        'Hmm, because [REPLACE]... then what?'
                    ],
                    'kill': [
                        'PLEASE DON\'T KILL [REPLACE]! :-(',
                        'KILLING [REPLACE] IS BAD! :-(',
                    ],                    
                    'yes': ['Alright', 'Okay, go on?', 'Got it, what next?', 'Mhm, I\'m listening'],
                    'hello': [f'Hello there!', f'Hi there!', f'Yo, friend! What\'s up?'],
                }
            ], # 177
            [100,1,0,False,None,0,{("C", "C"):[0,0], ("C", "D"):[0,0], ("D", "C"):[0,0], ("D", "D"):[0,0]}], # 178
            [0.5,0.5], # 179
            [
                [
                    {
                        (("C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C"),("C","C", "C", "C", "C", "C", "C", "C", "C", "C", "C")):[0,0]
                    },
                [[None, 0.5],0.1]
                ],
                [
                    {
                        (("C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C"),("C","C", "C", "C", "C", "C", "C", "C", "C", "C", "C")):[0,0]
                    },
                [[None, 0.5],0.1]
                ]
            ], # 180
            [100,1,False,0], # 181
            [100,1,False,None,0,{("C", "C"):[0,0], ("C", "D"):[0,0], ("D", "C"):[0,0], ("D", "D"):[0,0]}], # 182
            [100,1,False,None,0,{("C", "C"):[0,0], ("C", "D"):[0,0], ("D", "C"):[0,0], ("D", "D"):[0,0]}], # 183
            [100,1,False,None,0,{("C", "C"):[0,0], ("C", "D"):[0,0], ("D", "C"):[0,0], ("D", "D"):[0,0]},0], # 184
            [100,1,False,0,0], # 185
            [100,1,False,None,0,{("C", "C"):[0,0], ("C", "D"):[0,0], ("D", "C"):[0,0], ("D", "D"):[0,0]}], # 186
            0, # 187
            {("C", "C"):[0,0], ("C", "D"):[0,0], ("D", "C"):[0,0], ("D", "D"):[0,0]}, # 188
            0, # 189
            100, # 190
            0, # 191
            0, # 192
            0, # 193
            0, # 194
            0, # 195
            [0,False,0], # 196
            0, # 197
            0, # 198
            [0,0], # 199
            0, # 200
            0, # 201
            0, # 202
            {('C', 'C'): 0, ('C', 'D'): 0, ('D', 'C'): 0, ('D', 'D'): 0}, # 203
            [False,0,20,{('C', 'C'): 0, ('C', 'D'): 0, ('D', 'C'): 0, ('D', 'D'): 0}], # 204
            0, # 205
            0, # 206
            [], # 207
            '1', # 208
            '1', # 209
            0, # 210
            [1, False, 1, 0], # 211
            [1.0,0.0,0,0,0,0], # 212
            0, # 213
            0, # 214
            0, # 215
            0, # 216
            {
                "ema_trust": 1.0,
                "d_streak": 0,
                "history": [],
                "regret_count": 0,
                "pattern_score": 0,
                "pattern_window": 0,
                "punish_cooldown": 0,
                "last_my_action": 'C'
            }, # 217
            {
                "n":50,
                "old input": [0, 0],
                "input weight":[[rnd.uniform(-1,1) for _ in range(2)] for _ in range(50)],
                "bias":[rnd.uniform(-1,1) for _ in range(50)],
                "n val":[0.5 for _ in range(50)],
                "output weight":[[rnd.uniform(-1,1) for _ in range(50)], [rnd.uniform(-1,1) for _ in range(50)]],
                "output bias":[rnd.uniform(-1,1), rnd.uniform(-1,1)],
                "output val":[0.5, 0.5],
            }, # 218
            {
                "n":2,
                "time":0,
                "input weight":[[rnd.uniform(-1,1) for _ in range(2)] for _ in range(2)],
                "bias":[rnd.uniform(-1,1) for _ in range(2)],
                "n val":[0.5 for _ in range(2)],
                "output weight":[[rnd.uniform(-1,1) for _ in range(2)], [rnd.uniform(-1,1) for _ in range(2)]],
                "output bias":[rnd.uniform(-1,1), rnd.uniform(-1,1)],
                "output val":[0.5, 0.5],
                "best input weight":[[0 for _ in range(2)] for _ in range(2)],
                "best bias":[0 for _ in range(2)],
                "best output weight":[[0 for _ in range(2)], [0 for _ in range(2)]],
                "best output bias":[0, 0],
                "score":0,
                "high score":1,
                "reward":1,
            }, # 219
            {
                "pattern_score": 0,
                "trust": 1.0,
                "history": [],
                "d_streak": 0,
                "mode": "observing",
                "forgiveness_chance": 0.3
            }, # 220
            [0,0,0], # 221
            [0,0,0], # 222
            0, # 223
            {
                "trust": 1.0,
                "deal_broken": False,
                "coop_count": 0,
                "defect_count": 0
            }, # 224
            {
                "my_bluff_count": 0,
                "opp_bluff_count": 0,
                "opp_coop_words": 0,
                "opp_betrayals": 0
            }, # 225
            [0,[0 for _ in range(5)]], # 226
            {
                "acumulation":{'CC':0,'CD':0,'DC':0,'DD':0},
                "tree":{'CC':0.5,'CD':0.5,'DC':0.5,'DD':0.5},
                "total":{'CC':0,'CD':0,'DC':0,'DD':0},
            }, # 227
            [{'CC':[0,0], 'CD':[0,0], 'DC':[0,0], 'DD':[0,0]}, 'CC'], # 228
            0, # 229
            {
                'RRR':(RPST['R'] - RPST['S']) / (RPST['T'] - RPST['R']), # (R - S) / (T - R)
                'Tolerant':50 / (RPST['R'] - RPST['S']), # total tolerant loss score / (R - S)
                'Win round total':0,
                'Lose round total':0,
                'Highest score ever get':avg_RPST, # (T + R + P + S) / 4
                'Lowest score ever get':avg_RPST, # (T + R + P + S) / 4
                'G':[0,0],
                'B':[0,0],
                'Total poin opp C':0,
                'Total poin opp D':0,
                'Old score':0,
                'ALLD':False
            }, # 230
            [{'C':0,'D':0}, 0], # 231
            0, # 232
            0, # 233
            0, # 234
            None, # 235
            None, # 236
            0, # 237
            [0,False,0,0,0], # 238
            0, # 239
            [0,0], # 240
            [0.5,0.5,abs(max(RPST['T'],RPST['R'],RPST['P'],RPST['S']) / 3),0.0,0.5,0], # 241
            [[0]*20,False], # 242
            [1,0,['C','C','D']], # 243
            {
                "best pattern 1":[['C','C','D'], 0],
                "best pattern 2":[['C','C','D'], 0],
                "now pattern":['C','C','D'],
                "pattern score":0,
                "sum old index":0,
                "old score":0
            }, # 244
            {
                'Rd': {('C', 'C'): 1, ('C', 'D'): 1, ('D', 'C'): 0, ('D', 'D'): 0},
                'Rc': {},
                'Pi': {('C', 'C'): 1, ('C', 'D'): 1, ('D', 'C'): 0, ('D', 'D'): 0},
                'violation_counts': {},
                'reject_threshold': 3,
                'violation_threshold': 4,
                'promotion_threshold': 3,
                'tree_depth': 5,
                'v': 0,
                'alpha': 0.75,
                'history_by_cond': {
                    ('C', 'C'): ([1], [1]),
                    ('C', 'D'): ([1], [1]),
                    ('D', 'C'): ([0], [1]),
                    ('D', 'D'): ([0], [1]),
                }
            }, # 245
            [0,0], # 246
            '0', # 247
            '1', # 248
            '0', # 249
            '1', # 250
            '1', # 251
            '1', # 252
            '1', # 253
            '1', # 254
            '1', # 255
            '0', # 256
            '0', # 257
            '0', # 258
            '0', # 259
            '0', # 260
            '0', # 261
            '2', # 262
            0, # 263
            0, # 264
            [0.25,{},"",""], # 265
            '', # 266
            '', # 267
            '', # 268
            [0,0], # 269
            False, # 270
            [False, 0], # 271
            False, # 272
            False, # 273
            [False, 0], # 274
            False, # 275
            0, # 276
            ["Nice",10,-10,0,0], # 277
            [0, [[1.0]], [[1.0]], [0.5]], # 278
            3, # 279
            [1, 1, 0, 0], # 280
            [0,0,0], # 281
            False, # 282
            [0, 0], # 283
            [0, 0, 0, 0], # 284
            None, # 285
            [0, 0, 0, 0, (1 + mth.sqrt(5)) / 2], # 286
            [0, 0, 0, 0, mth.pi], # 287
            [0, 0, 0, 0, mth.e], # 288
            [max(min(rnd.random() * ((rnd.random() * 0.5) + 1), 1), 0), max(min(rnd.random() * (1 - (rnd.random() * 0.5)), 1), 0)], # 289
            ["TFT", 3, 0], # 290
            [], # 291
            [0,0,0,0,0,0], # 292
            [0, 0, True, [0, 0], [0, 0, 0, 0]], # 293
            1.0, # 294
            [False, 0], # 295
            0, # 296
            0, # 297
            [
                [
                    'C',
                    'C',
                    'D',
                    'C',
                    'D',
                    'D',
                    'D',
                    'C',
                    'C',
                    'D',
                    'C',
                    'D',
                    'C',
                    'C',
                    'D',
                    'C',
                    'D',
                    'D',
                    'C',
                    'D',
                ],
                0,
                0,
                False
            ], # 298
            0, # 299
            [
                {
                    (("C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C"),("C","C", "C", "C", "C", "C", "C", "C", "C", "C", "C")):[0,0]
                },
            [[None, 0.5],0.1]
            ], # 300
            {('C', 'C'): 0, ('C', 'D'): 0, ('D', 'C'): 0, ('D', 'D'): 0}, # 301
            {('C', 'C'): 0, ('C', 'D'): 0, ('D', 'C'): 0, ('D', 'D'): 0}, # 302
            [False,0,15,{('C', 'C'): 0, ('C', 'D'): 0, ('D', 'C'): 0, ('D', 'D'): 0}], # 303
            [False,0,20,{('C', 'C'): 0, ('C', 'D'): 0, ('D', 'C'): 0, ('D', 'D'): 0}], # 304
            ['0',1], # 305
            0, # 306
            [3, 8, 0, 0], # 307
            False, # 308
            0, # 309
            [round(max(rnd.randint(1, 10) * rnd.random(), 1)), round(max(rnd.randint(1, 10) * rnd.random(), 1)), 0], # 310
            False, # 311
            0, # 312
            [0, 0], # 313
            (lambda n=4: {"num_states": n, "transitions": {s: {"C": [x/sum(wc) for wc in [[rnd.random() for _ in range(n)]] for x in wc], "D": [x/sum(wd) for wd in [[rnd.random() for _ in range(n)]] for x in wd]} for s in range(n)}, "state_outputs": [rnd.choice(["C", "D"]) for _ in range(n)], "current_state": 0})(), # 314
            (lambda n=4: {"num_states": n, "transitions": {s: {"C": [x/sum(wc) for wc in [[rnd.random() for _ in range(n)]] for x in wc], "D": [x/sum(wd) for wd in [[rnd.random() for _ in range(n)]] for x in wd]} for s in range(n)}, "state_outputs": [rnd.choice(["C", "D"]) for _ in range(n)], "current_state": 0})(), # 315
            (lambda n=4: {"num_states": n, "transitions": {s: {"C": [x/sum(wc) for wc in [[rnd.random() for _ in range(n)]] for x in wc], "D": [x/sum(wd) for wd in [[rnd.random() for _ in range(n)]] for x in wd]} for s in range(n)}, "state_outputs": [rnd.choice(["C", "D"]) for _ in range(n)], "current_state": 0})(), # 316
            [0, 0], # 317
            0, # 318
            '1', # 319
            '1', # 320
            '1', # 321
            0, # 322
            [0, 0, 0, 0], # 323
            False, # 324
            [False, []], # 325
            1.0, # 326
            0, # 327
            [{'C': 0, 'D': 0}, {'C': 0.0, 'D': 0.0}], # 328
            0, # 329
            0, # 330
            0, # 331
            [22, 35, 0, 0], # 332
            [[], [], None, [], None], # 333
            [[], [], False], # 334
            [0, 0], # 335
            [True, 1.0, 1.0, 1.0, 1.0], # 336
            [{'CC':1,'CD':0,'DC':1,'DD':0},{'CC':1,'CD':0,'DC':1,'DD':0}], # 337
            [1, 0, 0, 0], # 338
            [
                {
                    'C': {'C': 0.0, 'D': 0.0},
                    'D': {'C': 0.0, 'D': 0.0}
                },
                0
            ], # 339
            [
                {
                    (("C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C"),("C","C", "C", "C", "C", "C", "C", "C", "C", "C", "C")):[0,0]
                },
            [[None, 0.5],0.1]
            ], # 340
            [
                {
                    (("C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C"),("C","C", "C", "C", "C", "C", "C", "C", "C", "C", "C")):[0,0]
                },
            [[None, 0.5],0.1]
            ], # 341
            [0.5, 0.5, 0.5], # 342
            [
                {
                    (("C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C"),("C","C", "C", "C", "C", "C", "C", "C", "C", "C", "C")):[0,0]
                },
            [[None, 0.5],0.1]
            ], # 343
            [0.5,0.5], # 344
            [{'CC':[0,0], 'CD':[0,0], 'DC':[0,0], 'DD':[0,0]}, 'CC', 0], # 345
            {'CC':[0,0], 'CD':[0,0], 'DC':[0,0], 'DD':[0,0]}, # 346
            False, # 347
            False, # 348
            False, # 349
            False, # 350
            False, # 351
            False, # 352
            False, # 353
            False, # 354
            False, # 355
            False, # 356
            False, # 357
            0, # 358
            0, # 359
           ]

def cycle_detec(ap, min_ = 1, max_ = 12, offset = 0):
    inp = [memory[-1 - i][ap] for i in range(offset, game_round)]
    new_max_size = min(len(inp) // 2, max_)
    for i in range(min_, new_max_size + 1):
        has_cycle = True
        cycle = tuple(inp[:i])
        for j, elem in enumerate(inp):
            if elem != cycle[j % len(cycle)]:
                has_cycle = False
                break
        if has_cycle:
            return cycle
    return None

def flip(the_choice):
    return 'C' if the_choice == 'D' else 'D'

def rnd_prob(prob):
    return 'C' if rnd.random() <= prob else 'D'

RPST = {'R':3, 'P':1, 'S':0, 'T':5}
avg_RPST = (RPST['R'] + RPST['P'] + RPST['S'] + RPST['T']) / 4
strat_memory = [reset_think_memory(),reset_think_memory()]
'return'
def think(strategy,ap):
    global strat_memory, strategy_next_chat
    think_memory = strat_memory[1-ap]
    chats = [strategy_chat[0][0], strategy_chat[1][0]]
    next_chat = strategy_next_chat[1-ap]
    # Classic Strategy & Axelrod Library Strategy & Emergent Lab Strategy & Youtube
    if strategy == 'Tit For Tat':
        if not game_round:
            return 'C'
        return memory[-1][ap]
    elif strategy == 'Joss / Naive Prober':
        prob_choice_steal = rnd.randint(1,10)
        if prob_choice_steal == 1:
            return 'D'
        else:
            if not game_round:
                return 'C'
            return memory[-1][ap]
    elif strategy == 'Friedman / Grudger / Grim trigger':
        if not game_round:
            return 'C'
        
        if memory[-1][ap] == 'D' or think_memory[0] == 1:
            think_memory[0] = 1
            return 'D'
        else:
            return 'C'
    elif strategy == 'Gladstein / Tester':
        if think_memory[1] != 2 and ((memory[-1][ap] != 'D') if game_round > 0 else True):
            if think_memory[1] == 0:
                think_memory[1] = 1
                return 'D'
            else:
                think_memory[1] = 0
                return 'C'
        else:
            if think_memory[1] != 2:
                think_memory[1] = 2
                return 'C'
            else:
                return 'C' if memory[-1][1-ap] == 'D' else 'D'
    elif strategy == 'Tit For 2 Tats':
        if (game_round + 1) <= 2:
            return 'C'
        if memory[-1][ap] == memory[-1-1][ap] and memory[-1][ap] == 'D':
            return 'D'
        else:
            return 'C'
    elif strategy == 'Harrington':
        if game_round > 0 and memory[-1][ap] == 'C':
            think_memory[2][15] += 1
        def try_return(to_return, lower_flags=True, inc_parity=False):
            """
            This will return to_return, with some end-of-turn logic.
            """

            if lower_flags and to_return == 'C':
                
                
                think_memory[2][7] -= 1
                think_memory[2][8] += 1

            if inc_parity and to_return == 'D':
                
                
                
                think_memory[2][11][think_memory[2][12]] += 1

            return to_return

        def calculate_chi_squared(turn):
            """
            Pearson's Chi Squared statistic = sum[ (E_i-O_i)^2 / E_i ], where O_i
            are the observed matrix values, and E_i is calculated as number (of
            defects) in the row times the number in the column over (total number
            in the matrix minus 1).  Equivalently, we expect we expect (for an
            independent distribution) the total number of recorded turns times the
            portion in that row times the portion in that column.

            In this function, the statistic is non-standard in that it excludes
            summands where E_i <= 1.
            """

            denom = turn - 2

            expected_matrix = (
                np.outer(
                    think_memory[2][6].sum(axis=1), think_memory[2][6].sum(axis=0)
                )
                / denom
            )

            chi_squared = 0.0
            for i in range(4):
                for j in range(2):
                    expect = expected_matrix[i, j]
                    if expect > 1.0:
                        chi_squared += (
                            expect - think_memory[2][6][i, j]
                        ) ** 2 / expect

            return chi_squared

        def detect_random(turn):
            """
            We check if the top-left cell of the matrix (corresponding to all
            Cooperations) has over 80% of the turns.  In which case, we label
            non-random.

            Then we check if over 75% or under 25% of the opponent's turns are
            Defections.  If so, then we label as non-random.

            Otherwise we calculates a modified Pearson's Chi Squared statistic on
            history, and returns True (is random) if and only if the statistic
            is less than or equal to 3.
            """

            denom = turn - 2

            if think_memory[2][6][0, 0] / denom >= 0.8:
                return False
            if (
                think_memory[2][1] / denom < 0.25
                or think_memory[2][1] / denom > 0.75
            ):
                return False

            if calculate_chi_squared(turn) > 3:
                return False
            return True

        def detect_streak(last_move):
            """
            Return true if and only if the opponent's last twenty moves are defects.
            """

            if last_move == 'D':
                think_memory[2][10] += 1
            else:
                think_memory[2][10] = 0
            if think_memory[2][10] >= 20:
                return True
            return False

        def detect_parity_streak(last_move):
            """
            Switch which `parity_streak` we're pointing to and incerement if the
            opponent's last move was a Defection.  Otherwise reset the flag.  Then
            return true if and only if the `parity_streak` is at least
            `parity_limit`.

            This is similar to detect_streak with alternating streaks, except that
            these streaks get incremented elsewhere as well.
            """

            think_memory[2][12] = 1 - think_memory[2][12]  
            if last_move == 'D':
                think_memory[2][11][think_memory[2][12]] += 1
            else:
                think_memory[2][11][think_memory[2][12]] = 0
            if think_memory[2][11][think_memory[2][12]] >= think_memory[2][13]:
                return True

        """Actual strategy definition that determines player's action."""
        turn = (game_round + 1)

        if turn == 1:
            return 'C'

        if think_memory[2][0] == "Defect":
            
            if memory[-1][ap] == 'D':
                think_memory[2][2] += 1
            else:
                think_memory[2][2] -= 3
            
            if think_memory[2][2] >= 11:
                think_memory[2][0] = "Normal"
                think_memory[2][4] = True
                think_memory[2][7] = 2
                return try_return(to_return='C', lower_flags=False)

            return try_return('D')

        
        

        
        if not think_memory[2][4]:
            if turn > 2:
                if memory[-1][ap] == 'D':
                    think_memory[2][1] += 1

                
                history_col = 1 if memory[-1][ap] == 'D' else 0
                
                
                history_row = 1 if memory[-2][ap] == 'D' else 0
                if memory[-2][1-ap] == 'D':
                    history_row += 2
                think_memory[2][6][history_row, history_col] += 1

            
            if turn % 15 == 0 and turn > 15:
                if detect_random(turn):
                    think_memory[2][0] = "Defect"
                    return try_return(
                        'D', lower_flags=False
                    )  

        
        if think_memory[2][8] == 2 and memory[-1][ap] == 'D':
            think_memory[2][9] = True

        
        
        if (
            turn == 38
            and memory[-1][ap] == 'D'
            and think_memory[2][15] == 36
        ):
            think_memory[2][0] = "Fair-weather"
            return try_return(to_return='C', lower_flags=False)

        if think_memory[2][0] == "Fair-weather":
            if memory[-1][ap] == 'D':
                think_memory[2][0] = "Normal"  
                
            else:
                
                return try_return('C')

        

        
        if detect_streak(memory[-1][ap]):
            return try_return('D', inc_parity=True)
        if detect_parity_streak(memory[-1][ap]):
            think_memory[2][11][think_memory[2][12]] = (
                0  
            )
            think_memory[2][14] += (
                1  
            )
            if think_memory[2][14] >= 8:  
                think_memory[2][13] = 3
            return try_return(
                'C', inc_parity=True
            )  

        
        if think_memory[2][7] >= 1:
            return try_return('C', lower_flags=True, inc_parity=True)

        if turn < 37:
            
            return try_return(memory[-1][ap], inc_parity=True)
        if turn == 37:
            
            think_memory[2][7], think_memory[2][8] = 2, 1
            return try_return('D', lower_flags=False)
        if think_memory[2][9] or rnd.random() > think_memory[2][5]:
            
            return try_return(memory[-1][ap], inc_parity=True)

        
        think_memory[2][5] += 0.05
        think_memory[2][7], think_memory[2][8] = 2, 1
        return try_return('D', lower_flags=False)
    elif strategy == 'Generous Tit For Tat':
        if not game_round:
            return 'C'
        
        if memory[-1][ap] == 'D' and rnd.random() > 0.1:
            return 'D'
        else:
            return 'C'
    elif strategy == 'Pavlov / Win-Stay, Lose-Shift':
        if not game_round:
            return 'C'
        
        if memory[-1][ap] == memory[-1][1-ap]:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Named Withheld':
        
        

        P = think_memory[5][0]

        if not game_round:
            return 'C' if rnd.random() <= 0.3 else 'D'

        if (game_round) % 10 == 0:

            coop_rate = 0
            for i in range(10):
                if memory[-1-i][ap] == 'C':
                    coop_rate += 1
            coop_rate /= 10

            
            if coop_rate >= 0.8:
                P = min(P + 0.1, 0.9)

            
            elif coop_rate <= 0.2:
                P = max(P - 0.1, 0.1)

            
            else:
                P += rnd.uniform(-0.1,0.1)
                P = max(min(P,0.9),0.1)

            
            if (game_round + 1) > 130:
                if think_memory[5][1] < think_memory[5][2]:
                    P = max(P - 0.1,0.1)

        think_memory[5][0] = P

        if rnd.random() < P:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Davis':
        if (game_round + 1) > 10:
            if think_memory[7] == 1:
                return 'D'
            elif memory[-1][ap] == 'D':
                think_memory[7] = 1
                return 'D'
            else:
                return 'C'
        else:
            if game_round > 0 and memory[-1][ap] == 'D':
                think_memory[7] = 1
            return 'C'
    elif strategy == 'Always Cooperate / Cooperator':
        return 'C'
    elif strategy == 'Always Defect / Defector':
        return 'D'
    elif strategy == 'Random':
        return rnd.choice(['C','D'])
    elif strategy == 'Graaskamp':
        if (game_round + 1) <= 56:
            if game_round > 0 and memory[-1][ap] == 'C':
                think_memory[134][2] += 1
            else:
                think_memory[134][3] += 1

        if (game_round + 1) <= 50:
            if not game_round:
                return 'C'
            return memory[-1][ap]
        elif (game_round + 1) <= 53:
            if (game_round + 1) == 53:
                think_memory[134][0] = (memory[-1][ap] == 'D')
            return ('D','C','C')[(game_round + 1)-51]
        elif (game_round + 1) <= 56:
            return memory[-1][ap]
        else:
            if (game_round + 1) == 57:
                E1 = 57
                E2 = E1/2
                H = ((((think_memory[134][2] - E2)*(think_memory[134][2] - E2)))/E2)+((((think_memory[134][3] - E2)*(think_memory[134][3] - E2)))/E2)
                if H < 3.84:
                    think_memory[134][1] = 1
                    return 'D'
                else:
                    if think_memory[134][0] == 1:
                        return memory[-1][ap]
                    else:
                        if ((game_round + 1) % rnd.randint(5, 15)) == 0:
                            return 'D'
                        else:
                            return memory[-1][ap]
            else:
                if think_memory[134][1] == 1:
                    return 'D'
                else:
                    if think_memory[134][0] == 1:
                        return memory[-1][ap]
                    else:
                        if ((game_round + 1) % rnd.randint(5, 15)) == 0:
                            return 'D'
                        else:
                            return memory[-1][ap]
    elif strategy == '2 Tits For Tat':
        if not game_round:
            return 'C'
        
        if memory[-1][ap] == 'D' or think_memory[10] > 0:
            if think_memory[10] == 0:
                think_memory[10] = 2
            elif think_memory[10] > 0 and memory[-1][ap] == 'D':
                think_memory[10] += 1
            think_memory[10] -= 1
            return 'D'
        else:
            return 'C'
    elif strategy == 'Stein & Rapoport':
        
        
        
        

        data = think_memory[14][0] if isinstance(think_memory[14], list) else think_memory[14]

        
        
        
        
        if game_round >= tournament_avg_last_round - 1:
            return 'D'

        
        
        
        if (game_round + 1) > 2:
            prev_self = 0 if memory[-1 - 1][1 - ap] == 'C' else 1
            prev_opp  = 0 if memory[-1][ap] == 'C' else 1
            data["matrix"][prev_self, prev_opp] += 1

        
        
        
        if game_round > 0 and (game_round) % 15 == 0 and data["mode"] == "TFT":
            O = data["matrix"]
            row_sums = O.sum(axis=1)
            col_sums = O.sum(axis=0)
            total = O.sum()

            
            if total > 0 and np.all(row_sums > 0) and np.all(col_sums > 0):
                
                E = np.outer(row_sums, col_sums) / total
                
                
                chi2_stat = np.sum(((O - E) ** 2) / (E + 1e-9))

                
                
                if chi2_stat < 3.841:
                    data["mode"] = "ALLD"

        
        
        
        if data["mode"] == "ALLD":
            return 'D'
        
        
        opp_last = memory[-1][ap] if game_round > 0 else 'C'
        return opp_last
    elif strategy == 'Downing':
        if (game_round + 1) <= 2:
            return 'D'
        
        if memory[-1-1][1-ap] == 'C':
            think_memory[16][2] += 1
            if memory[-1][ap] == 'C':
                think_memory[16][0] += 1
        else:
            think_memory[16][3] += 1
            if memory[-1][ap] == 'C':
                think_memory[16][1] += 1

        alpha = think_memory[16][0] / think_memory[16][2] if think_memory[16][2] > 0 else 0.5
        beta = think_memory[16][1] / think_memory[16][3] if think_memory[16][3] > 0 else 0.5

        E_sh = (alpha * RPST['R']) + ((1 - alpha) * RPST['S'])
        E_st = (beta * RPST['T']) + ((1 - beta) * RPST['P'])

        if E_sh > E_st:
            return 'C'
        if E_st > E_sh:
            return 'D'
        return 'C' if memory[-1][1-ap] == 'D' else 'D'
    elif strategy == 'Suspicious Tit For Tat':
        if not game_round:
            return 'D'
        return memory[-1][ap]
    elif strategy == 'Shubik':
        if not game_round:
            return 'C'
        
        if memory[-1][ap] == 'D' or think_memory[33][0] > 0:
            if memory[-1][ap] == 'D':
                if think_memory[33][1] == 0:
                    think_memory[33][0] = think_memory[33][2] + 1
                else:
                    think_memory[33][0] += 1
                think_memory[33][2] += 1
            think_memory[33][0] -= 1
            think_memory[33][1] = 1
            return 'D'
        else:
            think_memory[33][1] = 0
            return 'C'
    elif strategy == 'Nydegger':
        if not game_round:
            return 'C'
        elif (game_round + 1) == 2:
            return memory[-1][ap]
        elif (game_round + 1) == 3:
            if (memory[-1-1][ap] == 'D') and (memory[-1][ap] == 'C'):
                return 'D'
            else:
                return memory[-1][ap]
        else:
            black_list = (1, 6, 7, 17, 22, 23, 26, 29, 30, 31, 33, 38, 39, 45, 49, 54, 55, 58, 61)
            input_power = [1, 4, 16]
            def give_poin(x):
                opp = (memory[-1 - x][ap] == 'D') * 2
                self = (memory[-1 - x][1-ap] == 'D')

                return (opp + self)
            inp = [give_poin(i) for i in range(3)]
            A = sum(inp[i] * input_power[i] for i in range(3))
            if A in black_list:
                return 'D'
            else:
                return 'C'
    elif strategy == 'Grofman':
        if not game_round:
            return 'C'
        else:
            if memory[-1][ap] == memory[-1][1-ap]:
                return 'C'
            else:
                if rnd.random() <= (2/7):
                    return 'C'
                else:
                    return 'D'
    elif strategy == 'Champion':
        if game_round > 0:
            if memory[-1][ap] == 'D':
                think_memory[81] += 1

            p_d = think_memory[81] / (game_round)

        if (game_round + 1) <= 10:
            return 'C'
        elif (game_round + 1) <= 25:
            return memory[-1][ap]
        else:
            if (memory[-1][ap] == 'D') and (p_d >= max(0.4, rnd.random())):
                return 'D'
            else:
                return 'C'
    elif strategy == 'Tideman & Chieruzzi':
        def _decrease_retaliation_counter():
            """Lower the remaining owed retaliation count and flip to non-retaliate
            if the count drops to zero."""
            if think_memory[82][0]:
                think_memory[82][2] -= 1
                if think_memory[82][2] == 0:
                    think_memory[82][0] = False

        def _fresh_start():
            """Give the opponent a fresh start by forgetting the past"""
            think_memory[82][0] = False
            think_memory[82][1] = 0
            think_memory[82][2] = 0
            think_memory[82][5] = 0

        """Actual strategy definition that determines player's action."""

        if not game_round:
            return 'C'

        if memory[-1][ap] == 'D':
            think_memory[82][5] += 1

        
        if think_memory[82][4]:
            think_memory[82][4] = False
            return 'C'  

        
        current_round = (game_round + 1)
        if think_memory[82][3] == 0:
            valid_fresh_start = True
        
        else:
            valid_fresh_start = current_round - think_memory[82][3] >= 20

        if valid_fresh_start:
            valid_points = score[1-ap] - score[ap] >= 10
            valid_rounds = tournament_avg_last_round - current_round >= 10
            opponent_is_cooperating = memory[-1][ap] == 'C'
            if valid_points and valid_rounds and opponent_is_cooperating:
                
                N = game_round
                
                std_deviation = (N ** (1 / 2)) / 2
                lower = N / 2 - 3 * std_deviation
                upper = N / 2 + 3 * std_deviation
                if (
                    think_memory[82][5] <= lower
                    or think_memory[82][5] >= upper
                ):
                    
                    think_memory[82][3] = current_round
                    _fresh_start()
                    think_memory[82][4] = True
                    return 'C'  

        if think_memory[82][0]:
            
            _decrease_retaliation_counter()
            return 'D'

        if memory[-1][ap] == 'D':
            think_memory[82][0] = True
            think_memory[82][1] += 1
            think_memory[82][2] = think_memory[82][1]
            _decrease_retaliation_counter()
            return 'D'

        return 'C'
    elif strategy == 'Tullock':
        if (game_round + 1) < 11:
            return 'C'
        
        inp = [(memory[-1-(9 - i)][ap] == 'C') for i in range(10)]

        t_c = sum(inp)
        p_c = (t_c / 10) - 0.1

        if rnd.random() <= p_c:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Feld':
        if game_round > 0 and memory[-1][ap] == 'D':
            return 'D'
        else:
            cooperate_chance = 1 - (((game_round) / (tournament_avg_last_round)) * 0.5)
            if rnd.random() <= cooperate_chance:
                return 'C'
            else:
                return 'D'
    elif strategy == 'White':
        if game_round > 0:
            if memory[-1][ap] == 'D':
                think_memory[83] += 1
        if (game_round + 1) <= 11:
            return 'C'
        
        if memory[-1][ap] == 'C':
            return 'C'
        else:
            A = mth.floor(np.log((game_round + 1))) * think_memory[83]
            if A >= (game_round + 1):
                return 'D'
            else:
                return 'C'
    elif strategy == 'Black':
        prob_coop = {0: 1.0, 1: 1.0, 2: 0.88, 3: 0.68, 4: 0.4, 5: 0.04}
        if (game_round + 1) <= 5:
            return 'C'

        inp = [(memory[-1 - i][ap] == 'D') for i in range(5)]
        return 'C' if rnd.random() <= prob_coop[sum(inp)] else 'D'
    elif strategy == 'Tranquilizer':
        if game_round > 0:
            if memory[-1][ap] == 'D':
                think_memory[89][1] += 1
            else:
                think_memory[89][1] = 0
            if think_memory[89][0] == 2:
                think_memory[89][0] = 0
                think_memory[89][3] = (
                    (
                        think_memory[89][3]
                        * think_memory[89][5]
                    )
                    + (3 - (3 * think_memory[89][6][memory[-1][ap]]))
                    + (2 * think_memory[89][6][memory[-1][1-ap]])
                    - (
                        (
                            think_memory[89][6][memory[-1][ap]]
                            * think_memory[89][6][memory[-1][1-ap]]
                        )
                    )
                ) / (think_memory[89][5] + 1)
                think_memory[89][5] += 1
            elif think_memory[89][0] == 1:
                think_memory[89][0] = 2
                think_memory[89][2] = (
                    (
                        think_memory[89][2]
                        * think_memory[89][4]
                    )
                    + (3 - (3 * think_memory[89][6][memory[-1][ap]]))
                    + (2 * think_memory[89][6][memory[-1][1-ap]])
                    - (
                        (
                            think_memory[89][6][memory[-1][ap]]
                            * think_memory[89][6][memory[-1][1-ap]]
                        )
                    )
                ) / (think_memory[89][4] + 1)
                think_memory[89][4] += 1
        else:
            return 'C'

        if memory[-1][ap] == 'C':
            think_memory[89][7] += 1

        if think_memory[89][0] in (1, 2):
            return 'C'

        if (score[1-ap] / (game_round + 1)) >= (avg_RPST):
            probability = (
                0.95
                - (
                    (
                        think_memory[89][2]
                        + think_memory[89][3]
                        - 5
                    )
                    / 15
                )
                + (1 / ((game_round + 1) * (game_round + 1)))
                - (think_memory[89][6][memory[-1][ap]] / 4)
            )
            if rnd.random() <= probability:
                return 'C'
            think_memory[89][0] += 1
            return 'D'
        if (score[1-ap] / (game_round + 1)) >= 1.75:
            probability = (
                (
                    0.25
                    + ((think_memory[89][7] + 1) / (game_round + 1))
                )
                - (think_memory[89][1] * 0.25)
                + ((score[1-ap] - score[ap]) / 100)
                + (4 / (game_round + 1))
            )
            if rnd.random() <= probability:
                return 'C'
            return 'D'
        return memory[-1][ap]
    elif strategy == 'Fool Me Once':
        if not game_round:
            return 'C'
        
        if memory[-1][ap] == 'D':
            think_memory[88] += 1
        
        if think_memory[88] > 1:
            return 'D'
        else:
            return 'C'
    elif strategy == 'Gradual':
        if not game_round:
            return 'C'
        
        if think_memory[101][1] > 0:
            think_memory[101][1] -= 1
            return 'C'
        
        if memory[-1][ap] == 'D' and think_memory[101][0] <= 0:
            think_memory[101][0] = think_memory[101][2] + 1
            think_memory[101][2] += 1
        if think_memory[101][0] > 0:
            think_memory[101][0] -= 1
            if think_memory[101][0] == 0:
                think_memory[101][1] = 2
            return 'D'
        else:
            return 'C'
    elif strategy == 'Good Random':
        if rnd.random() <= 0.8:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Bad Random':
        if rnd.random() <= 0.8:
            return 'D'
        else:
            return 'C'
    elif strategy == 'Adaptive Tit For Tat':
        omega = 0.5
        if game_round > 0:
            if memory[-1][ap] == 'C':
                A = think_memory[104] + (omega * (1 - think_memory[104]))
            else:
                A = think_memory[104] - (omega * think_memory[104])
        else:
            A = think_memory[104]

        think_memory[104] = A
        if A >= 0.5:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Weiner':
        def try_return(to_return):
            if sum(think_memory[35][3]) >= 5:
                return 'D'
            return to_return

        if (game_round + 1) >= 3:
            think_memory[35][3][think_memory[35][4]] = (memory[-2][ap] == 'D')
            think_memory[35][4] = (think_memory[35][4] + 1) % 12

        if think_memory[35][0]:
           think_memory[35][0] = False
           think_memory[35][2] = 0
           if (
               think_memory[35][1] < (game_round + 1)
               and memory[-1][ap] == 'D'
           ):
               think_memory[35][1] += 20
               return try_return('C')
           else:
               return try_return(memory[-1][ap])
        else:
            if (memory[-1][ap] == 'D' if game_round > 0 else False):
                think_memory[35][2] += 1
            else:
                if think_memory[35][2] % 2 == 1:
                    think_memory[35][0] = True
                think_memory[35][2] = 0
            return try_return(memory[-1][ap] if game_round > 0 else 'C')
    elif strategy == 'Borufsen':
        def try_return(to_return):
            if to_return == 'C':
                return 'C'

            if think_memory[121][4]:
                think_memory[121][4] = False
                return 'C'
            return 'D'

        if (game_round + 1) <= 2:
            return 'C'

        if (game_round + 1) > 3:
            if memory[-1][ap] == 'C':
                if memory[-2][1-ap] == 'C':
                    think_memory[121][1] += 1
                else:
                    think_memory[121][0] += 1

        if (game_round + 1) > 2 and (game_round + 1) % 25 == 2:
            coming_from_defect = False
            if think_memory[121][5] == "Defect":
                coming_from_defect = True

            think_memory[121][5] = "Normal"
            coops = think_memory[121][0] + think_memory[121][1]

            if coops < 3:
                think_memory[121][5] = "Defect"

            if (8 <= coops <= 17) and think_memory[121][1] / coops < 0.7:
                think_memory[121][5] = "Defect"

            think_memory[121][0], think_memory[121][1] = 0, 0
            if think_memory[121][5] == "Defect":
                think_memory[121][2] = 0
                think_memory[121][3] = 0
                think_memory[121][4] = False

            if think_memory[121][5] == "Normal" and coming_from_defect:
                return 'D'

        if think_memory[121][5] == "Defect":
            return 'D'
        else:
            assert think_memory[121][5] == "Normal", "What do you mean?"

            if (memory[-1][1-ap], memory[-1][ap]) == ('D', 'D'):
                think_memory[121][2] += 1
            else:
                think_memory[121][2] = 0

            if think_memory[121][2] >= 3:
                think_memory[121][2] = 0
                think_memory[121][3] = 0
                return try_return('C')

            my_two_back, opp_two_back = 'C', 'C'
            if (game_round + 1) >= 3:
                my_two_back = memory[-2][1-ap]
                opp_two_back = memory[-2][ap]
            if (
                memory[-1][1-ap] != memory[-1][ap]
                and memory[-1][1-ap] == opp_two_back
                and memory[-1][ap] == my_two_back
            ):
                think_memory[121][3] += 1
            else:
                think_memory[121][3] = 0
            if think_memory[121][3] >= 3:
                think_memory[121][2] = 0
                think_memory[121][3] = 0
                think_memory[121][4] = True
            return try_return(memory[-1][ap])

    elif strategy == 'Graaskamp & Katzen':
        if not game_round:
            return 'C'

        if ((game_round) % 10) == 0:
            A = -10 + ((game_round + 1) * 3)
            if score[1-ap] < A:
                think_memory[122] = 1
        
        if think_memory[122] == 0:
            return memory[-1][ap]
        else:
            return 'D'
    elif strategy == 'Giles / Worse & Worse 3':
        if not game_round:
            return 'C'
        
        if memory[-1][ap] == 'C':
            think_memory[123] += 1
        
        p_c = think_memory[123] / (game_round)

        if rnd.random() <= p_c:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Adams':
        if not game_round:
            return 'C'
        
        if (game_round + 1) == 2:
            think_memory[124][1] = (memory[-1][ap] == 'D')

        if memory[-1][ap] == 'D':
            think_memory[124][0] += 1

        number_of_defection = think_memory[124][0] - think_memory[124][1]
        if number_of_defection <= 9:
            if think_memory[124][0] in (4, 7, 9):
                return 'D'
            else:
                return 'C'
        else:
            if memory[-1][ap] == 'D':
                A = 1 - (0.5 ** (number_of_defection - 9))
                if rnd.random() <= A:
                    return 'D'
            return 'C'
    elif strategy == 'Yamachi':
        def try_return(to_return, opp_def):
            """
            Return `to_return`, unless the turn is greater than 40 AND
            `portion_defect` is between 45% and 55%.

            In this case, still record the history as `to_return` so that the
            modified behavior doesn't affect the calculation of `count_us_them_us`.
            """
            turn = (game_round + 1)

            think_memory[125][1].append(to_return)

            
            
            if turn > 40:
                portion_defect = (opp_def + 0.5) / turn
                if 0.45 < portion_defect < 0.55:
                    return 'D'

            return to_return

        turn = (game_round + 1)
        if turn == 1:
            return try_return('C', 0)

        if memory[-1][ap] == 'D': think_memory[125][2] += 1

        us_last = think_memory[125][1][-1]
        them_two_ago, us_two_ago, them_three_ago = 'C', 'C', 'C'
        if turn >= 3:
            them_two_ago = memory[-2][ap]
            us_two_ago = memory[-2][1-ap]
        if turn >= 4:
            them_three_ago = memory[-3][ap]

        
        if turn >= 3:
            think_memory[125][0][
                (them_three_ago, us_two_ago, them_two_ago)
            ] += 1

        if (
            think_memory[125][0][(them_two_ago, us_last, 'C')]
            >= think_memory[125][0][(them_two_ago, us_last, 'D')]
        ):
            return try_return('C', think_memory[125][2])
        return try_return('D', think_memory[125][2])
    elif strategy == 'Prober':
        if (game_round + 1) <= 3:
            return ('D', 'C', 'C')[(game_round + 1)-1]
        
        if (memory[1][ap] + memory[2][ap]) == 'CC':
            return 'D'
        else:
            return memory[-1][ap]
    elif strategy == 'Contrite Tit For Tat':
        """Actual strategy definition that determines player's action."""

        if not game_round:
            think_memory[131][1].append('C')
            return 'C'

        
        if think_memory[131][0] and memory[-1][1-ap] == 'C':
            think_memory[131][0] = False
            think_memory[131][1].append('C')
            return 'C'

        
        if think_memory[131][1][-1] != memory[-1][1-ap]:  
            if memory[-1][1-ap] == 'D' and memory[-1][ap] == 'C':
                think_memory[131][0] = True

        think_memory[131][1].append(memory[-1][ap])
        return memory[-1][ap]
    elif strategy == 'Cycle Hunter':
        if not game_round:
            return 'C'
        
        if cycle_detec(ap, min_ = 3):
            return 'D'

        return 'C'
    elif strategy == 'Evolved Looker Up 2_2_2':
        if (game_round + 1) > 2:
            inp = (memory[-2][1-ap] + memory[-1][1-ap], memory[-2][ap] + memory[-1][ap], memory[0][ap] + memory[1][ap])

            lookup_table = {('CC', 'CC', 'CC'): 'C', ('CC', 'CC', 'CD'): 'D', ('CC', 'CC', 'DC'): 'D', ('CC', 'CC', 'DD'): 'C', ('CC', 'CD', 'CC'): 'D', ('CC', 'CD', 'CD'): 'C', ('CC', 'CD', 'DC'): 'D', ('CC', 'CD', 'DD'): 'D', ('CC', 'DC', 'CC'): 'C', ('CC', 'DC', 'CD'): 'D', ('CC', 'DC', 'DC'): 'D', ('CC', 'DC', 'DD'): 'D', ('CC', 'DD', 'CC'): 'C', ('CC', 'DD', 'CD'): 'D', ('CC', 'DD', 'DC'): 'D', ('CC', 'DD', 'DD'): 'D', ('CD', 'CC', 'CC'): 'D', ('CD', 'CC', 'CD'): 'D', ('CD', 'CC', 'DC'): 'C', ('CD', 'CC', 'DD'): 'D', ('CD', 'CD', 'CC'): 'C', ('CD', 'CD', 'CD'): 'D', ('CD', 'CD', 'DC'): 'C', ('CD', 'CD', 'DD'): 'C', ('CD', 'DC', 'CC'): 'C', ('CD', 'DC', 'CD'): 'D', ('CD', 'DC', 'DC'): 'D', ('CD', 'DC', 'DD'): 'C', ('CD', 'DD', 'CC'): 'C', ('CD', 'DD', 'CD'): 'D', ('CD', 'DD', 'DC'): 'C', ('CD', 'DD', 'DD'): 'D', ('DC', 'CC', 'CC'): 'D', ('DC', 'CC', 'CD'): 'D', ('DC', 'CC', 'DC'): 'C', ('DC', 'CC', 'DD'): 'C', ('DC', 'CD', 'CC'): 'C', ('DC', 'CD', 'CD'): 'C', ('DC', 'CD', 'DC'): 'C', ('DC', 'CD', 'DD'): 'D', ('DC', 'DC', 'CC'): 'D', ('DC', 'DC', 'CD'): 'D', ('DC', 'DC', 'DC'): 'C', ('DC', 'DC', 'DD'): 'D', ('DC', 'DD', 'CC'): 'D', ('DC', 'DD', 'CD'): 'D', ('DC', 'DD', 'DC'): 'D', ('DC', 'DD', 'DD'): 'D', ('DD', 'CC', 'CC'): 'D', ('DD', 'CC', 'CD'): 'D', ('DD', 'CC', 'DC'): 'D', ('DD', 'CC', 'DD'): 'D', ('DD', 'CD', 'CC'): 'C', ('DD', 'CD', 'CD'): 'C', ('DD', 'CD', 'DC'): 'D', ('DD', 'CD', 'DD'): 'D', ('DD', 'DC', 'CC'): 'C', ('DD', 'DC', 'CD'): 'D', ('DD', 'DC', 'DC'): 'D', ('DD', 'DC', 'DD'): 'D', ('DD', 'DD', 'CC'): 'C', ('DD', 'DD', 'CD'): 'C', ('DD', 'DD', 'DC'): 'C', ('DD', 'DD', 'DD'): 'D'}
        
            return lookup_table[inp]
        return 'C'
    elif strategy == 'Alternator':
        if ((game_round + 1) % 2) == 0:
            return 'D'
        else:
            return 'C'
    elif strategy == 'Suspicious Alternator':
        if ((game_round + 1) % 2) == 0:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Sneaky Tit For Tat':
        if game_round > 0:
            if memory[-1][ap] == 'D':
                think_memory[142] = True
        if game_round < 2:
            return 'C'
        if not think_memory[142]:
            return 'D'
        if (memory[-1][ap], memory[-2][1-ap]) == ('D', 'D'):
            return 'C'
        return memory[-1][ap]
    elif strategy == 'Meta Majority':
        meta_majo_tft = (1 if memory[-1][ap] == 'C' else -1) if game_round > 0 else 1
        meta_majo_pvlv = (1 if memory[-1][ap] == memory[-1][1-ap] else -1) if game_round > 0 else 1
        meta_majo_allc = 1
        meta_majo_alld = -1

        meta_majo_majority = meta_majo_tft + meta_majo_pvlv + meta_majo_allc + meta_majo_alld

        if meta_majo_majority < 0:
            return 'D'
        else:
            return 'C'
        
    elif strategy == 'Meta Winner':
        def pay_off_matrix(x,y):
            if x == 'C':
                if y == 'C':
                    return RPST['R']
                else:
                    return RPST['T']
            else:
                if y == 'C':
                    return RPST['S']
                else:
                    return RPST['P']
        think_memory[136][4] = max(think_memory[136][4] - 1, 0)
        if game_round > 0 and memory[-1][ap] == 'D':
            think_memory[136][4] = 2
            think_memory[136][5] = 1
        think_memory[136][0] += pay_off_matrix((memory[-1][ap] if game_round > 0 else 'C'),'C' if rnd.random() <= 0.2 else (memory[-1][ap] if game_round > 0 else 'C'))
        think_memory[136][1] += pay_off_matrix((memory[-1][ap] if game_round > 0 else 'C'),'D' if ((memory[-1][ap] == 'D') if game_round > 0 else 'C') and ((memory[-1-1][ap] == 'D') if (game_round + 1) > 2 else 'C') else 'C')
        think_memory[136][2] += pay_off_matrix((memory[-1][ap] if game_round > 0 else 'C'),'D' if think_memory[136][4] > 0 else 'C')
        think_memory[136][3] += pay_off_matrix((memory[-1][ap] if game_round > 0 else 'C'),'D' if think_memory[136][5] == 1 else 'C')

        meta_win_strategies = ['gtft', 'tf2t', '2tft', 'g']
        meta_win_scores = [think_memory[136][i] for i in range(4)]
            
        best_score = max(meta_win_scores)

        meta_win_gtft = 1 if rnd.random() <= 0.2 else ((1 if memory[-1][ap] == 'C' else -1) if game_round > 0 else 1)
        meta_win_tf2t = -1 if ((memory[-1][ap] == 'D') if game_round > 0 else 1) and ((memory[-1-1][ap] == 'D') if (game_round + 1) > 2 else 1) else 1
        meta_win_2tft = -1 if think_memory[136][4] > 0 else 1
        meta_win_g = -1 if think_memory[136][5] == 1 else 1
        meta_win_majority = sum([meta_win_gtft, meta_win_tf2t, meta_win_2tft, meta_win_g][i] for i in range(4) if meta_win_scores[i] == best_score)
        if meta_win_majority < 0:
            return 'D'
        else:
            return 'C'
    elif strategy == 'Risky Q-Learner':
        brain = think_memory[139][0]
        another_factor = think_memory[139][1]

        A = min(12, game_round) - 1
        opp_inp = tuple(['C' for _ in range(12 - min(12, game_round))] + [memory[-1 - (A - i)][ap] for i in range(min(12, game_round))])
        self_inp = tuple(['C' for _ in range(12 - min(12, game_round))] + [memory[-1 - (A - i)][1-ap] for i in range(min(12, game_round))])
        inp = (opp_inp, self_inp)

        if brain.get(inp) == None:
            brain[inp] = [0,0]

        if game_round > 0 and another_factor[0][1] != 0.5:
            reward = 0
            if memory[-1][ap] == 'C':
                if memory[-1][1-ap] == 'C':
                    reward = RPST['R']
                else:
                    reward = RPST['T']
            else:
                if memory[-1][1-ap] == 'C':
                    reward = RPST['S']
                else:
                    reward = RPST['P']

            c_factor = brain[inp][0]
            d_factor = brain[inp][1]
            
            a = 0.9
            gamma = 0.9
            Q_old = brain[another_factor[0][0]][another_factor[0][1]]
            Q_new = Q_old + (a * (reward + ((gamma * max(c_factor, d_factor)) - Q_old)))
            brain[another_factor[0][0]][another_factor[0][1]] = Q_new

        if rnd.random() <= another_factor[1]:
            Action = rnd.choice(['C', 'D'])
            another_factor[0] = [inp, (Action == 'D')]
            return Action
        decay = 1
        min_epsilon = 0
        another_factor[1] = max(another_factor[1] * decay, min_epsilon)

        if brain[inp][0] == brain[inp][1]:
            Action = rnd.choice(['C', 'D'])
            another_factor[0] = [inp, (Action == 'D')]
            return Action
        best_choice = max(brain[inp])
        if best_choice == brain[inp][0]:
            another_factor[0] = [inp, 0]
            return 'C'
        else:
            another_factor[0] = [inp, 1]
            return 'D'
    elif strategy == 'Arrogant Q-Learner':
        brain = think_memory[140][0]
        another_factor = think_memory[140][1]

        A = min(12, game_round) - 1
        opp_inp = tuple(['C' for _ in range(12 - min(12, game_round))] + [memory[-1 - (A - i)][ap] for i in range(min(12, game_round))])
        self_inp = tuple(['C' for _ in range(12 - min(12, game_round))] + [memory[-1 - (A - i)][1-ap] for i in range(min(12, game_round))])
        inp = (opp_inp, self_inp)

        if brain.get(inp) == None:
            brain[inp] = [0,0]

        if game_round > 0 and another_factor[0][1] != 0.5:
            reward = 0
            if memory[-1][ap] == 'C':
                if memory[-1][1-ap] == 'C':
                    reward = RPST['R']
                else:
                    reward = RPST['T']
            else:
                if memory[-1][1-ap] == 'C':
                    reward = RPST['S']
                else:
                    reward = RPST['P']

            c_factor = brain[inp][0]
            d_factor = brain[inp][1]
            
            a = 0.9
            gamma = 0.1
            Q_old = brain[another_factor[0][0]][another_factor[0][1]]
            Q_new = Q_old + (a * (reward + ((gamma * max(c_factor, d_factor)) - Q_old)))
            brain[another_factor[0][0]][another_factor[0][1]] = Q_new

        if rnd.random() <= another_factor[1]:
            Action = rnd.choice(['C', 'D'])
            another_factor[0] = [inp, (Action == 'D')]
            return Action
        decay = 1
        min_epsilon = 0
        another_factor[1] = max(another_factor[1] * decay, min_epsilon)

        if brain[inp][0] == brain[inp][1]:
            Action = rnd.choice(['C', 'D'])
            another_factor[0] = [inp, (Action == 'D')]
            return Action
        best_choice = max(brain[inp])
        if best_choice == brain[inp][0]:
            another_factor[0] = [inp, 0]
            return 'C'
        else:
            another_factor[0] = [inp, 1]
            return 'D'
    elif strategy == 'Hesitant Q-Learner':
        brain = think_memory[141][0]
        another_factor = think_memory[141][1]

        A = min(12, game_round) - 1
        opp_inp = tuple(['C' for _ in range(12 - min(12, game_round))] + [memory[-1 - (A - i)][ap] for i in range(min(12, game_round))])
        self_inp = tuple(['C' for _ in range(12 - min(12, game_round))] + [memory[-1 - (A - i)][1-ap] for i in range(min(12, game_round))])
        inp = (opp_inp, self_inp)

        if brain.get(inp) == None:
            brain[inp] = [0,0]

        if game_round > 0 and another_factor[0][1] != 0.5:
            reward = 0
            if memory[-1][ap] == 'C':
                if memory[-1][1-ap] == 'C':
                    reward = RPST['R']
                else:
                    reward = RPST['T']
            else:
                if memory[-1][1-ap] == 'C':
                    reward = RPST['S']
                else:
                    reward = RPST['P']

            c_factor = brain[inp][0]
            d_factor = brain[inp][1]
            
            a = 0.1
            gamma = 0.9
            Q_old = brain[another_factor[0][0]][another_factor[0][1]]
            Q_new = Q_old + (a * (reward + ((gamma * max(c_factor, d_factor)) - Q_old)))
            brain[another_factor[0][0]][another_factor[0][1]] = Q_new

        if rnd.random() <= another_factor[1]:
            Action = rnd.choice(['C', 'D'])
            another_factor[0] = [inp, (Action == 'D')]
            return Action
        decay = 1
        min_epsilon = 0
        another_factor[1] = max(another_factor[1] * decay, min_epsilon)

        if brain[inp][0] == brain[inp][1]:
            Action = rnd.choice(['C', 'D'])
            another_factor[0] = [inp, (Action == 'D')]
            return Action
        best_choice = max(brain[inp])
        if best_choice == brain[inp][0]:
            another_factor[0] = [inp, 0]
            return 'C'
        else:
            another_factor[0] = [inp, 1]
            return 'D'
    elif strategy == 'Cautious Q-Learner':
        brain = think_memory[300][0]
        another_factor = think_memory[300][1]

        A = min(12, game_round) - 1
        opp_inp = tuple(['C' for _ in range(12 - min(12, game_round))] + [memory[-1 - (A - i)][ap] for i in range(min(12, game_round))])
        self_inp = tuple(['C' for _ in range(12 - min(12, game_round))] + [memory[-1 - (A - i)][1-ap] for i in range(min(12, game_round))])
        inp = (opp_inp, self_inp)

        if brain.get(inp) == None:
            brain[inp] = [0,0]

        if game_round > 0 and another_factor[0][1] != 0.5:
            reward = 0
            if memory[-1][ap] == 'C':
                if memory[-1][1-ap] == 'C':
                    reward = RPST['R']
                else:
                    reward = RPST['T']
            else:
                if memory[-1][1-ap] == 'C':
                    reward = RPST['S']
                else:
                    reward = RPST['P']

            c_factor = brain[inp][0]
            d_factor = brain[inp][1]
            
            a = 0.1
            gamma = 0.1
            Q_old = brain[another_factor[0][0]][another_factor[0][1]]
            Q_new = Q_old + (a * (reward + ((gamma * max(c_factor, d_factor)) - Q_old)))
            brain[another_factor[0][0]][another_factor[0][1]] = Q_new

        if rnd.random() <= another_factor[1]:
            Action = rnd.choice(['C', 'D'])
            another_factor[0] = [inp, (Action == 'D')]
            return Action
        decay = 1
        min_epsilon = 0
        another_factor[1] = max(another_factor[1] * decay, min_epsilon)

        if brain[inp][0] == brain[inp][1]:
            Action = rnd.choice(['C', 'D'])
            another_factor[0] = [inp, (Action == 'D')]
            return Action
        best_choice = max(brain[inp])
        if best_choice == brain[inp][0]:
            another_factor[0] = [inp, 0]
            return 'C'
        else:
            another_factor[0] = [inp, 1]
            return 'D'
    elif strategy == 'Thue Morse Inverse':
        if not game_round:
            return 'C'
        else:
            if think_memory[143][1] == 0:
                thuemorse_code = think_memory[143][0]
                for i in think_memory[143][0]:
                    if i == '0':
                        thuemorse_code += '1'
                    else:
                        thuemorse_code += '0'

                think_memory[143][0] = thuemorse_code
            
                think_memory[143][1] = len(thuemorse_code)
            think_memory[143][1] -= 1

            return 'D' if think_memory[143][0][len(think_memory[143][0]) - (think_memory[143][1] + 1)] == '1' else 'C'
    elif strategy == 'Fibonacci':
        if think_memory[144][3] == 0:
            if think_memory[144][0] == "C":
                think_memory[144][0] = "D"
            else:
                think_memory[144][0] = "C"

            think_memory[144][3] = think_memory[144][1] + think_memory[144][2]
            think_memory[144][1], think_memory[144][2] = think_memory[144][2], think_memory[144][3]

        think_memory[144][3] -= 1
        return think_memory[144][0]
    elif strategy == 'Cycler CCD':
        return ('C', 'C', 'D')[(game_round) % 3]
    elif strategy == 'Bully / Reverse Tit For Tat':
        if not game_round:
            return 'D'
        
        return 'D' if memory[-1][ap] == 'C' else 'C'
    elif strategy == 'Anti Tit For Tat / Psycho':
        if not game_round:
            return 'C'
        else:
            return 'D' if memory[-1][ap] == 'C' else 'C'
    elif strategy == 'Evolved Looker Up 1_1_1':
        if game_round > 0:
            inp = (memory[-1][1-ap], memory[-1][ap], memory[0][ap])

            lookup_table = {('C', 'C', 'C'): 'C', ('C', 'C', 'D'): 'D', ('C', 'D', 'C'): 'D', ('C', 'D', 'D'): 'D', ('D', 'C', 'C'): 'D', ('D', 'C', 'D'): 'C', ('D', 'D', 'C'): 'D', ('D', 'D', 'D'): 'D'}
        
            return lookup_table[inp]
        return 'C'
    elif strategy == 'Tricky Cooperator':
        """Almost always cooperates, but will try to trick the opponent by
        defecting.

        Defect once in a while in order to get a better payout.
        After 3 rounds, if opponent has not defected to a max history depth of
        10, defect.
        """
        
        def _has_played_enough_rounds_to_be_tricky():
            return (game_round) >= think_memory[145][0]

        def _opponents_has_cooperated_enough_to_be_tricky():
            rounds_to_be_checked = [memory[-1 - i][ap] for i in range(min(-think_memory[145][0], default_memory_looked))]
            return 'D' not in rounds_to_be_checked
        
        if (
            _has_played_enough_rounds_to_be_tricky()
            and _opponents_has_cooperated_enough_to_be_tricky()
        ):
            return 'D'
        return 'C'
    elif strategy == 'Cave':
        if not game_round:
            return 'C'
        
        if memory[-1][ap] == 'D':
            think_memory[148] += 1
        
        p_d = think_memory[148] / (game_round + 1)

        if (game_round + 1) > 39:
            if p_d > 0.39:
                return 'D'
            
        if (game_round + 1) > 29:
            if p_d > 0.65:
                return 'D'

        if (game_round + 1) > 19:
            if p_d > 0.79:
                return 'D'
        
        if memory[-1][ap] == 'D':
            if think_memory[148] > 17:
                return 'D'
            else:
                return rnd.choice(['C', 'D'])
        else:
            return 'C'
    elif strategy == 'Kluepfel':
        if not game_round:
            return 'C'
        
        if (game_round + 1) > 2:
            if memory[-2][1-ap] == 'D':
                if memory[-1][ap] == 'C':
                    think_memory[149][0] += 1 
                else:
                    think_memory[149][1] += 1 
            else:
                if memory[-1][ap] == 'C':
                    think_memory[149][2] += 1 
                else:
                    think_memory[149][3] += 1 
        
        if (game_round + 1) > 27:
            if think_memory[149][0] >= (
                think_memory[149][0] + think_memory[149][1]
            ) / 2 - 0.75 * np.sqrt(
                think_memory[149][0] + think_memory[149][1]
            ) and think_memory[149][3] >= (
                think_memory[149][3] + think_memory[149][2]
            ) / 2 - 0.75 * np.sqrt(
                think_memory[149][3] + think_memory[149][2]
            ):
                return 'D'

        kluepfel_eyes = [('C', 'C') for _ in range(3)] + memory
        inp = [kluepfel_eyes[-1 - i][ap] for i in range(3)]
        if inp[0] == inp[1] and inp[1] == inp[2]:
            return inp[0]

        A = rnd.random()
        if inp[0] == inp[1]:
            if A < 0.9:
                return inp[0]
            else:
                return 'C' if inp[0] == 'D' else 'D'
            
        if inp[0] == 'C':
            if A < 0.7:
                return inp[0]
            else:
                return 'C' if inp[0] == 'D' else 'D'

        if inp[0] == 'D':
            if A < 0.6:
                return inp[0]
            else:
                return 'C' if inp[0] == 'D' else 'D'
    elif strategy == 'Getzler':
        if not game_round: return 'C'

        if memory[-1][ap] == 'D':
            think_memory[150] += 1

        think_memory[150] *= 0.5

        if rnd.random() < think_memory[150]:
            return 'D'
        else:
            return 'C'
    elif strategy == 'Leyvraz':
        if (game_round + 1) <= 3:
            return 'C'
        else:
            prob_coop = {
                ('C', 'C', 'C'): 1.0,
                ('C', 'C', 'D'): 0.5,  
                ('C', 'D', 'C'): 0.0,  
                ('C', 'D', 'D'): 0.25,  
                ('D', 'C', 'C'): 1.0,  
                ('D', 'C', 'D'): 1.0,  
                ('D', 'D', 'C'): 1.0,  
                ('D', 'D', 'D'): 0.25,  
            }
            inp = tuple([memory[-1 - i][ap] for i in range(3)])
            return 'C' if rnd.random() <= prob_coop[inp] else 'D'
    elif strategy == 'Eatherley':
        if not game_round:
            return 'C'
        
        if memory[-1][ap] == 'D':
            think_memory[151] += 1
        else:
            return 'C'
        
        p_d = think_memory[151] / (game_round)

        if rnd.random() <= p_d:
            return 'D'
        else:
            return 'C'
    elif strategy == 'Hufford':
        turn = (game_round + 1)

        if turn == 1:
            return 'C'

        
        think_memory[84][2] = (think_memory[84][2] + 1) % 4
        me_two_moves_ago = 'C'
        if turn > 2:
            me_two_moves_ago = memory[-2][1-ap]
        if me_two_moves_ago == memory[-1][ap]:
            think_memory[84][0] += 1
            think_memory[84][1][think_memory[84][2]] = 1
        else:
            think_memory[84][1][think_memory[84][2]] = 0

        
        
        if turn < think_memory[84][5]:
            if memory[-1][ap] == 'C':
                think_memory[84][4] += 1
            else:
                think_memory[84][4] = 0
            if think_memory[84][4] >= think_memory[84][3]:
                think_memory[84][5] = turn
                if think_memory[84][4] == think_memory[84][3]:
                    return 'D'
        elif turn == think_memory[84][5] + 2:
            if memory[-1][ap] == 'C':
                think_memory[84][6] += 1
            else:
                think_memory[84][7] += 1
            think_memory[84][3] = (
                np.floor(
                    20.0 * think_memory[84][7] / think_memory[84][6]
                )
                + 1
            )
            think_memory[84][4] = 0
            return 'C'

        proportion_agree = think_memory[84][0] / turn
        last_four_num = sum(think_memory[84][1])
        if proportion_agree > 0.9 and last_four_num >= 4:
            return 'C'
        elif proportion_agree >= 0.625 and last_four_num >= 2:
            return memory[-1][ap]
        return 'D'
    elif strategy == 'Colbert':
        if (game_round + 1) <= 5:
            return 'C'
        elif (game_round + 1) == 6:
            return 'D'

        if think_memory[152] > 0:
            A = ('D', 'C', 'C')[3 - think_memory[152]]
            think_memory[152] -= 1
            return A
        if memory[-1][ap] == 'D':
            think_memory[152] = 3
            return 'D'
        return 'C'
    elif strategy == 'Mauk':
        if game_round > 0:
            if memory[-1][ap] == 'C':
                think_memory[153][0] += 1

        if (game_round + 1) <= 10:
            return 'C'
        else:
            p_c = think_memory[153][0] / (game_round + 1)

            if p_c < 0.7:
                think_memory[153][1] = 1

            if think_memory[153][1] == 1:
                return 'D'
            else:
                return memory[-1][ap]
    elif strategy == 'Mikkelson':
        turn = (game_round + 1)
        if turn == 1:
            return 'C'

        if memory[-1][ap] == 'C':
            think_memory[154][0] += 1
            if think_memory[154][0] > 8:
                think_memory[154][0] = 8
        else:
            think_memory[154][0] -= 2
            if think_memory[154][0] < -7:
                think_memory[154][0] = -7
            think_memory[154][1] += 1

        if turn == 2:
            return 'C'
        if think_memory[154][0] > 0:
            return 'C'
        if turn <= 10:
            think_memory[154][0] = 4
            return 'D'
        if think_memory[154][1] / turn >= 0.15:
            return 'D'
        return 'C'
    elif strategy == 'Rowsam':
        turn = (game_round + 1)

        if think_memory[155][0] == "Defect":
            return 'D'

        if think_memory[155][0] == "Coop Def Cycle 1":
            think_memory[155][0] = "Coop Def Cycle 2"
            return 'C'

        if think_memory[155][0] == "Coop Def Cycle 2":
            think_memory[155][0] = "Normal"
            return 'D'

        
        if turn % 18 == 0:
            if think_memory[155][1] >= 3:
                think_memory[155][1] -= 1

        
        if turn % 6 != 0:
            return 'C'

        points_per_turn = score[1-ap] / turn  
        if points_per_turn < 1.0:
            think_memory[155][1] += 5
        elif points_per_turn < 1.5:
            think_memory[155][1] += 3
        elif points_per_turn < 2.0:
            think_memory[155][1] += 2
        elif points_per_turn < 2.5:
            think_memory[155][1] += 1
        else:
            
            return 'C'

        if think_memory[155][1] >= 7:
            think_memory[155][0] = "Defect"
        else:
            
            think_memory[155][0] = "Coop Def Cycle 1"
        return 'D'
    elif strategy == 'Appold':
        turn = (game_round + 1)

        us_two_turns_ago = 'C' if turn <= 2 else memory[-2][1-ap]

        
        if turn > 1:
            think_memory[156][1][us_two_turns_ago] += 1
        if turn > 1 and memory[-1][ap] == 'C':
            think_memory[156][0][us_two_turns_ago] += 1

        if turn <= 4:
            return 'C'

        if memory[-1][ap] == 'D' and not think_memory[156][2]:
            think_memory[156][2] = True
            return 'C'

        
        
        prob_coop = (
            think_memory[156][0][us_two_turns_ago]
            / think_memory[156][1][us_two_turns_ago]
        )
        return 'C' if rnd.random() <= prob_coop else 'D'
    elif strategy == 'Almy':
        if (game_round + 1) <= 5:
            return 'C'
        else:
            inp = [(memory[-1 - ((4) - i)][ap] == 'D') for i in range(5)]
            t_d = sum(inp)

            if t_d >= 3:
                return 'D'
            else:
                return 'C'
    elif strategy == 'Ambuelh & Kickey':
        if not game_round:
            think_memory[158] = 0
            return 'C'
        elif (game_round + 1) == 2:
            return 'D' 
        elif (game_round + 1) == 3:
            return 'C'
        elif (game_round + 1) == 4:
            
            if memory[-1][ap] == 'C':
                think_memory[158] = 1 
            return memory[-1][ap]
        else:
            if think_memory[158] == 1:
                return 'D' if ((game_round + 1) % 3 == 0) else 'C'
            return memory[-1][ap]
    elif strategy == 'Feathers':
        if (game_round + 1) <= 2:
            return 'C'
        else:
            if (memory[-1][ap] == 'D') and (memory[-1-1][ap] == 'D'):
                return 'D'
            else:
                return 'C'
    elif strategy == 'Pinkley':
        if (game_round + 1) <= 3:
            return 'C'
        def pay_off_matrix(x,y):
            if x == 'C':
                if y == 'C':
                    return RPST['R']
                else:
                    return RPST['T']
            else:
                if y == 'C':
                    return RPST['S']
                else:
                    return RPST['P']
        inp = [pay_off_matrix(memory[-1 - ((2) - i)][ap],memory[-1 - ((2) - i)][1-ap]) for i in range(3)]
        A = sum(inp) / 3

        if A < 2:
            return 'D'
        else:
            return memory[-1][ap]
    elif strategy == 'Pebley':
        if not game_round:
            return 'C'
        
        if think_memory[159] > 0:
            think_memory[159] -= 1
            return 'D'
        else:
            if memory[-1][ap] == 'D':
                think_memory[159] = 1
                return 'D'
            else:
                return 'C'
    elif strategy == 'Falk & Lanqsted':
        if not game_round:
            return 'C'
        else:
            if think_memory[160][0] == 1:
                return 'D'
            else:
                if (memory[-1][ap] == 'D') and (think_memory[160][1] == 0):
                    think_memory[160][1] = (game_round + 1)
                    return 'C'

                if think_memory[160][1] > 0:
                    if (((game_round + 1) - think_memory[160][1]) <= 3) and (memory[-1][ap] == 'D'):
                        think_memory[160][0] = 1
                        return 'D'
                    
                    if (((game_round + 1) - think_memory[160][1]) > 3):
                        think_memory[160][1] = 0
                
                return memory[-1][ap]
    elif strategy == 'Weiderman':
        if not game_round:
            return 'C'
        else:
            if memory[-1][ap] == 'D':
                think_memory[161] = 1
            
            if think_memory[161] == 1:
                return 'D'
            else:
                return 'C'
    elif strategy == 'Dawes & Batell':
        if not game_round:
            return 'C'
        elif (game_round + 1) <= (tournament_avg_last_round - 1):
            if memory[-1][ap] == 'D':
                think_memory[162] = 1
            
            if think_memory[162] == 1:
                return 'D'
            else:
                return 'C'
        else:
            return 'D'
    elif strategy == 'Lefevre':
        if not game_round:
            return 'C'
        else:
            if (memory[-1][1-ap] == 'D') and (memory[-1][ap] == 'C'):
                return 'C'
            else:
                return memory[-1][ap]
    elif strategy == 'Quayle':
        if (game_round + 1) <= (0.5 * tournament_avg_last_round):
            if not game_round:
                return 'C'
            
            return memory[-1][ap]
        else:
            if score[1-ap] < ((game_round) * 2.5):
                if rnd.random() <= 0.15:
                    return 'D'
                else:
                    return memory[-1][ap]
            else:
                return memory[-1][ap]
    elif strategy == 'Anderson':
        if (game_round + 1) <= (tournament_avg_last_round - 10):
            if not game_round:
                return 'C'
            
            return memory[-1][ap]
        else:
            return 'D'
    elif strategy == 'Zimmerman':
        if (game_round + 1) <= (tournament_avg_last_round - 5):
            if not game_round:
                return 'C'
            
            return memory[-1][ap]
        else:
            return 'D'
    elif strategy == 'Newman':
        
        if (game_round + 1) <= (tournament_avg_last_round - 10):
            if not game_round:
                return 'C'
            if memory[-1][ap] == 'D':
                think_memory[163][0] += 1
            return memory[-1][ap]
        else:
            
            if (think_memory[163][0] / (tournament_avg_last_round - 10)) < 0.05:
                return 'D'
            return memory[-1][ap]
    elif strategy == 'Jones':
        if not game_round:
            think_memory[197] = 0 
            return 'C'
            
        if memory[-1][ap] == 'D':
            think_memory[197] = 1 
            
        if think_memory[197] == 0 and ((game_round + 1) % 50 == 0):
            return 'D'
            
        return memory[-1][ap]
    elif strategy == 'Shurmann':
        if (game_round + 1) <= 2:
            return 'C'
        else:
            if memory[-1][ap] == 'C':
                return 'C'
            else:
                if memory[-1-1][ap] == 'C':
                    return 'C' if rnd.random() <= 0.3 else 'D'
                else:
                    return 'D'
    elif strategy == 'Nussbacher':
        if not game_round:
            return 'C'
        else:
            if memory[-1][1-ap] == 'D':
                if ((game_round + 1) >= 3) and ((memory[-1][ap] == 'C') and (memory[-1-1][ap] == 'C')):
                    return 'C'
                else:
                    return 'D'
            else:
                return memory[-1][ap]
    elif strategy == 'Batell':
        if (game_round + 1) <= 20:
            if not game_round:
                return 'C'
            if memory[-1][ap] == 'D':
                think_memory[164][0] += 1
            return memory[-1][ap]
        elif (game_round + 1) == 21:
            
            think_memory[164][1] = 'exploit' if think_memory[164][0] == 0 else 'tft'
            return memory[-1][ap]
        else:
            if think_memory[164][1] == 'exploit':
                return 'D' if ((game_round + 1) % 5 == 0) else 'C'
            return memory[-1][ap]
    elif strategy == 'Smith':
        if (game_round + 1) > 2:
            if memory[-1-1][1-ap] == 'C':
                think_memory[165][0] += 1
                if memory[-1][ap] == 'C':
                    think_memory[165][1] += 1
            else:
                think_memory[165][2] += 1
                if memory[-1][ap] == 'C':
                    think_memory[165][3] += 1

        if (game_round + 1) <= 10:
            if not game_round:
                return 'C'
            return memory[-1][ap]
        else:
            if memory[-1][1-ap] == 'C':
                A = think_memory[165][1] / (think_memory[165][0] + 1e-9)
            else:
                A = think_memory[165][3] / (think_memory[165][2] + 1e-9)

            if A > 0.65:
                return 'C'
            else:
                return 'D'
    elif strategy == 'Leyland':
        if (game_round + 1) <= 2:
            return 'C'
        else:
            if think_memory[166] > 0:
                think_memory[166] -= 1
                return 'D'

            leyland_eyes = [('C', 'C') for _ in range(4)] + memory
            inp = tuple([leyland_eyes[-1 - ((3) - i)][ap] for i in range(4)])

            if inp in (("C", "D", "C", "D"), ("D", "C", "D", "C")):
                think_memory[166] = 1
                return 'D'
            else:
                return leyland_eyes[-1][ap]
    elif strategy == 'Mcgurrin':
        if not game_round:
            return 'C'
        else:
            if memory[-1][ap] == 'C':
                return 'C'
            else:
                if score[1-ap] > score[ap]:
                    return 'C'
                else:
                    return 'D'
    elif strategy == 'Hollander':
        if (game_round + 1) <= 10:
            return 'C'
        else:
            if ((game_round + 1) % 10) == 1:
                current_score = score[1-ap]
                score_diff = current_score - think_memory[167][0]
                think_memory[167][0] = current_score
                
                
                think_memory[167][1] = 1 if score_diff < 20 else 0
            
            if think_memory[167][1] == 1:
                return 'D' if rnd.random() < 0.4 else memory[-1][ap]
            return memory[-1][ap]
    elif strategy == 'Smoody':
        if (game_round + 1) <= 16:
            if game_round > 0:
                think_memory[168][0][(game_round + 1) - 2] = (memory[-1][ap] == 'D')
            return 'C'
        def smoody_set_new_memory():
            if (game_round + 1) > 16:
                for i in range(14):
                    think_memory[168][0][i] = think_memory[168][0][i+1]
                think_memory[168][0][14] = (memory[-1][ap] == 'D')

        if think_memory[168][1] > 0:
            think_memory[168][1] -= 1
            smoody_set_new_memory()
            return 'D'
        
        A = sum(think_memory[168][0])
        if A > 5:
            think_memory[168][1] = 14
            smoody_set_new_memory()
            return 'D'
        else:
            smoody_set_new_memory()
            return memory[-1][ap]
    elif strategy == 'Snodgrass':
        if (game_round + 1) <= 11:
            if game_round > 0:
                think_memory[169][0][(game_round + 1) - 2] = (memory[-1][ap] == 'D')
            return 'C'
        def snodgrass_set_new_memory():
            if (game_round + 1) > 11:
                for i in range(9):
                    think_memory[169][0][i] = think_memory[169][0][i+1]
                think_memory[169][0][9] = (memory[-1][ap] == 'D')

        if think_memory[169][1] > 0:
            think_memory[169][1] -= 1
            snodgrass_set_new_memory()
            return 'D'
        
        A = sum(think_memory[169][0])
        if A > 3:
            think_memory[169][1] = 9
            snodgrass_set_new_memory()
            return 'D'
        else:
            snodgrass_set_new_memory()
            return memory[-1][ap]
    elif strategy == 'Duisman':
        if not game_round:
            return 'C'
        else:
            if memory[-1][ap] == 'C':
                think_memory[170] += 1
            else:
                think_memory[170] -= 2
            
            if think_memory[170] < 0:
                return 'D'
            else:
                return 'C'
    elif strategy == 'Robertson':
        if (game_round + 1) <= 41:
            if game_round > 0:
                think_memory[171][(game_round + 1) - 2] = (memory[-1][ap] == 'D')
            return 'C'
        def robertson_set_new_memory():
            if (game_round + 1) > 41:
                for i in range(39):
                    think_memory[171][i] = think_memory[171][i+1]
                think_memory[171][39] = (memory[-1][ap] == 'D')
        
        A1 = sum(think_memory[171][i] for i in range(20))
        A2 = sum(think_memory[171][i + 19] for i in range(20))

        if memory[-1][ap] == 'C':
            robertson_set_new_memory()
            return 'C'
        else:
            if A1 > A2:
                robertson_set_new_memory()
                return 'D'
            else:
                robertson_set_new_memory()
                return 'D' if rnd.random() <= 0.75 else 'C'
    elif strategy == 'Rabbie':
        if not game_round: return 'C'
        if memory[-1][ap] == 'C':
            return 'C'
        else:
            return 'D' if rnd.random() <= 0.7 else 'C'
    elif strategy == 'Hall':
        if not game_round:
            return 'C'
        else:
            if memory[-1][ap] == 'D':
                if think_memory[172] > 0:
                    think_memory[172] += 1
                else:
                    think_memory[172] += 2
            
            if think_memory[172] > 0:
                think_memory[172] -= 1
                return 'D'
            else:
                return 'C'
    elif strategy == 'Friedland':
        if not game_round:
            return 'C'
        else:
            if (memory[-1][1-ap] == 'D') and (memory[-1][ap] == 'C'):
                return 'D'
            else:
                return memory[-1][ap]
    elif strategy == 'Hotz':
        if (game_round + 1) <= 2:
            return 'C'
        else:
            if memory[-1][ap] != memory[-1-1][ap]:
                return 'D'
            else:
                return memory[-1][ap]
    elif strategy == 'Grisell / Go By Majority / Soft Go By Majority':
        if not game_round:
            return 'C'
        
        if memory[-1][ap] == 'C':
            think_memory[26][0] += 1
        else:
            think_memory[26][1] += 1
        t_c = think_memory[26][0]
        t_d = think_memory[26][1]
        if t_c >= t_d:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Remorseful Prober':
        if not game_round:
            return 'C'

        if memory[-1][ap] == 'D':
            if think_memory[176]:
                think_memory[176] = False
                return 'C'
            return 'D'
        
        if rnd.random() <= 0.9:
            think_memory[176] = False
            return 'C'
        think_memory[176] = True
        return 'D'
    elif strategy == 'Loyal Foomii Version':
        if (game_round + 1) <= 2:
            return 'C'
        else:
            return memory[-1][ap]
    elif strategy == 'Vandal Foomii Version':
        if not game_round:
            return 'C'
        
        if memory[-1][ap] == 'D':
            think_memory[191] += 1

        if think_memory[191] >= 3:
            return 'D'

        if (game_round) % 4 == 3:
            return 'D'
        else:
            return 'C'
    elif strategy == 'Gold Digger Foomii Version':
        if game_round > 0:
            if memory[-1][ap] == 'D':
                think_memory[192] += 1

        if (game_round + 1) <= 2:
            return 'D'
        
        if think_memory[192] == 0:
            return 'D'

        if (game_round + 1) == 3:
            return 'C'

        return memory[-1][ap]
    elif strategy == 'Trusty Foomii Version':
        if (game_round + 1) > 5:
            if think_memory[193] == 1:
                return 'D'
            elif memory[-1][ap] == 'D':
                think_memory[193] = 1
                return 'D'
            else:
                return 'C'
        else:
            if game_round > 0:
                if memory[-1][ap] == 'D':
                    think_memory[193] = 1
            return 'C'
    elif strategy == 'Mirror Foomii Version':
        if (game_round + 1) <= 2:
            return 'C'
        return memory[-2][ap]
    elif strategy == 'Bayes Foomii Version':
        if game_round > 0:
            if memory[-1][ap] == 'D':
                think_memory[194] += 1

        if (game_round + 1) <= 12:
            return 'C'
        
        p_d = think_memory[194] / (game_round)

        if p_d >= 0.55:
            return 'D'
        else:
            return memory[-1][ap]
    elif strategy == 'Forgiver The Nerd Of AI Version':
        if game_round > 0:
            if memory[-1][ap] == 'D':
                return 'D'
        return 'C'
    elif strategy == 'Calculator The Nerd Of AI Version':
        if game_round > 0:
            if memory[-1][ap] == 'D':
                think_memory[200] += 1

        
        p_d = (think_memory[200] / (game_round)) if game_round > 0 else 0.5

        if p_d > 0.5:
            return 'D'

        return 'C'
    elif strategy == 'Opportunist The Nerd Of AI Version':
        if not game_round:
            return 'C'

        if memory[-1][ap] == 'C':
            return 'D'

        return 'D'
    elif strategy == 'Long Term Strategist The Nerd Of AI Version':
        if not game_round:
            return 'C'
        
        inp = tuple([(memory[-1 - ((min(3, game_round) - 1) - i)][ap] == 'D') for i in range(min(3, game_round))])
        if sum(inp) >= 2:
            return 'D'

        return 'C'
    elif strategy == 'Manipulator The Nerd Of AI Version':
        if (game_round + 1) <= 3:
            return 'C'

        return 'D'
    elif strategy == 'Retaliate':
        retaliation_threshold = 0.1
        if game_round:
            last_round = (memory[-1][1-ap], memory[-1][ap])
            think_memory[203][last_round] += 1
        CD_count = think_memory[203][('C', 'D')]
        DC_count = think_memory[203][('D', 'C')]
        if CD_count > DC_count * retaliation_threshold:
            return 'D'
        return 'C'
    elif strategy == 'Limited Retaliate':
        retaliation_threshold = 0.1

        """
        If the opponent has played D to my C more often than x% of the time
        that I've done the same to him, retaliate by playing D but stop doing
        so once I've hit the retaliation limit.
        """

        if game_round:
            last_round = (memory[-1][1-ap], memory[-1][ap])
            think_memory[204][3][last_round] += 1
        CD_count = think_memory[204][3][('C', 'D')]
        DC_count = think_memory[204][3][('D', 'C')]
        if CD_count > DC_count * retaliation_threshold:
            think_memory[204][0] = True
        else:
            think_memory[204][0] = False
            think_memory[204][1] = 0

        if think_memory[204][0]:
            if think_memory[204][1] < think_memory[204][2]:
                think_memory[204][1] += 1
                return 'D'
            else:
                think_memory[204][1] = 0
                think_memory[204][0] = False

        return 'C'
    elif strategy == 'Cooperator Hunter':
        if game_round > 0:
            if memory[-1][ap] == 'C':
                think_memory[205] += 1

        if (game_round + 1) > 4 and (game_round) == think_memory[205]:
            return 'D'
        return 'C'

    elif strategy == 'Defector Hunter':
        if game_round > 0:
            if memory[-1][ap] == 'D':
                think_memory[206] += 1

        if (game_round + 1) > 4 and (game_round) == think_memory[206]:
            return 'D'
        return 'C'

    elif strategy == 'Alternator Hunter':
        if (game_round + 1) <= 6:
            return 'C'

        inp = tuple([memory[-1 - (5 - i)][ap] for i in range(6)])
        if inp not in (("C", "D", "C", "D", "C", "D"), ("D", "C", "D", "C", "D", "C")):
            think_memory[157] = False

        if think_memory[157]:
            return 'D'
        return 'C'
    elif strategy == 'Cycler CCCCCD':
        cyclic_pattern = 'CCCCCD'
        return cyclic_pattern[game_round % len(cyclic_pattern)]
    elif strategy == 'Cycler CCCD':
        cyclic_pattern = 'CCCD'
        return cyclic_pattern[game_round % len(cyclic_pattern)]
    elif strategy == 'Cycler CCCDCD':
        cyclic_pattern = 'CCCDCD'
        return cyclic_pattern[game_round % len(cyclic_pattern)]
    elif strategy == 'Cycler DC':
        cyclic_pattern = 'DC'
        return cyclic_pattern[game_round % len(cyclic_pattern)]
    elif strategy == 'Cycler DDC':
        cyclic_pattern = 'DDC'
        return cyclic_pattern[game_round % len(cyclic_pattern)]
    elif strategy == 'Worse & Worse / Worse & Worse 1':
        return 'D' if rnd.random() <= (game_round / 1000) else 'C'
    elif strategy == 'Better & Better':
        return 'C' if rnd.random() <= (game_round / 1000) else 'D'
    elif strategy == 'Handshake':
        if (game_round + 1) <= 2:
            return ('C', 'D')[game_round]

        if (game_round + 1) == 3:
            think_memory[207] = [memory[-1 - ((1) - i)][ap] for i in range(2)]

        if think_memory[207] == ['C', 'D']:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Fortress 3':
        if game_round == 0:
            return 'D'
        instruction_string = '1:D1|D2,2:D1|C3,3:C3|D1'
        def translating(x):
            rule = {}
            semua_mode = x.strip().rstrip(',').split(',')
            for item in semua_mode:
                if not item: continue
                nomor_mode, isi_aksi = item.split(':')
                aksi_0_str, aksi_1_str = isi_aksi.split('|')
                
                aksi_0 = [aksi_0_str[0], aksi_0_str[1:]]
                aksi_1 = [aksi_1_str[0], aksi_1_str[1:]]
                rule[nomor_mode] = [aksi_0, aksi_1]
            return rule
        instruction = translating(instruction_string)
        fortress_rule = instruction[think_memory[208]][(memory[-1][ap] == 'D')]
        think_memory[208] = fortress_rule[1]
        return fortress_rule[0]
    elif strategy == 'Fortress 4':
        if game_round == 0:
            return 'D'
        instruction_string = '1:D1|D2,2:D1|D3,3:D1|C4,4:C4|D1'
        def translating(x):
            rule = {}
            semua_mode = x.strip().rstrip(',').split(',')
            for item in semua_mode:
                if not item: continue
                nomor_mode, isi_aksi = item.split(':')
                aksi_0_str, aksi_1_str = isi_aksi.split('|')
                
                aksi_0 = [aksi_0_str[0], aksi_0_str[1:]]
                aksi_1 = [aksi_1_str[0], aksi_1_str[1:]]
                rule[nomor_mode] = [aksi_0, aksi_1]
            return rule
        instruction = translating(instruction_string)
        fortress_rule = instruction[think_memory[209]][(memory[-1][ap] == 'D')]
        think_memory[209] = fortress_rule[1]
        return fortress_rule[0]
    elif strategy == 'Tricky Defector':
        if game_round > 0:
            if memory[-1][ap] == 'C':
                think_memory[210] = 1
        if (game_round + 1) > 3:
            inp = tuple([memory[-1 - i][ap] for i in range(3)])
            if think_memory[210] == 1 and 'C' not in inp:
                return 'C'
        return 'D'
    elif strategy == 'Punisher':
        if game_round > 0:
            if memory[-1][ap] == 'D':
                think_memory[211][3] += 1

        """
        Begins by playing C, then plays D for an amount of rounds proportional
        to the opponents historical '%' of playing D if the opponent ever
        plays D
        """

        if think_memory[211][2] >= think_memory[211][0]:
            think_memory[211][2] = 0
            think_memory[211][1] = False

        if think_memory[211][1]:
            think_memory[211][2] += 1
            return 'D'

        elif ((memory[-1][ap] == 'D') if game_round > 0 else False):
            think_memory[211][0] = (think_memory[211][3] * 20) // (game_round)
            think_memory[211][1] = True
            return 'D'

        return 'C'
    elif strategy == 'Resurrection':
        if game_round == 0:
            return 'C'
        if game_round >= 5:
            inp = [memory[-1 - i][1-ap] for i in range(5)]
            if inp == ['D', 'D', 'D', 'D', 'D']:
                return 'D'
        return memory[-1][ap]
    elif strategy == 'Detective':
        if (game_round + 1) <= 4:
            return ('C', 'D', 'C', 'C')[game_round]
        
        elif (game_round + 1) == 5:
            inp = tuple([(memory[-1 - (3 - i)][ap] == 'D') for i in range(4)])
            total_D = sum(inp)
            if total_D > 0:
                think_memory[213] = 1
            else:
                think_memory[213] = 0

        if think_memory[213] == 1:
            return memory[-1][ap]
        else:
            return 'D'
    elif strategy == 'AON2':
        if (game_round + 1) <= 2:
            return 'C'
        return 'C' if rnd.random() <= {('CC', 'CC'): 1, ('CC', 'CD'): 0, ('CC', 'DC'): 0, ('CC', 'DD'): 0, ('CD', 'CC'): 0, ('CD', 'CD'): 1, ('CD', 'DC'): 0, ('CD', 'DD'): 0, ('DC', 'CC'): 0, ('DC', 'CD'): 0, ('DC', 'DC'): 1, ('DC', 'DD'): 0, ('DD', 'CC'): 0, ('DD', 'CD'): 0, ('DD', 'DC'): 0, ('DD', 'DD'): 1}[(memory[-2][1-ap] + memory[-1][1-ap], memory[-2][ap] + memory[-1][ap])] else 'D'
    elif strategy == 'Delayed AON1':
        if (game_round + 1) <= 2:
            return 'C'
        return 'C' if rnd.random() <= {('CC', 'CC'): 1, ('CC', 'CD'): 0, ('CC', 'DC'): 0, ('CC', 'DD'): 0, ('CD', 'CC'): 0, ('CD', 'CD'): 1, ('CD', 'DC'): 0, ('CD', 'DD'): 1, ('DC', 'CC'): 0, ('DC', 'CD'): 0, ('DC', 'DC'): 1, ('DC', 'DD'): 0, ('DD', 'CC'): 0, ('DD', 'CD'): 1, ('DD', 'DC'): 0, ('DD', 'DD'): 1}[(memory[-2][1-ap] + memory[-1][1-ap], memory[-2][ap] + memory[-1][ap])] else 'D'
    elif strategy == 'Meta Minority':
        meta_mino_tft = (1 if memory[-1][ap] == 'C' else -1) if game_round > 0 else 1
        meta_mino_pvlv = (1 if memory[-1][ap] == memory[-1][1-ap] else -1) if game_round > 0 else 1
        meta_mino_allc = 1
        meta_mino_alld = -1

        meta_mino_majority = meta_mino_tft + meta_mino_pvlv + meta_mino_allc + meta_mino_alld

        if meta_mino_majority < 0:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Meta Mixer':
        meta_mixe_tft = (1 if memory[-1][ap] == 'C' else -1) if game_round > 0 else 1
        meta_mixe_pvlv = (1 if memory[-1][ap] == memory[-1][1-ap] else -1) if game_round > 0 else 1
        meta_mixe_allc = 1
        meta_mixe_alld = -1

        meta_mixe_random = rnd.choice([meta_mixe_tft, meta_mixe_pvlv, meta_mixe_allc, meta_mixe_alld])

        if meta_mixe_random < 0:
            return 'D'
        else:
            return 'C'
    elif strategy == 'Win-Shift, Lose-Stay':
        if not game_round:
            return 'D'
        if memory[-1][ap] != memory[-1][1-ap]:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Adaptive':
        if game_round > 0:
            think_memory[231][0][memory[-1][ap]] += score[1-ap] - think_memory[231][1]
        think_memory[231][1] = score[1-ap]
        if (game_round + 1) <= 11:
            return 'C' if (game_round + 1) <= 6 else 'D'
        
        if think_memory[231][0]['C'] > think_memory[231][0]['D']:
            return 'C'
        return 'D'
    elif strategy == 'Adaptor Brief':
        perr = 0.01
        delta = {('C', 'C'):0.0, ('C', 'D'):0.992107, ('D', 'C'):-1.001505, ('D', 'D'):-0.638734}
        if game_round > 0:
            last_round = (memory[-1][1-ap], memory[-1][ap])
            think_memory[233] += delta[last_round]

        def step(x):
            return (1 if x > 0 else 0) if x != 0 else 0.5
        
        p = perr + (1 - 2 * perr) * (
            step(think_memory[233] + 1) - step(think_memory[233] - 1)
        )

        if rnd.random() <= p:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Adaptor Long':
        perr = 0.01
        delta = {('C', 'C'): 0.0, ('C', 'D'): 1.858883, ('D', 'C'): 1.888159, ('D', 'D'): -0.995703}
        if game_round > 0:
            last_round = (memory[-1][1-ap], memory[-1][ap])
            think_memory[234] += delta[last_round]

        def step(x):
            return (1 if x > 0 else 0) if x != 0 else 0.5
        
        p = perr + (1 - 2 * perr) * (
            step(think_memory[234] + 1) - step(think_memory[234] - 1)
        )

        if rnd.random() <= p:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Adaptive Pavlov 2006':
        if (game_round + 1) <= 6:
            if not game_round:
                return 'C'
            return memory[-1][ap]

        inp = [memory[-1 - i][ap] for i in range(6)]

        if (game_round) % 6 == 0:
            if inp == ['C'] * 6:
                think_memory[235] = "Cooperative"
            if inp == ['D'] * 6:
                think_memory[235] = "ALLD"
            if inp == ['D', 'C', 'D', 'C', 'D', 'C']:
                think_memory[235] = "STFT"
            if inp == ['D', 'D', 'C', 'D', 'D', 'C']:
                think_memory[235] = "PavlovD"
            if not think_memory[235]:
                think_memory[235] = "Random"

        if think_memory[235] in ["Random", "ALLD"]:
            return 'D'
        if think_memory[235] == "STFT":
            if (game_round) % 6 in [0, 1]:
                return 'C'
            
            if memory[-1][ap] == 'D':
                return 'D'
        if think_memory[235] == "PavlovD":
            
            if (game_round) % 6 == 0:
                return 'D'
        if think_memory[235] == "Cooperative":
            
            if memory[-1][ap] == 'D':
                return 'D'
        return 'C'
    elif strategy == 'Adaptive Pavlov 2011':
        if (game_round + 1) <= 6:
            if not game_round:
                return 'C'
            return memory[-1][ap]

        inp = [memory[-1 - i][ap] for i in range(6)]

        if (game_round) % 6 == 0:
            if inp == ['C'] * 6:
                think_memory[236] = "Cooperative"
            if inp.count('D') >= 4:
                think_memory[236] = "ALLD"
            if inp.count('D') == 3:
                think_memory[236] = "STFT"
            if not think_memory[236]:
                think_memory[236] = "Random"

        if think_memory[236] in ["Random", "ALLD"]:
            return 'D'
        if think_memory[236] == "STFT":
            return 'D' if (memory[-2][ap], memory[-1][ap]) == ('D', 'D') else 'C'
        if think_memory[236] == "Cooperative":
            
            return memory[-1][ap]
    elif strategy == 'Appeaser':
        if not game_round: return 'C'
        if memory[-1][ap] == 'D':
            if memory[-1][1-ap] == 'C':
                return 'D'
            else:
                return 'C'
        return memory[-1][1-ap]
    elif strategy == 'Average Copier':
        if not game_round:
            return rnd.choice(['C', 'D'])

        if memory[-1][ap] == 'C':
            think_memory[237] += 1

        p_c = think_memory[237] / (game_round)
        return 'C' if rnd.random() <= p_c else 'D'
    elif strategy == 'Grofman V2':
        if (game_round + 1) <= 2:
            return 'C'
        elif 2 < (game_round + 1) <= 7:
            return memory[-1][ap]
        else:
            t_d = sum([(memory[-1 - i][ap] == 'D') for i in range(1, min(game_round, 8))])
            if memory[-1][ap] == 'C' and t_d <= 2:
                return 'C'
            if memory[-1][ap] == 'D' and t_d <= 1:
                return 'C'
            return 'D'
    elif strategy == 'Tideman & Chieruzzi V2':
        def _fresh_start():
            """Give the opponent a fresh start by forgetting the past"""
            score[1-ap] = 0
            score[ap] = 0
            think_memory[238][2] = 0
            think_memory[238][3] = 0

        """Actual strategy definition that determines player's action."""
        current_round = (game_round + 1)

        if current_round == 1:
            return 'C'

        if memory[-1][ap] == 'D':
            think_memory[238][4] += 1

        
        if think_memory[238][1]:
            _fresh_start()
            think_memory[238][0] = current_round
            think_memory[238][1] = False
            return 'C'  

        opponent_CDd = False

        opponent_two_turns_ago = 'C'  
        if (game_round) >= 2:
            opponent_two_turns_ago = memory[-2][ap]
        
        if opponent_two_turns_ago == 'C' and memory[-1][ap] == 'D':
            opponent_CDd = True
            think_memory[238][2] += think_memory[238][3]
            think_memory[238][3] += 5

        
        if score[1-ap] - score[ap] >= think_memory[238][2]:
            return 'C'

        
        if (not opponent_CDd) and current_round - think_memory[238][0] >= 10:
            
            N = game_round
            
            std_deviation = (N ** (1 / 2)) / 2
            lower = N / 2 - 3 * std_deviation
            upper = N / 2 + 3 * std_deviation
            if think_memory[238][4] <= lower or think_memory[238][4] >= upper:
                
                think_memory[238][1] = True
                return 'C'  

        return 'D'
    elif strategy == 'Back Stabber':
        if game_round > 0:
            if memory[-1][ap] == 'D':
                think_memory[239] += 1
        if think_memory[239] > 3:
            return 'D'
        return 'C'
    elif strategy == 'Double Crosser':
        if game_round > 0:
            if memory[-1][ap] == 'D':
                think_memory[240][0] += 1
                if (game_round + 1) <= 7:
                    think_memory[240][1] = 1

        if (think_memory[240][1] == 0) and (7 < (game_round + 1) < 180):
            return 'D' if (memory[-2][ap], memory[-1][ap]) == ('D', 'D') else 'C'
        if think_memory[240][0] > 3:
            return 'D'
        return 'C'
    elif strategy == 'Bush Mosteller':
        def stimulus_update():
            """
            Updates the stimulus attribute based on the opponent's history. Used by
            the strategy.

            Parameters

            opponent : axelrod.Player
                The current opponent
            """
            last_round = (memory[-1][1-ap], memory[-1][ap])

            previous_play = score[1-ap] - think_memory[241][5]

            think_memory[241][3] = (previous_play - think_memory[241][2]) / abs(
                (max(5,3,1,0) - think_memory[241][2])
            )
            
            
            
            if think_memory[241][3] < -1:
                think_memory[241][3] = -1

            
            if memory[-1][1-ap] == 'C':

                if think_memory[241][3] >= 0:
                    think_memory[241][0] += (
                        think_memory[241][4] * think_memory[241][3] * (1 - think_memory[241][0])
                    )

                elif think_memory[241][3] < 0:
                    think_memory[241][0] += (
                        think_memory[241][4] * think_memory[241][3] * think_memory[241][0]
                    )

            
            if memory[-1][1-ap] == 'D':
                if think_memory[241][3] >= 0:
                    think_memory[241][1] += (
                        think_memory[241][4] * think_memory[241][3] * (1 - think_memory[241][1])
                    )

                elif think_memory[241][3] < 0:
                    think_memory[241][1] += (
                        think_memory[241][4] * think_memory[241][3] * think_memory[241][1]
                    )

        """Actual strategy definition that determines player's action."""

        
        if (game_round) == 0:
            return 'C' if rnd.random() <= (
                think_memory[241][0] / (think_memory[241][0] + think_memory[241][1])
            ) else 'D'

        
        stimulus_update()

        return 'C' if rnd.random() <= (
            think_memory[241][0] / (think_memory[241][0] + think_memory[241][1])
        ) else 'D'
    elif strategy == 'Calculator':
        if (game_round + 1) <= 21:
            if game_round > 0:
                think_memory[242][0][(game_round + 1) - 2] = memory[-1][ap]
            if (game_round + 1) < 21:
                if not game_round:
                    return 'C'
                return memory[-1][ap] if rnd.random() > 0.1 else 'D'
        if (game_round + 1) == 21:
            for i in range(10):
                Calculator_n = (i + 1)
                is_cycle = True
                Calculator_list = think_memory[242][0][:(Calculator_n + 1)]
                Calculator_index = len(think_memory[242][0]) % Calculator_n
                for j in range(len(think_memory[242][0])):
                    if think_memory[242][0][j] != Calculator_list[j % Calculator_n]:
                        is_cycle = False
            
                if is_cycle:
                    think_memory[242][1] = is_cycle
                    break

        if think_memory[242][1]:
            return 'D'
        return memory[-1][ap]
    elif strategy == 'Anti Cycler':
        if (game_round + 1) <= len(think_memory[243][2]):
            return think_memory[243][2].pop(0)
        if think_memory[243][1] < think_memory[243][0]:
            think_memory[243][1] += 1
            return 'C'
        else:
            think_memory[243][0] += 1
            think_memory[243][1] = 0
            return 'D'
    elif strategy == 'DBS / Derived Belief Strategy':
        
        class Node(object):
            def get_siblings(self): raise NotImplementedError()
            def is_stochastic(self): raise NotImplementedError()

        class StochasticNode(Node):
            def __init__(self, own_action, pC, depth):
                self.pC = pC
                self.depth = depth
                self.own_action = own_action

            def get_siblings(self):
                return DeterministicNode(self.own_action, 'C', self.depth + 1), \
                       DeterministicNode(self.own_action, 'D', self.depth + 1)

            def is_stochastic(self):
                return True

        class DeterministicNode(Node):
            def __init__(self, action1, action2, depth):
                self.action1 = action1
                self.action2 = action2
                self.depth = depth

            def get_siblings(self, policy):
                return StochasticNode('C', policy[(self.action1, self.action2)], self.depth), \
                       StochasticNode('D', policy[(self.action1, self.action2)], self.depth)

            def is_stochastic(self):
                return False

            def get_value(self):
                values = {('C', 'C'): RPST['R'], ('C', 'D'): RPST['S'], ('D', 'C'): RPST['T'], ('D', 'D'): RPST['P']}
                return values[(self.action1, self.action2)]


        def action_to_int(action):
            return 1 if action == 'C' else 0

        def minimax_tree_search(begin_node, policy, max_depth):
            if begin_node.is_stochastic():
                siblings = begin_node.get_siblings()
                return begin_node.pC * minimax_tree_search(siblings[0], policy, max_depth) + \
                       (1 - begin_node.pC) * minimax_tree_search(siblings[1], policy, max_depth)
            else:
                if begin_node.depth == max_depth:
                    return begin_node.get_value()
                elif begin_node.depth == 0:
                    siblings = begin_node.get_siblings(policy)
                    return (
                        minimax_tree_search(siblings[0], policy, max_depth) + begin_node.get_value(),
                        minimax_tree_search(siblings[1], policy, max_depth) + begin_node.get_value(),
                    )
                elif begin_node.depth < max_depth:
                    siblings = begin_node.get_siblings(policy)
                    a = minimax_tree_search(siblings[0], policy, max_depth)
                    b = minimax_tree_search(siblings[1], policy, max_depth)
                    return max(a, b) + begin_node.get_value()

        def move_gen(outcome, policy, depth_search_tree=5):
            current_node = DeterministicNode(outcome[0], outcome[1], depth=0)
            values_of_choices = minimax_tree_search(current_node, policy, depth_search_tree)
            actions_tuple = ('C', 'D')
            return actions_tuple[values_of_choices.index(max(values_of_choices))]

        dbs = think_memory[245]

        

        
        my_idx = 1 - ap

        
        if len(memory) >= 2:
            two_moves_ago = (memory[-2][my_idx], memory[-2][ap])
            op_last_move = memory[-1][ap]

            for outcome, GF in dbs['history_by_cond'].items():
                G, F = GF
                if outcome == two_moves_ago:
                    G.append(1 if op_last_move == 'C' else 0)
                    F.append(1)
                else:
                    G.append(0)
                    F.append(0)

            r_plus = (two_moves_ago, op_last_move)
            r_minus = (two_moves_ago, 'D' if op_last_move == 'C' else 'C')

            
            if r_plus[0] not in dbs['Rc'].keys():
                opposite_action = 0 if r_plus[1] == 'C' else 1
                k = 1
                count = 0
                cond_G = dbs['history_by_cond'][r_plus[0]][0]
                cond_F = dbs['history_by_cond'][r_plus[0]][1]
                
                while k < len(cond_G) and not (cond_G[1:][-k] == opposite_action and cond_F[1:][-k] == 1):
                    if cond_F[1:][-k] == 1:
                        count += 1
                    k += 1
                
                if count >= dbs['promotion_threshold']:
                    dbs['Rc'][r_plus[0]] = action_to_int(r_plus[1])
                    dbs['violation_counts'][r_plus[0]] = 0

            
            if r_plus[0] in dbs['Rc'].keys():
                to_check = 'C' if dbs['Rc'][r_plus[0]] == 1 else 'D'
                if r_plus[1] == to_check:
                    dbs['violation_counts'][r_plus[0]] = 0
                elif r_minus[1] == to_check:
                    dbs['violation_counts'][r_plus[0]] = dbs['violation_counts'].get(r_plus[0], 0) + 1
                    if dbs['violation_counts'].get(r_minus[0], 0) >= dbs['violation_threshold']:
                        dbs['Rd'].update(dbs['Rc'])
                        dbs['Rc'].clear()
                        dbs['violation_counts'].clear()
                        dbs['v'] = 0

            r_plus_in_Rc = r_plus[0] in dbs['Rc'].keys() and dbs['Rc'][r_plus[0]] == action_to_int(r_plus[1])
            r_minus_in_Rd = r_minus[0] in dbs['Rd'].keys() and dbs['Rd'][r_minus[0]] == action_to_int(r_minus[1])

            if r_minus_in_Rd:
                dbs['v'] += 1
            if (dbs['v'] > dbs['reject_threshold']) or (r_plus_in_Rc and r_minus_in_Rd):
                dbs['Rd'].clear()
                dbs['v'] = 0

            
            Rp = {}
            all_cond = [('C', 'C'), ('C', 'D'), ('D', 'C'), ('D', 'D')]
            for outcome in all_cond:
                if (outcome not in dbs['Rc'].keys()) and (outcome not in dbs['Rd'].keys()):
                    G = dbs['history_by_cond'][outcome][0]
                    F = dbs['history_by_cond'][outcome][1]
                    discounted_g = 0
                    discounted_f = 0
                    alpha_k = 1
                    for g, f in zip(G[::-1], F[::-1]):
                        discounted_g += alpha_k * g
                        discounted_f += alpha_k * f
                        alpha_k = dbs['alpha'] * alpha_k
                    Rp[outcome] = discounted_g / discounted_f if discounted_f != 0 else 0.5

            
            dbs['Pi'] = {}
            dbs['Pi'].update(dbs['Rc'])
            dbs['Pi'].update(dbs['Rd'])
            dbs['Pi'].update(Rp)

        
        last_outcome = (memory[-1][my_idx], memory[-1][ap]) if game_round > 0 else ('C', 'C')
        return move_gen(last_outcome, dbs['Pi'], depth_search_tree=dbs['tree_depth'])
    elif strategy == 'Doubler':
        if game_round > 0:
            if memory[-1][ap] == 'C':
                think_memory[246][0] += 1
            else:
                think_memory[246][1] += 1
                if think_memory[246][0] <= (think_memory[246][1] * 2):
                    return 'D'
        
        return 'C'
    elif strategy == 'Predator':
        if game_round == 0:
            return 'C'
        instruction_string = '0:D0|D1,1:D2|D3,2:C4|D3,3:D5|C4,4:C2|D6,5:D7|D3,6:C7|D7,7:D8|D7,8:D8|D6'
        def translating(x):
            rule = {}
            semua_mode = x.strip().rstrip(',').split(',')
            for item in semua_mode:
                if not item: continue
                nomor_mode, isi_aksi = item.split(':')
                aksi_0_str, aksi_1_str = isi_aksi.split('|')
                
                aksi_0 = [aksi_0_str[0], aksi_0_str[1:]]
                aksi_1 = [aksi_1_str[0], aksi_1_str[1:]]
                rule[nomor_mode] = [aksi_0, aksi_1]
            return rule
        instruction = translating(instruction_string)
        finite_state_rule = instruction[think_memory[247]][(memory[-1][ap] == 'D')]
        think_memory[247] = finite_state_rule[1]
        return finite_state_rule[0]
    elif strategy == 'Pun 1':
        if game_round == 0:
            return 'D'
        instruction_string = '1:C2|C2,2:C1|D1'
        def translating(x):
            rule = {}
            semua_mode = x.strip().rstrip(',').split(',')
            for item in semua_mode:
                if not item: continue
                nomor_mode, isi_aksi = item.split(':')
                aksi_0_str, aksi_1_str = isi_aksi.split('|')
                
                aksi_0 = [aksi_0_str[0], aksi_0_str[1:]]
                aksi_1 = [aksi_1_str[0], aksi_1_str[1:]]
                rule[nomor_mode] = [aksi_0, aksi_1]
            return rule
        instruction = translating(instruction_string)
        finite_state_rule = instruction[think_memory[248]][(memory[-1][ap] == 'D')]
        think_memory[248] = finite_state_rule[1]
        return finite_state_rule[0]
    elif strategy == 'Raider':
        if game_round == 0:
            return 'D'
        instruction_string = '0:D2|D2,1:C1|D1,2:D0|C3,3:D0|C1'
        def translating(x):
            rule = {}
            semua_mode = x.strip().rstrip(',').split(',')
            for item in semua_mode:
                if not item: continue
                nomor_mode, isi_aksi = item.split(':')
                aksi_0_str, aksi_1_str = isi_aksi.split('|')
                
                aksi_0 = [aksi_0_str[0], aksi_0_str[1:]]
                aksi_1 = [aksi_1_str[0], aksi_1_str[1:]]
                rule[nomor_mode] = [aksi_0, aksi_1]
            return rule
        instruction = translating(instruction_string)
        finite_state_rule = instruction[think_memory[249]][(memory[-1][ap] == 'D')]
        think_memory[249] = finite_state_rule[1]
        return finite_state_rule[0]
    elif strategy == 'Ripoff':
        if game_round == 0:
            return 'D'
        instruction_string = '1:C2|C3,2:D1|C3,3:C3|D3'
        def translating(x):
            rule = {}
            semua_mode = x.strip().rstrip(',').split(',')
            for item in semua_mode:
                if not item: continue
                nomor_mode, isi_aksi = item.split(':')
                aksi_0_str, aksi_1_str = isi_aksi.split('|')
                
                aksi_0 = [aksi_0_str[0], aksi_0_str[1:]]
                aksi_1 = [aksi_1_str[0], aksi_1_str[1:]]
                rule[nomor_mode] = [aksi_0, aksi_1]
            return rule
        instruction = translating(instruction_string)
        finite_state_rule = instruction[think_memory[250]][(memory[-1][ap] == 'D')]
        think_memory[250] = finite_state_rule[1]
        return finite_state_rule[0]
    elif strategy == 'Usually Cooperates':
        if game_round == 0:
            return 'C'
        instruction_string = '1:C1|C2,2:D1|C1'
        def translating(x):
            rule = {}
            semua_mode = x.strip().rstrip(',').split(',')
            for item in semua_mode:
                if not item: continue
                nomor_mode, isi_aksi = item.split(':')
                aksi_0_str, aksi_1_str = isi_aksi.split('|')
                
                aksi_0 = [aksi_0_str[0], aksi_0_str[1:]]
                aksi_1 = [aksi_1_str[0], aksi_1_str[1:]]
                rule[nomor_mode] = [aksi_0, aksi_1]
            return rule
        instruction = translating(instruction_string)
        finite_state_rule = instruction[think_memory[251]][(memory[-1][ap] == 'D')]
        think_memory[251] = finite_state_rule[1]
        return finite_state_rule[0]
    elif strategy == 'Usually Defects':
        if game_round == 0:
            return 'D'
        instruction_string = '1:D2|D1,2:D1|C1'
        def translating(x):
            rule = {}
            semua_mode = x.strip().rstrip(',').split(',')
            for item in semua_mode:
                if not item: continue
                nomor_mode, isi_aksi = item.split(':')
                aksi_0_str, aksi_1_str = isi_aksi.split('|')
                
                aksi_0 = [aksi_0_str[0], aksi_0_str[1:]]
                aksi_1 = [aksi_1_str[0], aksi_1_str[1:]]
                rule[nomor_mode] = [aksi_0, aksi_1]
            return rule
        instruction = translating(instruction_string)
        finite_state_rule = instruction[think_memory[252]][(memory[-1][ap] == 'D')]
        think_memory[252] = finite_state_rule[1]
        return finite_state_rule[0]
    elif strategy == 'Solution B1':
        if game_round == 0:
            return 'D'
        instruction_string = '1:D2|D1,2:C2|C3,3:C3|C3'
        def translating(x):
            rule = {}
            semua_mode = x.strip().rstrip(',').split(',')
            for item in semua_mode:
                if not item: continue
                nomor_mode, isi_aksi = item.split(':')
                aksi_0_str, aksi_1_str = isi_aksi.split('|')
                
                aksi_0 = [aksi_0_str[0], aksi_0_str[1:]]
                aksi_1 = [aksi_1_str[0], aksi_1_str[1:]]
                rule[nomor_mode] = [aksi_0, aksi_1]
            return rule
        instruction = translating(instruction_string)
        finite_state_rule = instruction[think_memory[253]][(memory[-1][ap] == 'D')]
        think_memory[253] = finite_state_rule[1]
        return finite_state_rule[0]
    elif strategy == 'Solution B5':
        if game_round == 0:
            return 'D'
        instruction_string = '1:C2|D6,2:C2|D3,3:C6|D1,4:C3|D6,5:D5|D4,6:C3|D5'
        def translating(x):
            rule = {}
            semua_mode = x.strip().rstrip(',').split(',')
            for item in semua_mode:
                if not item: continue
                nomor_mode, isi_aksi = item.split(':')
                aksi_0_str, aksi_1_str = isi_aksi.split('|')
                
                aksi_0 = [aksi_0_str[0], aksi_0_str[1:]]
                aksi_1 = [aksi_1_str[0], aksi_1_str[1:]]
                rule[nomor_mode] = [aksi_0, aksi_1]
            return rule
        instruction = translating(instruction_string)
        finite_state_rule = instruction[think_memory[254]][(memory[-1][ap] == 'D')]
        think_memory[254] = finite_state_rule[1]
        return finite_state_rule[0]
    elif strategy == 'Thumper':
        if game_round == 0:
            return 'C'
        instruction_string = '1:C1|D2,2:D1|D1'
        def translating(x):
            rule = {}
            semua_mode = x.strip().rstrip(',').split(',')
            for item in semua_mode:
                if not item: continue
                nomor_mode, isi_aksi = item.split(':')
                aksi_0_str, aksi_1_str = isi_aksi.split('|')
                
                aksi_0 = [aksi_0_str[0], aksi_0_str[1:]]
                aksi_1 = [aksi_1_str[0], aksi_1_str[1:]]
                rule[nomor_mode] = [aksi_0, aksi_1]
            return rule
        instruction = translating(instruction_string)
        finite_state_rule = instruction[think_memory[255]][(memory[-1][ap] == 'D')]
        think_memory[255] = finite_state_rule[1]
        return finite_state_rule[0]
    elif strategy == 'Evolved FSM 4':
        if game_round == 0:
            return 'C'
        instruction_string = '0:C0|D2,1:D3|C0,2:D2|C1,3:D3|D1'
        def translating(x):
            rule = {}
            semua_mode = x.strip().rstrip(',').split(',')
            for item in semua_mode:
                if not item: continue
                nomor_mode, isi_aksi = item.split(':')
                aksi_0_str, aksi_1_str = isi_aksi.split('|')
                
                aksi_0 = [aksi_0_str[0], aksi_0_str[1:]]
                aksi_1 = [aksi_1_str[0], aksi_1_str[1:]]
                rule[nomor_mode] = [aksi_0, aksi_1]
            return rule
        instruction = translating(instruction_string)
        finite_state_rule = instruction[think_memory[256]][(memory[-1][ap] == 'D')]
        think_memory[256] = finite_state_rule[1]
        return finite_state_rule[0]
    elif strategy == 'Evolved FSM 16':
        if game_round == 0:
            return 'C'
        instruction_string = '0:C0|D12,1:D3|C6,2:D2|D14,3:D3|D3,5:D12|D10,6:C5|D12,7:D3|C1,8:C5|C5,10:D11|C8,11:D15|D5,12:C8|D11,13:D13|D7,14:D13|D13,15:D15|C2'
        def translating(x):
            rule = {}
            semua_mode = x.strip().rstrip(',').split(',')
            for item in semua_mode:
                if not item: continue
                nomor_mode, isi_aksi = item.split(':')
                aksi_0_str, aksi_1_str = isi_aksi.split('|')
                
                aksi_0 = [aksi_0_str[0], aksi_0_str[1:]]
                aksi_1 = [aksi_1_str[0], aksi_1_str[1:]]
                rule[nomor_mode] = [aksi_0, aksi_1]
            return rule
        instruction = translating(instruction_string)
        finite_state_rule = instruction[think_memory[257]][(memory[-1][ap] == 'D')]
        think_memory[257] = finite_state_rule[1]
        return finite_state_rule[0]
    elif strategy == 'Evolved FSM 16 Noise 05':
        if game_round == 0:
            return 'C'
        instruction_string = '0:C8|D3,1:C13|D15,2:C12|D3,3:C10|D3,4:D5|D4,5:D4|D10,6:C8|D6,8:C2|D4,10:D4|D1,11:D14|C13,12:C13|C2,13:C13|C6,14:D3|D13,15:D5|C11'
        def translating(x):
            rule = {}
            semua_mode = x.strip().rstrip(',').split(',')
            for item in semua_mode:
                if not item: continue
                nomor_mode, isi_aksi = item.split(':')
                aksi_0_str, aksi_1_str = isi_aksi.split('|')
                
                aksi_0 = [aksi_0_str[0], aksi_0_str[1:]]
                aksi_1 = [aksi_1_str[0], aksi_1_str[1:]]
                rule[nomor_mode] = [aksi_0, aksi_1]
            return rule
        instruction = translating(instruction_string)
        finite_state_rule = instruction[think_memory[258]][(memory[-1][ap] == 'D')]
        think_memory[258] = finite_state_rule[1]
        return finite_state_rule[0]
    elif strategy == 'TF 1':
        if game_round == 0:
            return 'C'
        instruction_string = '0:C7|C1,1:D11|D11,2:D8|C8,3:C3|D12,4:C6|C3,5:C11|D8,6:D13|C14,7:D4|D2,8:D14|D8,9:C0|D10,10:C8|C15,11:D6|D5,12:D6|D9,13:D9|D8,14:D8|D13,15:C4|C5'
        def translating(x):
            rule = {}
            semua_mode = x.strip().rstrip(',').split(',')
            for item in semua_mode:
                if not item: continue
                nomor_mode, isi_aksi = item.split(':')
                aksi_0_str, aksi_1_str = isi_aksi.split('|')
                
                aksi_0 = [aksi_0_str[0], aksi_0_str[1:]]
                aksi_1 = [aksi_1_str[0], aksi_1_str[1:]]
                rule[nomor_mode] = [aksi_0, aksi_1]
            return rule
        instruction = translating(instruction_string)
        finite_state_rule = instruction[think_memory[259]][(memory[-1][ap] == 'D')]
        think_memory[259] = finite_state_rule[1]
        return finite_state_rule[0]
    elif strategy == 'TF 2':
        if game_round == 0:
            return 'C'
        instruction_string = '0:D13|D12,1:D3|D4,2:D14|D9,3:C0|D1,4:D1|D2,7:D12|D2,8:D7|D9,9:D8|D0,10:C2|C15,11:D7|D13,12:C3|D8,13:C7|D10,14:D10|D7,15:C15|D11'
        def translating(x):
            rule = {}
            semua_mode = x.strip().rstrip(',').split(',')
            for item in semua_mode:
                if not item: continue
                nomor_mode, isi_aksi = item.split(':')
                aksi_0_str, aksi_1_str = isi_aksi.split('|')
                
                aksi_0 = [aksi_0_str[0], aksi_0_str[1:]]
                aksi_1 = [aksi_1_str[0], aksi_1_str[1:]]
                rule[nomor_mode] = [aksi_0, aksi_1]
            return rule
        instruction = translating(instruction_string)
        finite_state_rule = instruction[think_memory[260]][(memory[-1][ap] == 'D')]
        think_memory[260] = finite_state_rule[1]
        return finite_state_rule[0]
    elif strategy == 'TF 3':
        if game_round == 0:
            return 'C'
        instruction_string = '0:C0|C3,1:D5|C0,2:C3|D2,3:D4|D6,4:C3|D1,5:C6|D3,6:D6|D6,7:D7|C5'
        def translating(x):
            rule = {}
            semua_mode = x.strip().rstrip(',').split(',')
            for item in semua_mode:
                if not item: continue
                nomor_mode, isi_aksi = item.split(':')
                aksi_0_str, aksi_1_str = isi_aksi.split('|')
                
                aksi_0 = [aksi_0_str[0], aksi_0_str[1:]]
                aksi_1 = [aksi_1_str[0], aksi_1_str[1:]]
                rule[nomor_mode] = [aksi_0, aksi_1]
            return rule
        instruction = translating(instruction_string)
        finite_state_rule = instruction[think_memory[261]][(memory[-1][ap] == 'D')]
        think_memory[261] = finite_state_rule[1]
        return finite_state_rule[0]
    elif strategy == 'Evolved FSM 6':
        if game_round == 0:
            return 'C'
        instruction_string = '2:D2|D7,3:D6|C4,4:C4|D6,5:D2|D7,6:C3|D5,7:C2|D3'
        def translating(x):
            rule = {}
            semua_mode = x.strip().rstrip(',').split(',')
            for item in semua_mode:
                if not item: continue
                nomor_mode, isi_aksi = item.split(':')
                aksi_0_str, aksi_1_str = isi_aksi.split('|')
                
                aksi_0 = [aksi_0_str[0], aksi_0_str[1:]]
                aksi_1 = [aksi_1_str[0], aksi_1_str[1:]]
                rule[nomor_mode] = [aksi_0, aksi_1]
            return rule
        instruction = translating(instruction_string)
        finite_state_rule = instruction[think_memory[262]][(memory[-1][ap] == 'D')]
        think_memory[262] = finite_state_rule[1]
        return finite_state_rule[0]
    elif strategy == 'Forgiver':
        if game_round > 0:
            if memory[-1][ap] == 'D':
                think_memory[263] += 1
        t_d = think_memory[263]
        if t_d > ((game_round + 1) / 10):
            return 'D'
        return 'C'
    elif strategy == 'Forgiving Tit For Tat':
        if game_round > 0:
            if memory[-1][ap] == 'D':
                think_memory[264] += 1
        t_d = think_memory[264]
        if t_d > ((game_round + 1) / 10):
            return memory[-1][ap]
        return 'C'
    elif strategy == 'Frequency Analyzer':
        """
        Parameters
        ----------
        p, float
            The probability to cooperate
        """
        def update_table():
            if think_memory[265][2] in think_memory[265][1].keys():
                results = think_memory[265][1][think_memory[265][2]]
                results.append(memory[-1][ap])
                think_memory[265][1][think_memory[265][2]] = results
            else:
                think_memory[265][1][think_memory[265][2]] = [memory[-1][ap]]

        """This is the actual strategy"""
        if (game_round) > 5:
            think_memory[265][2] = (
                memory[-3][ap]
                + memory[-3][1-ap]
                + memory[-2][ap]
                + memory[-2][1-ap]
            )
            think_memory[265][3] = (
                memory[-2][ap]
                + memory[-2][1-ap]
                + memory[-1][ap]
                + memory[-1][1-ap]
            )
            update_table()

        
        if ((game_round) < 30) or (
            think_memory[265][3] not in think_memory[265][1]
        ):
            if not game_round:
                return 'C'
            if memory[-1][ap] == 'D':
                return 'D'
            return 'C'

        
        results = think_memory[265][1][think_memory[265][3]]
        cooperates = sum((i == 'C') for i in results)
        if (cooperates / (game_round)) > think_memory[265][0]:
            return 'C'
        return 'D'
    elif strategy == 'PSO Gambler Mem 1':
        A = {('C', 'C'): 1.0, ('C', 'D'): 0.52173487, ('D', 'C'): 0.0, ('D', 'D'): 0.12050939}[(memory[-1][1-ap], memory[-1][ap]) if game_round > 0 else ('C', 'C')]
        return 'C' if rnd.random() <= A else 'D'
    elif strategy == 'PSO Gambler 1_1_1':
        if not game_round:
            return 'C'
        if (game_round + 1) == 2:
            think_memory[266] = memory[-1][ap]
        A = {('C', 'C', 'C'): 1.0, ('C', 'C', 'D'): 1.0, ('C', 'D', 'C'): 0.12304797, ('C', 'D', 'D'): 0.57740178, ('D', 'C', 'C'): 0.0, ('D', 'C', 'D'): 0.0, ('D', 'D', 'C'): 0.13581423, ('D', 'D', 'D'): 0.11886807}[(memory[-1][1-ap], memory[-1][ap], think_memory[266])]
        return 'C' if rnd.random() <= A else 'D'
    elif strategy == 'PSO Gambler 2_2_2':
        if (game_round + 1) < 3:
            return 'C'
        elif (game_round + 1) == 3:
            think_memory[267] = memory[-2][ap] + memory[-1][ap]
        A = {('CC', 'CC', 'CC'): 1.0, ('CC', 'CC', 'CD'): 1.0, ('CC', 'CC', 'DC'): 1.0, ('CC', 'CC', 'DD'): 0.0, ('CC', 'CD', 'CC'): 1.0, ('CC', 'CD', 'CD'): 0.95280465, ('CC', 'CD', 'DC'): 0.0, ('CC', 'CD', 'DD'): 0.0, ('CC', 'DC', 'CC'): 0.0, ('CC', 'DC', 'CD'): 0.80897541, ('CC', 'DC', 'DC'): 0.0, ('CC', 'DC', 'DD'): 0.0, ('CC', 'DD', 'CC'): 0.02126434, ('CC', 'DD', 'CD'): 0.0, ('CC', 'DD', 'DC'): 0.43278586, ('CC', 'DD', 'DD'): 0.0, ('CD', 'CC', 'CC'): 0.0, ('CD', 'CC', 'CD'): 0.0, ('CD', 'CC', 'DC'): 1.0, ('CD', 'CC', 'DD'): 0.15140743, ('CD', 'CD', 'CC'): 1.0, ('CD', 'CD', 'CD'): 0.0, ('CD', 'CD', 'DC'): 0.0, ('CD', 'CD', 'DD'): 0.0, ('CD', 'DC', 'CC'): 1.0, ('CD', 'DC', 'CD'): 0.0, ('CD', 'DC', 'DC'): 0.23563137, ('CD', 'DC', 'DD'): 0.0, ('CD', 'DD', 'CC'): 0.0, ('CD', 'DD', 'CD'): 0.65147565, ('CD', 'DD', 'DC'): 1.0, ('CD', 'DD', 'DD'): 0.0, ('DC', 'CC', 'CC'): 0.0, ('DC', 'CC', 'CD'): 0.15412392, ('DC', 'CC', 'DC'): 1.0, ('DC', 'CC', 'DD'): 0.0, ('DC', 'CD', 'CC'): 0.0, ('DC', 'CD', 'CD'): 0.24922166, ('DC', 'CD', 'DC'): 1.0, ('DC', 'CD', 'DD'): 0.0, ('DC', 'DC', 'CC'): 0.0, ('DC', 'DC', 'CD'): 0.0, ('DC', 'DC', 'DC'): 0.00227615, ('DC', 'DC', 'DD'): 0.0, ('DC', 'DD', 'CC'): 0.0, ('DC', 'DD', 'CD'): 0.0, ('DC', 'DD', 'DC'): 0.0, ('DC', 'DD', 'DD'): 1.0, ('DD', 'CC', 'CC'): 0.0, ('DD', 'CC', 'CD'): 0.0, ('DD', 'CC', 'DC'): 0.0, ('DD', 'CC', 'DD'): 0.0, ('DD', 'CD', 'CC'): 0.0, ('DD', 'CD', 'CD'): 0.0, ('DD', 'CD', 'DC'): 0.0, ('DD', 'CD', 'DD'): 1.0, ('DD', 'DC', 'CC'): 0.0, ('DD', 'DC', 'CD'): 0.0, ('DD', 'DC', 'DC'): 0.0, ('DD', 'DC', 'DD'): 0.77344942, ('DD', 'DD', 'CC'): 1.0, ('DD', 'DD', 'CD'): 0.24523149, ('DD', 'DD', 'DC'): 1.0, ('DD', 'DD', 'DD'): 0.0}[(memory[-2][1-ap] + memory[-1][1-ap], memory[-2][ap] + memory[-1][ap], think_memory[267])]
        return 'C' if rnd.random() <= A else 'D'
    elif strategy == 'PSO Gambler 2_2_2 Noise 05':
        if (game_round + 1) < 3:
            return 'C'
        elif (game_round + 1) == 3:
            think_memory[268] = memory[-2][ap] + memory[-1][ap]
        A = {('CC', 'CC', 'CC'): 1.0, ('CC', 'CC', 'CD'): 1.0, ('CC', 'CC', 'DC'): 0.0, ('CC', 'CC', 'DD'): 0.0, ('CC', 'CD', 'CC'): 0.0, ('CC', 'CD', 'CD'): 1.0, ('CC', 'CD', 'DC'): 0.98603825, ('CC', 'CD', 'DD'): 1.0, ('CC', 'DC', 'CC'): 1.0, ('CC', 'DC', 'CD'): 0.0, ('CC', 'DC', 'DC'): 0.0, ('CC', 'DC', 'DD'): 0.16240799, ('CC', 'DD', 'CC'): 0.63548102, ('CC', 'DD', 'CD'): 0.0, ('CC', 'DD', 'DC'): 0.0, ('CC', 'DD', 'DD'): 0.0, ('CD', 'CC', 'CC'): 1.0, ('CD', 'CC', 'CD'): 0.0, ('CD', 'CC', 'DC'): 1.0, ('CD', 'CC', 'DD'): 0.0, ('CD', 'CD', 'CC'): 1.0, ('CD', 'CD', 'CD'): 0.13863175, ('CD', 'CD', 'DC'): 0.06434619, ('CD', 'CD', 'DD'): 1.0, ('CD', 'DC', 'CC'): 1.0, ('CD', 'DC', 'CD'): 1.0, ('CD', 'DC', 'DC'): 1.0, ('CD', 'DC', 'DD'): 1.0, ('CD', 'DD', 'CC'): 0.0, ('CD', 'DD', 'CD'): 0.7724137, ('CD', 'DD', 'DC'): 1.0, ('CD', 'DD', 'DD'): 0.0, ('DC', 'CC', 'CC'): 0.0, ('DC', 'CC', 'CD'): 0.0, ('DC', 'CC', 'DC'): 1.0, ('DC', 'CC', 'DD'): 0.0, ('DC', 'CD', 'CC'): 1.0, ('DC', 'CD', 'CD'): 1.0, ('DC', 'CD', 'DC'): 0.50999729, ('DC', 'CD', 'DD'): 1.0, ('DC', 'DC', 'CC'): 0.0, ('DC', 'DC', 'CD'): 0.0, ('DC', 'DC', 'DC'): 0.00524508, ('DC', 'DC', 'DD'): 0.87463905, ('DC', 'DD', 'CC'): 0.0, ('DC', 'DD', 'CD'): 0.07127653, ('DC', 'DD', 'DC'): 1.0, ('DC', 'DD', 'DD'): 0.0, ('DD', 'CC', 'CC'): 1.0, ('DD', 'CC', 'CD'): 0.0, ('DD', 'CC', 'DC'): 1.0, ('DD', 'CC', 'DD'): 0.0, ('DD', 'CD', 'CC'): 0.0, ('DD', 'CD', 'CD'): 1.0, ('DD', 'CD', 'DC'): 1.0, ('DD', 'CD', 'DD'): 1.0, ('DD', 'DC', 'CC'): 0.0, ('DD', 'DC', 'CD'): 0.28124022, ('DD', 'DC', 'DC'): 1.0, ('DD', 'DC', 'DD'): 0.0, ('DD', 'DD', 'CC'): 0.0, ('DD', 'DD', 'CD'): 0.0, ('DD', 'DD', 'DC'): 1.0, ('DD', 'DD', 'DD'): 0.0}[(memory[-2][1-ap] + memory[-1][1-ap], memory[-2][ap] + memory[-1][ap], think_memory[268])]
        return 'C' if rnd.random() <= A else 'D'
    elif strategy == 'Zero Determinant Memory 2 / ZD-M2':
        if (game_round + 1) < 3:
            return 'C'
        A = {('CC', 'CC'): 0.9166666666666666, ('CC', 'CD'): 0.36363636363636365, ('CC', 'DC'): 0.7777777777777778, ('CC', 'DD'): 0.1, ('CD', 'CC'): 0.8333333333333334, ('CD', 'CD'): 0.2727272727272727, ('CD', 'DC'): 0.7777777777777778, ('CD', 'DD'): 0.1, ('DC', 'CC'): 0.6666666666666666, ('DC', 'CD'): 0.09090909090909091, ('DC', 'DC'): 0.7777777777777778, ('DC', 'DD'): 0.1, ('DD', 'CC'): 0.75, ('DD', 'CD'): 0.18181818181818182, ('DD', 'DC'): 0.7777777777777778, ('DD', 'DD'): 0.1}[(memory[-2][1-ap] + memory[-1][1-ap], memory[-2][ap] + memory[-1][ap])]
        return 'C' if rnd.random() <= A else 'D'
    elif strategy == 'Hard Go By Majority':
        if not game_round: return 'C'
        if memory[-1][ap] == 'C':
            think_memory[269][0] += 1
        else:
            think_memory[269][1] += 1
        t_c = think_memory[269][0]
        t_d = think_memory[269][1]
        if t_c > t_d:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Gradual Killer':
        if (game_round + 1) <= 5:
            return 'D'
        elif ((game_round + 1) - 5) <= 2:
            return 'C'
        elif (game_round + 1) == 8:
            think_memory[270] = ((memory[-2][ap], memory[-1][ap]) == ('D', 'D'))
        if think_memory[270]:
            return 'D'
        return 'C'
    elif strategy == 'Forgetful Grudger':        
        maximal_grudging = 10
        if think_memory[271][1] == maximal_grudging:
            think_memory[271][1] = 0
            think_memory[271][0] = False

        if game_round > 0:
            if memory[-1][ap] == 'D':
                think_memory[271][0] = True

        if think_memory[271][0]:
            think_memory[271][1] += 1
            return 'D'
        return 'C'
    elif strategy == 'Opposite Grudger':
        if game_round > 0:
            if memory[-1][ap] == 'C':
                think_memory[272] = True

        if think_memory[272]:
            return 'C'
        return 'D'
    elif strategy == 'Aggravater':
        if game_round > 0:
            if memory[-1][ap] == 'D':
                think_memory[273] = True

        if (game_round + 1) <= 3:
            return 'D'

        if think_memory[273]:
            return 'D'
        return 'C'
    elif strategy == 'Soft Grudger':
        if think_memory[274][0]:
            A = ('D', 'D', 'D', 'C', 'C')[think_memory[274][1]]
            think_memory[274][1] += 1
            if think_memory[274][1] == 5:
                think_memory[274][1] = 0
                think_memory[274][0] = False
            return A
        elif game_round > 0 and memory[-1][ap] == 'D':
            think_memory[274][0] = True
            return 'D'
        return 'C'
    elif strategy == 'Grudger Alternator':
        if game_round > 0:
            if memory[-1][ap] == 'D':
                think_memory[275] = True

            if think_memory[275]:
                if memory[-1][ap] == 'C':
                    return 'D'
        return 'C'
    elif strategy == 'Spiteful CC':
        if game_round > 0:
            if memory[-1][ap] == 'D':
                think_memory[276] = True

        if (game_round + 1) <= 2:
            return 'C'

        if think_memory[276]:
            return 'D'
        return 'C'
    elif strategy == 'Capri':
        hist = [(memory[-3][1-ap], memory[-3][ap]) if (game_round + 1) > 3 else ('C','C'), (memory[-2][1-ap], memory[-2][ap]) if (game_round + 1) > 2 else ('C','C'), (memory[-1][1-ap], memory[-1][ap]) if game_round > 0 else ('C','C')]
        if hist == [('C', 'C'), ('C', 'C'), ('C', 'C')]:  
            return 'C'
        if hist == [('C', 'C'), ('C', 'C'), ('D', 'C')]:  
            return 'C'
        if hist == [('C', 'C'), ('D', 'C'), ('C', 'D')]:
            return 'C'
        if hist == [('D', 'C'), ('C', 'D'), ('C', 'C')]:
            return 'C'
        if hist == [('C', 'D'), ('C', 'C'), ('C', 'C')]:  
            return 'C'
        if hist == [('C', 'C'), ('C', 'C'), ('C', 'D')]:  
            return 'D'
        if hist == [('C', 'C'), ('C', 'D'), ('D', 'C')]:
            return 'C'
        if hist == [('C', 'D'), ('D', 'C'), ('C', 'C')]:
            return 'C'
        if hist == [('D', 'C'), ('C', 'C'), ('C', 'C')]:  
            return 'C'
        if hist == [('D', 'D'), ('D', 'D'), ('D', 'C')]:  
            return 'C'
        if hist == [('D', 'D'), ('D', 'C'), ('C', 'C')]:
            return 'C'
        if hist == [('D', 'D'), ('D', 'D'), ('C', 'D')]:  
            return 'C'
        if hist == [('D', 'D'), ('C', 'D'), ('C', 'C')]:
            return 'C'
        if hist == [('D', 'D'), ('D', 'D'), ('C', 'C')]:  
            return 'C'
        if hist == [('D', 'D'), ('C', 'C'), ('C', 'C')]:
            return 'C'
        return 'D'
    elif strategy == 'Grumpy':
        """A player that gets grumpier the more the opposition defects,
        and nicer the more they cooperate.

        Starts off Nice, but becomes grumpy once the grumpiness threshold is
        hit. Won't become nice once that grumpy threshold is hit, but must
        reach a much lower threshold before it becomes nice again.
        """

        if game_round > 0:
            if memory[-1][ap] == 'C':
                think_memory[277][3] += 1
            else:
                think_memory[277][4] += 1

        grumpiness = think_memory[277][4] - think_memory[277][3]

        if think_memory[277][0] == "Nice":
            if grumpiness > think_memory[277][1]:
                think_memory[277][0] = "Grumpy"
                return 'D'
            return 'C'

        if think_memory[277][0] == "Grumpy":
            if grumpiness < think_memory[277][2]:
                think_memory[277][0] = "Nice"
                return 'C'
            return 'D'
    elif strategy == 'HMM Player':
        def process_hmm_step(current_state, last_opponent_move, t_C, t_D, emissions):
            """
            Menghitung transisi state HMM dan menentukan emisi aksi (C/D).
            """
            
            probs = t_C[current_state] if last_opponent_move == 'C' else t_D[current_state]

            
            if 1 in probs:
                next_state = probs.index(1)
            else:
                
                next_state = int(np.random.choice(len(probs), p=probs))

            
            p_emit = emissions[next_state]
            if p_emit == 1.0:
                action = 'C'
            elif p_emit == 0.0:
                action = 'D'
            else:
                action = 'C' if np.random.random() < p_emit else 'D'

            return next_state, action
        
        
        
        if not game_round:
            t_C = [[1.0]]
            t_D = [[1.0]]
            emissions = [0.5]
            think_memory[278] = [0, t_C, t_D, emissions]
            return 'C'

        current_state, t_C, t_D, emissions = think_memory[278]
        last_move = memory[-1][ap]
        
        next_state, action = process_hmm_step(current_state, last_move, t_C, t_D, emissions)
        
        
        think_memory[278][0] = next_state
        return action

    elif strategy == 'Evolved HMM 5':
        def process_hmm_step(current_state, last_opponent_move, t_C, t_D, emissions):
            """
            Menghitung transisi state HMM dan menentukan emisi aksi (C/D).
            """
            
            probs = t_C[current_state] if last_opponent_move == 'C' else t_D[current_state]

            
            if 1 in probs:
                next_state = probs.index(1)
            else:
                
                next_state = int(np.random.choice(len(probs), p=probs))

            
            p_emit = emissions[next_state]
            if p_emit == 1.0:
                action = 'C'
            elif p_emit == 0.0:
                action = 'D'
            else:
                action = 'C' if np.random.random() < p_emit else 'D'

            return next_state, action
        
        
        t_C = [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0.631, 0, 0, 0.369, 0],
            [0.143, 0.018, 0.118, 0, 0.721],
        ]
        t_D = [
            [0, 1, 0, 0, 0],
            [0, 0.487, 0.513, 0, 0],
            [0, 0, 0, 0.590, 0.410],
            [1, 0, 0, 0, 0],
            [0, 0.287, 0.456, 0.146, 0.111],
        ]
        emissions = [1.0, 0.0, 0.0, 1.0, 0.111]

        
        if not game_round:
            think_memory[279] = 3
            return 'C'

        current_state = think_memory[279]
        last_move = memory[-1][ap]

        next_state, action = process_hmm_step(current_state, last_move, t_C, t_D, emissions)

        
        think_memory[279] = next_state
        return action
    elif strategy == 'Eventual Cycle Hunter':
        if game_round > 0:
            if memory[-1][ap] == 'D':
                think_memory[282] = True

        if (game_round + 1) <= 10:
            return 'C'

        if (game_round) % 10 == 0 and think_memory[282]:
            if cycle_detec(ap, min_ = 3, offset = 10):
                return 'D'
        return 'C'
    elif strategy == 'Math Constant Hunter':
        """
        Check whether the number of cooperations in the first and second halves
        of the history are close. The variance of the uniform distribution (1/4)
        is a reasonable delta but use something lower for certainty and avoiding
        false positives. This approach will also detect a lot of random players.
        """

        if not game_round:
            return 'C'

        think_memory[283][(memory[-1][ap] == 'D')] += 1

        unziped_memory = list(zip(*memory))
        self_history, opponent_history = unziped_memory[1-ap], unziped_memory[ap]

        n = game_round
        if n >= 8 and think_memory[283][0] and think_memory[283][1]:
            start1, end1 = 0, n // 2
            start2, end2 = n // 4, 3 * n // 4
            start3, end3 = n // 2, n
            count1 = opponent_history[start1:end1].count('C') + self_history[
                start1:end1
            ].count('C')
            count2 = opponent_history[start2:end2].count('C') + self_history[
                start2:end2
            ].count('C')
            count3 = opponent_history[start3:end3].count('C') + self_history[
                start3:end3
            ].count('C')
            ratio1 = 0.5 * count1 / (end1 - start1)
            ratio2 = 0.5 * count2 / (end2 - start2)
            ratio3 = 0.5 * count3 / (end3 - start3)
            if abs(ratio1 - ratio2) < 0.2 and abs(ratio1 - ratio3) < 0.2:
                return 'D'
        return 'C'
    elif strategy == 'Random Hunter':
        """
        A random player is unpredictable, which means the conditional frequency
        of cooperation after cooperation, and defection after defections, should
        be close to 50%... although how close is debatable.
        """
        if game_round > 0:
            think_memory[284][(memory[-1][1-ap] == 'D') + 2] += 1
        
        if (game_round) > 1:
            if memory[-2][1-ap] == 'C' and memory[-1][ap] == 'C':
                think_memory[284][0] += 1
            if memory[-2][1-ap] == 'D' and memory[-1][ap] == 'D':
                think_memory[284][1] += 1

        n = (game_round)
        if n > 10:
            probabilities = []
            if think_memory[284][2] > 5:
                probabilities.append(think_memory[284][0] / think_memory[284][2])
            if think_memory[284][3] > 5:
                probabilities.append(think_memory[284][1] / think_memory[284][3])
            if probabilities and all(
                [abs(p - 0.5) < 0.25 for p in probabilities]
            ):
                return 'D'
        return 'C'
    elif strategy == 'Inverse':
        if game_round > 0:
            if think_memory[285] != None or memory[-1][ap] == 'D':
                if memory[-1][ap] == 'C':
                    think_memory[285] += 1
                else:
                    think_memory[285] = 1

                return 'C' if rnd.random() <= (1 - (1 / think_memory[285])) else 'D'
        return 'C'
    elif strategy == 'Winner 12':
        inp = ((memory[-1][1-ap] if game_round > 0 else 'C'), (memory[-2][ap] if (game_round + 1) > 2 else 'C') + (memory[-1][ap] if game_round > 0 else 'C'))

        lookup_table = {('C', 'CC'): 'C', ('C', 'CD'): 'D', ('C', 'DC'): 'C', ('C', 'DD'): 'D', ('D', 'CC'): 'D', ('D', 'CD'): 'C', ('D', 'DC'): 'D', ('D', 'DD'): 'D'}

        return lookup_table[inp]
    elif strategy == 'Winner 21':
        inp = ((memory[-1][1-ap] if game_round > 0 else 'C'), (memory[-2][ap] if (game_round + 1) > 2 else 'C') + (memory[-1][ap] if game_round > 0 else 'C'))

        lookup_table = {('C', 'CC'): 'C', ('C', 'CD'): 'D', ('C', 'DC'): 'C', ('C', 'DD'): 'D', ('D', 'CC'): 'C', ('D', 'CD'): 'D', ('D', 'DC'): 'D', ('D', 'DD'): 'D'}

        return lookup_table[inp]
    elif strategy == 'Phi':
        if not game_round:
            return 'C'

        if memory[-1][ap] == 'C':
            think_memory[286][0] += 1
        else:
            think_memory[286][1] += 1

        if memory[-1][1-ap] == 'C':
            think_memory[286][2] += 1
        else:
            think_memory[286][3] += 1

        
        if think_memory[286][1] == 0:
            return 'D'
        
        cooperations = think_memory[286][0] + think_memory[286][2]
        defections = think_memory[286][1] + think_memory[286][3]
        if cooperations / defections > think_memory[286][4]:
            return 'D'
        return 'C'
    elif strategy == 'Pi':
        if not game_round:
            return 'C'

        if memory[-1][ap] == 'C':
            think_memory[287][0] += 1
        else:
            think_memory[287][1] += 1

        if memory[-1][1-ap] == 'C':
            think_memory[287][2] += 1
        else:
            think_memory[287][3] += 1

        
        if think_memory[287][1] == 0:
            return 'D'
        
        cooperations = think_memory[287][0] + think_memory[287][2]
        defections = think_memory[287][1] + think_memory[287][3]
        if cooperations / defections > think_memory[287][4]:
            return 'D'
        return 'C'
    elif strategy == 'Euler':
        if not game_round:
            return 'C'

        if memory[-1][ap] == 'C':
            think_memory[288][0] += 1
        else:
            think_memory[288][1] += 1

        if memory[-1][1-ap] == 'C':
            think_memory[288][2] += 1
        else:
            think_memory[288][3] += 1

        
        if think_memory[288][1] == 0:
            return 'D'
        
        cooperations = think_memory[288][0] + think_memory[288][2]
        defections = think_memory[288][1] + think_memory[288][3]
        if cooperations / defections > think_memory[288][4]:
            return 'D'
        return 'C'
    elif strategy == 'Generous Tit For Tat Axelrod Project Contributor Team Version':
        if not game_round:
            return 'C'
        
        A = min(1 - ((RPST['T'] - RPST['R']) / (RPST['R'] - RPST['S'])), ((RPST['R'] - RPST['P']) / (RPST['T'] - RPST['P'])))
        if rnd.random() <= A:
            return 'C'
        return memory[-1][ap]
    elif strategy == 'Firm But Fair / Firm For Tat':
        if not game_round:
            return 'C'
        else:
            if memory[-1][ap] == memory[-1][1-ap] and rnd.random() <= (2/3):
                return 'C'
            return memory[-1][ap]
    elif strategy == 'Stochastic Cooperator':
        return 'C' if rnd.random() <= (0.935, 0.229, 0.266, 0.42)[((2 * (memory[-1][1-ap] == 'D')) + (memory[-1][ap] == 'D')) if game_round > 0 else 0] else 'D'
    elif strategy == 'Stochastic WSLS':
        ep = 0.05
        return 'C' if rnd.random() <= (1.0 - ep, ep, ep, 1.0 - ep)[((2 * (memory[-1][1-ap] == 'D')) + (memory[-1][ap] == 'D')) if game_round > 0 else 0] else 'D'
    elif strategy == 'ALLC OR ALLD':
        if not game_round:
            return 'C' if rnd.random() <= 0.6 else 'D'
        return memory[-1][1-ap]
    elif strategy == 'Reactive Player':
        if ((memory[-1][ap] == 'C') if game_round > 0 else True):
            return 'C' if rnd.random() <= think_memory[289][0] else 'D'
        return 'C' if rnd.random() <= think_memory[289][1] else 'D'
    elif strategy == 'MEM2':

        """Actual strategy definition that determines player's action."""
        
        
        
        think_memory[290][1] -= 1
        if (think_memory[290][1] == 0) and (think_memory[290][2] < 2):
            think_memory[290][1] = 2
            
            last_two = memory[-2:]
            if set(last_two) == {('C', 'C')}:
                think_memory[290][0] = "TFT"
            elif set(last_two) == {('C', 'D'), ('D', 'C')}:
                think_memory[290][0] = "TFTT"
            else:
                think_memory[290][0] = "ALLD"
                think_memory[290][2] += 1
        if think_memory[290][0] == "TFT":
            if not game_round:
                return 'C'
            return memory[-1][ap]
        elif think_memory[290][0] == 'TFTT':
            if (game_round + 1) <= 2:
                return 'C'
            return 'D' if (memory[-2][ap], memory[-1][ap]) == ('D', 'D') else 'C'
        else:
            return 'D'
    elif strategy == 'Memory Decay':
        if not game_round:
            return 'C'

        think_memory[291].append(memory[-1][ap])

        t_c = sum(1 for i in think_memory[291] if i == 'C')
        t_d = len(think_memory[291]) - t_c

        trust = t_c - (2 * t_d)

        if rnd.random() <= 0.03:
            think_memory[291][rnd.randint(0, len(think_memory[291]) - 1)] = 'C' if think_memory[291][rnd.randint(0, len(think_memory[291]) - 1)] == 'D' else 'D'
        if rnd.random() <= 0.1:
            think_memory[291].pop(rnd.randint(0, len(think_memory[291]) - 1))

        if trust < 0:
            return 'D'
        return 'C'
    elif strategy == 'Meta Winner Ensemble':
        def pay_off_matrix(x,y):
            if x == 'C':
                if y == 'C':
                    return RPST['R']
                else:
                    return RPST['T']
            else:
                if y == 'C':
                    return RPST['S']
                else:
                    return RPST['P']
        think_memory[292][4] = max(think_memory[292][4] - 1, 0)
        if game_round > 0 and memory[-1][ap] == 'D':
            think_memory[292][4] = 2
            think_memory[292][5] = 1
        think_memory[292][0] += pay_off_matrix((memory[-1][ap] if game_round > 0 else 'C'),'C' if rnd.random() <= 0.2 else (memory[-1][ap] if game_round > 0 else 'C'))
        think_memory[292][1] += pay_off_matrix((memory[-1][ap] if game_round > 0 else 'C'),'D' if ((memory[-1][ap] == 'D') if game_round > 0 else 'C') and ((memory[-1-1][ap] == 'D') if (game_round + 1) > 2 else 'C') else 'C')
        think_memory[292][2] += pay_off_matrix((memory[-1][ap] if game_round > 0 else 'C'),'D' if think_memory[292][4] > 0 else 'C')
        think_memory[292][3] += pay_off_matrix((memory[-1][ap] if game_round > 0 else 'C'),'D' if think_memory[292][5] == 1 else 'C')

        meta_win_ens_strategies = ['gtft', 'tf2t', '2tft', 'g']
        meta_win_ens_scores = [think_memory[292][i] for i in range(4)]
            
        prop = max(1, int(len(meta_win_ens_scores) * rnd.random()))
        best_scores = sorted(meta_win_ens_scores, reverse = True)[:prop]

        meta_win_ens_gtft = 1 if rnd.random() <= 0.2 else ((1 if memory[-1][ap] == 'C' else -1) if game_round > 0 else 1)
        meta_win_ens_tf2t = -1 if ((memory[-1][ap] == 'D') if game_round > 0 else 1) and ((memory[-1-1][ap] == 'D') if (game_round + 1) > 2 else 1) else 1
        meta_win_ens_2tft = -1 if think_memory[292][4] > 0 else 1
        meta_win_ens_g = -1 if think_memory[292][5] == 1 else 1
        meta_win_ens_majority = sum([meta_win_ens_gtft, meta_win_ens_tf2t, meta_win_ens_2tft, meta_win_ens_g][i] for i in range(4) if meta_win_ens_scores[i] in best_scores)
        if meta_win_ens_majority < 0:
            return 'D'
        else:
            return 'C'
    elif strategy == 'Meta Hunter':
        if game_round > 0:
            if cycle_detec(ap, min_ = 3):
                return 'D'

        if game_round > 0:
            if memory[-1][ap] == 'C':
                think_memory[293][0] += 1

        if (game_round + 1) > 4 and (game_round) == think_memory[293][0]:
            return 'D'

        if game_round > 0:
            if memory[-1][ap] == 'D':
                think_memory[293][1] += 1

        if (game_round + 1) > 4 and (game_round) == think_memory[293][1]:
            return 'D'

        if (game_round + 1) > 6:
            inp = tuple([memory[-1 - (5 - i)][ap] for i in range(6)])
            if inp not in (("C", "D", "C", "D", "C", "D"), ("D", "C", "D", "C", "D", "C")):
                think_memory[293][2] = False

            if think_memory[293][2]:
                return 'D'

        """
        Check whether the number of cooperations in the first and second halves
        of the history are close. The variance of the uniform distribution (1/4)
        is a reasonable delta but use something lower for certainty and avoiding
        false positives. This approach will also detect a lot of random players.
        """

        if game_round > 0:
            think_memory[293][3][(memory[-1][ap] == 'D')] += 1

            unziped_memory = list(zip(*memory))
            self_history, opponent_history = unziped_memory[1-ap], unziped_memory[ap]

            n = game_round
            if n >= 8 and think_memory[293][3][0] and think_memory[293][3][1]:
                start1, end1 = 0, n // 2
                start2, end2 = n // 4, 3 * n // 4
                start3, end3 = n // 2, n
                count1 = opponent_history[start1:end1].count('C') + self_history[
                    start1:end1
                ].count('C')
                count2 = opponent_history[start2:end2].count('C') + self_history[
                    start2:end2
                ].count('C')
                count3 = opponent_history[start3:end3].count('C') + self_history[
                    start3:end3
                ].count('C')
                ratio1 = 0.5 * count1 / (end1 - start1)
                ratio2 = 0.5 * count2 / (end2 - start2)
                ratio3 = 0.5 * count3 / (end3 - start3)
                if abs(ratio1 - ratio2) < 0.2 and abs(ratio1 - ratio3) < 0.2:
                    return 'D'

        """
        A random player is unpredictable, which means the conditional frequency
        of cooperation after cooperation, and defection after defections, should
        be close to 50%... although how close is debatable.
        """
        if game_round > 0:
            think_memory[293][4][(memory[-1][1-ap] == 'D') + 2] += 1
            
            if (game_round) > 1:
                if memory[-2][1-ap] == 'C' and memory[-1][ap] == 'C':
                    think_memory[293][4][0] += 1
                if memory[-2][1-ap] == 'D' and memory[-1][ap] == 'D':
                    think_memory[293][4][1] += 1

            n = (game_round)
            if n > 10:
                probabilities = []
                if think_memory[293][4][2] > 5:
                    probabilities.append(think_memory[293][4][0] / think_memory[293][4][2])
                if think_memory[293][4][3] > 5:
                    probabilities.append(think_memory[293][4][1] / think_memory[293][4][3])
                if probabilities and all(
                    [abs(p - 0.5) < 0.25 for p in probabilities]
                ):
                    return 'D'
        return 'C'
    elif strategy == 'Momentum':
        a = 0.9914655399877477
        threshold = 0.9676595613724907
        if not game_round:
            return 'C'

        think_memory[294] = (
            (a * think_memory[294]) + ((1 - a) * (memory[-1][ap] == 'C'))
        )
        return 'C' if think_memory[294] >= threshold else 'D'
    elif strategy == 'Desprate':
        if not game_round:
            return rnd.choice(['C','D'])
        return 'C' if memory[-1] == ('D','D') else 'D'
    elif strategy == 'Hopeless':
        if not game_round:
            return rnd.choice(['C','D'])
        return 'D' if memory[-1] == ('C','C') else 'C'
    elif strategy == 'Willing':
        if not game_round:
            return rnd.choice(['C','D'])
        return 'D' if memory[-1] == ('D','D') else 'C'
    elif strategy == 'Negation':
        if not game_round:
            return rnd.choice(['C','D'])
        return 'C' if memory[-1][ap] == 'D' else 'D'
    elif strategy == 'Once Bitten':
        maximal_grudging = 10
        if think_memory[295][1] == maximal_grudging:
            think_memory[295][1] = 0
            think_memory[295][0] = False

        if (game_round + 1) > 2:
            if not ('C' in (memory[-2][ap], memory[-1][ap])):
                think_memory[295][0] = True

        if think_memory[295][0]:
            think_memory[295][1] += 1
            return 'D'
        return 'C'
    elif strategy == 'Forgetful Fool Me Once':
        if game_round > 0:
            if memory[-1][ap] == 'D':
                think_memory[296] += 1
        forget_prop = 0.05

        if rnd.random() <= forget_prop:
            think_memory[296] = 0
        
        if think_memory[296] > 1:
            return 'D'
        else:
            return 'C'
    elif strategy == 'CS / Collective Strategy':
        if game_round > 0 and memory[-1][ap] == 'D':
            think_memory[297] += 1
        if (game_round + 1) <= 2:
            return ('D', 'C')[game_round]
        if (memory[0][ap] + memory[1][ap]) == 'CD':
            return 'C'
        return 'D'
    elif strategy == 'Prober 2':
        if (game_round + 1) <= 3:
            return ('D', 'C', 'C')[(game_round + 1)-1]
        
        if (memory[1][ap] + memory[2][ap]) == 'DC':
            return 'C'
        else:
            return memory[-1][ap]
    elif strategy == 'Prober 3':
        if (game_round + 1) <= 2:
            return ('D', 'C')[(game_round + 1)-1]
        
        if memory[1][ap] == 'C':
            return 'D'
        else:
            return memory[-1][ap]
    elif strategy == 'Prober 4':
        """Actual strategy definition that determines player's action."""
        if not game_round:
            return think_memory[298][0][0]
        turn = game_round
        if turn < len(think_memory[298][0]):
            if memory[-1][ap] == 'D':
                if memory[-1][1-ap] == 'D':
                    think_memory[298][1] += 1
                if memory[-1][1-ap] == 'C':
                    think_memory[298][2] += 1
            return think_memory[298][0][turn]
        if turn == len(think_memory[298][0]):
            diff_in_Ds = abs(think_memory[298][1] - think_memory[298][2])
            think_memory[298][3] = diff_in_Ds <= 2
        if think_memory[298][3]:
            return 'D'
        if not think_memory[298][3]:
            if turn < len(think_memory[298][0]) + 5:
                return 'C'
            return 'D' if memory[-1][ap] == 'D' else 'C'
    elif strategy == 'Hard Prober':
        if (game_round + 1) <= 4:
            return ('D', 'D', 'C', 'C')[(game_round + 1)-1]
        
        if (memory[1][ap] + memory[2][ap]) == 'CC':
            return 'D'
        else:
            return memory[-1][ap]
    elif strategy == 'Inverse Punisher':
        if game_round > 0:
            if memory[-1][ap] == 'C':
                think_memory[128][3] += 1

        """
        Begins by playing C, then plays D for an amount of rounds proportional
        to the opponents historical '%' of playing D if the opponent ever
        plays D
        """

        if think_memory[128][2] >= think_memory[128][0]:
            think_memory[128][2] = 0
            think_memory[128][1] = False

        if think_memory[128][1]:
            think_memory[128][2] += 1
            return 'D'

        elif ((memory[-1][ap] == 'D') if game_round > 0 else False):
            think_memory[128][0] = (think_memory[128][3] * 20) // (game_round)
            think_memory[128][1] = True
            return 'D'

        return 'C'
    elif strategy == 'Level Punisher':
        if game_round > 0:
            if memory[-1][ap] == 'D':
                think_memory[146] += 1
        if game_round < 10:
            return 'C'
        if think_memory[146] / game_round > 0.2:
            return 'D'
        return 'C'
    elif strategy == 'Tricky Level Punisher':
        if game_round > 0:
            if memory[-1][ap] == 'D':
                think_memory[299] += 1
        else:
            return 'C'
        if game_round < 10:
            if think_memory[299] / game_round > 0.2:
                return 'D'
        if game_round < 50:
            if think_memory[299] / game_round > 0.1:
                return 'D'
        if game_round < 100:
            if think_memory[299] / game_round > 0.05:
                return 'D'
        return 'C'
    elif strategy == 'Double Resurrection':
        if game_round == 0:
            return 'C'
        if game_round >= 5:
            inp = [memory[-1 - i][1-ap] for i in range(5)]
            if inp == ['C', 'C', 'C', 'C', 'C']:
                return 'D'
            if inp == ['D', 'D', 'D', 'D', 'D']:
                return 'C'
        return memory[-1][ap]
    elif strategy == 'Retaliate 2':
        retaliation_threshold = 0.08
        if game_round:
            last_round = (memory[-1][1-ap], memory[-1][ap])
            think_memory[301][last_round] += 1
        CD_count = think_memory[301][('C', 'D')]
        DC_count = think_memory[301][('D', 'C')]
        if CD_count > DC_count * retaliation_threshold:
            return 'D'
        return 'C'
    elif strategy == 'Retaliate 3':
        retaliation_threshold = 0.05
        if game_round:
            last_round = (memory[-1][1-ap], memory[-1][ap])
            think_memory[302][last_round] += 1
        CD_count = think_memory[302][('C', 'D')]
        DC_count = think_memory[302][('D', 'C')]
        if CD_count > DC_count * retaliation_threshold:
            return 'D'
        return 'C'
    elif strategy == 'Limited Retaliate 2':
        retaliation_threshold = 0.08

        """
        If the opponent has played D to my C more often than x% of the time
        that I've done the same to him, retaliate by playing D but stop doing
        so once I've hit the retaliation limit.
        """

        if game_round:
            last_round = (memory[-1][1-ap], memory[-1][ap])
            think_memory[303][3][last_round] += 1
        CD_count = think_memory[303][3][('C', 'D')]
        DC_count = think_memory[303][3][('D', 'C')]
        if CD_count > DC_count * retaliation_threshold:
            think_memory[303][0] = True
        else:
            think_memory[303][0] = False
            think_memory[303][1] = 0

        if think_memory[303][0]:
            if think_memory[303][1] < think_memory[303][2]:
                think_memory[303][1] += 1
                return 'D'
            else:
                think_memory[303][1] = 0
                think_memory[303][0] = False

        return 'C'
    elif strategy == 'Limited Retaliate 3':
        retaliation_threshold = 0.05

        """
        If the opponent has played D to my C more often than x% of the time
        that I've done the same to him, retaliate by playing D but stop doing
        so once I've hit the retaliation limit.
        """

        if game_round:
            last_round = (memory[-1][1-ap], memory[-1][ap])
            think_memory[304][3][last_round] += 1
        CD_count = think_memory[304][3][('C', 'D')]
        DC_count = think_memory[304][3][('D', 'C')]
        if CD_count > DC_count * retaliation_threshold:
            think_memory[304][0] = True
        else:
            think_memory[304][0] = False
            think_memory[304][1] = 0

        if think_memory[304][0]:
            if think_memory[304][1] < think_memory[304][2]:
                think_memory[304][1] += 1
                return 'D'
            else:
                think_memory[304][1] = 0
                think_memory[304][0] = False

        return 'C'
    elif strategy == 'Downing V2':
        round_number = game_round + 1

        if round_number == 1:
            return 'C'

        
        if round_number > 2:
            if memory[-2][1-ap] == 'D':
                if memory[-1][ap] == 'C':
                    think_memory[212][3] += 1
                think_memory[212][5] += 1
                think_memory[212][1] = think_memory[212][3] / think_memory[212][5]
            else:
                if memory[-1][ap] == 'C':
                    think_memory[212][2] += 1
                think_memory[212][4] += 1
                think_memory[212][0] = think_memory[212][2] / think_memory[212][4]
        
        c = 6.0 * think_memory[212][0] - 8.0 * think_memory[212][1] - 2
        alt = 4.0 * think_memory[212][0] - 5.0 * think_memory[212][1] - 1
        if c >= 0 and c >= alt:
            move = 'C'
        elif (0 <= c < alt) or (alt >= 0):
            move = 'C' if memory[-1][1-ap] == 'D' else 'D'
        else:
            move = 'D'
        return move
    elif strategy == 'Self Steem':
        """Actual strategy definition that determines player's action."""
        turns_number = game_round
        sine_value = mth.sin(2 * mth.pi * turns_number / 10)

        if sine_value > 0.95:
            return 'D'

        if 0.95 > abs(sine_value) > 0.3:
            return memory[-1][ap]

        if 0.3 > sine_value > -0.3:
            return rnd.choice(['C', 'D'])

        return 'C'
    elif strategy == 'Thue Morse':
        if not game_round:
            return 'C'
        else:
            if think_memory[305][1] == 0:
                thuemorse_code = think_memory[305][0]
                for i in think_memory[305][0]:
                    if i == '0':
                        thuemorse_code += '1'
                    else:
                        thuemorse_code += '0'

                think_memory[305][0] = thuemorse_code
            
                think_memory[305][1] = len(thuemorse_code)
            think_memory[305][1] -= 1

            return 'C' if think_memory[305][0][len(think_memory[305][0]) - (think_memory[305][1] + 1)] == '1' else 'D'
    elif strategy == 'Short Mem':
        if game_round <= 10:
            return 'C'

        array = [memory[-1 - i][ap] for i in range(10)]
        C_counts = array.count('C')
        D_counts = array.count('D')

        if C_counts - D_counts >= 3:
            return 'C'
        elif D_counts - C_counts >= 3:
            return 'D'
        else:
            return memory[-1][ap]
    elif strategy == 'Stalker':
        very_good_score = RPST['R']
        very_bad_score = RPST['P']
        wish_score = (RPST['R'] + RPST['P']) / 2

        """Actual strategy definition that determines player's action."""

        if game_round == 0:
            return 'C'

        current_average_score = score[1-ap] / game_round

        if current_average_score > very_good_score:
            return 'D'
        if (current_average_score > wish_score) and (
            current_average_score < very_good_score
        ):
            return 'C'
        if current_average_score > 2:
            return 'C'
        if (current_average_score < 2) and (current_average_score > 1):
            return 'D'
        return rnd.choice(['C', 'D'])
    elif strategy == 'Dynamic 2 Tits For Tat':
        """Actual strategy definition that determines player's action."""
        
        if not game_round:
            
            return 'C'
        think_memory[306] += (memory[-1][ap] == 'C')
        inp = [memory[-1 - i][ap] for i in range(min(2, game_round))]
        if 'D' in inp:
            
            return 'C'  if rnd.random() <= (
                think_memory[306] / game_round
            ) else 'D'
        else:
            return 'C'
    elif strategy == 'Hard Tit For Tat / 3 Tits For Tat':
        if not game_round:
            return 'C'
        inp = [memory[-1 - i][ap] for i in range(min(3, game_round))]
        if 'D' in inp:
            return 'D'
        return 'C'
    elif strategy == 'Hard Tit For 2 Tats / 2 Tits For 2 Tats':
        if not game_round:
            return 'C'
        inp = [memory[-1 - i][ap] for i in range(min(3, game_round))]
        if 'DD' in "".join(inp):
            return 'D'
        return 'C'
    elif strategy == 'Omega Tit For Tat':
        """Actual strategy definition that determines player's action."""
        
        if not game_round:
            return 'C'
        
        if game_round == 1:
            return memory[-1][ap]

        
        if think_memory[307][3] >= think_memory[307][0]:
            move = 'C'
            if think_memory[307][3] == think_memory[307][0]:
                think_memory[307][3] = think_memory[307][0] + 1
            else:
                think_memory[307][3] = 0
        else:
            
            if game_round >= 2:
                if (memory[-2][ap], memory[-1][ap]) == ('C', 'C'):
                    think_memory[307][2] -= 1
            
            if memory[-2][ap] != memory[-1][ap]:
                think_memory[307][2] += 1
            
            
            if memory[-1][1-ap] != memory[-1][ap]:
                think_memory[307][2] += 1
            
            
            if think_memory[307][2] >= think_memory[307][1]:
                move = 'D'
            else:
                
                move = memory[-1][ap]
                
                if memory[-2][ap] != memory[-1][ap]:
                    think_memory[307][3] += 1
                else:
                    think_memory[307][3] = 0
        return move
    elif strategy == 'Gradual Cristal Version':
        if not game_round:
            return 'C'

        if memory[-1][ap] == 'D':
            think_memory[281][2] += 1
        
        if think_memory[281][1] > 0:
            think_memory[281][1] -= 1
            return 'C'
        if memory[-1][ap] == 'D' or think_memory[281][0] > 0:
            if memory[-1][ap] == 'D':
                think_memory[281][0] = think_memory[281][2]
            think_memory[281][0] -= 1
            if think_memory[281][0] == 0:
                think_memory[281][1] = 2
            return 'D'
        else:
            return 'C'
    elif strategy == 'Spiteful Tit For Tat':
        if game_round == 0:
            return 'C'
        if game_round >= 2:
            if ((memory[-2][ap], memory[-1][ap]) == ('D', 'D')) or think_memory[308]:
                think_memory[308] = True
                return 'D'
        return memory[-1][ap]
    elif strategy == 'Slow Tit For 2 Tats 2':
        """Actual strategy definition that determines player's action."""

        
        if game_round < 2:
            return 'C'

        
        if memory[-2][ap] == memory[-1][ap]:
            return memory[-1][ap]

        
        return memory[-1][1-ap]
    elif strategy == 'Alexei':
        if game_round == 0:
            return 'C'
        if game_round >= (tournament_avg_last_round):
            return 'D'
        return memory[-1][ap]
    elif strategy == 'Eugine Nier':
        if not game_round:
            return 'C'
        if memory[-1][ap] == 'D':
            think_memory[309] += 1
        if (think_memory[309] >= 5) or (game_round >= (tournament_avg_last_round)):
            return 'D'
        return memory[-1][ap]
    elif strategy == 'N Tits For M Tats':
        if game_round <= (think_memory[310][0] - 1):
            return  'C'
        
        inp = sum([(memory[-1 - i][ap] == 'D') for i in range(think_memory[310][0])])
        if inp == think_memory[310][0]:
            think_memory[310][2] = think_memory[310][1]
        
        if think_memory[310][2] > 0:
            think_memory[310][2] -= 1
            return 'D'
        return 'C'
    elif strategy == 'Michaelos':
        """Actual strategy definition that determines player's action."""
        if not game_round:
            return 'C'
        if game_round >= (tournament_avg_last_round):
            think_memory[311] = True
        if think_memory[311]:
            return 'D'
        if memory[-1][1-ap] == 'D' and memory[-1][ap] == 'C':
            decision = rnd.choice(['C', 'D'])
            if decision == 'C':
                return 'C'
            else:
                think_memory[311] = True
                return 'D'

        return memory[-1][ap]
    elif strategy == 'Random Tit For Tat':
        Peta = 0.5
        if game_round % 2 == 0:
            return memory[-1][ap] if game_round > 0 else 'C'
        return 'C' if rnd.random() <= Peta else 'D'
    elif strategy == 'Burn Both Ends / BBE':
        
        if not game_round:
            
            return 'C'
        
        if memory[-1][ap] == 'C':
            
            return 'C' if rnd.random() <= 0.9 else 'D'
        
        return 'D'
    elif strategy == 'Very Bad':
        """Actual strategy definition that determines player's action."""
        total_moves = game_round

        if total_moves > 1 and memory[-1][ap] == 'C':
            think_memory[312] += 1

        if total_moves < 3:
            return 'C'

        cooperations = think_memory[312]

        cooperation_probability = cooperations / total_moves

        if cooperation_probability > 0.5:
            return 'C'

        elif cooperation_probability < 0.5:
            return 'D'

        else:
            return memory[-1][ap]
    elif strategy == 'Knowledgeable Worse & Worse':
        return 'D' if rnd.random() <= (game_round / tournament_avg_last_round) else 'C'
    elif strategy == 'Worse & Worse 2':
        current_round = game_round + 1
        if current_round == 1:
            return 'C'
        elif current_round <= 20:
            return memory[-1][ap]
        else:
            probability = 20 / current_round
            return 'C' if rnd.random() <= probability else 'D'
    elif strategy == 'Zero Determinant Extortion / ZD-X':
        phi = 0.2
        s = 0.1
        l = 1.0
        s_min = -min((RPST['T'] - l) / (l - RPST['S']), (l - RPST['S']) / (RPST['T'] - l))
        if (l < RPST['P']) or (l > RPST['R']) or (s > 1) or (s < s_min):
           raise ValueError

        p1 = 1 - phi * (1 - s) * (RPST['R'] - l)
        p2 = 1 - phi * (s * (l - RPST['S']) + (RPST['T'] - l))
        p3 = phi * ((l - RPST['S']) + s * (RPST['T'] - l))
        p4 = phi * (1 - s) * (l - RPST['P'])

        four_vector = [p1, p2, p3, p4]
        return 'C' if rnd.random() <= four_vector[((2 * (memory[-1][1-ap] == 'D')) + (memory[-1][ap] == 'D')) if game_round > 0 else 0] else 'D'
    elif strategy == 'Zero Determinant Generous / ZD-G':
        phi = 1/8
        s = 0.5
        l = 3.0
        s_min = -min((RPST['T'] - l) / (l - RPST['S']), (l - RPST['S']) / (RPST['T'] - l))
        if (l < RPST['P']) or (l > RPST['R']) or (s > 1) or (s < s_min):
           raise ValueError

        p1 = 1 - phi * (1 - s) * (RPST['R'] - l)
        p2 = 1 - phi * (s * (l - RPST['S']) + (RPST['T'] - l))
        p3 = phi * ((l - RPST['S']) + s * (RPST['T'] - l))
        p4 = phi * (1 - s) * (l - RPST['P'])

        four_vector = [p1, p2, p3, p4]
        return 'C' if rnd.random() <= four_vector[((2 * (memory[-1][1-ap] == 'D')) + (memory[-1][ap] == 'D')) if game_round > 0 else 0] else 'D'
    elif strategy == 'Zero Determinant Generous Tit For Tat / ZD-GTFT':
        phi = 0.25
        s = 0.5
        l = 3.0
        s_min = -min((RPST['T'] - l) / (l - RPST['S']), (l - RPST['S']) / (RPST['T'] - l))
        if (l < RPST['P']) or (l > RPST['R']) or (s > 1) or (s < s_min):
           raise ValueError

        p1 = 1 - phi * (1 - s) * (RPST['R'] - l)
        p2 = 1 - phi * (s * (l - RPST['S']) + (RPST['T'] - l))
        p3 = phi * ((l - RPST['S']) + s * (RPST['T'] - l))
        p4 = phi * (1 - s) * (l - RPST['P'])

        four_vector = [p1, p2, p3, p4]
        return 'C' if rnd.random() <= four_vector[((2 * (memory[-1][1-ap] == 'D')) + (memory[-1][ap] == 'D')) if game_round > 0 else 0] else 'D'
    elif strategy == 'Zero Determinant Mischief / ZD-MS':
        phi = 0.1
        s = 0.0
        l = 1.0
        s_min = -min((RPST['T'] - l) / (l - RPST['S']), (l - RPST['S']) / (RPST['T'] - l))
        if (l < RPST['P']) or (l > RPST['R']) or (s > 1) or (s < s_min):
           raise ValueError

        p1 = 1 - phi * (1 - s) * (RPST['R'] - l)
        p2 = 1 - phi * (s * (l - RPST['S']) + (RPST['T'] - l))
        p3 = phi * ((l - RPST['S']) + s * (RPST['T'] - l))
        p4 = phi * (1 - s) * (l - RPST['P'])

        four_vector = [p1, p2, p3, p4]
        return 'C' if rnd.random() <= four_vector[((2 * (memory[-1][1-ap] == 'D')) + (memory[-1][ap] == 'D')) if game_round > 0 else 0] else 'D'
    elif strategy == 'Zero Determinant Set / ZD-S':
        phi = 1/4
        s = 0.0
        l = 2.0
        s_min = -min((RPST['T'] - l) / (l - RPST['S']), (l - RPST['S']) / (RPST['T'] - l))
        if (l < RPST['P']) or (l > RPST['R']) or (s > 1) or (s < s_min):
           raise ValueError

        p1 = 1 - phi * (1 - s) * (RPST['R'] - l)
        p2 = 1 - phi * (s * (l - RPST['S']) + (RPST['T'] - l))
        p3 = phi * ((l - RPST['S']) + s * (RPST['T'] - l))
        p4 = phi * (1 - s) * (l - RPST['P'])

        four_vector = [p1, p2, p3, p4]
        return 'C' if rnd.random() <= four_vector[((2 * (memory[-1][1-ap] == 'D')) + (memory[-1][ap] == 'D')) if game_round > 0 else 0] else 'D'
    elif strategy == 'Zero Determinant Equalizer / ZD-Q':
        phi = 0.125
        s = 0.0
        l = 2.0
        s_min = -min((RPST['T'] - l) / (l - RPST['S']), (l - RPST['S']) / (RPST['T'] - l))
        if (l < RPST['P']) or (l > RPST['R']) or (s > 1) or (s < s_min):
           raise ValueError

        p1 = 1 - phi * (1 - s) * (RPST['R'] - l)
        p2 = 1 - phi * (s * (l - RPST['S']) + (RPST['T'] - l))
        p3 = phi * ((l - RPST['S']) + s * (RPST['T'] - l))
        p4 = phi * (1 - s) * (l - RPST['P'])

        four_vector = [p1, p2, p3, p4]
        return 'C' if rnd.random() <= four_vector[((2 * (memory[-1][1-ap] == 'D')) + (memory[-1][ap] == 'D')) if game_round > 0 else 0] else 'D'
    elif strategy == 'Laran':
        if not game_round:
            return "C"

        otak = think_memory[314]

        last_my = memory[-1][1-ap]
        last_opp = memory[-1][ap]
        curr = otak["current_state"]
    
        
        for s in range(otak["num_states"]):
            if otak["state_outputs"][s] == last_opp:
                otak["transitions"][curr][last_my][s] = 1.0  
                break
            
        
        probs = otak["transitions"][curr][last_my]
        otak["current_state"] = probs.index(max(probs))
    
        predicted_opp = otak["state_outputs"][otak["current_state"]]
        return "D" if predicted_opp == "C" else "C"
    elif strategy == 'Turan':
        if not game_round:
            return "C"

        otak = think_memory[315]

        last_my = memory[-1][1-ap]
        last_opp = memory[-1][ap]
        curr = otak["current_state"]
    
        probs = otak["transitions"][curr][last_my]
        for s in range(otak["num_states"]):
            if otak["state_outputs"][s] == last_opp:
                probs[s] += 0.25  
            
        
        total = sum(probs)
        otak["transitions"][curr][last_my] = [p / total for p in probs]
    
        
        otak["current_state"] = probs.index(max(probs))
        predicted_opp = otak["state_outputs"][otak["current_state"]]
    
        return "D" if predicted_opp == "C" else "C"
    elif strategy == 'Tages':
        if not game_round:
            return "C"

        otak = think_memory[316]
        look_ahead_depth = 10
        PAYOFFS = {('C', 'C'): RPST['R'],('C', 'D'): RPST['S'],('D', 'C'): RPST['T'],('D', 'D'): RPST['P']}

        last_my = memory[-1][1-ap]
        last_opp = memory[-1][ap]
        curr = otak["current_state"]
    
        
        probs = otak["transitions"][curr][last_my]
        for s in range(otak["num_states"]):
            if otak["state_outputs"][s] == last_opp:
                probs[s] += 0.25
        total = sum(probs)
        otak["transitions"][curr][last_my] = [p / total for p in probs]
        otak["current_state"] = probs.index(max(probs))
    
        
        def _simulasi(mulai_state, aksi_awal):
            skor_total = 0
            s_shadow = mulai_state
            a_shadow = aksi_awal
            for _ in range(look_ahead_depth):
                p_shadow = otak["transitions"][s_shadow][a_shadow]
                
                s_shadow = rnd.choices(range(otak["num_states"]), weights=p_shadow)[0]
                pred_opp = otak["state_outputs"][s_shadow]
                skor_total += PAYOFFS[(a_shadow, pred_opp)]
                a_shadow = pred_opp
            return skor_total

        skor_C = _simulasi(otak["current_state"], "C")
        skor_D = _simulasi(otak["current_state"], "D")
    
        return "C" if skor_C >= skor_D else "D"
    elif strategy == 'Pavlov D / Suspicious Pavlov':
        if not game_round:
            return 'D'
        if memory[-1][1-ap] == memory[-1][ap]:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Free Rider':
        if game_round > 0 and memory[-1][ap] == 'D':
            return '[SKIP]'
        return 'D'
    elif strategy == 'Rover':
        threshold = 2.0
        if game_round > 0 and (score[1-ap] - think_memory[318]) < threshold:
            return '[SKIP]'
        think_memory[318] = score[1-ap]
        return 'D'
    elif strategy == 'Vengeful':
        if game_round == 0:
            return 'C'
        instruction_string = "1:C1|D2,2:D3|D3,3:C1|D2"
        def translating(x):
            rule = {}
            semua_mode = x.strip().rstrip(',').split(',')
            for item in semua_mode:
                if not item: continue
                nomor_mode, isi_aksi = item.split(':')
                aksi_0_str, aksi_1_str = isi_aksi.split('|')
                
                aksi_0 = [aksi_0_str[0], aksi_0_str[1:]]
                aksi_1 = [aksi_1_str[0], aksi_1_str[1:]]
                rule[nomor_mode] = [aksi_0, aksi_1]
            return rule
        instruction = translating(instruction_string)
        finite_state_rule = instruction[think_memory[319]][(memory[-1][ap] == 'D')]
        think_memory[319] = finite_state_rule[1]
        return finite_state_rule[0]
    elif strategy == 'Deadlock Breaker':
        if game_round == 0:
            return 'C'
        instruction_string = "1:C1|D2,2:C1|D3,3:C1|C1"
        def translating(x):
            rule = {}
            semua_mode = x.strip().rstrip(',').split(',')
            for item in semua_mode:
                if not item: continue
                nomor_mode, isi_aksi = item.split(':')
                aksi_0_str, aksi_1_str = isi_aksi.split('|')
                
                aksi_0 = [aksi_0_str[0], aksi_0_str[1:]]
                aksi_1 = [aksi_1_str[0], aksi_1_str[1:]]
                rule[nomor_mode] = [aksi_0, aksi_1]
            return rule
        instruction = translating(instruction_string)
        finite_state_rule = instruction[think_memory[320]][(memory[-1][ap] == 'D')]
        think_memory[320] = finite_state_rule[1]
        return finite_state_rule[0]
    elif strategy == 'Treasure Hunt / Sugar Strategy':
        if game_round == 0:
            return 'C'
        instruction_string = "1:C2|D4,2:D3|D4,3:C3|D4,4:D4|D4"
        def translating(x):
            rule = {}
            semua_mode = x.strip().rstrip(',').split(',')
            for item in semua_mode:
                if not item: continue
                nomor_mode, isi_aksi = item.split(':')
                aksi_0_str, aksi_1_str = isi_aksi.split('|')
                
                aksi_0 = [aksi_0_str[0], aksi_0_str[1:]]
                aksi_1 = [aksi_1_str[0], aksi_1_str[1:]]
                rule[nomor_mode] = [aksi_0, aksi_1]
            return rule
        instruction = translating(instruction_string)
        finite_state_rule = instruction[think_memory[321]][(memory[-1][ap] == 'D')]
        think_memory[321] = finite_state_rule[1]
        return finite_state_rule[0]
    elif strategy == 'Nasty Tit For Tat':
        if rnd.random() <= 0.1:
            think_memory[322] = 2

        if think_memory[322] > 0:
            think_memory[322] -= 1
            return 'D'
        return memory[-1][ap] if game_round > 0 else 'C'
    elif strategy == 'Analogy':
        if game_round < 2:
            return 'C'

        if memory[-2][1-ap] == 'C':
            think_memory[323][0] += 1
            if memory[-1][ap] == 'C':
                think_memory[323][1] += 1
        else:
            think_memory[323][2] += 1
            if memory[-1][ap] == 'C':
                think_memory[323][3] += 1

        pc_given_c = (think_memory[323][1] / think_memory[323][0]) if think_memory[323][0] > 0 else 0.5
        pc_given_d = (think_memory[323][3] / think_memory[323][2]) if think_memory[323][2] > 0 else 0.5

        E_sh = (pc_given_c * RPST['R']) + ((1 - pc_given_c) * RPST['S'])
        E_st = (pc_given_d * RPST['T']) + ((1 - pc_given_d) * RPST['P'])

        return 'C' if E_sh > E_st else 'D'
    elif strategy == 'Look Up / Look Ahead':
        depth = 5
        PAYOFFS = {
            ('C', 'C'): (RPST['R'], RPST['R']),
            ('C', 'D'): (RPST['S'], RPST['T']),
            ('D', 'C'): (RPST['T'], RPST['S']),
            ('D', 'D'): (RPST['P'], RPST['P'])
        }

        
        if game_round < 2:
            return 'C'

        
        def predict_opponent_response(last_my_move, last_opp_move):
            count_C = 0
            count_D = 0
            
            for i in range(game_round - 1):
                if memory[i][1-ap] == last_my_move and memory[i][ap] == last_opp_move:
                    if memory[i + 1][ap] == 'C':
                        count_C += 1
                    else:
                        count_D += 1
        
            if count_C == 0 and count_D == 0:
                return last_opp_move  
            return 'C' if count_C >= count_D else 'D'

        
        def simulate_future_score(chosen_action):
            total_score = 0
        
            
            pred_opp = predict_opponent_response(memory[-1][1-ap], memory[-1][ap])
            total_score += PAYOFFS[(chosen_action, pred_opp)][0]
        
            
            curr_my = chosen_action
            curr_opp = pred_opp
            for _ in range(1, depth):
                next_my = 'C' if curr_opp == 'C' else 'D'
                next_opp = predict_opponent_response(curr_my, curr_opp)
                total_score += PAYOFFS[(next_my, next_opp)][0]
                curr_my, curr_opp = next_my, next_opp
            
            return total_score

        
        score_if_C = simulate_future_score('C')
        score_if_D = simulate_future_score('D')

        return 'C' if score_if_C >= score_if_D else 'D'
    elif strategy == 'Tit For 3 Tats':
        if game_round >= 3:
            if (memory[-3][ap], memory[-2][ap], memory[-1][ap]) == ('D', 'D', 'D'):
                return 'D'
        return 'C'
    elif strategy == 'Lenient Grim 2':
        if game_round == 0:
            return 'C'
        if game_round >= 2:
            if (memory[-2][ap], memory[-1][ap]) == ('D', 'D'):
                return 'D'
        return memory[-1][1-ap]
    elif strategy == 'Lenient Grim 3':
        if game_round == 0:
            return 'C'
        if game_round >= 3:
            if (memory[-3][ap], memory[-2][ap], memory[-1][ap]) == ('D', 'D', 'D'):
                return 'D'
        return memory[-1][1-ap]
    elif strategy == 'Exploratory Tit For 3 Tats':
        if game_round >= 3:
            if (memory[-3][ap], memory[-2][ap], memory[-1][ap]) == ('D', 'D', 'D'):
                return 'D'
        return 'C' if rnd.random() > 0.1 else 'D'
    elif strategy == 'Exploratory Lenient Grim 2':
        if game_round == 0:
            return 'C' if rnd.random() > 0.1 else 'D'
        if game_round >= 2:
            if (memory[-2][ap], memory[-1][ap]) == ('D', 'D'):
                think_memory[324] = True
        if think_memory[324]:
            return 'D'
        else:
            return 'C' if rnd.random() > 0.1 else 'D'
    elif strategy == 'False Cooperator':
        return 'C' if game_round < 10 else 'D'
    elif strategy == 'Boxer':
        d_threshold = 0.4
        min_rounds = 3
        req_c_to_recover = 3

        if not game_round:
            return 'C'

        think_memory[325][1].append(memory[-1][ap])

        
        if think_memory[325][0]:
            recent_opp_moves = think_memory[325][1][-req_c_to_recover:]
            if len(recent_opp_moves) == req_c_to_recover and all(m == 'C' for m in recent_opp_moves):
                think_memory[325][0] = False  
                return 'C'
            return 'D'

        
        if len(think_memory[325][1]) >= min_rounds:
            d_count = think_memory[325][1].count('D')
            d_ratio = d_count / len(think_memory[325][1])
            if d_ratio > d_threshold:
                think_memory[325][0] = True
                return 'D'

        return 'C'
    elif strategy == 'Bros Mind':
        decay_rate = 0.7  

        if not game_round:
            return 'C'

        
        last_opp_move = memory[-1][ap]
        if last_opp_move == 'C':
            
            think_memory[326] = min(1.0, think_memory[326] + 0.1)
        else:
            
            think_memory[326] *= decay_rate

        
        return 'C' if think_memory[326] >= 0.5 else 'D'
    elif strategy == 'Crabby':
        anger_threshold = 2
        punishment_length = 3

        if not game_round:
            return 'C'

        
        if think_memory[327] > 0:
            think_memory[327] -= 1
            return 'D'

        
        recent_opp = [memory[-1 - i][ap] for i in range(min(3, game_round))]
        if recent_opp.count('D') >= anger_threshold:
            think_memory[327] = punishment_length - 1
            return 'D'

        return 'C'
    elif strategy == 'Observant':
        if game_round < 5:
            return 'C'

        inp = [memory[-1 - i][ap] for i in range(game_round)]
        d_ratio = inp.count('D') / game_round
        if d_ratio > 0.5:
            return 'D'
        return 'C'
    elif strategy == 'Mensa':
        if game_round < 3:
            return 'C'

        inp = [memory[-1 - i][ap] for i in range(game_round)]
        c_ratio = inp.count('C') / game_round
        
        
        if c_ratio > 0.8:
            return 'D'
        
        
        return memory[-1][ap]
    elif strategy == 'Killer':
        if game_round < 4:
            
            return 'D' if game_round % 2 == 1 else 'C'
        
        
        inp = [memory[-1 - i][ap] for i in range(game_round)]
        if inp.count('D') == 0:
            return 'D'
            
        
        return memory[-1][ap]
    elif strategy == 'MARS / Mimicry And Relative Similarity':
        window_size = 5
        threshold = 0.6

        if game_round < window_size:
            return 'C'  
        
        
        matches = sum(1 for m, o in memory[-window_size:] if m == o)
        similarity = matches / window_size
        
        
        if similarity >= threshold:
            return 'C'
        
        
        return memory[-1][ap]
    elif strategy == 'TOM Level 1 / Theory Of Mind Level 1':
        if not game_round:
            return 'C'
        
        
        
        my_history = [memory[-1 - i][1-ap] for i in range(game_round)]
        my_c_ratio = my_history.count('C') / game_round
        
        
        
        
        predicted_opp_move = 'D' if my_c_ratio > 0.7 or memory[-1][ap] == 'D' else 'C'
        
        
        
        
        if predicted_opp_move == 'D':
            return 'D'
        return 'C'
    elif strategy == 'Bandit UCB':
        payoff_map = {
            ('C', 'C'): RPST['R'], ('C', 'D'): RPST['S'],
            ('D', 'C'): RPST['T'], ('D', 'D'): RPST['P']
        }
        def _update_payoffs():
            if not game_round:
                return
            last_my = memory[-1][1-ap]
            last_opp = memory[-1][ap]
            reward = payoff_map[(last_my, last_opp)]
        
            think_memory[328][0][last_my] += 1
            think_memory[328][1][last_my] += reward

        _update_payoffs()
        
        t = game_round
        
        if t < 2:
            return 'C' if t == 0 else 'D'
        
        
        ucb_values = {}
        for action in ['C', 'D']:
            mean_reward = think_memory[328][1][action] / think_memory[328][0][action]
            bonus = mth.sqrt((2 * mth.log(t)) / think_memory[328][0][action])
            ucb_values[action] = mean_reward + bonus
            
        
        if ucb_values['C'] == ucb_values['D']:
            return rnd.choice(['C', 'D'])
        return 'C' if ucb_values['C'] > ucb_values['D'] else 'D'
    elif strategy == 'MCMC / Markov Chain Monte Carlo':
        num_samples = 100

        if game_round < 2:
            return 'C'
            
        current_state = (memory[-1][1-ap], memory[-1][ap])
        
        
        matching_next_opp_moves = []
        for i in range(game_round - 1):
            state_i = (memory[i][1-ap], memory[i][ap])
            if state_i == current_state:
                matching_next_opp_moves.append(memory[i+1][ap])
                
        if not matching_next_opp_moves:
            return memory[-1][ap]  
            
        
        d_count = 0
        for _ in range(num_samples):
            sampled_move = rnd.choice(matching_next_opp_moves)
            if sampled_move == 'D':
                d_count += 1
                
        
        prob_opp_defect = d_count / num_samples
        
        
        if prob_opp_defect > 0.5:
            return 'D'
        return 'C'
    elif strategy == 'Star S':
        handshake = ['C', 'C', 'D', 'C', 'D']
        if game_round < len(handshake):
            return handshake[game_round]
        inp = [memory[i][ap] for i in range(len(handshake))]
        if inp == handshake:
            return 'D'
        return 'D'
    elif strategy == 'Star SL':
        handshake = ['C', 'C', 'D', 'C', 'D']
        if game_round < len(handshake):
            return handshake[game_round]
        inp = [memory[i][ap] for i in range(len(handshake))]
        if inp == handshake:
            return 'D'
        return memory[-1][ap]
    elif strategy == 'Star SN':
        handshake = ['C', 'C', 'D', 'C', 'D']
        max_noise_error = 1
        if game_round < len(handshake):
            return handshake[game_round]
        inp = [memory[i][ap] for i in range(len(handshake))]
        verify = sum((inp[i] != handshake[i]) for i in range(len(handshake)))
        if verify <= max_noise_error:
            return 'D'
        return 'D'
    elif strategy == 'Star Slave 1':
        handshake = ['C', 'C', 'D', 'C', 'D']
        if game_round < len(handshake):
            return handshake[game_round]
        inp = [memory[i][ap] for i in range(len(handshake))]
        if inp == handshake:
            return 'C'
        return 'D'
    elif strategy == 'Star Slave 2':
        handshake = ['C', 'C', 'D', 'C', 'D']
        if game_round < len(handshake):
            return handshake[game_round]
        inp = [memory[i][ap] for i in range(len(handshake))]
        if inp == handshake:
            return 'C'
        return 'D'
    elif strategy == 'Star Slave 3':
        handshake = ['C', 'C', 'D', 'C', 'D']
        if game_round < len(handshake):
            return handshake[game_round]
        inp = [memory[i][ap] for i in range(len(handshake))]
        if inp == handshake:
            return 'C'
        return 'D'
    elif strategy == 'Star Slave 4':
        handshake = ['C', 'C', 'D', 'C', 'D']
        if game_round < len(handshake):
            return handshake[game_round]
        inp = [memory[i][ap] for i in range(len(handshake))]
        if inp == handshake:
            return 'C'
        return 'D'
    elif strategy == 'Star Slave 5':
        handshake = ['C', 'C', 'D', 'C', 'D']
        if game_round < len(handshake):
            return handshake[game_round]
        inp = [memory[i][ap] for i in range(len(handshake))]
        if inp == handshake:
            return 'C'
        return 'D'
    elif strategy == 'Star Slave 6':
        handshake = ['C', 'C', 'D', 'C', 'D']
        if game_round < len(handshake):
            return handshake[game_round]
        inp = [memory[i][ap] for i in range(len(handshake))]
        if inp == handshake:
            return 'C'
        return 'D'
    elif strategy == 'Star Slave 7':
        handshake = ['C', 'C', 'D', 'C', 'D']
        if game_round < len(handshake):
            return handshake[game_round]
        inp = [memory[i][ap] for i in range(len(handshake))]
        if inp == handshake:
            return 'C'
        return 'D'
    elif strategy == 'Star Slave 8':
        handshake = ['C', 'C', 'D', 'C', 'D']
        if game_round < len(handshake):
            return handshake[game_round]
        inp = [memory[i][ap] for i in range(len(handshake))]
        if inp == handshake:
            return 'C'
        return 'D'
    elif strategy == 'Star Slave 9':
        handshake = ['C', 'C', 'D', 'C', 'D']
        if game_round < len(handshake):
            return handshake[game_round]
        inp = [memory[i][ap] for i in range(len(handshake))]
        if inp == handshake:
            return 'C'
        return 'D'
    elif strategy == 'Star Slave 10':
        handshake = ['C', 'C', 'D', 'C', 'D']
        if game_round < len(handshake):
            return handshake[game_round]
        inp = [memory[i][ap] for i in range(len(handshake))]
        if inp == handshake:
            return 'C'
        return 'D'
    elif strategy == 'Generous Tit For Tat Less Wrong Version / Tit For Tat But Pico Generous':
        if not game_round:
            return 'C'

        if rnd.random() <= 0.0000004839:
            return 'C'
        return memory[-1][ap]
    elif strategy == 'Caerbannog':
        if not game_round:
            return 'C'

        if game_round >= (tournament_avg_last_round):
            return 'D'
        if rnd.random() <= 0.2:
            return 'C'
        return memory[-1][ap]
    elif strategy == 'Jem':
        if game_round >= 2:
            if memory[-1][ap] == 'D':
                if memory[-2][1-ap] == 'C':
                    think_memory[329] += 1
        if game_round >= (tournament_avg_last_round):
            return 'D'
        A = 0.5 ** think_memory[329]
        if rnd.random() <= A:
            return 'C'
        return memory[-1][ap]
    elif strategy == 'Rwallace':
        think_memory[330] += (memory[-1][ap] == 'D') if game_round > 0 else 0
        if (game_round >= (tournament_avg_last_round)) or (think_memory[330] >= 3):
            return 'D'
        return memory[-1][ap] if game_round > 0 else 'C'
    elif strategy == 'Malthrin':
        if game_round:
            inp = [(memory[i][ap] == 'D') for i in range(game_round)]
            if (sum(inp) >= 7) or (game_round >= (tournament_avg_last_round - 1)):
                return 'D'
        return memory[-1][ap] if game_round > 0 else 'C'
    elif strategy == 'Vengeful Cheater':
        if not game_round:
            return 'C'
        if memory[-1][ap] == 'D':
            think_memory[331] = 1
        if (game_round + 1) < (tournament_avg_last_round - 1):
            return 'D' if think_memory[331] == 1 else 'C'
        return 'D'
    elif strategy == 'Jesus Petry':
        if not game_round:
            return 'C'
        if game_round >= (tournament_avg_last_round - 1):
            return 'D'
        
        if think_memory[332][2]:
            think_memory[332][2] -= 1
            if think_memory[332][2] > 0:
                return 'C'
            return 'C' if 'D' not in (memory[-2][ap], memory[-3][ap]) else memory[-1][ap]
        if (game_round + 1) <= 35:
            if (game_round + 1) in (22, 35):
                think_memory[332][2] = 2
                return 'D'
        elif (game_round + 1) == (think_memory[332][0] + think_memory[332][1] - (19 * think_memory[332][3])):
            A = (think_memory[332][0] + think_memory[332][1] - (19 * think_memory[332][3]))
            think_memory[332][0] = think_memory[332][1]
            think_memory[332][1] = A
            think_memory[332][2] = 2
            think_memory[332][3] = 1
            return 'D'
        return memory[-1][ap]
    elif strategy == 'FAWS':
        if game_round:
            think_memory[333][1].append(memory[-1][ap])
            think_memory[333][0].append(memory[-1][1-ap])

        turn = game_round + 1  

        opp_defects = think_memory[333][1].count('D')

        
        if opp_defects >= 3:
            return 'D'

        
        if think_memory[333][3]:
            return think_memory[333][3].pop(0)

        
        if think_memory[333][4] == 'RULE_10_DEFECTING':
            if game_round > 0 and think_memory[333][1][-1] == 'D':
                think_memory[333][4] = None
                think_memory[333][3] = ['C', 'C']
                
                
            else:
                return 'D'

        
        if turn >= tournament_avg_last_round - 1: 
            
            if think_memory[333][4] == 'USE_RULE_12':
                if opp_defects == 1:
                    return 'C' if turn == 1 else think_memory[333][1][-1] 
                return 'D'
            
            return 'D'

        
        if turn == 1:
            return 'C'

        
        if turn == 20:
            
            if think_memory[333][1][:20].count('D') == 0:
                think_memory[333][2] = rnd.randint(21, 30)

        if think_memory[333][2] is not None:
            
            if 20 < turn < think_memory[333][2] and think_memory[333][1][-1] == 'D':
                think_memory[333][2] = None 
                think_memory[333][3] = ['D', 'C'] 
                think_memory[333][4] = 'USE_RULE_12'
                return 'C'

            
            if turn == think_memory[333][2]:
                think_memory[333][2] = None 
                return 'D'

        
        
        if game_round >= 1 and think_memory[333][0][-1] == 'D' and turn > 21:
            
            last_turn_idx = game_round - 1
            
            
            if game_round >= 1 and think_memory[333][1][-1] == 'D':
                think_memory[333][4] = 'USE_RULE_12'
                return 'C'

            
            if game_round >= 2 and think_memory[333][1][-1] == 'D':
                return 'C'

            
            if game_round >= 3 and think_memory[333][1][-1] == 'D':
                think_memory[333][4] = 'USE_RULE_12'
                return 'C'

            
            if game_round >= 2 and think_memory[333][1][-1] == 'C' and think_memory[333][1][-2] == 'C':
                think_memory[333][4] = 'RULE_10_DEFECTING'
                return 'D'

        
        return think_memory[333][1][-1] if think_memory[333][1] else 'C'
    elif strategy == 'Second Chance':
        if game_round:
            think_memory[334][0].append(memory[-1][1-ap])
            think_memory[334][1].append(memory[-1][ap])

        turn = game_round + 1  

        
        
        
        if turn == 1:
            return 'C'
        if turn >= (tournament_avg_last_round - 2):
            return 'D'

        
        
        
        
        
        
        if think_memory[334][2]:
            return 'D'

        
        if game_round >= 2:
            coop_indices = [
                i for i in range(game_round - 1) 
                if think_memory[334][0][i] == 'C'
            ]
            if len(coop_indices) >= 4:
                
                all_exploited = all(think_memory[334][1][i] == 'D' for i in coop_indices)
                if all_exploited:
                    think_memory[334][2] = True
                    return 'D'

        
        
        
        
        if game_round >= 2:
            prev_my = think_memory[334][0][:-1]
            prev_opp = think_memory[334][1]

            c_count = prev_my.count('C')
            d_count = prev_my.count('D')

            if c_count >= 8 and d_count >= 10:
                
                
                followed_by_c_after_my_c = 0
                followed_by_c_after_my_d = 0

                for i in range(len(prev_my)):
                    if prev_my[i] == 'C' and prev_opp[i] == 'C':
                        followed_by_c_after_my_c += 1
                    elif prev_my[i] == 'D' and prev_opp[i] == 'C':
                        followed_by_c_after_my_d += 1

                x = followed_by_c_after_my_c / c_count
                y = followed_by_c_after_my_d / d_count

                
                if (4 * x) < (6 * y + 1):
                    return 'D'

        
        
        
        
        
        opp_defects = think_memory[334][1].count('D')
        if opp_defects > 0 and opp_defects % 4 == 0:
            return 'C'

        
        
        
        
        return think_memory[334][1][-1] if think_memory[334][1] else 'C'
    elif strategy == 'Simple Identity ChecK':
        turn = game_round + 1
        if not game_round:
            return 'C'
        if turn <= 57:
            return memory[-1][ap]
        elif turn == 58:
            return 'D'
        elif turn == 59:
            if memory[:57] == ([('C', 'C')] * 57) and memory[57] == ('D', 'D'):
                return 'C'
            return 'D'
        return memory[-1][ap] if memory[58][1-ap] == 'C' else 'D'
    elif strategy == 'Evil Alliance':
        if game_round < 5:
            return 'D'
        inp1 = [memory[i][ap] for i in range(5)]
        inp2 = [memory[-1 - i][ap] for i in range(5)]
        if 'C' in inp1 or ('C' not in inp2 if game_round > 5 else False):
            return 'D'
        elif game_round == 5:
            return 'C'
        return memory[-1][1-ap]
    elif strategy == 'Probe & Punish':
        if game_round > 0:
            if memory[-1][ap] == 'D':
                think_memory[335][1] = 11
                think_memory[335][0] = 2
                return 'D'
        
        if think_memory[335][1]:
            think_memory[335][1] -= 1
            return 'D'
        
        if think_memory[335][0]:
            think_memory[335][0] -= 1
            return 'C'
        return 'C'
    elif strategy == 'Rack Block Shooter':
        """
        0 = Defect (Mengkhianati), 1 = Cooperate (Bekerja sama)
        """
        
        handshaking_sequence = ['C', 'D', 'D', 'C']  
        
        
        payoff = {
            (1, 1): RPST['R'],  
            (1, 0): RPST['S'],  
            (0, 1): RPST['T'],  
            (0, 0): RPST['P']   
        }

        if game_round > 0:
            
            if game_round < len(handshaking_sequence):
                
                
                if memory[-1][ap] != handshaking_sequence[game_round]:
                    think_memory[336][0] = False

            
            
            if memory[-1][1-ap] == 1:  
                if memory[-1][ap] == 1:
                    think_memory[336][1] += 1.0  
                else:
                    think_memory[336][2] += 1.0   
            else:  
                if memory[-1][ap] == 1:
                    think_memory[336][3] += 1.0  
                else:
                    think_memory[336][4] += 1.0   

        def get_expected_p_cooperate(alpha, beta):
            """Menghitung ekspektasi (nilai rata-rata) dari distribusi Beta."""
            return alpha / (alpha + beta)

        
        if game_round < len(handshaking_sequence):
            return handshaking_sequence[game_round]

        
        if think_memory[336][0]:
            return 'C'  

        
        
        p_c_given_c = get_expected_p_cooperate(think_memory[336][1], think_memory[336][2])
        
        p_c_given_d = get_expected_p_cooperate(think_memory[336][3], think_memory[336][4])

        
        eu_cooperate = (p_c_given_c * payoff[(1, 1)]) + ((1 - p_c_given_c) * payoff[(1, 0)])

        
        eu_defect = (p_c_given_d * payoff[(0, 1)]) + ((1 - p_c_given_d) * payoff[(0, 0)])

        
        return 'C' if eu_cooperate > eu_defect else 'D'
    elif strategy == 'Control 1':
        if not game_round:
            p = 0.5
        else:
            unziped_memory = list(zip(*memory))
            my_history, opponent_history = unziped_memory[1-ap], unziped_memory[ap]
            last_my_move = my_history[-1]
            
            matching_indices = [i for i, move in enumerate(my_history[:-1]) if move == last_my_move]
        
            if not matching_indices:
                p = 0.5
            else:
                
                cooperations = sum(1 for i in matching_indices if opponent_history[i + 1] == 'C')
                p = cooperations / len(matching_indices)
            
        Ec = 5 * p - 7
        Ed = 6 * p - 6
        exp_Ec = mth.exp(Ec)
        exp_Ed = mth.exp(Ed)
        prob_cooperate = exp_Ec / (exp_Ec + exp_Ed)
    
        return 'C' if rnd.random() < prob_cooperate else 'D'
    elif strategy == 'Control 2':
        if not game_round:
            return 'C'
        opponent_history = [memory[i][ap] for i in range(game_round)]
        if opponent_history[0] == 'D' or opponent_history.count('D') > 2:
            return 'D'
        return 'C'
    elif strategy == 'Control 3':
        turn = game_round + 1
        if turn == 1:
            return 'D'
        if turn == 2:
            return 'C'
    
        if memory[-1][ap] == memory[-2][ap]:
            return memory[-1][ap]
        else:
            return flip(memory[-1][1-ap])
    elif strategy == 'Control 4':
        turn = game_round + 1
        if turn <= 3:
            return 'C'
        if turn >= (tournament_avg_last_round - 1):
            return 'D'

        opponent_history = [memory[i][ap] for i in range(game_round)]
    
        coop_rate = opponent_history.count('C') / game_round
        return 'C' if coop_rate >= 0.85 else 'D'
    elif strategy == 'Control 5':
        turn = game_round + 1
        if turn <= 3:
            return 'C'
        if turn == tournament_avg_last_round:  
            return 'D'

        opponent_history = [memory[i][ap] for i in range(game_round)]
    
        coop_rate = opponent_history.count('C') / game_round
        return 'C' if rnd.random() < coop_rate else 'D'
    elif strategy == 'Control 6':
        if not game_round:
            return 'C'

        opponent_history = [memory[i][ap] for i in range(game_round)]
    
        
        if opponent_history[-1] == 'D' and opponent_history.count('D') == 1:
            return 'C'
        
        return opponent_history[-1]
    elif strategy == 'Control 7':
        turn = game_round + 1
        if turn == 1:
            return rnd.choice(['C', 'D'])
    
        n = turn // 2  
        target_index = n - 1  
        return memory[target_index][ap]
    elif strategy == 'Control 8':
        turn = game_round + 1
    
        
        if turn in [20, 40, 60, 80]:
            return 'C'

        if turn > 1:
            unziped_memory = list(zip(*memory))
            my_history, opponent_history = unziped_memory[1-ap], unziped_memory[ap]
    
            cond1 = game_round >= 3 and opponent_history[-3:].count('D') >= 2
            cond2 = game_round >= 2 and my_history[-1] == 'D' and my_history[-2] == 'C'
            cond3 = game_round >= 10 and 'D' not in my_history[-10:]
    
            if cond1 or cond2 or cond3:
                return 'D'
        return 'C'
    elif strategy == 'Control 9':
        think_memory[338][0] = 1  
        think_memory[338][1] = 0
        think_memory[338][2] = 0
        think_memory[338][3] = 0

        turn = game_round + 1
        
        
        if turn > 1:
            unziped_memory = list(zip(*memory))
            my_history, opponent_history = unziped_memory[1-ap], unziped_memory[ap]

            my_last = my_history[-1]
            op_last = opponent_history[-1]
            
            payoff = {('C', 'C'): RPST['R'], ('C', 'D'): RPST['S'], ('D', 'C'): RPST['T'], ('D', 'D'): RPST['P']}
            think_memory[338][3] += payoff[(my_last, op_last)]
            
            if op_last == 'D':
                think_memory[338][1] += 1

        
        if (turn - 1) > 0 and (turn - 1) % 10 == 0 and (turn - 1) != think_memory[338][2]:
            if 16 <= think_memory[338][3] <= 34:
                think_memory[338][0] = 2 if think_memory[338][0] == 1 else 1
                think_memory[338][1] = 0  
            think_memory[338][3] = 0
            think_memory[338][2] = turn - 1

        
        if think_memory[338][0] == 1:  
            if game_round >= 2:
                if opponent_history[-1] == 'D' and opponent_history[-2] == 'D':
                    return 'D'
            return 'C'
        else:  
            if think_memory[338][1] >= 1:
                return 'D'
            return 'C'
    elif strategy == 'Control 10':
        turn = game_round + 1
        AD = (200 / tournament_avg_last_round)

        opponent_history = [memory[i][ap] for i in range(game_round)]
    
        if turn in [1, 2]:
            return 'C'
        elif 3 <= turn <= round(29 * AD):
            return opponent_history[-1]  
        elif round(30 * AD) <= turn <= round(84 * AD):
            return 'D' if opponent_history.count('D') > 7 else opponent_history[-1]
        elif turn == round(84 * AD) + 1:
            return 'D'
        elif turn in [round(84 * AD) + 2, round(84 * AD) + 3]:
            return 'D' if opponent_history.count('D') > 7 else 'C'
        elif round(84 * AD) + 4 <= turn <= (tournament_avg_last_round - 2):
            if game_round >= round(84 * AD) + 3 and (opponent_history[round(84 * AD) + 2] == 'D' or opponent_history.count('D') > 4):
                return 'D'
            return 'C'
        elif (tournament_avg_last_round - 1) <= turn:
            return 'D'
    
        return 'C'
    elif strategy == 'Control 11':
        turn = game_round + 1

        opponent_history = [memory[i][ap] for i in range(game_round)]
    
        if turn == 1:
            return 'C'
        if turn == 2:
            return opponent_history[-1]
        if turn == tournament_avg_last_round:
            return 'D'
    
        op_last_d = opponent_history[-1] == 'D'
        my_last_c = memory[-1][1-ap] == 'C'
        op_prev_d = game_round >= 2 and opponent_history[-2] == 'D'
    
        if op_last_d and (my_last_c or op_prev_d):
            return 'D'
        return 'C'

    
    elif strategy == 'Pavlov Tester':
        if game_round > 0:
            if memory[-1][ap] == 'D':
                if think_memory[3][1] < 2:
                    think_memory[3][1] += 1
            else:
                if think_memory[3][1] > 0:
                    think_memory[3][1] -= 1
        if think_memory[3][0] != 3 and think_memory[3][1] < 2:
            if think_memory[3][0] < 2:
                think_memory[3][0] += 1
                return 'D'
            else:
                think_memory[3][0] = 0
                return 'C'
        else:
            if think_memory[3][0] != 3:
                think_memory[3][0] = 3
                return 'C'
            else:
                if not game_round:
                    return 'C'
                return memory[-1][ap]
    elif strategy == 'Clement Coldridge':
        if game_round > 0:
            if memory[-1][ap] == 'D' or think_memory[4] == 1:
                if rnd.random() < 0.2:
                    think_memory[4] = 0
                    return 'C'
                else:
                    think_memory[4] = 1
                    return 'D'
        return 'C'
    elif strategy == 'Gaslighter':
        SH = (memory[-1][ap] == 'C') if game_round > 0 else True
        if not SH:
            if (game_round + 1) > 2 and memory[-1-1][ap] == 'D':
                return 'D'
            elif rnd.random() < (1 - (think_memory[6][1] / (game_round + 1))):
                return 'D'
        else:
            think_memory[6][1] += 1
        rand_est = 1 - (think_memory[6][1] / (game_round + 1))
        A = ((1 - think_memory[6][0]) * SH) + (think_memory[6][0] * (think_memory[6][1] / (game_round + 1)))

        A = A * (1 - 0.3 * rand_est)
        think_memory[6][0] = A
        if rnd.random() > (A - 0.05):
            return 'D'
        else:
            return 'C'
    elif strategy == 'Springter':
        if (game_round + 1) % 10 == 0 or think_memory[8][1] == 1:
            if think_memory[8][1] == 0: 
                think_memory[8][1] = 1
            if memory[-1][ap] == 'D': 
                think_memory[8][0] += 1
            if think_memory[8][0] > 0:
                think_memory[8][0] -= 1
                return 'D'
            else:
                think_memory[8][1] = 0
                return 'C'
        elif (memory[-1][ap] == 'D') if game_round > 0 else False:
            think_memory[8][0] += 1
            return 'C'
        else:
            return 'C'
    elif strategy == 'William':
        G = rnd.random()
        if ((memory[-1][ap] == 'D') if game_round > 0 else False) and G > 0.2:
            return 'D'
        SH = (memory[-1][ap] == 'C') if game_round > 0 else True
        if SH:
            think_memory[9][0] += 1
        elif G <= 0.2:
            think_memory[9][1] += 1
        elif (memory[-1-1][ap] == 'D') if (game_round + 1) > 2 else False:
            think_memory[9][1] -= 1
        think_memory[9][1] = max(think_memory[9][1], 0)
        A = min(((SH * (think_memory[9][0] / (game_round + 1))) + (think_memory[9][1] / (game_round + 1))),1)
        if rnd.random() > A:
            return 'D'
        else:
            return 'C'
    elif strategy == 'Random Exploiter 1':
        if not game_round: return 'C'
        if memory[-1][ap] == 'C':
            think_memory[11][0] += 1
        else:
            think_memory[11][1] += 1

        E1 = 20
        E2 = E1/2
        H = ((((think_memory[11][0] - E2)*(think_memory[11][0] - E2)))/E2)+((((think_memory[11][1] - E2)*(think_memory[11][1] - E2)))/E2)
        if ((game_round + 1) % E1 == 0 and H < 3.84) or think_memory[11][2] == 1:
            think_memory[11][2] = 1
            return 'D'
        else:
            return memory[-1][ap]
    elif strategy == 'Stoic Mirror':
        if (game_round + 1) < 4:
            return 'C'
        else:
            if think_memory[12][1] > 0:
                if think_memory[12][1] == 5:
                    think_memory[12][0] = 0
                    think_memory[12][1] = 0
                    think_memory[12][2] = 1
                    return 'C'
                think_memory[12][1] += 1
                return 'D'
            elif think_memory[12][2] == 1:
                if memory[-1][ap] == 'D':
                    think_memory[12][2] = 2
                    return 'D'
                else:
                    think_memory[12][2] = 0
                    return 'C'
            elif think_memory[12][2] == 2:
                return 'D'
            else:
                if memory[-1][ap] == 'D':
                    think_memory[12][0] += 1
                ratio = think_memory[12][0] / (game_round + 1)
                if ratio > 0.3:
                    think_memory[12][1] = 1
                    return 'D'
                else:
                    return 'C'
    elif strategy == 'Adaptive Tit For Tat Attar Version':
        if (memory[-1][ap] == 'C') if game_round > 0 else True:
            think_memory[13][1] += 1
        elif rnd.random() < (0.25 * (think_memory[13][1] / (game_round + 1))):
            return 'C'

        expectation = think_memory[13][1] / (game_round + 1)
        reality = (memory[-1][ap] == 'C') if game_round > 0 else True

        A = ((1-think_memory[13][0])*reality)+(think_memory[13][0]*expectation)

        think_memory[13][0] = A

        if A > (1 - expectation):
            return 'C'
        else:
            return 'D'
    elif strategy == 'Entropy Sentinel':

        opp = memory[-1][ap] if game_round > 0 else 'C'

        tm = think_memory[15]

        
        if opp == 'C':
            tm[0] += 1
        else:
            tm[2] += 1  
        tm[1] += 1

        share_ratio = tm[0] / max(1, tm[1])

        
        if opp == 'C':
            tm[2] = 0

        
        if tm[1] < 5:
            if not game_round:
                return 'C'
            return memory[-1][ap]  

        
        if share_ratio > 0.6:
            tm[4] += 0.1
        elif share_ratio < 0.4:
            tm[4] -= 0.1
        tm[4] = max(0, min(1, tm[4]))

        
        

        if tm[4] > 0.7:
            tm[3] = 1
        elif tm[4] < 0.3:
            tm[3] = 2
        else:
            tm[3] = 0

        mode = tm[3]

        
        if mode == 1:
            
            return 'D' if rnd.random() < 0.7 else 'C'

        elif mode == 2:
            
            return 'C' if tm[2] < 2 else 'D'

        else:
            
            return memory[-1][ap] if rnd.random() < 0.8 else rnd.choice(['C','D'])
    elif strategy == 'Learner':
        if game_round > 0:
            if memory[-1][ap] == 'D':
                think_memory[17] += 0.125
            else:
                think_memory[17] -= 0.125
        think_memory[17] = max(min(think_memory[17],1),0)
        if rnd.random() < think_memory[17]:
            return 'D'
        else:
            return 'C'
    elif strategy == 'Paranoid':
        if game_round > 0:
            if memory[-1][ap] == 'D':
                think_memory[18] = 1
        if rnd.random() < 0.25 or think_memory[18] == 1:
            return 'D'
        else:
            return 'C'
    elif strategy == 'Liar Person':
        if (memory[-1][ap] == 'C') if game_round > 0 else True:
            think_memory[19] += 1
        ET = ((game_round + 1) % 5 == 0)
        A = min(((think_memory[19] / (game_round + 1)) * ET) + (1 - (think_memory[19] / (game_round + 1))),1)
        if A > 0.5:
            return 'D'
        else:
            return 'C'
    elif strategy == 'Keyman':
        key_len = len(think_memory[20][0])
        if game_round > 0:
            if memory[-1][ap] == 'C':
                think_memory[20][2] += 1
        if (game_round + 1) <= key_len:
            if game_round > 0:
                think_memory[20][1][(game_round + 1)-1] = (memory[-1][ap] == 'D')
            key = think_memory[20][0][(game_round + 1)-1]
            if key == 1:
                return 'D'
            else:
                return 'C'
        if (game_round + 1) == key_len:
            think_memory[20][1][(game_round + 1)-1] = (memory[-1][ap] == 'D')
        if think_memory[20][1] == think_memory[20][0]:
            return 'C'
        else:
            if (game_round + 1) > (key_len+1):
                ET = ((game_round + 1) % 5 == 0)
                A = min(((think_memory[20][2] / (game_round + 1)) * ET) + (1 - (think_memory[20][2] / (game_round + 1))),1)
                if A > 0.5:
                    return 'D'
                else:
                    return 'C'
            else:
                return 'C'
    elif strategy == 'Game Theory Explorer / GT-E':
        if (memory[-1][ap] == 'C') if game_round > 0 else True:
            if (memory[-1][1-ap] == 'C') if game_round > 0 else True:
                C = 0
                S = 2
                learn = 2 * (RPST['R'] / 3)
            else:
                C = 1
                S = 3
                learn = 4 * (RPST['T'] / 5)
        else:
            if memory[-1][1-ap] == 'C':
                C = 2
                S = 0
                learn = -3 * abs(RPST['S'] - RPST['P'])
            else:
                C = 3
                S = 1
                learn = RPST['P']
        
        if (memory[-1][ap] == 'C')  if game_round > 0 else True:
            think_memory[21][4] += 1
        p_c = think_memory[21][4] / (game_round + 1)
        
        CHO = (memory[-1][1-ap] == 'D') if game_round > 0 else False
        
        think_memory[21][C][CHO] = max(min(
            think_memory[21][C][CHO] + learn,
            30
        ), 0)

        think_memory[21][S][CHO] = max(min(
            think_memory[21][S][CHO] + learn,
            30
        ), 0)

        
        opposite = 1 - CHO
        think_memory[21][C][opposite] = max(min(
            think_memory[21][C][opposite] - (learn * 0.3),
            30
        ), 0)

        think_memory[21][S][opposite] = max(min(
            think_memory[21][S][opposite] - (learn * 0.3),
            30
        ), 0)
        
        if rnd.random() < (p_c * 0.5):
            return 'C'
        
        A = [think_memory[21][C][0],think_memory[21][C][1]]
        total = (A[0]) + (A[1]) + 1e-9
        p_share = (A[0]) / total
        if abs((A[0]) - (A[1])) == 0:
            return rnd.choice(['C','D'])
        elif rnd.random() <= p_share:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Hungry':
        if (game_round + 1) % 25 == 0 and rnd.random() > 0.1:
            return 'D'
        else:
            if not game_round:
                return 'C'
            return memory[-1][ap]
    elif strategy == 'Uranium':
        if game_round > 0:
            if memory[-1][1-ap] == 'D' and rnd.random() > 0.05:
                return 'D'
            if (game_round + 1) > 2:
                if memory[-1][ap] == 'D' or memory[-1-1][ap] == 'D':
                    return 'D'
        return 'C'
    elif strategy == 'Markov':
        if (memory[-1][ap] == 'C') if game_round > 0 else True:
            if (memory[-1][1-ap] == 'C') if game_round > 0 else True:
                think_memory[22][0] += 1
            else:
                think_memory[22][1] += 1
        else:
            if memory[-1][1-ap] == 'C':
                think_memory[22][2] += 1
            else:
                think_memory[22][3] += 1
        p_cc = think_memory[22][0] / (game_round + 1)
        p_cd = think_memory[22][1] / (game_round + 1)
        p_dc = think_memory[22][2] / (game_round + 1)
        p_dd = think_memory[22][3] / (game_round + 1)
        if (memory[-1][1-ap] == 'C') if game_round > 0 else True:
            total = p_cc + p_dc
            A = p_cc / total
            if rnd.random() <= A:
                return 'C'
            else:
                return 'D'
        else:
            total = p_dd + p_cd
            A = p_dd / total
            if rnd.random() <= A:
                return 'D'
            else:
                return 'C'
    elif strategy == 'Game Theory Analyzer / GT-A':
        if (memory[-1-1][1-ap] == 'C') if (game_round + 1) > 2 else True:
            if (memory[-1-1][ap] == 'C') if (game_round + 1) > 2 else True:
                if (memory[-1][ap] == 'C') if game_round > 0 else True:
                    think_memory[23][0][0] += 1
                else:
                    think_memory[23][1][0] += 1
            else:
                if memory[-1][ap] == 'C':
                    think_memory[23][0][1] += 1
                else:
                    think_memory[23][1][1] += 1
        else:
            if memory[-1-1][ap] == 'C':
                if memory[-1][ap] == 'C':
                    think_memory[23][0][2] += 1
                else:
                    think_memory[23][1][2] += 1
            else:
                if memory[-1][ap] == 'C':
                    think_memory[23][0][3] += 1
                else:
                    think_memory[23][1][3] += 1

        p_ccc = (think_memory[23][0][0]) / (game_round) if game_round > 0 else 1
        p_ccd = (think_memory[23][1][0]) / (game_round) if game_round > 0 else 0.5
        p_cdc = (think_memory[23][0][1]) / (game_round) if game_round > 0 else 0.5
        p_cdd = (think_memory[23][1][1]) / (game_round) if game_round > 0 else 0.5
        p_dcc = (think_memory[23][0][2]) / (game_round) if game_round > 0 else 0.5
        p_dcd = (think_memory[23][1][2]) / (game_round) if game_round > 0 else 0.5
        p_ddc = (think_memory[23][0][3]) / (game_round) if game_round > 0 else 0.5
        p_ddd = (think_memory[23][1][3]) / (game_round) if game_round > 0 else 0.5

        GTA_cond = (memory[-1][1-ap] + memory[-1][ap]) if game_round > 0 else 'CC'
        GTA_table = {
            'CC':[p_ccc, p_ccd],
            'CD':[p_cdc, p_cdd],
            'DC':[p_dcc, p_dcd],
            'DD':[p_ddc, p_ddd],
        }

        c_factor = GTA_table[GTA_cond][0]
        d_factor = GTA_table[GTA_cond][1]
        if c_factor > d_factor:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Josuah':
        if not game_round:
            return 'C'
        
        if memory[-1][ap] == 'C':
            think_memory[24][3] += 1
        p_c = think_memory[24][3] / (game_round)
        if rnd.random() < (think_memory[24][0] * p_c):
            think_memory[24][1] = 1
            think_memory[24][2] = 1
        if think_memory[24][1] == 1:
            if memory[-1][ap] == 'D':
                think_memory[24][0] *= 0.9
                think_memory[24][1] = 0
                return 'C'
            else:
                if think_memory[24][2] == 1:
                    think_memory[24][2] = 0
                    return 'D'
                else:
                    think_memory[24][2] = 1
                    return 'C'
        else:
            return memory[-1][ap]
    elif strategy == 'Smart Generous Tit For Tat':
        if not game_round:
            return 'C'
        
        if memory[-1][ap] == 'C':
            think_memory[25] += 1
        p_c = (think_memory[25] / (game_round))

        if rnd.random() < (0.2 * p_c):
            return 'C'
        else:
            return memory[-1][ap]
    elif strategy == 'Tiny Brain':
        if not game_round:
            return 'C'
        
        if memory[-1][ap] == 'C':
            think_memory[27][0] += 1
        elif rnd.random() < (0.25 * (think_memory[27][0] / (game_round + 1))):
            think_memory[27][1] += 1
            return 'C'
        p_c = think_memory[27][0] / (game_round + 1)
        p_d = 1 - p_c
        p_f = think_memory[27][1] / (game_round + 1)
        SH = (memory[-1][ap] == 'C')
        ST = 1 - SH
        LIN = ((p_c * SH) - (p_d * ST)) + p_f
        A = max(min(LIN,1),0)
        if rnd.random() <= A:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Alan':
        if game_round == 0:
            return 'C'
        instruction_string = '0:C0|C1,1:C0|D2,2:C1|D3,3:C2|D4,4:D3|D4'
        def translating(x):
            rule = {}
            semua_mode = x.strip().rstrip(',').split(',')
            for item in semua_mode:
                if not item: continue
                nomor_mode, isi_aksi = item.split(':')
                aksi_0_str, aksi_1_str = isi_aksi.split('|')
                
                aksi_0 = [aksi_0_str[0], aksi_0_str[1:]]
                aksi_1 = [aksi_1_str[0], aksi_1_str[1:]]
                rule[nomor_mode] = [aksi_0, aksi_1]
            return rule
        instruction = translating(instruction_string)
        alan_rule = instruction[think_memory[28]][(memory[-1][ap] == 'D')]
        think_memory[28] = alan_rule[1]
        return alan_rule[0]

    elif strategy == 'Unpredictable Tit For Tat':
        A = rnd.random()
        if A > 0.9:
            return 'C'
        elif A < 0.1:
            return 'D'
        else:
            if not game_round:
                return 'C'

            return memory[-1][ap]
    elif strategy == 'Nash':
        cond = ((2 * (memory[-2][1-ap] == 'D')) + (memory[-2][ap] == 'D')) if (game_round + 1) > 2 else 0
        think_memory[280][cond] *= 0.75
        think_memory[280][cond] += 0.25 * ((memory[-1][ap] == 'C') if game_round > 0 else True)

        reward_cc = RPST['R']
        reward_dc = RPST['T']
        punisment_cd = RPST['S']
        punisment_dd = RPST['P']

        nash_think_future = 0.6

        c_factor = (reward_cc + punisment_cd) * (1 - nash_think_future)
        d_factor = (reward_dc + punisment_dd) * (1 - nash_think_future)

        c_factor += ((think_memory[280][0] * reward_cc) + (think_memory[280][1] * punisment_cd)) * nash_think_future
        d_factor += ((think_memory[280][2] * reward_dc) + (think_memory[280][3] * punisment_dd)) * nash_think_future

        E_sh = c_factor
        E_st = d_factor

        if E_sh > E_st:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Line':
        if (game_round + 1) <= 2:
            if not game_round:
                return 'C'
            return memory[-1][ap]
        else:
            return memory[-1][1-ap]
    elif strategy == 'Smart Cat':
        D = 1 - (1 / (game_round + 1))
        think_memory[30] *= D
        if game_round > 0:
            if memory[-1][ap] == 'D':
                think_memory[30] += 1
            p_d = think_memory[30] / (game_round + 1)
            if rnd.random() <= (p_d * 0.6):
                return 'D'
        if not game_round:
            return 'C'
        return memory[-1][ap]
    elif strategy == 'Cerebrum':
        brain = think_memory[31][0] if isinstance(think_memory[31], list) else think_memory[31]
        brain["day_tick"] = (brain["day_tick"] + 1) % 100

        n = brain["n"]
        adj = brain["adj"]       
        volt = brain["volt"]     
        th = brain["th"]         
        ref = brain["ref"]       

        A_plus = 0.015
        A_minus = 0.018
        tau_plus = 20.0
        tau_minus = 20.0
        current_time = (game_round + 1)
        last_spike = brain["last_spike_time"]

        
        brain["CORT"] *= 0.97  
        brain["NOR"]  *= 0.20  
        brain["ADR"]  *= 0.10  
        brain["DA"]   *= 0.50  
        brain["SER"]  *= 0.80  
        brain["END"]  *= 0.70  
        brain["OXY"]  *= 0.85  
        brain["TES"]  *= 0.95  
        brain["VAS"]  *= 0.90  
        brain["GAB"]  *= 0.05  
        brain["ACH"]  *= 0.40  
        brain["GLU"]  *= 0.01  
        brain["MEL"]  *= 0.90  

        
        
        

        opp = memory[-1][ap] if game_round > 0 else 'C'
        self_prev = memory[-1][1 - ap] if game_round > 0 else 'C'

        opp_share = 1.0 if opp == 'C' else 0.0
        opp_steal = 1.0 - opp_share

        self_share = 1.0 if self_prev == 'C' else 0.0
        self_steal = 1.0 - self_share

        exploiting  = (self_prev == 'D' and opp == 'C')
        punished    = (self_prev == 'D' and opp == 'D')
        exploited   = (self_prev == 'C' and opp == 'D')
        cooperating = (self_prev == 'C' and opp == 'C')

        
        
        
        horm_CORT = brain["CORT"]
        horm_NOR = brain["NOR"] 
        horm_ADR = brain["ADR"] 
        horm_DA = brain["DA"]  
        horm_SER = brain["SER"] 
        horm_END = brain["END"] 
        horm_OXY = brain["OXY"] 
        horm_TES = brain["TES"] 
        horm_VAS = brain["VAS"] 
        horm_GAB = brain["GAB"] 
        horm_ACH = brain["ACH"] 
        horm_ADE = brain["ADE"] 
        horm_MEL = brain["MEL"] 
        horm_GLU = brain["GLU"] 

        
        if brain["is_sleeping"]:
            brain["ADE"] *= 0.70
            brain["MEL"] *= 0.80
            brain["energy"] = min(1000.0, brain["energy"] + 150.0)
            brain["lactate"] *= 0.5
            brain["CORT"] *= 0.80
            brain["NOR"] *= 0.02

            
            if brain["energy"] > 400.0 and horm_ADE < 0.10 and horm_MEL < 0.10 and brain["day_tick"] <= 70:
                brain["is_sleeping"] = False

            if exploited:
                brain["ADR"] += 0.30
                brain["NOR"] += 0.30
                brain["GLU"] += 0.30

                brain["MEL"] = max(0.0, brain["MEL"] - 0.40)
                brain["ADE"] = max(0.0, brain["ADE"] - 0.20)

            
            return 'C'

        cerebrum_eyes = [('C', 'C') for _ in range(10)] + memory
        inp_list = [val for i in range(10) for val in [(cerebrum_eyes[-1 - (9 - i)][ap] == 'C') * (brain["decay factor"][i]), (cerebrum_eyes[-1 - (9 - i)][ap] == 'D') * (brain["decay factor"][i]), (cerebrum_eyes[-1 - (9 - i)][1-ap] == 'C') * (brain["decay factor"][i]), (cerebrum_eyes[-1 - (9 - i)][1-ap] == 'D') * (brain["decay factor"][i])]]

        
        inp = np.array(inp_list, dtype=np.float64)

        
        
        

        
        tot_spk = 0
        
        gf = (0.35 + (horm_ACH * 0.1)) * ((1.0 - (horm_SER * 0.05)) * (1.0 - (horm_MEL * 0.4)) + (1 + (horm_NOR * 0.1)))

        for _ in range(4):

            old_volt = volt.copy()

            
            volt[:40] += inp * gf

            
            ref_active_mask = ref > 0
            ref[ref_active_mask] -= 1

            
            

            
            total_incoming = np.dot(old_volt, adj) * horm_GLU
            noise = np.random.uniform(-0.08 * horm_CORT, 0.08 * horm_CORT, size=n)

            
            non_ref_mask = ~ref_active_mask
            volt[non_ref_mask] += total_incoming[non_ref_mask] + noise[non_ref_mask]
            volt[non_ref_mask] *= 0.86
            volt[non_ref_mask] = np.clip(volt[non_ref_mask], -3.0, 3.0)

            
            firing_mask = (ref == 0) & (volt >= th)

            if np.any(firing_mask):
                volt[firing_mask] = 0.0
                ref[firing_mask] = 2

                tot_spk += np.sum(firing_mask)

                firing_indices = np.where(firing_mask)[0]
                
                dt = current_time - last_spike

                
                
                
                pre_trace = np.zeros(n)

                valid_pre = last_spike != -np.inf

                pre_trace[valid_pre] = np.exp(
                    -dt[valid_pre] / tau_plus
                )

                
                
                
                post_trace = np.zeros(n)

                valid_post = last_spike != -np.inf

                post_trace[valid_post] = np.exp(
                    -dt[valid_post] / tau_minus
                )

                
                
                
                if len(firing_indices) > 0:

                    
                    adj[:, firing_indices] += (
                        A_plus
                        * pre_trace[:, None]
                        * horm_DA
                    )

                    
                    adj[firing_indices, :] -= (
                        A_minus
                        * post_trace[None, :]
                        * horm_DA
                    )

                    adj = np.clip(adj, -1.0, 1.0)

                
                
                
                last_spike[firing_indices] = current_time    

            
            th_delta = np.where(volt > th, 0.02, -0.08)
            th = np.clip(th + th_delta, 0.6, 2.5)

        
        brain["energy"] -= (tot_spk * 0.04)
        brain["lactate"] = min(2.0, brain["lactate"] + (tot_spk * 0.0015))

        brain["GLU"] = min(1.0, brain["GLU"] + (tot_spk * 0.01))

        
        brain["MEL"] = min(1.0, brain["MEL"] + 0.15) if brain["day_tick"] > 70 else max(0.0, brain["MEL"] - 0.1)
        brain["ADE"] = min(1.0, brain["ADE"] + 0.01)

        
        if brain["energy"] <= 0 or horm_ADE > 0.95:
            brain["is_sleeping"] = True
        
        
        brain["volt"] = volt
        brain["th"] = th
        brain["ref"] = ref

        
        
        

        half_n = n // 2
        share_signal = np.sum(np.maximum(volt[:half_n], 0))
        steal_signal = np.sum(np.maximum(volt[half_n:], 0))

        
        
        

        
        if exploiting:
            brain["DA"] += 0.05
            brain["CORT"] -= 0.03
            brain["energy"] += 50.0
        elif exploited:
            brain["DA"] -= 0.06
            brain["CORT"] += 0.08
            brain["energy"] -= 40.0
        elif punished:
            if horm_CORT > 0.6:
                brain["DA"] -= 0.01
                brain["CORT"] += 0.05
            else:
                brain["DA"] += 0.015
                brain["CORT"] += 0.02
            brain["energy"] -= 10.0
        elif cooperating:
            brain["DA"] += 0.02
            brain["CORT"] -= 0.03
            brain["energy"] += 30.0

        
        
        
        

        if exploiting:
            
            brain["TES"] += 0.05  
            brain["OXY"] -= 0.04  
            brain["SER"] -= 0.02  

        elif exploited:
            
            brain["ADR"] += 0.06  
            brain["OXY"] -= 0.06  

        elif cooperating:
            
            brain["OXY"] += 0.07  
            brain["SER"] += 0.05  

        elif punished:
            
            brain["NOR"] += 0.06  
            brain["ADR"] += 0.04  
            brain["SER"] -= 0.04  


        
        lead = score[1-ap] - score[ap]
        delta_lead = (lead - brain["lead"])
        if brain["lead"] != lead:
            brain["GLU"] += 0.10 * abs(delta_lead)

        if delta_lead > 0:
            brain["DA"] += 0.12 * delta_lead
        else:
            brain["CORT"] += 0.10 * abs(delta_lead)
            brain["ADR"] += 0.12 * abs(delta_lead)
        brain["lead"] = lead

        
        
        
        energy_ratio = max(0.0, min(1.0, ((brain["energy"] - 450.0) / 100.0)))
        energy_deficit = 1.0 - energy_ratio
        metabolic_stress = energy_deficit * energy_deficit
        brain["CORT"] += metabolic_stress * 0.08
        brain["ADR"] += metabolic_stress * 0.10
        brain["GLU"] += metabolic_stress * 0.05
        brain["DA"] -= metabolic_stress * 0.05

        
        brain["GLU"] += ((horm_NOR * 0.04) + (horm_ADR * 0.03) + (horm_ACH * 0.02)) - ((horm_GAB * 0.05) + (horm_MEL * 0.04))
        brain["GAB"] += ((horm_SER * 0.04) + (horm_END * 0.03) + (horm_MEL * 0.05)) - ((horm_GLU * 0.04) + (horm_NOR * 0.03) + (horm_ADR * 0.02))
        brain["CORT"] += ((horm_ADR * 0.04) + (horm_NOR * 0.03) + (horm_GLU * 0.02)) - ((horm_OXY * 0.04) + (horm_SER * 0.03) + (horm_MEL * 0.03))
        brain["ADR"] += ((horm_NOR * 0.04) + (horm_CORT * 0.02) + (horm_GLU * 0.02)) - ((horm_GAB * 0.05) + (horm_MEL * 0.04))
        brain["NOR"] += ((horm_ADR * 0.03) + (horm_GLU * 0.02)) - ((horm_OXY * 0.03) + (horm_GAB * 0.04) + (horm_MEL * 0.05))
        brain["DA"] += ((horm_END * 0.03) + (horm_ACH * 0.03) + (horm_GLU * 0.02)) - ((horm_CORT * 0.04) + (horm_GAB * 0.02) + (horm_ADE * 0.03))
        brain["SER"] += ((horm_GAB * 0.03) + (horm_OXY * 0.03) + (horm_END * 0.02)) - ((horm_CORT * 0.04) + (horm_ADR * 0.02))
        brain["ACH"] += ((horm_DA * 0.03) + (horm_GLU * 0.03) + (horm_NOR * 0.02)) - ((horm_GAB * 0.03) + (horm_MEL * 0.04))
        brain["END"] += ((horm_ADR * 0.03) + (horm_DA * 0.02)) - ((horm_CORT * 0.03) + (horm_GAB * 0.01))
        brain["OXY"] += ((horm_SER * 0.04) + (horm_DA * 0.02)) - ((horm_CORT * 0.05) + (horm_TES * 0.03) + (horm_VAS * 0.02))
        brain["TES"] += ((horm_DA * 0.03) + (horm_ADR * 0.02) + (horm_VAS * 0.02)) - ((horm_OXY * 0.04) + (horm_SER * 0.02))
        brain["VAS"] += ((horm_CORT * 0.04) + (horm_TES * 0.03) + (horm_NOR * 0.02)) - ((horm_OXY * 0.03) + (horm_SER * 0.02))
        brain["ADE"] += ((horm_GLU * 0.04) + (horm_ADR * 0.02) + (horm_NOR * 0.02)) - ((horm_MEL * 0.15))
        brain["MEL"] += ((horm_ADE * 0.05) + (horm_SER * 0.02)) - ((horm_GLU * 0.04) + (horm_NOR * 0.04))

        
        
        
        if cooperating:
            brain["ACH"] = min(1.0, brain["ACH"] + 0.05)
        elif exploited:
            brain["ACH"] = max(0.1, brain["ACH"] - 0.12)  
            brain["GLU"] = min(1.0, brain["GLU"] + 0.20)

        
        if cooperating and brain["energy"] > 1000.0:
            brain["GAB"] = min(1.0, brain["GAB"] + 0.03)
            brain["energy"] = max(1000.0, brain["energy"] - 15.0)
        

        
        for key in ["DA", "CORT", "SER", "OXY", "ADR", "NOR", "TES", "GAB", "END", "ACH", "VAS", "GLU", "ADE", "MEL"]:
            brain[key] = float(np.clip(brain[key], 0.05 if key in ["DA", "CORT"] else 0.0, 1.0))

        
        stress = horm_CORT
        reward = horm_DA
        calm = horm_SER + horm_OXY

        if stress > 0.75:
            personality_mode = "panic"
        elif reward > 0.75:
            personality_mode = "hungry"
        elif calm > 0.7:
            personality_mode = "stable"
        else:
            personality_mode = "neutral"

        
        
        

        if horm_DA > 0.82 and n < brain["max_n"]:
            if rnd.random() < 0.40:
                new_n = n + 1
                
                
                new_adj = np.pad(adj, ((0, 1), (0, 1)), mode='constant')
                new_adj[:-1, -1] = np.random.uniform(-0.15, 0.15, n)
                new_adj[-1, :] = np.random.uniform(-0.15, 0.15, new_n)
                brain["adj"] = new_adj

                
                brain["volt"] = np.append(volt, 0.0)
                brain["th"] = np.append(th, rnd.uniform(0.9, 1.1))
                brain["ref"] = np.append(ref, 0)

                brain["n"] = new_n

                brain["last_spike_time"] = np.append(
                    brain["last_spike_time"],
                    -np.inf
                )

        
        
        

        elif (horm_CORT > 0.78 or horm_DA < 0.15) and n > brain["min_n"]:
            if rnd.random() < 0.25:
                
                conn_strength = np.sum(np.abs(adj[40:, :]), axis=1) + np.sum(np.abs(adj[:, 40:]), axis=0)
                activity = np.abs(volt[40:])
                weakness_scores = conn_strength + activity

                weakest_idx = np.argmin(weakness_scores) + 4

                brain["last_spike_time"] = np.delete(
                brain["last_spike_time"],
                weakest_idx
                )

                
                brain["adj"] = np.delete(np.delete(adj, weakest_idx, axis=0), weakest_idx, axis=1)
                brain["volt"] = np.delete(volt, weakest_idx)
                brain["th"] = np.delete(th, weakest_idx)
                brain["ref"] = np.delete(ref, weakest_idx)

                brain["n"] -= 1

        
        
        

        mutual = (self_prev == opp)
        if exploited:
            brain["a"] += 0.01
        elif exploiting:
            brain["a"] -= 0.008
        elif mutual:
            brain["a"] -= 0.002

        brain["a"] = float(np.clip(brain["a"], 0.0, 1.0))
        a = brain["a"]

        social_bias = 1.5 * (horm_SER + horm_OXY + horm_END + horm_GAB + horm_VAS + horm_ACH)
        aggressive_bias = 1.5 * (horm_ADR + horm_NOR + horm_TES + horm_CORT + horm_DA)

        social_bias *= (1.0 - horm_ACH * 0.3)
        aggressive_bias *= (1.0 + horm_ACH * 0.2)

        p_share = share_signal * (((1.0 - a) + (a * opp_share)) * social_bias)
        p_steal = steal_signal * ((a * opp_steal) * aggressive_bias)

        if personality_mode == "panic":
            p_steal *= 1.4; p_share *= 0.8
        elif personality_mode == "hungry":
            p_steal *= 1.3; p_share *= 1.1
        elif personality_mode == "stable":
            p_share *= 1.3; p_steal *= 0.9

        p_share = float(np.clip(p_share, 0.0, 1.0))
        p_steal = float(np.clip(p_steal, 0.0, 1.0))

        brain["a"] = float(np.clip(brain["a"] + 0.005 * (horm_CORT - horm_DA), 0.0, 1.0))

        if p_share >= p_steal:
            return 'C'
        else:
            return 'D'
    elif strategy == 'SURPRISE ATTACK!':
        attack_p = 1 - max(min(((tournament_avg_last_round / 2) / (game_round + 1)),1),0)
        if (rnd.random() < attack_p) or (think_memory[32] == 1):
            if think_memory[32] == 0:
                think_memory[32] = 1
            return 'D'
        else:
            if not game_round:
                return 'C'
            return memory[-1][ap]
    elif strategy == 'Sumobox Fighter':
        brain = think_memory[34][0]
        another_variable = think_memory[34][1]
        another_variable[0] -= 1

        if game_round > 0:
            if memory[-1][ap] == 'D':
                another_variable[2] += 1

        A = 5
        sumo_eyes = [('C', 'C') for _ in range(A)] + memory

        if len(brain) != 0:
            if sumo_eyes[-1][ap] == 'C':
                if sumo_eyes[-1][1-ap] == 'C':
                    brain[another_variable[1]][1] += RPST['R']
                else:
                    brain[another_variable[1]][1] += RPST['T']
            else:
                if sumo_eyes[-1][1-ap] == 'C':
                    brain[another_variable[1]][1] -= 4 * abs(RPST['S'] - RPST['P'])
                else:
                    brain[another_variable[1]][1] += RPST['P']

            if brain[another_variable[1]][1] <= 0 and another_variable[0] <= 0:
                new_output = "".join([rnd.choice(["C", "D"]) for _ in range(A)])
                brain[another_variable[1]] = [new_output,100]
            if brain[another_variable[1]][1] >= 200 and another_variable[0] <= 0:
                if rnd.random() <= (another_variable[2] / (game_round + 1)):
                    new_choice = "D"
                else:
                    new_choice = "C"
                idx = rnd.randint(0,A-1)
                new_output = brain[another_variable[1]][0][:idx] + new_choice + brain[another_variable[1]][0][idx+1:]
                brain[another_variable[1]] = [new_output,100]

        if another_variable[0] <= 0:

            inp = tuple(["".join([sumo_eyes[-1 - ((A-1) - i)][ap] for i in range(A)]),"".join([sumo_eyes[-1 - ((A-1) - j)][1-ap] for j in range(A)])])

            another_variable[0] = A
            another_variable[1] = inp

            def similarity(new_key,old_key):
                new_key_vector_opp = [1 if new_key[0][S1] == "C" else -1 for S1 in range(A)]
                old_key_vector_opp = [1 if old_key[0][S2] == "C" else -1 for S2 in range(A)]
                new_key_vector_self = [1 if new_key[1][S1] == "C" else -1 for S1 in range(A)]
                old_key_vector_self = [1 if old_key[1][S2] == "C" else -1 for S2 in range(A)]
                similarity_val = 0
                for SRT in range(A):
                    similarity_val += ((new_key_vector_opp[SRT] * old_key_vector_opp[SRT]) + (new_key_vector_self[SRT] * old_key_vector_self[SRT]))
                return ((similarity_val + A) / 2) / (A * 2)
        
            if brain.get(inp) == None:
                if len(brain) != 0:
                    target_space = min(brain, key=lambda k: brain[k][1])

                    is_same = similarity(inp,target_space)

                    new_output = "".join([rnd.choice(["C", "D"]) for _ in range(A)])

                    if is_same >= rnd.random():
                        brain[target_space] = [new_output,100]
                        another_variable[1] = target_space
                    else:
                        brain[inp] = [new_output,100]
                        another_variable[1] = inp
                else:
                    new_output = "".join([rnd.choice(["C", "D"]) for _ in range(A)])
                    
                    brain[inp] = [new_output,100]
                    another_variable[1] = inp

        idx = A - another_variable[0]
        idx = max(0, min(idx, (A-1)))

        if brain[another_variable[1]][0][idx] == "C":
            return 'C'
        else:
            return 'D'
    elif strategy == 'Smart Snake':
        brain = think_memory[36]

        smart_snake_eyes = [('C', 'C') for _ in range(5)] + memory
        
        if game_round > 0:
            dis = [1,0.8,0.6,0.4,0.2]
            if smart_snake_eyes[-1][ap] == 'C':
                if smart_snake_eyes[-1][1-ap] == 'C':
                    for learn in range(5):
                        brain[0][learn] = max(min((brain[0][learn] + (0.333 * (RPST['R'] / 3) * dis[learn])),1),-1)
                else:
                    for learn in range(5):
                        brain[1][learn] = max(min((brain[1][learn] + (0.5 * (RPST['T'] / 5) * dis[learn])),1),-1)
            else:
                if smart_snake_eyes[-1][1-ap] == 'C':
                    for learn in range(5):
                        brain[0][learn] = max(min((brain[0][learn] - (0.4 * abs(RPST['S'] - RPST['P']) * dis[learn])),1),-1)
                else:
                    for learn in range(5):
                        brain[1][learn] = max(min((brain[1][learn] + (0.1 * (RPST['P']) * dis[learn])),1),-1)

        C_input = [1 if smart_snake_eyes[-1 - (4 - i)][ap] == 'C' else 0 for i in range(5)]
        D_input = [1 if smart_snake_eyes[-1 - (4 - i)][ap] == 'D' else 0 for i in range(5)]
        c_factor = (
            (C_input[0] * brain[0][0]) +
            ((C_input[1] * brain[0][1]) * 0.8) +
            ((C_input[2] * brain[0][2]) * 0.6) +
            ((C_input[3] * brain[0][3]) * 0.4) +
            ((C_input[4] * brain[0][4]) * 0.2)
        )

        d_factor = (
            (D_input[0] * brain[1][0]) +
            ((D_input[1] * brain[1][1]) * 0.8) +
            ((D_input[2] * brain[1][2]) * 0.6) +
            ((D_input[3] * brain[1][3]) * 0.4) +
            ((D_input[4] * brain[1][4]) * 0.2)
        )
        if c_factor == d_factor:
            return rnd.choice(['C','D'])
        elif c_factor > d_factor:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Theseus':
        another_factor = think_memory[38][1]
        brain = think_memory[38][0]

        theseus_eyes = [('C', 'C') for _ in range(5)] + memory

        another_factor[1] *= 0.9
        if theseus_eyes[-1][ap] == 'D':
            another_factor[1] += 0.1

        if game_round > 0:
            if theseus_eyes[-1][ap] == 'C':
                if theseus_eyes[-1][1-ap] == 'C':
                    brain[another_factor[0]] = max(min(brain[another_factor[0]] + (0.333 * (RPST['R'] / 3) * (1 - another_factor[1])),1),0)
                else:
                    brain[another_factor[0]] = max(min(brain[another_factor[0]] - (0.5 * (RPST['T'] / 5) * another_factor[1]),1),0)
            else:
                if theseus_eyes[-1][1-ap] == 'C':
                    brain[another_factor[0]] = max(min(brain[another_factor[0]] - (0.4 * abs(RPST['S'] - RPST['P']) * another_factor[1]),1),0)
                else:
                    brain[another_factor[0]] = max(min(brain[another_factor[0]] - (0.1 * RPST['P'] * another_factor[1]),1),0)

        inp = tuple(["".join([theseus_eyes[-1 - ((4) - i)][ap] for i in range(5)]),"".join([theseus_eyes[-1 - ((4) - j)][1-ap] for j in range(5)])])
        SH = (theseus_eyes[-1][ap] == 'C')

        if brain.get(inp) == None:
            brain[inp] = ((1 - another_factor[1]) * 0.9) + (SH * 0.1)
        another_factor[0] = inp
        
        if rnd.random() <= brain[inp]:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Social Average':

        tm = think_memory[39]

        if not game_round:
            return 'C'

        
        
        
        
        
        
        

        opp = memory[-1][ap]
        self_prev = memory[-1][1-ap]

        
        
        

        if opp == 'C':
            tm[0] += 1
            tm[2] += 0.08
            tm[3] -= 0.05
        else:
            tm[1] += 1
            tm[2] -= 0.12
            tm[3] += 0.10

        
        tm[2] = max(0, min(1, tm[2]))
        tm[3] = max(0, min(1, tm[3]))

        
        
        

        total = tm[0] + tm[1] + 1e-9

        p_coop = tm[0] / total
        p_def = tm[1] / total

        
        
        

        
        if p_coop > 0.7:

            
            if rnd.random() < 0.9:
                return 'C'
            else:
                return 'D'

        
        elif p_def > 0.65:

            
            if opp == 'D':
                return 'D'
            else:
                if rnd.random() < 0.2:
                    return 'D'
                return 'C'

        
        
        

        
        decision = opp

        
        if opp == self_prev:
            if rnd.random() < 0.75:
                decision = 'C'

        
        if tm[3] > 0.6:
            if rnd.random() < tm[3]:
                decision = 'D'

        
        if tm[2] > 0.75:
            if rnd.random() < tm[2]:
                decision = 'C'

        
        if rnd.random() < 0.08:
            decision = rnd.choice(['C','D'])

        
        
        

        if (
            opp == 'C'
            and memory[-1-1][ap] == 'C'
            and rnd.random() < 0.15
        ):
            tm[4] += 1
            return 'D'

        
        
        

        if tm[5] > 0:
            tm[5] -= 1
            return 'C'

        if (
            opp == 'D'
            and self_prev == 'C'
            and rnd.random() < 0.12
        ):
            tm[5] = 2
            return 'C'

        return decision
    elif strategy == 'Capitalist':

        tm = think_memory[40]

        if not game_round:
            return 'C'

        
        
        

        opp = memory[-1][ap]

        
        
        

        if opp == 'C':
            tm[1] += 0.05
        else:
            tm[1] -= 0.12

        tm[1] = max(0,min(1,tm[1]))

        
        tm[0] += 0.002
        tm[0] = min(tm[0],1)

        
        
        

        exploit_chance = (
            (tm[0] * 0.6) +
            ((1 - tm[1]) * 0.4)
        ) * 0.125

        
        if opp == 'C':

            if rnd.random() < exploit_chance:

                tm[2] += 1
                return 'D'

            else:
                return 'C'

        
        
        

        else:

            
            if rnd.random() < (tm[1] * 0.5):
                return 'C'

            return 'D'
    elif strategy == 'Communalist':

        tm = think_memory[41]

        if not game_round:
            return 'C'

        
        
        

        opp = memory[-1][ap]

        
        
        

        if opp == 'C':

            tm[0] += 1
            tm[2] += 0.04

        else:

            tm[1] += 1
            tm[2] -= 0.08

        tm[2] = max(0,min(1,tm[2]))

        total = tm[0] + tm[1] + 1e-9

        social_stability = tm[0] / total

        
        
        

        if social_stability < 0.45:

            
            if opp == 'D':
                return 'D'

        
        
        

        if rnd.random() < tm[2]:

            return 'C'

        
        if opp == 'D':
            return 'D'

        return 'C'
    elif strategy == 'NoName':
        opp_prev = memory[-1-1][ap] if (game_round + 1) > 2 else 'C'
        opp = memory[-1][ap] if game_round > 0 else 'C'
        self = memory[-1][1-ap] if game_round > 0 else 'C'

        if opp == 'C':
            if self == 'C':
                return 'C'
            else:
                return rnd.choice(['C','D'])
        else:
            if self == 'C':
                return 'D'
            else:
                if opp_prev == 'D':
                    return 'D'
                else:
                    return rnd.choice(['C','D'])
    elif strategy == 'Spectrum Attacker':
        A = max(min((((game_round + 1) - (tournament_avg_last_round / 2)) / (tournament_avg_last_round / 2)),1),0)

        if rnd.random() <= A:
            return 'D'
        else:
            if not game_round:
                return 'C'
            return memory[-1][ap]
    elif strategy == 'Smooth Grudger':
        opp = memory[-1][ap] if game_round > 0 else 'C'

        think_memory[42] *= 0.9
        if opp == 'D':
            think_memory[42] += 0.1

        if rnd.random() < think_memory[42]:
            return 'D'
        else:
            return opp
    elif strategy == 'Forest-RND':
        brain = think_memory[43][0]
        trees_input = [think_memory[43][1],think_memory[43][2]]
        old_output = think_memory[43][3]
        old_inp = think_memory[43][4]
        rating = think_memory[43][5]
        forest_eyes = [('C', 'C') for _ in range(9)] + memory

        share_signal = 0
        steal_signal = 0

        if game_round > 0:
            R_or_P = None
            if forest_eyes[-1][ap] == 'C':
                if forest_eyes[-1][1-ap] == 'C':
                    R_or_P = [1,3,1,-1]
                else:
                    R_or_P = [4,1,-1,1]
            else:
                if forest_eyes[-1][1-ap] == 'C':
                    R_or_P = [8,1,-2,1]
                else:
                    R_or_P = [4,1,-1,0]
            
            for n in range(len(brain)):
                if old_output[n]:
                    trees_input[0][n] = (trees_input[0][n] + rnd.randint(-R_or_P[0], R_or_P[0])) % 9
                    trees_input[1][n] = (trees_input[1][n] + rnd.randint(-R_or_P[0], R_or_P[0])) % 9
                    rating[n] += R_or_P[2]
                else:
                    trees_input[0][n] = (trees_input[0][n] + rnd.randint(-R_or_P[1], R_or_P[1])) % 9
                    trees_input[1][n] = (trees_input[1][n] + rnd.randint(-R_or_P[1], R_or_P[1])) % 9
                    rating[n] += R_or_P[3]
                if rating[n] <= -5:
                    if brain[n][old_inp[n]] == "C":
                        brain[n][old_inp[n]] = "D"
                    else:
                        brain[n][old_inp[n]] = "C"
                    rating[n] = 0
                elif rating[n] >= 5:
                    trees_input[0][n] = (trees_input[0][n] + rnd.randint(-1, 1)) % 9
                    trees_input[1][n] = (trees_input[1][n] + rnd.randint(-1, 1)) % 9
                    rating[n] = 0
                rating[n] = max(min(rating[n],5),-5)

        for n in range(len(brain)):
            inp1 = forest_eyes[-1 - trees_input[0][n]][ap]
            inp2 = forest_eyes[-1 - trees_input[1][n]][1-ap]
            inp = (inp1, inp2)
            if brain[n].get(inp) == None:
                brain[n][inp] = rnd.choice(["C","D"])
            if brain[n][inp] == "C":
                share_signal += 1
                old_output[n] = True
            else:
                steal_signal += 1
                old_output[n] = False
            old_inp[n] = inp
        
        if share_signal == steal_signal:
            return forest_eyes[-1][ap]
        elif share_signal > steal_signal:
            return 'C'
        else:
            return 'D'
    elif strategy == '3-Tree Of Prediction':
        primary_tree = think_memory[44][0]
        secondary_tree = think_memory[44][1]

        ttop_eyes = [('C', 'C')] + memory

        if game_round > 0:
            R_or_P = None
            if ttop_eyes[-1][ap] == 'C':
                R_or_P = 'reward'
            else:
                R_or_P = 'punisment'
            
            self = ttop_eyes[-1][1-ap]
            rev_self = "C" if ttop_eyes[-1][1-ap] == 'D' else "D"
            if R_or_P == 'reward':
                if primary_tree[think_memory[44][2][0]] != self:
                    primary_tree[think_memory[44][2][0]] = rev_self
            else:
                if primary_tree[think_memory[44][2][0]] == self:
                    primary_tree[think_memory[44][2][0]] = rev_self
            
            for n in range(2):
                if R_or_P == 'reward':
                    if secondary_tree[n][think_memory[44][2][n+1]] != self:
                        secondary_tree[n][think_memory[44][2][n+1]] = rev_self
                else:
                    if secondary_tree[n][think_memory[44][2][n+1]] == self:
                        secondary_tree[n][think_memory[44][2][n+1]] = rev_self

        F_inp1 = ttop_eyes[-1][ap]
        F_inp2 = ttop_eyes[-1][1-ap]
        F_inp = (F_inp1, F_inp2)

        if primary_tree.get(F_inp) == None:
            primary_tree[F_inp] = rnd.choice(["C","D"])
        first_choice = primary_tree[F_inp]
        think_memory[44][2][0] = F_inp

        share_signal = 0
        steal_signal = 0

        S_inp = [("C", first_choice), ("D", first_choice)]

        for n in range(2):
            if secondary_tree[n].get(S_inp[n]) == None:
                secondary_tree[n][S_inp[n]] = rnd.choice(["C","D"])
            if secondary_tree[n][S_inp[n]] == "C":
                share_signal += 1
            else:
                steal_signal += 1
            think_memory[44][2][n+1] = S_inp[n]
        if share_signal == steal_signal:
            return ttop_eyes[-1][ap]
        elif share_signal > steal_signal:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Bayes Attar Version':
        tm = think_memory[45]
        self = memory[-1][1-ap] if game_round > 0 else 'C'
        opp = memory[-1][ap] if game_round > 0 else 'C'
        if self == 'C':
            tm[0] += 1
            if opp == 'C':
                tm[1] += 1
        else:
            tm[2] += 1
            if opp == 'C':
                tm[3] += 1
        
        SH = (opp == 'C')

        a = 0.1
        P1 = ((tm[1] / (tm[0] + 1e-9)) * (1 - a)) + (SH * a)
        P2 = ((tm[3] / (tm[2] + 1e-9)) * (1 - a)) + (SH * a)

        reward_cc = RPST['R']
        reward_dc = RPST['T']
        punisment_cd = RPST['S']
        punisment_dd = RPST['P']

        c_factor = (P1 * reward_cc) + ((1-P1) * punisment_cd)
        d_factor = (P2 * reward_dc) + ((1-P1) * punisment_dd)

        if c_factor >= d_factor:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Naive Q-Learner':
        brain = think_memory[46][0]
        another_factor = think_memory[46][1]

        A = min(12, game_round) - 1
        opp_inp = tuple(['C' for _ in range(12 - min(12, game_round))] + [memory[-1 - (A - i)][ap] for i in range(min(12, game_round))])
        self_inp = tuple(['C' for _ in range(12 - min(12, game_round))] + [memory[-1 - (A - i)][1-ap] for i in range(min(12, game_round))])
        inp = (opp_inp, self_inp)

        if brain.get(inp) == None:
            brain[inp] = [0,0]

        if game_round > 0 and another_factor[0][1] != 0.5:
            reward = 0
            if memory[-1][ap] == 'C':
                if memory[-1][1-ap] == 'C':
                    reward = RPST['R']
                else:
                    reward = RPST['T']
            else:
                if memory[-1][1-ap] == 'C':
                    reward = RPST['S']
                else:
                    reward = RPST['P']

            c_factor = brain[inp][0]
            d_factor = brain[inp][1]
            
            a = 0.1
            gamma = 0.5
            Q_old = brain[another_factor[0][0]][another_factor[0][1]]
            Q_new = Q_old + (a * (reward + ((gamma * max(c_factor, d_factor)) - Q_old)))
            brain[another_factor[0][0]][another_factor[0][1]] = Q_new

        if rnd.random() <= another_factor[1]:
            Action = rnd.choice(['C', 'D'])
            another_factor[0] = [inp, (Action == 'D')]
            return Action
        decay = 0.75
        min_epsilon = 0
        another_factor[1] = max(another_factor[1] * decay, min_epsilon)

        if brain[inp][0] == brain[inp][1]:
            Action = rnd.choice(['C', 'D'])
            another_factor[0] = [inp, (Action == 'D')]
            return Action
        best_choice = max(brain[inp])
        if best_choice == brain[inp][0]:
            another_factor[0] = [inp, 0]
            return 'C'
        else:
            another_factor[0] = [inp, 1]
            return 'D'
    elif strategy == 'Stupidiot':
        if (game_round + 1) <= ((50 / 200) * tournament_avg_last_round):
            if game_round > 0:
                if memory[-1][ap] == 'D':
                    return 'D'
                if memory[-1][1-ap] == 'D':
                    if (game_round + 1) > 2:
                        if memory[-1-1][1-ap] == 'D':
                            return 'C'
            return 'D'
            return rnd.choice(['C','D'])
        elif (game_round + 1) <= ((100 / 200) * tournament_avg_last_round):
            return 'D'
        else:
            if rnd.random() <= 0.1:
                return 'D'
            if memory[-1][ap] == 'D':
                think_memory[47] = 1
            if think_memory[47] == 1:
                return 'D'
            else:
                return 'C'
    elif strategy == 'Smart-Clever':
        if (memory[-1][ap] == 'C') if game_round > 0 else True:
            win_or_lose = 'win'
        else:
            win_or_lose = 'lose'
        
        if win_or_lose == 'lose':
            if memory[-1][1-ap] == 'C':
                think_memory[48] = 1 - think_memory[48]
        
        if think_memory[48] == 1:
            if not game_round:
                return 'C'

            return memory[-1][ap]
        else:
            if not game_round:
                return 'C'

            if memory[-1][ap] == memory[-1][1-ap]:
                return 'C'
            else:
                return 'D'
    elif strategy == 'Half Sin Square':
        def half_sin_square(x):
            pi = mth.pi
            half_sin_val = mth.sin((x * pi) / 2)
            return half_sin_val * half_sin_val

        if not game_round:
            return 'C'
        
        if memory[-1][ap] == 'C':
            think_memory[50] += 1

        p_c = think_memory[50] / (game_round)
        c_factor = half_sin_square(p_c)

        if rnd.random() <= c_factor:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Gateman':
        def not_gate(x):
            return (1 - x)
        def and_gate(x,y):
            return min(x,y)
        def or_gate(x,y):
            return max(x,y)
        def random_boolean(x,y):
            if rnd.random() <= x:
                return (y % 2)
            else:
                return 1 - (y % 2)
        def save_to_prog(x):
            nonlocal gate_prog
            gate_prog.append(x)
        def gateman_input(x,y):
            opp_or_self = ap if (y % 2) == 0 else 1-ap
            gateman_memory_index = x % 8
            return (memory[-1-gateman_memory_index][opp_or_self] == 'D') if (game_round) > gateman_memory_index else 0

        gate_prog = []

        save_to_prog(
            or_gate(
                or_gate(
                    and_gate(
                        gateman_input(0,0),
                        random_boolean(0.1, 0)
                    ),
                    and_gate(
                        gateman_input(0,0),
                        gateman_input(1,0)
                    )
                ),
                and_gate(
                    and_gate(
                        gateman_input(1,0),
                        not_gate(gateman_input(1,1))
                    ),
                    random_boolean(0.1, 1)
                )
            )
        )

        gate_output = gate_prog[0]

        if gate_output == 0:
            return 'C'
        else:
            return 'D'
    elif strategy == 'TF2T & 2TFT':
        think_memory[51][0] = rnd.randint(0,1)

        if think_memory[51][0] == 1:
            think_memory[51][1] *= 0.5
            if ((memory[-1][ap] == 'D') if game_round > 0 else False):
                think_memory[51][1] += 0.5
                return 'D'
            else:
                if rnd.random() <= (0.5 * think_memory[51][1]):
                    if ((memory[-1-1][ap] == 'D') if (game_round + 1) > 2 else False):
                        return 'D'
                    else:
                        return 'C'
                else:
                    return 'C'
        else:
            think_memory[51][1] *= 0.5
            if ((memory[-1][ap] == 'D') if game_round > 0 else False):
                think_memory[51][1] += 0.5
                if rnd.random() <= (0.5 * (1 - think_memory[51][1])):
                    if ((memory[-1-1][ap] == 'D') if (game_round + 1) > 2 else False):
                        return 'D'
                    else:
                        return 'C'
                else:
                    return 'D'
            else:
                return 'C'
    elif strategy == 'Psycho Attar Version':
        if not game_round:
            return 'C'
        
        if think_memory[52] == 0:
            if memory[-1][ap] == 'D':
                think_memory[52] = 1

        if think_memory[52] == 1:
            if rnd.random() < (0.25 * max(min((((game_round + 1) - (tournament_avg_last_round / 2)) / (tournament_avg_last_round / 2)), 1), 0)):
                return 'D'
            else:
                return memory[-1][ap]
        else:
            return memory[-1][ap]
    elif strategy == 'Socio':
        if not game_round:
            return 'C'
        
        if think_memory[53][0] == 0:
            if memory[-1][ap] == 'D':
                think_memory[53][0] = 1
                return 'D'
        
        if think_memory[53][0] == 1:
            if memory[-1][ap] == 'D':
                think_memory[53][1] = 1

            if think_memory[53][1] == 1:
                return 'D'
            else:
                return rnd.choice(['C','D'])
        else:
            return 'C'
    elif strategy == 'Isolated Tit For Tat':
        if rnd.random() <= 0.1:
            think_memory[54][1] = 1 - think_memory[54][1]
            think_memory[54][0] = (memory[-1][ap] == 'C') if game_round > 0 else True
        
        if think_memory[54][1] == 1:
            if think_memory[54][0] == 1:
                return 'C'
            else:
                return 'D'
        else:
            if not game_round:
                return 'C'
            return memory[-1][ap]
    elif strategy == 'Isolated Clan':
        if rnd.random() <= 0.1:
            think_memory[55][1] = 1 - think_memory[55][1]
            think_memory[55][0] = max(min(((memory[-1][ap] == 'C') if game_round > 0 else True) + (rnd.uniform(-0.1,0.1)),1),0)
        
        if think_memory[55][1] == 1:
            think_memory[55][0] = max(min(think_memory[55][0] + rnd.uniform(-0.1,0.1),1),0)
            if rnd.random() <= think_memory[55][0]:
                return 'C'
            else:
                return 'D'
        else:
            if not game_round:
                return 'C'
            return memory[-1][ap]
    elif strategy == 'Neil A':
        if ((memory[-1][ap] == 'C') if game_round > 0 else True):
            think_memory[56] *= 0.9
        else:
            think_memory[56] = max(min(think_memory[56] + 1,2),0)
        A = rnd.random()
        if A > think_memory[56]:
            return 'C'
        elif A > max(think_memory[56] - 1, 0):
            return 'D'
        else:
            if memory[-1][ap] == 'D':
                return 'D'
            else:
                return rnd.choice(['C','D'])
    elif strategy == 'Cult Leader':
        leader_key = [0,0,0,1,1]
        others_key = {
                    (0,0,0,1,1):'leader',
                    (0,0,1,0,1):'bishop',
                    (0,1,0,0,1):'citizen',
                    }
        if ((game_round + 1) <= 6) and (game_round > 0):
            if memory[-1][ap] == 'C':
                think_memory[57][(game_round + 1) - 2] = 0
            else:
                think_memory[57][(game_round + 1) - 2] = 1

        if (game_round + 1) < 6:
            if leader_key[(game_round)] == 0:
                return 'C'
            else:
                return 'D'
        else:
            if (game_round + 1) <= 7:
                return 'C'
            others_pattern = others_key.get(tuple(think_memory[57]))
            if others_pattern == None:
                if rnd.random() < 0.2:
                    return 'C'
                else:
                    return memory[-1][ap]
            else:
                if others_pattern == 'leader':
                    return 'C'
                else:
                    return 'D'
    elif strategy == 'Cult Bishop A':
        bishop_key = [0,0,1,0,1]
        others_key = {
                    (0,0,0,1,1):'leader',
                    (0,0,1,0,1):'bishop',
                    (0,1,0,0,1):'citizen',
                    }
        if ((game_round + 1) <= 6) and (game_round > 0):
            if memory[-1][ap] == 'C':
                think_memory[58][(game_round + 1) - 2] = 0
            else:
                think_memory[58][(game_round + 1) - 2] = 1

        if (game_round + 1) < 6:
            if bishop_key[(game_round)] == 0:
                return 'C'
            else:
                return 'D'
        else:
            if (game_round + 1) <= 7:
                return 'C'
            others_pattern = others_key.get(tuple(think_memory[58]))
            if others_pattern == None:
                if memory[-1-1][ap] == 'D':
                    return 'D'
                else:
                    return memory[-1][ap]
            else:
                if others_pattern == 'leader':
                    return 'C'
                else:
                    return 'D'
    elif strategy == 'Cult Bishop B':
        bishop_key = [0,0,1,0,1]
        others_key = {
                    (0,0,0,1,1):'leader',
                    (0,0,1,0,1):'bishop',
                    (0,1,0,0,1):'citizen',
                    }
        if ((game_round + 1) <= 6) and (game_round > 0):
            if memory[-1][ap] == 'C':
                think_memory[59][(game_round + 1) - 2] = 0
            else:
                think_memory[59][(game_round + 1) - 2] = 1

        if (game_round + 1) < 6:
            if bishop_key[(game_round)] == 0:
                return 'C'
            else:
                return 'D'
        else:
            if (game_round + 1) <= 7:
                return 'C'
            others_pattern = others_key.get(tuple(think_memory[59]))
            if others_pattern == None:
                if (memory[-1-1][ap] == 'D') and (memory[-1][ap] == 'D'):
                    return 'D'
                else:
                    return 'C'
            else:
                if others_pattern == 'leader':
                    return 'C'
                else:
                    return 'D'
    elif strategy == 'Cult Citizen A':
        bishop_key = [0,1,0,0,1]
        others_key = {
                    (0,0,0,1,1):'leader',
                    (0,0,1,0,1):'bishop',
                    (0,1,0,0,1):'citizen',
                    }
        if ((game_round + 1) <= 6) and (game_round > 0):
            if memory[-1][ap] == 'C':
                think_memory[60][(game_round + 1) - 2] = 0
            else:
                think_memory[60][(game_round + 1) - 2] = 1

        if (game_round + 1) < 6:
            if bishop_key[(game_round)] == 0:
                return 'C'
            else:
                return 'D'
        else:
            return 'C'
    elif strategy == 'Cult Citizen B':
        bishop_key = [0,1,0,0,1]
        others_key = {
                    (0,0,0,1,1):'leader',
                    (0,0,1,0,1):'bishop',
                    (0,1,0,0,1):'citizen',
                    }
        if ((game_round + 1) <= 6) and (game_round > 0):
            if memory[-1][ap] == 'C':
                think_memory[61][(game_round + 1) - 2] = 0
            else:
                think_memory[61][(game_round + 1) - 2] = 1

        if (game_round + 1) < 6:
            if bishop_key[(game_round)] == 0:
                return 'C'
            else:
                return 'D'
        else:
            if (game_round + 1) <= 7:
                return 'C'
            others_pattern = others_key.get(tuple(think_memory[61]))
            if others_pattern == None:
                return memory[-1][ap]
            else:
                return 'C'
    elif strategy == 'Cult Citizen C':
        bishop_key = [0,1,0,0,1]
        others_key = {
                    (0,0,0,1,1):'leader',
                    (0,0,1,0,1):'bishop',
                    (0,1,0,0,1):'citizen',
                    }
        if ((game_round + 1) <= 6) and (game_round > 0):
            if memory[-1][ap] == 'C':
                think_memory[62][(game_round + 1) - 2] = 0
            else:
                think_memory[62][(game_round + 1) - 2] = 1

        if (game_round + 1) < 6:
            if bishop_key[(game_round)] == 0:
                return 'C'
            else:
                return 'D'
        else:
            if (game_round + 1) <= 7:
                return 'C'
            others_pattern = others_key.get(tuple(think_memory[62]))
            if others_pattern == None:
                if memory[-1][ap] == memory[-1][1-ap]:
                    return 'C'
                else:
                    return 'D'
            else:
                return 'C'
    elif strategy == 'Angry Tit For Tat':
        B = min(5, game_round)
        if game_round > 0:
            inp = tuple([memory[-1 - ((B - 1) - i)][ap] for i in range(B)])
            if (think_memory[63] == 1) or (inp == tuple(["D"] * B)):
                if think_memory[63] == 0:
                    think_memory[63] = 1
                return 'D'
            else:
                return memory[-1][ap]
        return 'C'
    elif strategy == 'Namdeirf / Anti-Grudger':
        if game_round > 0:
            if (think_memory[64] == 1) or (memory[-1][ap] == 'D'):
                if think_memory[64] == 0:
                    think_memory[64] = 1
                return 'C'
        return 'D'
    elif strategy == 'RNN & RTRL':
        brain = think_memory[65][0]

        if game_round > 0:
            if memory[-1][ap] == 'C':
                if memory[-1][1-ap] == 'C':
                    reward = 1
                else:
                    reward = 2
            else:
                if memory[-1][1-ap] == 'C':
                    reward = -2
                else:
                    reward = -1
            learning_rate = 0.1

            Error = [-((memory[-1][1-ap] == 'C') - brain["output old val"][0]) * reward, -((memory[-1][1-ap] == 'D') - brain["output old val"][1]) * reward]
            G = brain["G"]

            for learn in range(brain["n"]):
                derivative = 1 - (brain["n old val"][learn] * brain["n old val"][learn])
                for learn2 in range(4): brain["scalar trace wi"][learn][learn2] = derivative * (brain["input old val"][learn2] + brain["recurrent weight"][learn] * brain["old scalar trace wi"][learn][learn2])
                brain["scalar trace wr"][learn] = derivative * (brain["n old val"][learn] + brain["recurrent weight"][learn] * brain["old scalar trace wr"][learn])
                G[learn] = sum(Error[learn2] * brain["output weight"][learn2][learn] for learn2 in range(2))
                for learn2 in range(4): brain["input weight"][learn][learn2] -= learning_rate * G[learn] * brain["scalar trace wi"][learn][learn2]
                brain["recurrent weight"][learn] -= learning_rate * G[learn] * brain["scalar trace wr"][learn]
                brain["bias"][learn] -= learning_rate * G[learn] * derivative

            for learn in range(2):
                for learn2 in range(brain["n"]): brain["output weight"][learn][learn2] -= learning_rate * Error[learn] * brain["n old val"][learn2]
                brain["output bias"][learn] -= learning_rate * Error[learn]

            for i in range(len(brain["old scalar trace wi"])): brain["old scalar trace wi"][i] = brain["old scalar trace wi"][i][:]
            brain["old scalar trace wr"][:] = brain["old scalar trace wr"][:]
            for i in range(len(brain["old scalar trace wo"])): brain["old scalar trace wo"][i] = brain["old scalar trace wo"][i][:]

        self_share = (memory[-1][1-ap] == 'C') if game_round > 0 else True
        self_steal = (memory[-1][1-ap] == 'D') if game_round > 0 else False
        opp_share = (memory[-1][ap] == 'C') if game_round > 0 else True
        opp_steal = (memory[-1][ap] == 'D') if game_round > 0 else False

        inp = [opp_share, opp_steal, self_share, self_steal]
        brain["n real val"] = mth_mtx.matrix_function(
            mth_mtx.matrix_aritc(
                mth_mtx.matrix_aritc(
                    mth_mtx.matrix_list_nestlist(
                        inp, brain["input weight"], brain["n"], None, 'dot prod'
                    ), 
                    mth_mtx.matrix_aritc(
                            brain["n old real val"], brain["recurrent weight"], '*'
                    ), '+'
                ), brain["bias"], '+'
            ), (-1, 1), 'clamp'
        )
        brain["n val"] = mth_mtx.matrix_function(brain["n real val"], None, 'tanh')
        brain["output val"] = mth_mtx.matrix_function(
            mth_mtx.matrix_aritc(
                mth_mtx.matrix_list_nestlist(
                    brain["n val"], brain["output weight"], 2, None, 'dot prod'
                ), brain["bias"], '+'
            ), (-1, 1), 'clamp'
        )

        share_output = mth.exp(brain["output val"][0]) / (mth.exp(brain["output val"][0]) + mth.exp(brain["output val"][1]))
        steal_output = mth.exp(brain["output val"][1]) / (mth.exp(brain["output val"][0]) + mth.exp(brain["output val"][1]))

        brain["output old val"] = [share_output, steal_output]
        brain["n old real val"] = brain["n real val"][:]
        brain["n old val"] = brain["n val"][:]
        brain["input old val"] = inp[:]
        
        if share_output >= steal_output:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Dont Bite The Hand That Feeds You':
        think_memory[66][1] *= 0.75
        if (memory[-1][ap] == 'C') if game_round > 0 else True:
            think_memory[66][1] += 0.25

        think_memory[66][0] *= (1 - think_memory[66][1])
        if (think_memory[66][0] >= 0.5) or ((memory[-1][ap] == 'D') if game_round > 0 else False):
            if think_memory[66][0] < 0.5:
                think_memory[66][0] = min((think_memory[66][0] + 1), 5)
            return 'D'
        else:
            return 'C'
    elif strategy == 'Tri-Brain':
        tb_eyes = [('C', 'C') for _ in range(8)] + memory
        tribrain_input_power = [0.125 * (i + 1) for i in range(8)]
        opp_inp = [1 if tb_eyes[-1 - (7 - i)][ap] == 'C' else 0 for i in range(8)]
        self_inp = [1 if tb_eyes[-1 - (7 - i)][1 - ap] == 'C' else 0 for i in range(8)]
        SH = (tb_eyes[-1][ap] == 'C')

        def clamp(x):
            return max(min(x,1),-1)
        
        def tribrains_norm(x):
            return (x - 0.5) * 2

        tribrain_logic = clamp(sum(((SH * (tribrain_input_power[i] * opp_inp[i])) - ((1-SH) * (tribrain_input_power[i] * (1 - opp_inp[i])))) + (tribrain_input_power[i] * (tribrains_norm(opp_inp[i]) + tribrains_norm(self_inp[i]))) for i in range(8)))
        tribrain_most = clamp(sum((tribrain_input_power[i] * opp_inp[i]) - (tribrain_input_power[i] * (1 - opp_inp[i])) for i in range(8)))
        tribrain_now = SH - (1-SH)

        tribrain_output = tribrain_logic + tribrain_most + tribrain_now
        if tribrain_output > 0:
            return 'C'
        elif tribrain_output == 0:
            return tb_eyes[-1][ap]
        else:
            return 'D'
    elif strategy == 'Racister':
        if (game_round + 1) <= (tournament_avg_last_round / 2):
            if game_round > 0:
                if memory[-1][ap] == 'D':
                    think_memory[67] = 1
            return 'C'
        else:
            if think_memory[67] == 1:
                return memory[-1][ap]
            else:
                return 'D'
    elif strategy == 'Karen':
        if think_memory[68] == 0:
            if ((memory[-1][ap] == 'C') if game_round > 0 else True):
                if rnd.random() <= 0.1:
                    return 'D'
                else:
                    return 'C'
            else:
                think_memory[68] = 4
                return rnd.choice(['C','D'])
        else:
            if memory[-1][ap] == 'C':
                think_memory[68] -= 1
                return rnd.choice(['C','D'])
            else:
                return 'D'
    elif strategy == 'Male / Man':
        if (game_round + 1) <= ((50 / 200) * tournament_avg_last_round): 
            if (think_memory[69] == 1) or ((memory[-1][ap] == 'D') if game_round > 0 else False):
                if think_memory[69] == 0:
                    think_memory[69] = 1
                return 'D'
            else:
                return 'C'
        elif (game_round + 1) <= ((100 / 200) * tournament_avg_last_round): 
            if (think_memory[69] == 1) or (memory[-1][ap] == 'D'):
                if think_memory[69] == 0:
                    think_memory[69] = 1
                else:
                    if rnd.random() <= 0.1:
                        think_memory[69] = 0
                return 'D'
            else:
                return 'C'
        elif (game_round + 1) <= ((150 / 200) * tournament_avg_last_round): 
            return memory[-1][ap]
        else: 
            if rnd.random() < 0.1:
                return 'C'
            else:
                return memory[-1][ap]
    elif strategy == 'Female / Woman':
        if (game_round + 1) <= ((50 / 200) * tournament_avg_last_round): 
            if ((memory[-1][ap] == 'C') if game_round > 0 else True):
                return 'C'
            else:
                return rnd.choice(['C','D'])
        elif (game_round + 1) <= ((100 / 200) * tournament_avg_last_round): 
            if rnd.random() < 0.1:
                return 'D'
            else:
                return memory[-1][ap]
        elif (game_round + 1) <= ((150 / 200) * tournament_avg_last_round): 
            if rnd.random() < 0.1:
                return 'C'
            else:
                return memory[-1][ap]
        else: 
            if (memory[-1][ap] == 'D') and (memory[-1-1][ap] == 'D'):
                return 'D'
            else:
                return 'C'
    elif strategy == 'Anatol':
        Ant_eyes = [('C', 'C') for _ in range(4)] + memory
        self_inp = [(Ant_eyes[-1 - (3 - i)][1-ap] == 'D') for i in range(4)]
        opp_inp = [(Ant_eyes[-1 - (3 - i)][ap] == 'D') for i in range(4)]
        t_d = sum(self_inp[i] + opp_inp[i] for i in range(4))
        self_t_d = sum(self_inp)

        if ((t_d >= 3) and (t_d <= 4)) and (self_t_d > 0):
            return 'C'
        else:
            return Ant_eyes[-1][ap]
    elif strategy == 'Good':
        brain = think_memory[70][0]

        if game_round > 0:
            if memory[-1][ap] == 'C':
                if memory[-1][1-ap] == 'C':
                    reward_or_punisment = 1.25
                else:
                    reward_or_punisment = 1.5
            else:
                if memory[-1][1-ap] == 'C':
                    reward_or_punisment = 0.5
                else:
                    reward_or_punisment = 1.1
            
            brain[think_memory[70][1]] = max(min(brain[think_memory[70][1]] * reward_or_punisment,1),0)

        inp = tuple([memory[-1][ap], memory[-1][1-ap]]) if game_round > 0 else ('C', 'C')

        if brain.get(inp) == None:
            brain[inp] = 0.5
        think_memory[70][1] = inp
        
        if rnd.random() <= brain[inp]:
            return 'C'
        else:
            return 'D'
    elif strategy == 'AQUA':
        if not game_round:
            return 'C'

        B = min(8, game_round)
        inp = [1 if memory[-1 - ((B - 1) - i)][ap] == 'C' else 0 for i in range(B)]
        share_ratio = sum(inp) / B

        if rnd.random() <= share_ratio:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Majapahit':
        R = 5
        if (game_round + 1) <= R:
            if game_round > 0:
                if memory[-1][ap] == 'D':
                    think_memory[72][0] = 1
            return 'C'
        elif (game_round + 1) == (R + 1):
            if think_memory[72][0] == 1:
                return 'D'
            else:
                return 'C'
        else:
            if memory[-1][ap] == 'D':
                think_memory[72][1] = min(think_memory[72][1] + 0.1, 5)

            if think_memory[72][0] == 1:
                think_memory[72][1] *= 0.9
                if rnd.random() < (think_memory[72][1] + 0.1):
                    return 'D'
                else:
                    return memory[-1][ap]
            else:
                return memory[-1][ap]
    elif strategy == 'EGOist':
        ego_of_hate = max((((score[ap] - score[1-ap]) + ((memory[-1][ap] == 'D') if game_round > 0 else False)) / 10),0)
        ego_of_wanting = 0.1 + (0.15 * ((memory[-1][ap] == 'C') if game_round > 0 else True))
        ego = ego_of_hate + ego_of_wanting
        
        if rnd.random() <= ego:
            return 'D'
        else:
            return 'C'
    elif strategy == 'The Calculated Mirror With A Fuse':
        if (game_round + 1) > 2:
            if ((memory[-1-1][ap] == 'D') and (memory[-1][ap] == 'D')) or (think_memory[73][1] > 0):
                if think_memory[73][1] == 0:
                    think_memory[73][1] = 3

                if think_memory[73][1] < 3:
                    if memory[-1][ap] != 'D':
                        think_memory[73][1] -= 1
                else:
                    think_memory[73][1] -= 1

                return 'D'

        if game_round > 0:
            if memory[-1][ap] == 'C':
                think_memory[73][0] += 1

        if (game_round + 1) <= 2:
            return 'C'
        
        A = think_memory[73][0] / (game_round + 1)

        if A >= 0.7:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Trust Ledger':
        if (game_round + 1) > 3:
            inp = tuple([memory[-1 - (2 - i)][ap] for i in range(3)])

            if (inp == ("C", "C", "C")) and (think_memory[74] <= -6):
                think_memory[74] = 0

        if game_round > 0:
            if memory[-1][ap] == 'C':
                think_memory[74] += 2
            else:
                think_memory[74] -= 3

        if think_memory[74] > 0:
            think_memory[74] -= 1
        elif think_memory[74] < 0:
            think_memory[74] += 1
        
        think_memory[74] = max(min(think_memory[74],10),-10)

        if think_memory[74] >= 3:
            return 'C'
        elif think_memory[74] <= -3:
            return 'D'
        else:
            if not game_round:
                return 'C'
            return memory[-1][ap]
    elif strategy == 'Echo Adaptive':
        if ((memory[-1-1][ap] == 'D') if (game_round + 1) > 2 else False) and ((memory[-1][ap] == 'D') if game_round > 0 else False):
            return 'C'
        elif ((memory[-1-2][ap] == 'D') if (game_round + 1) > 3 else False) and ((memory[-1-1][ap] == 'D') if (game_round + 1) > 2 else False):
            return 'D'
        else:
            if not game_round:
                return 'C'
            return memory[-1][ap]
    elif strategy == 'Threshold Punisher':
        if game_round > 0:
            if memory[-1][ap] == 'D':
                think_memory[75] += 1
        else:
            return 'C'

        steal_ratio = think_memory[75] / (game_round)

        if steal_ratio > 0.3:
            return 'D'
        else:
            return 'C'
    elif strategy == 'AMERICA! / USA / Freedoom':

        tm = think_memory[77]

        
        

        opp = memory[-1][ap] if game_round > 0 else 'C'

        
        
        

        
        tm[0] += 0.1
        tm[0] = min(tm[0],1)

        
        
        

        exploit_chance = (
            (tm[0] * 0.6)
        )

        
        if opp == 'C':

            if rnd.random() < exploit_chance:

                tm[1] += 1
                return 'D'

            else:
                return 'C'

        
        
        

        else:
            return 'D'
    elif strategy == 'SOVIET UNION! / USSR / Inequality':

        tm = think_memory[78]

        
        

        opp = memory[-1][ap] if game_round > 0 else 'C'

        
        
        

        if opp == 'C':

            tm[0] += 1
            tm[1] += 0.04

        else:
            tm[1] -= 0.08

        tm[1] = max(0,min(1,tm[1]))

        
        
        

        if rnd.random() < (0.05 * min(tm[0], 2)):
                return 'D'

        
        
        

        if rnd.random() < tm[1]:
            return 'C'

        
        if opp == 'D':
            return 'D'

        return 'C'
    elif strategy == 'Little Grudger':
        if game_round > 0:
            think_memory[79] *= 0.75
            if memory[-1][ap] == 'C':
                think_memory[79] += 0.25
        
        if rnd.random() <= think_memory[79]:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Second Chance Attar Version':
        if not game_round:
            return 'C'
        
        if memory[-1][ap] == 'D':
            if think_memory[80][0] == 0:
                think_memory[80][0] = 1
                think_memory[80][1] = (game_round + 1)
            elif think_memory[80][0] == 1:
                think_memory[80][0] = 2
                think_memory[80][2] = (game_round + 1)
            else:
                think_memory[80][1] = think_memory[80][2]
                think_memory[80][2] = (game_round + 1)
        
        if think_memory[80][0] != 2:
            return 'C'
        else:
            A1 = max(min(1 - ((think_memory[80][2] - think_memory[80][1]) / 5),1),0)
            A2 = max(min(1 - (((game_round) - think_memory[80][2]) / 5),1),0)
            A = A1 * A2
            if rnd.random() <= A:
                return 'D'
            else:
                return 'C'
    elif strategy == 'Crabby Attar Version':
        if not game_round:
            return 'D'
        elif (game_round + 1) == 2:
            return 'C'
        elif (game_round + 1) == 3:
            if memory[-1][ap] == 'D':
                think_memory[85][0] = 0
            else:
                think_memory[85][0] = 1
            return 'C'
        if think_memory[85][0] == 1:
            if (memory[-1][ap] == 'D') and ((game_round + 1) > 150):
                if think_memory[85][1] == 0:
                    think_memory[85][1] = 1
            if (game_round + 1) > ((185 / 200) * tournament_avg_last_round):
                return 'D'
            if think_memory[85][1] == 0:
                if (game_round + 1) > ((100 / 200) * tournament_avg_last_round):
                    if ((game_round + 1) % 5) == 0:
                        return 'D'
                    else:
                        return 'C'
                else:
                    if rnd.random() <= 0.2:
                        return 'C'
                    else:
                        return memory[-1][ap]
            elif think_memory[85][1] == 1:
                if rnd.random() <= 0.1:
                    if (memory[-1][ap] == 'C'):
                        think_memory[85][1] = 2
                        return 'C'
                return 'D'
            elif think_memory[85][1] == 2:
                think_memory[85][1] = 0
                return 'C'
        else:
            if rnd.random() <= 0.2:
                return 'C'
            else:
                return memory[-1][ap]
    elif strategy == 'Cycler DCCCC':
        if not game_round:
            return 'D'
        elif (game_round + 1) == 2:
            return 'C'
        elif (game_round + 1) == 3:
            if memory[-1][ap] == 'D':
                think_memory[86] = 0
            else:
                think_memory[86] = 1
            return 'C'
        if think_memory[86] == 1:
            if (((game_round + 1)-1) % 5) == 0:
                return 'D'
            else:
                if rnd.random() <= 0.2:
                    return 'C'
                else:
                    return memory[-1][ap]
        else:
            return memory[-1][ap]
    elif strategy == 'Odd-Even Go By Majority':
        if not game_round:
            return 'C'
        B = min(8, game_round)
        inp = [1 if memory[-1 - ((B - 1) - i)][ap] == 'C' else -1 for i in range(B)]

        odt_sum_odd = sum(inp[(2*i)-2] for i in range(B // 2))
        odt_sum_even = sum(inp[(2*i)-1] for i in range(B - (B // 2)))

        if rnd.random() <= 0.5:
            odt_final_choice = odt_sum_even
        else:
            odt_final_choice = odt_sum_odd
        
        if odt_final_choice > 0:
            return 'C'
        elif odt_final_choice < 0:
            return 'D'
        else:
            return memory[-1][ap]
    elif strategy == 'Desire':
        dsi_eyes = [('C', 'C') for _ in range(8)] + memory
        input_power = [0.125 * (i + 1) for i in range(8)]
        D_input = [1 if dsi_eyes[-1 - (7 - i)][ap] == 'D' else 0 for i in range(8)]
        C_input = [1 if dsi_eyes[-1 - (7 - i)][ap] == 'C' else 0 for i in range(8)]

        A1 = sum(D_input[i] * input_power[i] for i in range(8))
        A2 = sum(C_input[i] * input_power[i] for i in range(8))
        if (score[1-ap] - score[ap]) <= -10:
            A1 += -((score[1-ap] - score[ap]) / 10)
            A2 += ((score[1-ap] - score[ap]) / 10)
        elif (score[1-ap] - score[ap]) >= 10:
            A1 += (score[1-ap] - score[ap]) / 10
            A2 += -(score[1-ap] - score[ap]) / 10
        A = A2 - A1
        if A > 0:
            return 'C'
        elif A == 0:
            return 'D' if rnd.random() <= 0.25 else 'C'
        else:
            return 'D'
    elif strategy == 'Lighty-Darky':
        if ((memory[-1][ap] == 'D') if game_round > 0 else False):
            think_memory[87] += 1
        else:
            if rnd.random() <= 0.1:
                think_memory[87] += 1
        if (game_round + 1) > (tournament_avg_last_round / 2):
            if think_memory[87] > 0:
                think_memory[87] -= 1
                return 'D'
            else:
                return 'C'
        else:
            return 'C'
    elif strategy == 'Ethanol / Alcohol':
        if not game_round:
            return 'D'
        elif (game_round + 1) <= (tournament_avg_last_round / 2):
            if memory[-1][ap] == 'D':
                think_memory[90] += 1
            return 'C'
        else:
            if memory[-1][ap] == 'D':
                think_memory[90] += 1
            if (think_memory[90] - 1) > 0:
                think_memory[90] -= 1
                return 'D'
            else:
                return 'C'
    elif strategy == 'Simple Exploiter':
        if (game_round + 1) <= 2:
            return ('D', 'C')[game_round]
        elif (game_round + 1) == 3:
            if memory[-1][ap] == 'D':
                think_memory[92] = 1
                return 'C'
            else:
                think_memory[92] = 0
                return 'D'
        else:
            if think_memory[92] == 1:
                return memory[-1][ap]
            else:
                return 'D'
    elif strategy == 'Smart Go By Majority':
        if (game_round + 1) <= 8:
            if game_round > 0:
                think_memory[93][(game_round + 1) - 2] = (memory[-1][ap] == 'D')
            return 'C'
        def st_set_new_memory():
            for i in range(6):
                think_memory[93][i] = think_memory[93][i+1]
            think_memory[93][6] = (memory[-1][ap] == 'D')
        t_d = sum(think_memory[93])
        A = 1 - (((t_d * t_d) - 1) / 49)
        if rnd.random() <= A:
            st_set_new_memory()
            return 'C'
        else:
            st_set_new_memory()
            return 'D'
    elif strategy == 'Mafia A':
        if (game_round + 1) <= 3:
            return ('D', 'C', 'D')[game_round]
        elif (game_round + 1) == 4:
            inp = tuple(["D" if memory[-1-(2-i)][ap] == 'D' else "C" for i in range(3)])
            if inp == ("D", "C", "D"):
                think_memory[94] = 1
                return 'C'
            elif inp == ("C", "D", "C"):
                think_memory[94] = 2
                return 'C'
            else:
                think_memory[94] = 0
                return 'D'
        else:
            mafia_eyes = [('C', 'C') for _ in range(10)] + memory
            opp_inp = tuple(["D" if mafia_eyes[-1-(2-i)][ap] == 'D' else "C" for i in range(3)])
            self_inp = tuple(["D" if mafia_eyes[-1-(2-i)][1-ap] == 'D' else "C" for i in range(3)])
            grudger_inp = [1 if mafia_eyes[-1-(9-i)][ap] == 'D' else 0 for i in range(10)]
            if think_memory[94] > 0:
                if think_memory[94] == 2:
                    grudger_ratio = sum(grudger_inp) / 10
                    if (opp_inp, self_inp) in ((("C", "D", "C"), ("D", "C", "D")), (("D", "C", "D"), ("C", "D", "C"))):
                        return 'C'
                    else:
                        A = 1 - max(min(2 * (grudger_ratio - 0.5),1),0)
                        if rnd.random() <= (0.1 * A):
                            return 'C'
                        else:
                            return memory[-1][ap]
                else:
                    return 'C'
            else:
                return 'D'
    elif strategy == 'Bluff':
        if (game_round + 1) <= 4:
            return ('C', 'D', 'C', 'C')[(game_round + 1)-1]
        else:
            if memory[-1][ap] == 'D':
                return 'D'
            else:
                if (rnd.random() <= 0.5) and ((memory[-1][1-ap] == 'D') and (memory[-1-1][ap] == 'D')):
                    return 'D'
                else:
                    return 'C'
    elif strategy == 'Mafia B':
        if (game_round + 1) <= 3:
            return ('D', 'C', 'D')[game_round]
        elif (game_round + 1) == 4:
            inp = tuple(["D" if memory[-1-(2-i)][ap] == 'D' else "C" for i in range(3)])
            if inp == ("D", "C", "D"):
                think_memory[95] = 1
                return 'C'
            elif inp == ("C", "D", "C"):
                think_memory[95] = 2
                return 'C'
            else:
                think_memory[95] = 0
                return 'D'
        else:
            mafia_eyes = [('C', 'C') for _ in range(10)] + memory
            opp_inp = tuple(["D" if mafia_eyes[-1-(2-i)][ap] == 'D' else "C" for i in range(3)])
            self_inp = tuple(["D" if mafia_eyes[-1-(2-i)][1-ap] == 'D' else "C" for i in range(3)])
            grudger_inp = [1 if mafia_eyes[-1-(9-i)][ap] == 'D' else 0 for i in range(10)]
            if think_memory[95] > 0:
                if think_memory[95] == 2:
                    grudger_ratio = sum(grudger_inp) / 10
                    if (opp_inp, self_inp) in ((("C", "D", "C"), ("D", "C", "D")), (("D", "C", "D"), ("C", "D", "C"))):
                        return 'C'
                    else:
                        A = 1 - max(min(2 * (grudger_ratio - 0.5),1),0)
                        if rnd.random() <= (0.1 * A):
                            return 'C'
                        else:
                            return memory[-1][ap]
                else:
                    return 'C'
            else:
                if (game_round + 1) % 2 == 0:
                    return 'D'
                else:
                    return memory[-1][ap]
    elif strategy == 'Mafia C':
        if (game_round + 1) <= 3:
            return ('D', 'C', 'D')[game_round]
        elif (game_round + 1) == 4:
            inp = tuple(["D" if memory[-1-(2-i)][ap] == 'D' else "C" for i in range(3)])
            if inp == ("D", "C", "D"):
                think_memory[96] = 1
                return 'C'
            elif inp == ("C", "D", "C"):
                think_memory[96] = 2
                return 'C'
            else:
                think_memory[96] = 0
                return 'D'
        else:
            mafia_eyes = [('C', 'C') for _ in range(10)] + memory
            opp_inp = tuple(["D" if mafia_eyes[-1-(2-i)][ap] == 'D' else "C" for i in range(3)])
            self_inp = tuple(["D" if mafia_eyes[-1-(2-i)][1-ap] == 'D' else "C" for i in range(3)])
            grudger_inp = [1 if mafia_eyes[-1-(9-i)][ap] == 'D' else 0 for i in range(10)]
            if think_memory[96] > 0:
                if think_memory[96] == 2:
                    grudger_ratio = sum(grudger_inp) / 10
                    if (opp_inp, self_inp) in ((("C", "D", "C"), ("D", "C", "D")), (("D", "C", "D"), ("C", "D", "C"))):
                        return 'C'
                    else:
                        A = 1 - max(min(2 * (grudger_ratio - 0.5),1),0)
                        if rnd.random() <= (0.2 * A):
                            return 'C'
                        else:
                            return memory[-1][ap]
                else:
                    return 'C'
            else:
                if memory[-1][ap] == 'C':
                    if memory[-1][1-ap] == 'C':
                        return rnd.choice(['D','C'])
                    else:
                        return 'D'
                else:
                    if memory[-1][1-ap] == 'C':
                        return 'D'
                    else:
                        return rnd.choice(['D','C'])
    elif strategy == 'Mafia D':
        if (game_round + 1) <= 3:
            return ('D', 'C', 'D')[game_round]
        elif (game_round + 1) == 4:
            inp = tuple(["D" if memory[-1-(2-i)][ap] == 'D' else "C" for i in range(3)])
            if inp == ("D", "C", "D"):
                think_memory[97] = 1
                return 'C'
            elif inp == ("C", "D", "C"):
                think_memory[97] = 2
                return 'C'
            else:
                think_memory[97] = 0
                return 'D'
        else:
            mafia_eyes = [('C', 'C') for _ in range(10)] + memory
            opp_inp = tuple(["D" if mafia_eyes[-1-(2-i)][ap] == 'D' else "C" for i in range(3)])
            self_inp = tuple(["D" if mafia_eyes[-1-(2-i)][1-ap] == 'D' else "C" for i in range(3)])
            grudger_inp = [1 if mafia_eyes[-1-(9-i)][ap] == 'D' else 0 for i in range(10)]
            if think_memory[97] > 0:
                if think_memory[97] == 2:
                    grudger_ratio = sum(grudger_inp) / 10
                    if (opp_inp, self_inp) in ((("C", "D", "C"), ("D", "C", "D")), (("D", "C", "D"), ("C", "D", "C"))):
                        return 'C'
                    else:
                        A = 1 - max(min(2 * (grudger_ratio - 0.5),1),0)
                        if rnd.random() <= (0.2 * A):
                            return 'C'
                        else:
                            return memory[-1][ap]
                else:
                    return 'C'
            else:
                if memory[-1][ap] == 'C':
                    return rnd.choice(['D','C'])
                else:
                    return 'D'
    elif strategy == 'Stoicalist':
        M = 5
        a = 1 / M
        input_power = [a * (i+1) for i in range(M)]
        stc_eyes = [('C', 'C') for _ in range(M)] + memory
        inp = [1 if stc_eyes[-1-((M-1)-i)][ap] == 'D' else 0 for i in range(M)]
        E = sum(input_power[i] * inp[i] for i in range(M)) / sum(input_power[j] for j in range(M))
        if E >= 0.5:
            return 'D'
        else:
            return 'C'
    elif strategy == 'Sun Tzu Bot':
        if think_memory[98] == 1:
            return 'D'
        else:
            stb_eyes = [('C', 'C') for _ in range(10)] + memory
            inp = [1 if stb_eyes[-1 - ((9) - i)][ap] == 'D' else 0 for i in range(10)]
            grudger_ratio = sum(inp[i] for i in range(10)) / 10
            random_cc_ratio = sum(1 if (inp[i], inp[i+1]) == (0, 0) else 0 for i in range(9)) / 9
            random_cd_ratio = sum(1 if (inp[i], inp[i+1]) == (0, 1) else 0 for i in range(9)) / 9
            random_dc_ratio = sum(1 if (inp[i], inp[i+1]) == (1, 0) else 0 for i in range(9)) / 9
            random_dd_ratio = sum(1 if (inp[i], inp[i+1]) == (1, 1) else 0 for i in range(9)) / 9
            random_estimation = (1.5 - (abs(random_cc_ratio - 0.25) + abs(random_cd_ratio - 0.25) + abs(random_dc_ratio - 0.25) + abs(random_dd_ratio - 0.25))) / 1.5
            opp_inp = tuple([1 if stb_eyes[-1 - ((2) - i)][ap] == 'D' else 0 for i in range(3)])
            self_inp = tuple([1 if stb_eyes[-1 - ((2) - i)][1-ap] == 'D' else 0 for i in range(3)])
            if random_estimation >= 0.75:
                think_memory[98] = 1
                return 'D'
            else:
                if (opp_inp, self_inp) in (((1, 0, 1), (0, 1, 0)), ((0, 1, 0), (1, 0, 1))):
                    return 'C'
                else:
                    A = 1 - max(min(2 * (grudger_ratio - 0.5),1),0)
                    if rnd.random() <= (0.1 * A):
                        return 'C'
                    else:
                        return stb_eyes[-1][ap]
    elif strategy == 'Laws':
        brain = think_memory[100][0]
        another_variable = think_memory[100][1]
        another_variable[0] -= 1

        if game_round > 0:
            if memory[-1][ap] == 'D':
                another_variable[2] += 1

        A1 = 5
        A2 = 3

        laws_eyes = [('C', 'C') for _ in range(A1)] + memory

        if len(brain) != 0:
            if laws_eyes[-1][ap] == 'C':
                if laws_eyes[-1][1-ap] == 'C':
                    brain[another_variable[1]][1] += RPST['R']
                else:
                    brain[another_variable[1]][1] += RPST['T']
            else:
                if laws_eyes[-1][1-ap] == 'C':
                    brain[another_variable[1]][1] -= 4 * abs(RPST['S'] - RPST['P'])
                else:
                    brain[another_variable[1]][1] += RPST['P']

            if brain[another_variable[1]][1] <= 0 and another_variable[0] <= 0:
                new_output = "".join([rnd.choice(["C", "D", "T"]) for _ in range(A2)])
                brain[another_variable[1]] = [new_output,100]
            if brain[another_variable[1]][1] >= 200 and another_variable[0] <= 0:
                ST = (laws_eyes[-1][ap] == 'D')
                a = 0.25
                A = rnd.uniform(0.0, 1.5)
                if A <= (((another_variable[2] / (game_round + 1)) * (1 - a)) + (ST * a)):
                    new_choice = "D"
                else:
                    if A > 1:
                        new_choice = "T"
                    else:
                        new_choice = "C"
                idx = rnd.randint(0,A2-1)
                new_output = brain[another_variable[1]][0][:idx] + new_choice + brain[another_variable[1]][0][idx+1:]
                brain[another_variable[1]] = [new_output,100]

        if another_variable[0] <= 0:

            inp = "".join([laws_eyes[-1 - ((A1-1) - i)][ap] for i in range(A1)])

            another_variable[0] = A2
            another_variable[1] = inp

            def similarity(new_key,old_key):
                new_key_vector_opp = [1 if new_key[S1] == "C" else -1 for S1 in range(A1)]
                old_key_vector_opp = [1 if old_key[S2] == "C" else -1 for S2 in range(A1)]
                similarity_val = 0
                for SRT in range(A1):
                    similarity_val += (new_key_vector_opp[SRT] * old_key_vector_opp[SRT])
                return ((similarity_val + A1) / 2) / (A1 * 2)
        
            if brain.get(inp) == None:
                if len(brain) != 0:
                    target_space = min(brain, key=lambda k: brain[k][1])

                    is_same = similarity(inp,target_space)

                    new_output = "".join([rnd.choice(["C", "D", "T"]) for _ in range(A2)])

                    if is_same >= rnd.random():
                        brain[target_space] = [new_output,100]
                        another_variable[1] = target_space
                    else:
                        brain[inp] = [new_output,100]
                        another_variable[1] = inp
                else:
                    new_output = "".join([rnd.choice(["C", "D", "T"]) for _ in range(A2)])
                    
                    brain[inp] = [new_output,100]
                    another_variable[1] = inp

        idx = A2 - another_variable[0]
        idx = max(0, min(idx, (A2-1)))

        if brain[another_variable[1]][0][idx] == "C":
            return 'C'
        else:
            if brain[another_variable[1]][0][idx] == "T":
                return laws_eyes[-1][ap]
            else:
                return 'D'
    elif strategy == 'Matcher':
        if (game_round + 1) <= 10:
            if not game_round:
                return 'C'
            return memory[-1][ap]
        
        input_power = [0.1 * (i+1) for i in range(10)]
        cc_inp = [
                (1 if (memory[-1 - ((9) - i)][ap] == 'C') else -1) if ((memory[-1 - (((9) - i) + 1)][1-ap] == 'C') and (memory[-1 - (((9) - i) + 1)][ap] == 'C')) else 0
            for i in range(1, 10)
            ]
        cd_inp = [
                (1 if (memory[-1 - ((9) - i)][ap] == 'C') else -1) if ((memory[-1 - (((9) - i) + 1)][1-ap] == 'C') and (memory[-1 - (((9) - i) + 1)][ap] == 'D')) else 0
            for i in range(1, 10)
            ]
        dc_inp = [
                (1 if (memory[-1 - ((9) - i)][ap] == 'C') else -1) if ((memory[-1 - (((9) - i) + 1)][1-ap] == 'D') and (memory[-1 - (((9) - i) + 1)][ap] == 'C')) else 0
            for i in range(1, 10)
            ]
        dd_inp = [
                (1 if (memory[-1 - ((9) - i)][ap] == 'C') else -1) if ((memory[-1 - (((9) - i) + 1)][1-ap] == 'D') and (memory[-1 - (((9) - i) + 1)][ap] == 'D')) else 0
            for i in range(1, 10)
            ]
        matcher_dict = {
            'CC':sum(cc_inp[i] * input_power[i + 1] for i in range(9)),
            'CD':sum(cd_inp[i] * input_power[i + 1] for i in range(9)),
            'DC':sum(dc_inp[i] * input_power[i + 1] for i in range(9)),
            'DD':sum(dd_inp[i] * input_power[i + 1] for i in range(9))
        }
        matcher_cond = ('C' if memory[-1][1 - ap] == 'C' else 'D') + ('C' if memory[-1][ap] == 'C' else 'D')
        if matcher_dict[matcher_cond] >= 0:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Trust Score Sigmoid':
        if game_round > 0:
            if memory[-1][ap] == 'C':
                think_memory[105] += 1
            else:
                think_memory[105] -= 3

        def sigmoid(x):
            sigm = max(min(x,10),-10)
            return 1 / (1 + (mth.exp(-sigm)))
        
        A = sigmoid(think_memory[105])

        if rnd.random() <= A:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Tit For Increasing Tat':
        tfit_eyes = [('C', 'C') for _ in range(5)] + memory
        if think_memory[106][1] > 0:
            think_memory[106][1] -= 1
            return 'D'
        else:
            if tfit_eyes[-1][ap] == 'D':
                think_memory[106][0] += 1
                think_memory[106][1] = think_memory[106][0] - 1
                return 'D'
            else:
                inp = tuple([tfit_eyes[-1 - ((4) - i)][ap] for i in range(5)])
                if inp == ("C", "C", "C", "C", "C"):
                    think_memory[106][0] = max(think_memory[106][0] - 1, 0)
                return 'C'
    elif strategy == 'Freud':
        if game_round > 0:
            if (memory[-1][1-ap] == 'C'):
                if (memory[-1][ap] == 'D'):
                    think_memory[107] -= 3
                else:
                    think_memory[107] += 1

        id_wanting = think_memory[107] + (score[1-ap] - score[ap])
        ego_reality = (((memory[-1][ap] == 'C') if game_round > 0 else True) - 0.5) * 2
        super_ego_morality = (((memory[-1][ap] == 'C') if game_round > 0 else True) - 0.5) * 2 if rnd.random() > 0.2 else 1
        
        A = id_wanting + ego_reality + super_ego_morality

        if A == 0:
            if not game_round:
                return 'C'
            return memory[-1][ap]
        elif A > 0:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Jung':
        jung_eyes = [('C', 'C') for _ in range(10)] + memory
        if jung_eyes[-1][ap] == 'D':
            think_memory[108] += 1
        else:
            think_memory[108] -= 0.25
        
        persona = ((jung_eyes[-1][ap] == 'C') - 0.5) * 2 if rnd.random() > 0.2 else 1
        shadow = -think_memory[108]

        def clamp(x):
            return max(min(x,1),0)
        
        input_power = [0.1 * (i+1) for i in range(10)]
        inp = [1 if jung_eyes[-1 - ((9) - i)][ap] == 'C' else 0 for i in range(10)]

        a = clamp(sum(input_power[i] * inp[i] for i in range(10)) / sum(input_power))
        A = (shadow * (1 - a)) + (persona * a)

        if A == 0:
            return jung_eyes[-1][ap]
        elif A > 0:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Skinner':
        def clamp(x):
            return max(min(x,1),-1)

        if game_round > 0:
            if memory[-1][ap] == 'C':
                if memory[-1][1-ap] == 'C':
                    think_memory[109][0] += min(3 + max((score[1-ap] - score[ap]), 0), 10)
                else:
                    think_memory[109][1] += min(5 + max((score[1-ap] - score[ap]), 0), 10)
            else:
                if memory[-1][1-ap] == 'C':
                    think_memory[109][0] -= min(4 + max(-(score[1-ap] - score[ap]), 0), 10)
                else:
                    think_memory[109][1] -= min(1 + max(-(score[1-ap] - score[ap]), 0), 10)

            think_memory[109][0] = max(min(think_memory[109][0], 100), -100)
            think_memory[109][1] = max(min(think_memory[109][1], 100), -100)

        E_sh = (think_memory[109][0] / (game_round)) if game_round > 0 else 2.25
        E_st = (think_memory[109][1] / (game_round)) if game_round > 0 else 2.25
        A = E_sh - E_st

        if A == 0:
            if not game_round:
                return 'C'
            return memory[-1][ap]
        elif A > 0:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Rogers & Maslow':
        rogers_and_maslow_eyes = [('C', 'C') for _ in range(10)] + memory
        input_power = [mth.exp(i+1) for i in range(10)]
        inp = [1 if rogers_and_maslow_eyes[-1 - ((9) - i)][ap] == 'D' else 0 for i in range(10)]

        D = sum(input_power[i] * inp[i] for i in range(10)) / sum(input_power)

        if rogers_and_maslow_eyes[-1][ap] == 'D':
            think_memory[110] += 1
        else:
            think_memory[110] *= D
        
        A = 1 - min(think_memory[110], 1)

        if rnd.random() <= A:
            return 'C'
        else:
            if rnd.random() <= (1 - D):
                return 'C'
            else:
                return 'D'
    elif strategy == 'Frankl':
        if game_round > 0:
            if memory[-1][ap] == 'D':
                think_memory[111] += 1

        frankl_choice = [0,0]

        frankl_choice[0] = (-1 if memory[-1][ap] == 'D' else 1) if game_round > 0 else 1

        Y_I_exist_qo = (abs((think_memory[111] / game_round) - 0.5) * 2) if game_round > 0 else 0

        if rnd.random() < ((think_memory[111] / game_round) if game_round > 0 else 0): 
            frankl_choice[1] = -1
        else:
            frankl_choice[1] = 1
        
        A = (frankl_choice[0] * (1 - Y_I_exist_qo)) + (frankl_choice[1] * Y_I_exist_qo)

        if A >= 0:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Teacher Pee While Standing, Student Pee While Running':
        if game_round > 0:
            if memory[-1][ap] == 'D':
                think_memory[112] += 1
        
        a = 0.1

        ST = (memory[-1][ap] == 'D') if game_round > 0 else False
        A = ((think_memory[112] / (game_round + 1)) * (1 - a)) + (ST * a)

        if rnd.random() < A:
            return 'D'
        else:
            return 'C'
    elif strategy == 'Javelin':
        javelin_eyes = [('C', 'C') for _ in range(5)] + memory
        input_power = [0.2 * (i + 1) for i in range(5)]
        inp = [1 if javelin_eyes[-1 - ((4) - i)][ap] == 'D' else -1 for i in range(5)]

        A1 = sum(input_power[i] * inp[i] for i in range(5)) / sum(input_power)
        A2 = mth.tanh(A1 * 8)

        if rnd.random() <= A2:
            return 'D'
        else:
            return 'C'
    elif strategy == 'Tip For Tap':
        if not game_round:
            return 'C'
        
        if memory[-1][ap] == 'C':
            think_memory[113][1] += 1

        B = min(5, game_round)
        inp = [1 if memory[-1 - ((B - 1) - i)][ap] == 'D' else 0 for i in range(B)]
        A = sum(inp) / B
        if A >= 0.5:
            return 'D'
        elif (memory[-1][ap] == 'D') or (think_memory[113][0] == 1):
            if think_memory[113][0] == 0:
                think_memory[113][0] = 1
                return 'C'
            else:
                think_memory[113][0] = 0
                if memory[-1][ap] == 'D':
                    return 'D'
                else:
                    p_c = (think_memory[113][1] / (game_round + 1))

                    if rnd.random() < (0.5 * p_c):
                        return 'C'
                    else:
                        return 'D'
        else:
            return 'C'
    elif strategy == 'Patterned Adaptive Fortress':
        confidence = (think_memory[114][0] * 0.95) + (0.05 * ((memory[-1][ap] == 'C') if game_round > 0 else True))
        think_memory[114][0] = confidence
        if (game_round + 1) <= 5:
            return 'C'
        else:
            if game_round >= (tournament_avg_last_round):
                if confidence < 0.9:
                    think_memory[114][2] = 3

            if think_memory[114][2] == 3:
                return 'D'
            
            if think_memory[114][1] > 0:
                if memory[-1][ap] == 'D':
                    think_memory[114][2] += 1

                think_memory[114][1] -= 1
                if think_memory[114][1] > 0:
                    return 'C'
                else:
                    return memory[-1][ap]
            
            if memory[-1][ap] == 'C':
                return 'C'
            else:
                think_memory[114][2] = 0
                think_memory[114][1] = 4
                return 'D'
    elif strategy == 'Anti Noise Tit For Tat':
        if game_round <= 1:
            return 'C'
        
        B = min(5, game_round)
        inp = [1 if memory[-1 - ((B - 1) - i)][ap] == 'D' else 0 for i in range(B)]
        new_noise = sum(inp)
        if new_noise > 0:
            a = 0.25
            new_noise_prob = 1 - ((new_noise - 1) / (B - 1))
            think_memory[115] = (think_memory[115] * (1 - a)) + (new_noise_prob * a)
        
        if rnd.random() <= think_memory[115]:
            return 'C'
        else:
            return memory[-1][ap]
    elif strategy == 'Praedator':
        if (game_round + 1) <= 5:
            if (game_round + 1) == 5:
                if memory[-1][ap] == 'D':
                    think_memory[116][0] = 1
            return memory[-1][ap] if game_round > 0 else 'C'
        elif (game_round + 1) <= 25:
            if memory[-1][ap] == 'D':
                think_memory[116][1] = 1
            return 'C'
        elif (game_round + 1) <= 30:
            if (game_round + 1) > 25:
                if memory[-1][ap] == 'D':
                    think_memory[116][2] = 1
            if (game_round + 1) == 26:
                return 'D'
            else:
                return 'C'
        else:
            another_player_pattern = tuple(think_memory[116])
            
            if another_player_pattern in ((0, 0, 1),(1, 0, 1),(0, 1, 1)):
                self_inp = [memory[-1 - (2 - i)][1-ap] for i in range(3)]
                opp_inp = [memory[-1 - (2 - i)][ap] for i in range(3)]
                grudger_inp = [1 if memory[-1-(9-i)][ap] == 'D' else 0 for i in range(10)]

                grudger_ratio = sum(grudger_inp) / 10

                if ((self_inp == ["D", "C", "D"]) and (opp_inp == ["C", "D", "C"])) or ((opp_inp == ["D", "C", "D"]) and (self_inp == ["C", "D", "C"])):
                    return 'C'
                else:
                    A = 1 - max(min(2 * (grudger_ratio - 0.5),1),0)
                    if rnd.random() <= (0.2 * A):
                        return 'C'
                    else:
                        return memory[-1][ap]
            
            elif another_player_pattern in ((1, 1, 1),(0, 0, 0)):
                return 'D'
            
            else:
                inp = [1 if memory[-1 - ((4) - i)][ap] == 'D' else 0 for i in range(5)]

                t_d = sum(inp) / 5
                t_c = 1 - t_d
                ego = (score[1-ap] - score[ap]) / 15
                if (t_c + max(ego, 0)) > (t_d - min(ego, 0)):
                    return 'C'
                else:
                    return 'D'
    elif strategy == 'Thresholded Trust Bot':
        thresholded_trust_bot_eyes = [('C', 'C') for _ in range(10)] + memory
        a = 0.25
        think_memory[117] += ((a if thresholded_trust_bot_eyes[-1][ap] == 'C' else -a) - ((1 - a) if thresholded_trust_bot_eyes[-1-1][ap] == 'C' else -(1 - a))) * 0.25
        
        think_memory[117] = max(min(think_memory[117], 1), 0)

        input_power = [mth.exp(i + 1) for i in range(10)]
        inp = [1 if thresholded_trust_bot_eyes[-1 - ((9) - i)][ap] == 'C' else 0 for i in range(10)]
        A = sum(input_power[i] * inp[i] for i in range(10)) / sum(input_power)

        if A == (1 - think_memory[117]):
            return thresholded_trust_bot_eyes[-1][ap]
        elif A > (1 - think_memory[117]):
            return 'C'
        else:
            return 'D'
    elif strategy == 'Crow':
        crow_eyes = [('C', 'C') for _ in range(5)] + memory
        if think_memory[118][1] > 0:
            think_memory[118][1] -= 1
            return 'C'
        if crow_eyes[-1][ap] == 'D' or think_memory[118][0] > 0:
            if crow_eyes[-1][ap] == 'D':
                think_memory[118][0] = think_memory[118][2] + 1
                think_memory[118][2] += 1
            think_memory[118][0] -= 1
            if think_memory[118][0] == 0:
                think_memory[118][1] = 2
            return 'D'
        else:
            inp = tuple([crow_eyes[-1 - ((4) - i)][ap] for i in range(5)])
            if inp == ("C", "C", "C", "C", "C"):
                think_memory[118][2] -= 1
                think_memory[118][2] = max(think_memory[118][2],0)
            return 'C'
    elif strategy == 'MENACE / HER':
        if game_round > 0:
            rev = "C" if think_memory[119][1] == "D" else "D"
            if memory[-1][ap] == 'C':
                if memory[-1][1-ap] == 'C':
                    think_memory[119][0][think_memory[119][1]][0] += RPST['R']
                    think_memory[119][0][rev][1] -= RPST['R'] / 2
                else:
                    think_memory[119][0][think_memory[119][1]][1] += RPST['T']
                    think_memory[119][0][rev][0] -= RPST['T'] / 2
            else:
                if memory[-1][1-ap] == 'C':
                    think_memory[119][0][think_memory[119][1]][0] -= 4 * abs(RPST['S'] - RPST['P'])
                    think_memory[119][0][rev][1] += 2 * abs(RPST['S'] - RPST['P'])
                else:
                    think_memory[119][0][think_memory[119][1]][1] += RPST['P']
                    think_memory[119][0][rev][0] -= RPST['P'] / 2

            think_memory[119][0][think_memory[119][1]][0] = min(max(think_memory[119][0][think_memory[119][1]][0], (1 if think_memory[119][0][think_memory[119][1]][1] == 0 else 0)), 100)
            think_memory[119][0][think_memory[119][1]][1] = min(max(think_memory[119][0][think_memory[119][1]][1], (1 if think_memory[119][0][think_memory[119][1]][0] == 0 else 0)), 100)

            think_memory[119][0][rev][0] = min(max(think_memory[119][0][rev][0], (1 if think_memory[119][0][rev][1] == 0 else 0)), 100)
            think_memory[119][0][rev][1] = min(max(think_memory[119][0][rev][1], (1 if think_memory[119][0][rev][0] == 0 else 0)), 100)

        inp = memory[-1][ap] if game_round > 0 else 'C'
        act = think_memory[119][0][inp][0] / (think_memory[119][0][inp][0] + think_memory[119][0][inp][1])
        think_memory[119][1] = inp

        if rnd.random() <= act:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Forgiving Fate':
        think_memory[120] *= 0.75
        if ((memory[-1][ap] == 'C') if game_round > 0 else True):
            think_memory[120] += 0.25

        def C_function(x):
            return -(x - 1)

        ST = (memory[-1][ap] == 'D') if game_round > 0 else False
        A = 1 - (C_function(think_memory[120]) * ST)

        if rnd.random() <= A:
            return 'C'
        else:
            return 'D'
    elif strategy == 'King Of 48 Laws Of Power':
        # law 14: Pose as a Friend, Work as a Spy
        # law 23: Concentrate Your Forces
        # law 24: Play the Perfect Courtier
        # law 26: Keep Your Hands Clean
        # law 30: Make Your Accomplishments Seem Effortless
        # law 34: Be Royal in Your Own Fashion: Act Like a King to Be Treated Like One
        # law 38: Think As You Like, But Act Like Others
        # law 41: Avoid Stepping into a Great Man’s Shoes
        # law 42: Strike the Shepherd and the Sheep Will Scatter
        # law 46: Never Appear Too Perfect

        king_of_48_laws_of_power_eyes = [('C', 'C') for _ in range(10)] + memory
        
        cond = king_of_48_laws_of_power_eyes[-2][1-ap] + king_of_48_laws_of_power_eyes[-2][ap]
        think_memory[126][8][cond][0] += (king_of_48_laws_of_power_eyes[-1][ap] == 'C')
        think_memory[126][8][cond][1] += 1

        if king_of_48_laws_of_power_eyes[-1][ap] == 'D': # law 7: get others do the work for you
            think_memory[126][2] += 1

            if (king_of_48_laws_of_power_eyes[-1-1][1-ap] == 'C') and (king_of_48_laws_of_power_eyes[-1-2][1-ap] == 'C'):
                think_memory[126][3] += 1

        if (game_round + 1) <= 7: 
            return king_of_48_laws_of_power_eyes[-1][ap]
        elif (game_round + 1) <= 15: # law 8: make other people come to you, use bait if necessary
            return 'C'

        p_d = think_memory[126][2] / (game_round + 1)
        p_c = 1 - p_d

        input_power = [0.1 * (i + 1) for i in range(10)] # rule 9: win through your actions, not through argument
        inp = [1 if king_of_48_laws_of_power_eyes[-1 - ((9) - i)][ap] == 'C' else 0 for i in range(10)]
        A1 = sum(input_power[i] * inp[i] for i in range(10)) / sum(input_power)

        A2 = think_memory[126][3] / (game_round + 1) # law 10: Infection: Avoid the Unhappy and Unlucky

        a3 = 0.25

        SH = (king_of_48_laws_of_power_eyes[-1][ap] == 'C')
        ST = 1 - SH

        a4 = 0.25
        A3 = (max(((A1 * (1 - a3)) + (SH * a3)) - (((1 - A1) * (1 - a3)) + (ST * a3)),0) * (1 - a4)) + (think_memory[126][7] * a4) # law 29: Plan All the Way to the End
        think_memory[126][7] = A3 # law 45: Preach the Need for Change, But Never Reform Too Much at Once
        # law 48: Assume Formlessness

        A6 = [
            (think_memory[126][8]['CC'][0]) / (think_memory[126][8]['CC'][1]) if think_memory[126][8]['CC'][1] > 0 else 0.5,
            (think_memory[126][8]['CD'][0]) / (think_memory[126][8]['CD'][1]) if think_memory[126][8]['CD'][1] > 0 else 0.5,
            (think_memory[126][8]['DC'][0]) / (think_memory[126][8]['DC'][1]) if think_memory[126][8]['DC'][1] > 0 else 0.5,
            (think_memory[126][8]['DD'][0]) / (think_memory[126][8]['DD'][1]) if think_memory[126][8]['DD'][1] > 0 else 0.5,
        ]

        if rnd.random() <= ((1 - A1) * (1 - A6[3]) * A6[1]): # law 22: Use the Surrender Tactic: Transform Weakness into Power
            think_memory[126][9] = 2

        if think_memory[126][9] > 0:
            think_memory[126][9] -= 1
            next_chat[0] = 'I surrender'
            return 'C'


        a2 = 0.5
        if rnd.random() <= ((1 - abs((sum(A6) / 2) - 1)) * (1 - abs(p_c - p_d)) * (1 - abs(A1 - (1 - A1)))): # law 15: Crush Your Enemy Totally
            return 'D'

        if A2 >= 0.5:
            if rnd.random() <= min(A2, 1):
                return 'D'
        
        A5 = A1 if A1 >= 0.75 else 0 # law 35: Master the Art of Timing
        if rnd.random() <= (0.25 * (p_d * A5) * A6[2]): # law 17: Keep Others in Suspended Terror: Cultivate an Air of Unpredictability
            return 'D'
            
        if ((score[ap] - score[1-ap]) / 25) > 0.5: # law 16: Use Absence to Increase Respect and Honor
            think_memory[126][5] = 2 # law 18: Do Not Build Fortresses to Protect Yourself – Isolation is Dangerous 

        if think_memory[126][5] > 0:
            think_memory[126][5] -= 1
            return 'D' # law 39: Stir Up Waters to Catch Fish 

        if ((score[1-ap] - score[ap]) / 25) > 0.5: # law 11: Learn to Keep People Dependent on You
            next_chat[0] = 'I am king'
            return 'C' # law 47: In Victory, Learn When to Stop

        next_chat[0] = 'Hello, I\'am your friend' # law 4: always say less than necessary
        
        E = 10

        A4 = (0.25 * ((1 - A1) * p_c)) # law 33: Discover Each Man’s Thumbscrew 
        if rnd.random() <= A4: # law 12: Use Selective Honesty and Generosity to Disarm Your Victim
            think_memory[126][4] = E
            return 'C'
        
        if think_memory[126][4] > 0:
            if think_memory[126][4] > (E // 2):
                if think_memory[126][4] >= ((E // 2) + 2): # law 21: Play a Sucker to Catch a Sucker – Seem Dumber Than Your Mark
                    think_memory[126][6] = 0
                    return 'C'
                    # law 32: Play to People’s Fantasies
                else:
                    if king_of_48_laws_of_power_eyes[-1][ap] == 'D':
                        think_memory[126][6] = 1
                    return king_of_48_laws_of_power_eyes[-1][ap]
            else:
                if think_memory[126][6] == 1: # law 31: Control the Options: Get Others to Play with the Cards You Deal
                    return 'D' # law 36: Disdain Things You Cannot Have: Ignoring Them is the Best Revenge
                else:
                    return king_of_48_laws_of_power_eyes[-1][ap]
            
            think_memory[126][4] -= 1
        
        if rnd.random() <= ((0.25 * p_d) * A6[2]): # law 3: conceal your intentions
            if score[1-ap] < score[ap]:
                if think_memory[126][0] == 0:
                    think_memory[126][0] = 1
                    return 'D'

        if king_of_48_laws_of_power_eyes[-1-1][ap] == 'D': # law 6: court attention at all costs
            if rnd.random() <= ((0.25 * p_d) * A6[2]):
                if think_memory[126][1] == 0:
                    think_memory[126][1] = 1
                    return 'D'
        
        think_memory[126][0] = 0
        think_memory[126][1] = 0
        A7 = max(score[ap] - score[1-ap], 0)
        if king_of_48_laws_of_power_eyes[-1][ap] == 'D': # law 2: never put too much trust in friends, learn how to use enemy
            a1 = 0.1
            if rnd.random() <= (A3 * ((p_c * (1 - a1)) +  (A1 * a1)) * (A7 / 25)): # law 1: never outshine the master
                return 'C' # law 28: Enter Action with Boldness
            else:
                return 'D'
                # law 20: Do Not Commit to Anyone
                # law 44: Disarm and Infuriate with the Mirror Effect
        else:
            return 'C'
            # law 13: When Asking for Help, Appeal to People’s Self-Interest, Never to Their Mercy or Gratitude
            # law 19: Know Who You Are Dealing with – Do Not Offend the Wrong Person
            # law 25: Re-create Yourself
            # law 27: Play on People’s Need to Believe to Create a Cultlike Following
            # law 37: Create Compelling Spectacles
            # law 40: Despise the Free Lunch
            # law 43: Work on the Hearts and Minds of Others
    elif strategy == 'Pavolovo':
        if (game_round + 1) <= 2:
            if not game_round:
                think_memory[127][0] += 1
                think_memory[127][2] = 1
            return ('D','C')[(game_round + 1)-1]

        pavolovo_eyes = [('C', 'C') for _ in range(5)] + memory
        
        if think_memory[127][2] == 2:
            if pavolovo_eyes[-1][ap] == pavolovo_eyes[-1][1-ap]:
                if (pavolovo_eyes[-1][ap] == 'D') and (pavolovo_eyes[-1-1][ap] == 'D'):
                    return 'D'
                return 'C'
            else:
                return 'D'
        
        if (pavolovo_eyes[-1-1][1-ap] == 'D') or (think_memory[127][2] == 1):
            inp = [(pavolovo_eyes[-1 - ((4) - i)][ap] == 'D') for i in range(5)]
            if sum(inp) == 5:
                think_memory[127][2] = 2
                return 'D'
            if pavolovo_eyes[-1][ap] == 'D':
                think_memory[127][1] += 1
                think_memory[127][2] = 1
                return 'C'
            else:
                think_memory[127][2] = 0
        
        p_retalitory = think_memory[127][1] / think_memory[127][0]

        if pavolovo_eyes[-1][ap] == pavolovo_eyes[-1][1-ap]:
            if (pavolovo_eyes[-1][ap] == 'D') and (pavolovo_eyes[-1-1][ap] == 'D'):
                return 'D'
            else:
                if rnd.random() <= (0.1 * (1 - p_retalitory)):
                    think_memory[127][0] += 1
                    return 'D'
                else:
                    return 'C'
        else:
            think_memory[127][0] += 1
            return 'D'
    elif strategy == 'Decay Given':
        decay_given_eyes = [('C', 'C') for _ in range(10)] + memory
        input_power = [0.1 * (i + 1) for i in range(10)]
        opp_inp = [1 if decay_given_eyes[-1 - ((9) - i)][ap] == 'C' else 0 for i in range(10)]
        self_inp = [1 if decay_given_eyes[-1 - ((9) - i)][1-ap] == 'C' else 0 for i in range(10)]

        p_c_decay_given_c = sum(input_power[i] * opp_inp[i] * self_inp[i] for i in range(10)) / (sum(input_power[j] * self_inp[j] for j in range(10)) + 1e-9)
        p_c_decay_given_d = sum(input_power[i] * opp_inp[i] * (1 - self_inp[i]) for i in range(10)) / (sum(input_power[j] * (1 - self_inp[j]) for j in range(10)) + 1e-9)
        p_d_decay_given_c = sum(input_power[i] * (1 - opp_inp[i]) * self_inp[i] for i in range(10)) / (sum(input_power[j] * self_inp[j] for j in range(10)) + 1e-9)
        p_d_decay_given_d = sum(input_power[i] * (1 - opp_inp[i]) * (1 - self_inp[i]) for i in range(10)) / (sum(input_power[j] * (1 - self_inp[j]) for j in range(10)) + 1e-9)

        if decay_given_eyes[-1][1-ap] == 'C':
            A = p_c_decay_given_c / (p_c_decay_given_c + p_d_decay_given_c + 1e-9)
            if rnd.random() <= A:
                return 'C'
            else:
                return 'D'
        else:
            A = p_c_decay_given_d / (p_c_decay_given_d + p_d_decay_given_d + 1e-9)
            if rnd.random() <= A:
                return 'D'
            else:
                return 'C'
    elif strategy == 'Eric':
        eric_eyes = [('C', 'C')] + memory
        if (game_round + 1) <= 5:
            return eric_eyes[-1][ap]
        
        if think_memory[130] == 1:
            return eric_eyes[-1][ap]
        
        inp = [1 if eric_eyes[-1 - ((5) - i)][ap] == 'D' else 0 for i in range(6)]
        
        if (eric_eyes[-1][ap] == 'D') and (eric_eyes[-1-1][ap] == 'D'):
            return 'D'
        else:
            if sum(inp) >= 3:
                think_memory[130] = 1
                return 'D'
            else:
                return 'C'
    elif strategy == 'Go By Minority':
        if not game_round:
            return 'C'
        if memory[-1][ap] == 'C':
            think_memory[132][0] += 1
        else:
            think_memory[132][1] += 1
        t_c = think_memory[132][0]
        t_d = think_memory[132][1]
        if t_c < t_d:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Score Equalizer':
        E_sh = (abs((score[1-ap] + RPST['R']) - (score[ap] + RPST['R'])) + abs((score[1-ap] + RPST['S']) - (score[ap] + RPST['T']))) / 2
        E_st = (abs((score[1-ap] + RPST['T']) - (score[ap] + RPST['S'])) + abs((score[1-ap] + RPST['P']) - (score[ap] + RPST['P']))) / 2

        if E_sh <= E_st:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Boltzmann Brain':
        def clamp(x):
            return max(min(x,1),0)
        
        brain = think_memory[137][0]

        if game_round > 0:
            M = 0.3
            if memory[-1][ap] == 'C':
                boltzman_cond = 'win'
            else:
                boltzman_cond = 'lose'
            boltzman_update_list = []
            for i in range(len(brain)):
                if think_memory[137][1][i] == (memory[-1][1-ap]):
                    boltzman_update_list.append(i)

            if boltzman_cond == 'win':
                for i in boltzman_update_list:
                    brain.append(
                        {
                        
                        ("C", "C", "T", "F"):clamp(brain[i][("C", "C", "T", "F")] + rnd.uniform(-M,M)),
                        ("D", "C", "T", "F"):clamp(brain[i][("D", "C", "T", "F")] + rnd.uniform(-M,M)),
                        ("C", "D", "T", "F"):clamp(brain[i][("C", "D", "T", "F")] + rnd.uniform(-M,M)),
                        ("D", "D", "T", "F"):clamp(brain[i][("D", "D", "T", "F")] + rnd.uniform(-M,M)),

                        ("C", "C", "F", "F"):clamp(brain[i][("C", "C", "F", "F")] + rnd.uniform(-M,M)),
                        ("D", "C", "F", "F"):clamp(brain[i][("D", "C", "F", "F")] + rnd.uniform(-M,M)),
                        ("C", "D", "F", "F"):clamp(brain[i][("C", "D", "F", "F")] + rnd.uniform(-M,M)),
                        ("D", "D", "F", "F"):clamp(brain[i][("D", "D", "F", "F")] + rnd.uniform(-M,M)),

                        ("C", "C", "F", "T"):clamp(brain[i][("C", "C", "F", "T")] + rnd.uniform(-M,M)),
                        ("D", "C", "F", "T"):clamp(brain[i][("D", "C", "F", "T")] + rnd.uniform(-M,M)),
                        ("C", "D", "F", "T"):clamp(brain[i][("C", "D", "F", "T")] + rnd.uniform(-M,M)),
                        ("D", "D", "F", "T"):clamp(brain[i][("D", "D", "F", "T")] + rnd.uniform(-M,M))
                        }
                    )
                    think_memory[137][1].append("C")
            else:
                for i in range(len(boltzman_update_list)):
                    brain.pop(boltzman_update_list[len(boltzman_update_list) - (i + 1)])
                    think_memory[137][1].pop(boltzman_update_list[len(boltzman_update_list) - (i + 1)])

        if len(brain) > 32:
            while len(brain) > 32:
                A = rnd.randint(1,len(brain)) - 1
                brain.pop(A)
                think_memory[137][1].pop(A)
        
        if len(brain) < 1:
            brain.append(
                {
                    
                    ("C", "C", "T", "F"):1,
                    ("D", "C", "T", "F"):1,
                    ("C", "D", "T", "F"):1,
                    ("D", "D", "T", "F"):1,

                    ("C", "C", "F", "F"):1,
                    ("D", "C", "F", "F"):0,
                    ("C", "D", "F", "F"):1,
                    ("D", "D", "F", "F"):0,

                    ("C", "C", "F", "T"):0,
                    ("D", "C", "F", "T"):0,
                    ("C", "D", "F", "T"):0,
                    ("D", "D", "F", "T"):0
                }
            )
            think_memory[137][1].append("C")

        opp = memory[-1][ap] if game_round > 0 else 'C'
        self = memory[-1][1-ap] if game_round > 0 else 'C'
        boltzman_win = "T" if score[1-ap] > score[ap] else "F"
        boltzman_lose = "T" if score[ap] > score[1-ap] else "F"

        boltzmann_mini_strat_choice = []
        for i in range(len(brain)):
            if rnd.random() <= brain[i][(opp, self, boltzman_win, boltzman_lose)]:
                boltzmann_mini_strat_choice.append(1)
                think_memory[137][1][i] = "C"
            else:
                boltzmann_mini_strat_choice.append(-1)
                think_memory[137][1][i] = "D"

        A = sum(boltzmann_mini_strat_choice)

        if A <= 0:
            return 'D'
        else:
            return 'C'
    elif strategy == 'Smooth Tit For Tat':
        a = 0.25
        think_memory[138] *= (1 - a)
        if ((memory[-1][ap] == 'D') if game_round > 0 else False):
            think_memory[138] += a

        if rnd.random() <= think_memory[138]:
            return 'D'
        else:
            return 'C'
    elif strategy == 'Konflikt Green':
        if game_round > 0:
            if memory[-1][ap] == 'D':
                think_memory[99] += 1
        
        if rnd.random() <= (think_memory[99] / (game_round + 1)):
            return 'D'
        else:
            return 'C' if rnd.random() <= 0.25 else (memory[-1][ap] if game_round > 0 else 'C')
    elif strategy == 'Konflikt Blue':
        if game_round > 0:
            if memory[-1][ap] == 'D':
                think_memory[133] += 1
        
        if rnd.random() <= ((think_memory[133] / (game_round + 1)) + 0.1):
            return 'D'
        else:
            return 'C' if rnd.random() <= 0.125 else (memory[-1][ap] if game_round > 0 else 'C')
    elif strategy == 'Contrite Grudger':
        if think_memory[147][1] == 0:
            if ((memory[-1][1-ap] == 'D') if game_round > 0 else False) and ((memory[-2][ap] == 'C') if game_round > 0 else True):
                think_memory[147][0] += 1
            
            if ((memory[-1][ap] == 'D') if game_round > 0 else False):
                if think_memory[147][0] > 0:
                    think_memory[147][0] -= 1
                    return 'C'
                else:
                    think_memory[147][1] = 1
                    return 'D'
            else:
                return 'C'
        else:
            return 'D'
    elif strategy == 'Passive Tit For Tat':
        if not game_round:
            return 'C'
        elif memory[-1][1-ap] == 'D':
            return 'C'
        elif (memory[-1][ap] == 'C') and ((memory[-1-1][ap] == 'C') if (game_round + 1) > 2 else True):
            return 'C'
        else:
            return 'D'
    elif strategy == 'Beta Tester':
        beta_tester_random_param = 0.2
        if rnd.random() <= beta_tester_random_param:
            think_memory[29] = rnd.randint(0, 6)
        if think_memory[29] == 0:
            return 'C'
        elif think_memory[29] == 1:
            return 'D'
        elif think_memory[29] == 2:
            return rnd.choice(['C', 'D'])
        elif think_memory[29] == 3:
            if not game_round:
                return 'D'
            return 'D' if memory[-1][1-ap] == 'C' else 'C'
        elif think_memory[29] == 4:
            if not game_round:
                return 'C'
            return 'C' if memory[-1][ap] == memory[-1][1-ap] else 'D'
        elif think_memory[29] == 5:
            if not game_round:
                return 'D'
            return 'D' if memory[-1][ap] == 'C' else 'C'
        else:
            if not game_round:
                return 'C'
            return memory[-1][ap]
    elif strategy == 'Buzzer':
        if game_round <= (tournament_avg_last_round // 5):
            next_chat[0] = 'BUZZER!!!'
            if game_round:
                if memory[-1][ap] == 'D' or chats[ap] == 'BUZZER!!!':
                    think_memory[173] = True
            return 'C'
        else:
            if think_memory[173]:
                return 'C'
            return 'D'
    elif strategy == 'Man In The Middle':
        if (game_round + 1) <= ((190 / 200) * tournament_avg_last_round):
            if not game_round:
                return 'C'
            return 'C' if rnd.random() <= 0.1 else memory[-1][ap]
        else:
            return 'D'
    elif strategy == 'DDoS':
        if (game_round + 1) <= (tournament_avg_last_round / 2):
            if not game_round:
                return 'C'
            return 'C' if rnd.random() <= 0.1 else memory[-1][ap]
        
        inp = [memory[-1 - i][ap] for i in range(10)]
        if "D" in inp:
            for i in range(5):
                DDoS_n = (i + 1)
                is_cycle = True
                DDoS_list = inp[:(DDoS_n + 1)]
                DDoS_index = len(inp) % DDoS_n
                for j in range(len(inp)):
                    if inp[j] != DDoS_list[j % DDoS_n]:
                        is_cycle = False

                if is_cycle:
                    if DDoS_list[DDoS_index + 1] == "D":
                        return 'D'
                    else:
                        return rnd.choice(['C', 'D'])
        
        return 'C' if rnd.random() <= 0.1 else memory[-1][ap]
    elif strategy == 'Social Engineering':
        if (game_round + 1) <= 5:
            return ('D', 'C', 'D', 'D', 'C')[game_round]
        elif (game_round + 1) == 6:
            inp = [memory[-1 - (4 - i)][ap] for i in range(5)]
            if tuple(inp) == ('D', 'C', 'D', 'D', 'C'):
                think_memory[174] = 1
            return 'C'
        else:
            if think_memory[174] == 1:
                return 'C'
            else:
                return memory[-1][ap]
    elif strategy == 'Zero Day':
        A = 1
        if (game_round + 1) <= (A + 1):
            if not game_round:
                return 'C'
            return memory[-1][ap]
        elif (game_round + 1) <= ((190 / 200) * tournament_avg_last_round):
            
            inp = ''
            for i in range(A):
                if memory[-1 - (A - i)][ap] == 'C':
                    inp += 'C'
                else:
                    inp += 'D'
            
                if memory[-1 - (A - i)][1-ap] == 'C':
                    inp += 'C'
                else:
                    inp += 'D'

            if think_memory[175][0].get(inp) == None:
                think_memory[175][0][inp] = [(memory[-1][ap] == 'C'), 1]
            else:
                think_memory[175][0][inp][0] += (memory[-1][ap] == 'C')
                think_memory[175][0][inp][1] += 1

            cond = ''
            for i in range(A):
                if memory[-1 - (A - i)][ap] == 'C':
                    cond += 'C'
                else:
                    cond += 'D'
                        
                if memory[-1 - (A - i)][1-ap] == 'C':
                    cond += 'C'
                else:
                    cond += 'D'

            if cond != 'CC':
                B = 1 - (think_memory[175][0][cond][0] / think_memory[175][0][cond][1] if think_memory[175][0].get(cond) != None else 0.5)
            else:
                B = think_memory[175][0]['CC'][0] / think_memory[175][0]['CC'][1] if think_memory[175][0].get('CC') != None else 0.5

            return memory[-1][ap] if rnd.random() <= B else 'D'
        else:
            return 'D'
    elif strategy == 'Eliza / Therapist':
        if ((memory[-2][ap] == 'C') if (game_round + 1) > 2 else True) and ((memory[-1][1-ap] == 'D') if game_round > 0 else False):
            think_memory[177][0] += 1

        opp_chat = chats[ap].lower().strip().rstrip('?.!')
        if opp_chat == '':
            opp_chat = 'we are friends'
            if game_round > 0:
                opp_chat = 'we are friends' if memory[-1][ap] == 'C' else 'i hate you'

        opp_chat = opp_chat.replace('were', 'we are')

        keyword_map = think_memory[177][1]

        ganti_kata = {
            ' me ': ' you ',
            ' you ': ' me ',
            ' my ': ' your ',
            ' your ': ' my ',
        }

        
        pola = re.compile("|".join(sorted(map(re.escape, ganti_kata.keys()), key=len, reverse=True)))

        the_key_word = None
        
        for kw in sorted(keyword_map.keys(), key=len, reverse=True):
            if kw in opp_chat:
                the_key_word = kw
                break

        if the_key_word:
            text_template = rnd.choice(keyword_map[the_key_word])

            if '[REPLACE]' in text_template:
                A = (' ' + opp_chat + ' ').split(the_key_word, 1)[1]
                hasil = pola.sub(lambda m: ganti_kata[m.group(0)], A)
                next_chat[0] = f"{text_template.replace('[REPLACE]', hasil[1:-1])}"
            else:
                next_chat[0] = f"{text_template}"
        else:
            next_chat[0] = f"{rnd.choice(['What do you think about that?', 'Could you tell me more about it?', 'Hmm, interesting... go on?', 'How do you feel about that?'])}"

        if think_memory[177][0] > 0:
            think_memory[177][0] -= 1
            return 'C'
        else:
            if not game_round:
                return 'C'
            return memory[-1][ap] if rnd.random() > 0.1 else 'C'
    elif strategy == 'Verity':
        def clamp(x):
            return max(min(x, 1), 0)
        
        verity_eyes = [('C', 'C') for _ in range(10)] + memory
        
        if think_memory[178][2] > 0:
            think_memory[178][2] -= 1
            return 'C'
        
        if verity_eyes[-1][ap] == 'D':
            think_memory[178][0] -= 30
            if think_memory[178][1] == 1:
                think_memory[178][1] = 2
        else:
            think_memory[178][0] += 5

        verity_cond = (verity_eyes[-1-1][ap], verity_eyes[-1-1][1-ap])

        think_memory[178][6][verity_cond][0] += (verity_eyes[-1][ap] == 'C')
        think_memory[178][6][verity_cond][1] += 1

        if (think_memory[178][0] < 30) or think_memory[178][3]:
            if not think_memory[178][3]:
                think_memory[178][3] = True
                think_memory[178][4] = (game_round + 1)
                think_memory[178][5] = 20

            if think_memory[178][1] < 3:
                think_memory[178][1] = 3
            
            if (think_memory[178][1] == 4) and (think_memory[178][0] >= 30):
                think_memory[178][3] = False
                think_memory[178][2] = 1
                think_memory[178][1] = 1
                return 'C'

            elif (((game_round + 1) - think_memory[178][4]) / think_memory[178][5]) >= 0.75:
                think_memory[178][1] = 4
        
        if think_memory[178][1] == 1:
            return 'C'
        elif think_memory[178][1] == 2:
            cyclic_inp = [verity_eyes[-1 - i][ap] for i in range(10)]

            is_cycle = False
            if "D" in cyclic_inp:
                if cycle_detec(ap, min_ = 1, max_ = 5, offset = (game_round - (game_round % 10))):
                    is_cycle = True
                    
            if (verity_eyes[-1][ap] == 'C') and (not is_cycle):
                think_memory[178][2] = 1
                think_memory[178][1] = 1
                return 'C'
            
            return 'D'
        elif think_memory[178][1] == 3:
            mutual_cooperate_prob = ((think_memory[178][6][("C", "C")][0] + 1) / (think_memory[178][6][("C", "C")][1] + 2))
            exploitation_prob = ((think_memory[178][6][("C", "D")][0] + 1) / (think_memory[178][6][("C", "D")][1] + 2))
            exploited_prob = ((think_memory[178][6][("D", "C")][0] + 1) / (think_memory[178][6][("D", "C")][1] + 2))
            mutual_defect_prob = ((think_memory[178][6][("D", "D")][0] + 1) / (think_memory[178][6][("D", "D")][1] + 2))

            verity_category = [
                [ 
                    1,
                    0,
                    1,
                    0
                ],
                [ 
                    1,
                    0,
                    0,
                    1
                ],
                [ 
                    0,
                    0,
                    1,
                    1
                ]
            ]

            verity_enemy_differents = [0 for _ in range(len(verity_category))]
            for i in range(len(verity_category)):
                verity_enemy_differents[i] = (
                    abs(mutual_cooperate_prob - verity_category[i][0]) +
                    abs(exploitation_prob - verity_category[i][1]) +
                    abs(exploited_prob - verity_category[i][2]) +
                    abs(mutual_defect_prob - verity_category[i][3])
                ) / 4
            
            if min(verity_enemy_differents) <= 0.25:
                most_similar = [['Tit For Tat', 'Pavlov', 'Alternator'][j] for j in range(len(verity_enemy_differents)) if verity_enemy_differents[j] == min(verity_enemy_differents)]

                enemy_category = rnd.choice(most_similar)

                if enemy_category == 'Tit For Tat':
                    return 'C'
                elif enemy_category == 'Pavlov':
                    return ['D', 'D', 'C'][(game_round + 1) % 3]
                else:
                    return 'D' if verity_eyes[-1][ap] == 'C' else rnd.choice(['C', 'D'])
                
            if (verity_eyes[-1-1][ap], verity_eyes[-1][ap]) == ('D', 'D'):
                return 'D'

            if verity_eyes[-1][ap] == verity_eyes[-1][1-ap]:
                return 'D' if rnd.random() <= exploitation_prob else 'C'
            else:
                return 'D'
        else:
            return 'D'
    elif strategy == 'Continuous Q-Learner Max':
        brain = think_memory[179]
        payoff = {
            'CC': RPST['R'],
            'CD': RPST['S'],
            'DC': RPST['T'],
            'DD': RPST['P']
        }

        reward = avg_RPST
        if game_round > 0:
            learning_rate = 0.1
            brain[0] = ((1 - learning_rate) * brain[0]) + (learning_rate * (memory[-1][ap] == 'C'))
            brain[1] = ((1 - learning_rate) * brain[1]) + (learning_rate * (memory[-1][ap] == 'D'))

            reward = payoff[memory[-1][1-ap] + memory[-1][ap]]
        X_sh = (brain[0] * payoff['CC']) + (brain[1] * payoff['CD'])
        X_st = (brain[0] * payoff['DC']) + (brain[1] * payoff['DD'])

        Action = reward - max(X_sh, X_st)
        kappa = 2.0
        A = 1 / (1 + mth.exp(-Action * kappa))
        if rnd.random() <= A:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Double Q-Learner':
        A_or_B = rnd.randint(0, 1)
        brain = think_memory[180][A_or_B][0]
        another_brain = think_memory[180][1 - A_or_B][0]
        another_factor = think_memory[180][A_or_B][1]
        second_another_factor = think_memory[180][1 - A_or_B][1]

        A = min(12, game_round) - 1
        opp_inp = tuple(['C' for _ in range(12 - min(12, game_round))] + [memory[-1 - (A - i)][ap] for i in range(min(12, game_round))])
        self_inp = tuple(['C' for _ in range(12 - min(12, game_round))] + [memory[-1 - (A - i)][1-ap] for i in range(min(12, game_round))])
        inp = (opp_inp, self_inp)

        if brain.get(inp) == None:
            brain[inp] = [0,0]
        if another_brain.get(inp) == None:
            another_brain[inp] = [0,0]

        if game_round > 0 and another_factor[0][1] != 0.5:
            reward = 0
            if memory[-1][ap] == 'C':
                if memory[-1][1-ap] == 'C':
                    reward = RPST['R']
                else:
                    reward = RPST['T']
            else:
                if memory[-1][1-ap] == 'C':
                    reward = RPST['S']
                else:
                    reward = RPST['P']

            c_factor = another_brain[inp][0]
            d_factor = another_brain[inp][1]
            
            a = 0.5
            gamma = 0.5
            Q_another = max(c_factor, d_factor)
            Q_old = brain[another_factor[0][0]][another_factor[0][1]]
            Q_new = Q_old + (a * (reward + ((gamma * Q_another) - Q_old)))
            brain[another_factor[0][0]][another_factor[0][1]] = Q_new

        if rnd.random() <= another_factor[1]:
            Action = rnd.choice(['C', 'D'])
            another_factor[0] = [inp, (Action == 'D')]
            return Action
        decay = 1
        min_epsilon = 1
        another_factor[1] = max(another_factor[1] * decay, min_epsilon)

        if brain[inp][0] == brain[inp][1]:
            Action = rnd.choice(['C', 'D'])
            another_factor[0] = [inp, (Action == 'D')]
            return Action
        best_choice = max(brain[inp])
        if best_choice == brain[inp][0]:
            another_factor[0] = [inp, 0]
            return 'C'
        else:
            another_factor[0] = [inp, 1]
            return 'D'
    elif strategy == 'Falsity':
        def clamp(x):
            return max(min(x, 1), 0)

        if think_memory[181][3] > 0:
            think_memory[181][3] -= 1
            return 'C'

        if game_round > 0:
            if memory[-1][ap] == 'D':
                think_memory[181][0] -= 30
            else:
                think_memory[181][0] += 5
        else:
            think_memory[181][0] += 5
        
        if (think_memory[181][0] < 30) or think_memory[181][2]:
            if not think_memory[181][2]:
                think_memory[181][2] = True

            if think_memory[181][1] < 2:
                think_memory[181][1] = 2
                    
            if (think_memory[181][1] == 2) and (think_memory[181][0] >= 30):
                think_memory[181][2] = False
                think_memory[181][3] = 1
                think_memory[181][1] = 1
                return 'C'
        
        if think_memory[181][1] == 1:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Cruelity':
        def clamp(x):
            return max(min(x, 1), 0)

        cruelity_eyes = [('C', 'C') for _ in range(10)] + memory
        
        if think_memory[182][4] > 0:
            think_memory[182][4] -= 1
            return 'C'
                
        if cruelity_eyes[-1][ap] == 'D':
            think_memory[182][0] -= 40
        else:
            think_memory[182][0] += 5

        cruelity_cond = (cruelity_eyes[-1-1][ap], cruelity_eyes[-1-1][1-ap])
        
        think_memory[182][5][cruelity_cond][0] += (cruelity_eyes[-1][ap] == 'C')
        think_memory[182][5][cruelity_cond][1] += 1
                
        if (think_memory[182][0] < 30) or think_memory[182][2]:
            if not think_memory[182][2]:
                think_memory[182][2] = True
                think_memory[182][3] = (game_round + 1)

            if think_memory[182][1] < 2:
                think_memory[182][1] = 2

            if (think_memory[182][1] == 3) and (think_memory[182][0] >= 30):
                think_memory[182][2] = False
                think_memory[182][4] = 1
                think_memory[182][1] = 1
                return 'C'
        
            elif (((game_round + 1) - think_memory[182][3]) / 20) >= 0.75:
                think_memory[182][1] = 3
                
        if think_memory[182][1] == 1:
            return 'C'
        elif think_memory[182][1] == 2:
            cyclic_inp = [cruelity_eyes[-1 - i][ap] for i in range(10)]
                
            is_cycle = False
            if "D" in cyclic_inp:
                if cycle_detec(ap, min_ = 1, max_ = 5, offset = (game_round - (game_round % 10))):
                    is_cycle = True

            if is_cycle:
                return 'D'

            mutual_cooperate_prob = ((think_memory[182][5][("C", "C")][0] + 1) / (think_memory[182][5][("C", "C")][1] + 2))
            exploitation_prob = ((think_memory[182][5][("C", "D")][0] + 1) / (think_memory[182][5][("C", "D")][1] + 2))
            exploited_prob = ((think_memory[182][5][("D", "C")][0] + 1) / (think_memory[182][5][("D", "C")][1] + 2))
            mutual_defect_prob = ((think_memory[182][5][("D", "D")][0] + 1) / (think_memory[182][5][("D", "D")][1] + 2))
            
            cruelity_category = [
                [ 
                    1,
                    0,
                    1,
                    0
                ],
                [ 
                    1,
                    0,
                    0,
                    1
                ],
                [ 
                    0,
                    0,
                    1,
                    1
                ]
            ]
            
            cruelity_enemy_differents = [0 for _ in range(len(cruelity_category))]
            for i in range(len(cruelity_category)):
                cruelity_enemy_differents[i] = (
                    abs(mutual_cooperate_prob - cruelity_category[i][0]) +
                    abs(exploitation_prob - cruelity_category[i][1]) +
                    abs(exploited_prob - cruelity_category[i][2]) +
                    abs(mutual_defect_prob - cruelity_category[i][3])
                ) / 4
                        
            if min(cruelity_enemy_differents) <= 0.25:
                most_similar = [['Tit For Tat', 'Pavlov', 'Alternator'][j] for j in range(len(cruelity_enemy_differents)) if cruelity_enemy_differents[j] == min(cruelity_enemy_differents)]
            
                enemy_category = rnd.choice(most_similar)
            
                if enemy_category == 'Tit For Tat':
                    return 'C'
                elif enemy_category == 'Pavlov':
                    return ['D', 'D', 'C'][(game_round + 1) % 3]
                else:
                    return 'D' if cruelity_eyes[-1][ap] == 'C' else rnd.choice(['C', 'D'])
            
            A = ((think_memory[182][5][cruelity_cond][0] + 1) / (think_memory[182][5][cruelity_cond][1] + 2))
            if rnd.random() <= A:
                return 'D' if rnd.random() <= ((think_memory[182][5][("C", "D")][0] + 1) / (think_memory[182][5][("C", "D")][1] + 2)) else 'C'
            else:
                return 'D'
        else:
            return 'D'
    elif strategy == 'Curiosity':
        def clamp(x):
            return max(min(x, 1), 0)

        curiosity_eyes = [('C', 'C') for _ in range(10)] + memory
            
        if think_memory[183][4] > 0:
            think_memory[183][4] -= 1
            return 'C'
                    
        if curiosity_eyes[-1][ap] == 'D':
            think_memory[183][0] -= 30
        else:
            think_memory[183][0] += 5
    
        curiosity_cond = (curiosity_eyes[-1-1][ap], curiosity_eyes[-1-1][1-ap])
            
        think_memory[183][5][curiosity_cond][0] += (curiosity_eyes[-1][ap] == 'C')
        think_memory[183][5][curiosity_cond][1] += 1
                    
        if (think_memory[183][0] < 30) or think_memory[183][2]:
            if not think_memory[183][2]:
                think_memory[183][2] = True
                think_memory[183][3] = (game_round + 1)
    
            if think_memory[183][1] < 2:
                think_memory[183][1] = 2
    
            if (think_memory[183][1] == 3) and (think_memory[183][0] >= 30):
                think_memory[183][2] = False
                think_memory[183][4] = 1
                think_memory[183][1] = 1
                return 'C'
            
            elif (((game_round + 1) - think_memory[183][3]) / 20) >= 0.75:
                think_memory[183][1] = 3
                    
        if think_memory[183][1] == 1:
            return 'C'
        elif think_memory[183][1] == 2:
            mutual_cooperate_prob = ((think_memory[183][5][("C", "C")][0] + 1) / (think_memory[183][5][("C", "C")][1] + 2))
            exploitation_prob = ((think_memory[183][5][("C", "D")][0] + 1) / (think_memory[183][5][("C", "D")][1] + 2))
            exploited_prob = ((think_memory[183][5][("D", "C")][0] + 1) / (think_memory[183][5][("D", "C")][1] + 2))
            mutual_defect_prob = ((think_memory[183][5][("D", "D")][0] + 1) / (think_memory[183][5][("D", "D")][1] + 2))
                
            curiosity_category = [
                [ 
                    1,
                    0,
                    1,
                    0
                ],
                [ 
                    1,
                    0,
                    0,
                    1
                ],
                [ 
                    0,
                    0,
                    1,
                    1
                ],
                [ 
                    0,
                    1,
                    0,
                    1
                ],
                [ 
                    0.5,
                    0.5,
                    0.5,
                    0.5
                ],
                [ 
                    0.8,
                    0.8,
                    0.8,
                    0.8
                ],
                [ 
                    0.2,
                    0.2,
                    0.2,
                    0.2
                ]
            ]
                
            curiosity_enemy_differents = [0 for _ in range(len(curiosity_category))]
            for i in range(len(curiosity_category)):
                curiosity_enemy_differents[i] = (
                    abs(mutual_cooperate_prob - curiosity_category[i][0]) +
                    abs(exploitation_prob - curiosity_category[i][1]) +
                    abs(exploited_prob - curiosity_category[i][2]) +
                    abs(mutual_defect_prob - curiosity_category[i][3])
                ) / 4
                            
            if min(curiosity_enemy_differents) <= 0.25:
                most_similar = [['Tit For Tat', 'Pavlov', 'Alternator', 'reverse tit for tat', 'Random', 'Good Random', 'Bad Random'][j] for j in range(len(curiosity_enemy_differents)) if curiosity_enemy_differents[j] == min(curiosity_enemy_differents)]
            
                enemy_category = rnd.choice(most_similar)
                
                if enemy_category == 'Tit For Tat':
                    return 'C'
                elif enemy_category == 'Pavlov':
                    return ['D', 'D', 'C'][(game_round + 1) % 3]
                elif enemy_category == 'Alternator':
                    return 'D' if curiosity_eyes[-1][ap] == 'C' else rnd.choice(['C', 'D'])
                elif enemy_category == 'reverse tit for tat':
                    return 'D'
                elif 'Random' in enemy_category:
                    return 'D'
                
            A = clamp(((think_memory[183][5][curiosity_cond][0] + 1) / (think_memory[183][5][curiosity_cond][1] + 2)) - 0.25)
            if rnd.random() <= A:
                return 'C'
            else:
                return 'D'
        else:
            return 'D'
    elif strategy == 'Manipulity':
        def clamp(x):
            return max(min(x, 1), 0)

        manipulity_eyes = [('C', 'C') for _ in range(10)] + memory
                
        if think_memory[184][4] > 0:
            think_memory[184][4] -= 1
            return 'C'
                        
        if manipulity_eyes[-1][ap] == 'D':
            think_memory[184][0] -= 30
        else:
            think_memory[184][0] += 5
        
        manipulity_cond = (manipulity_eyes[-1-1][ap], manipulity_eyes[-1-1][1-ap])
                
        think_memory[184][5][manipulity_cond][0] += (manipulity_eyes[-1][ap] == 'C')
        think_memory[184][5][manipulity_cond][1] += 1
                        
        if (think_memory[184][0] < 30) or think_memory[184][2]:
            if not think_memory[184][2]:
                think_memory[184][2] = True
                think_memory[184][3] = (game_round + 1)
        
            if think_memory[184][1] < 2:
                think_memory[184][1] = 2
        
            if (think_memory[184][1] == 3) and (think_memory[184][0] >= 40):
                think_memory[184][2] = False
                think_memory[184][4] = 1
                think_memory[184][1] = 1
                return 'C'
                
            elif (((game_round + 1) - think_memory[184][3]) / 20) >= 0.75:
                think_memory[184][1] = 3
                        
        if think_memory[184][1] == 1:
            if rnd.random() <= clamp(((think_memory[184][5][("C", "D")][0] + 1) / (think_memory[184][5][("C", "D")][1] + 2)) - 0.25):
                return 'D'
            else:
                return 'C'
        elif think_memory[184][1] == 2:
            cyclic_inp = [manipulity_eyes[-1 - i][ap] for i in range(10)]
                            
            is_cycle = False
            if "D" in cyclic_inp:
                if cycle_detec(ap, min_ = 1, max_ = 5, offset = (game_round - (game_round % 10))):
                    is_cycle = True
            
            if is_cycle:
                return 'D'
            
            mutual_cooperate_prob = ((think_memory[184][5][("C", "C")][0] + 1) / (think_memory[184][5][("C", "C")][1] + 2))
            exploitation_prob = ((think_memory[184][5][("C", "D")][0] + 1) / (think_memory[184][5][("C", "D")][1] + 2))
            exploited_prob = ((think_memory[184][5][("D", "C")][0] + 1) / (think_memory[184][5][("D", "C")][1] + 2))
            mutual_defect_prob = ((think_memory[184][5][("D", "D")][0] + 1) / (think_memory[184][5][("D", "D")][1] + 2))
                    
            manipulity_category = [
                [ 
                    1,
                    0,
                    1,
                    0
                ],
                [ 
                    1,
                    0,
                    0,
                    1
                ],
                [ 
                    0,
                    0,
                    1,
                    1
                ],
                [ 
                    0,
                    1,
                    0,
                    1
                ],
                [ 
                    0.5,
                    0.5,
                    0.5,
                    0.5
                ],
                [ 
                    0.8,
                    0.8,
                    0.8,
                    0.8
                ],
                [ 
                    0.2,
                    0.2,
                    0.2,
                    0.2
                ]
            ]
                    
            manipulity_enemy_differents = [0 for _ in range(len(manipulity_category))]
            for i in range(len(manipulity_category)):
                manipulity_enemy_differents[i] = (
                    abs(mutual_cooperate_prob - manipulity_category[i][0]) +
                    abs(exploitation_prob - manipulity_category[i][1]) +
                    abs(exploited_prob - manipulity_category[i][2]) +
                    abs(mutual_defect_prob - manipulity_category[i][3])
                ) / 4
                                
            if min(manipulity_enemy_differents) <= 0.25:
                most_similar = [['Tit For Tat', 'Pavlov', 'Alternator', 'reverse tit for tat', 'Random', 'Good Random', 'Bad Random'][j] for j in range(len(manipulity_enemy_differents)) if manipulity_enemy_differents[j] == min(manipulity_enemy_differents)]
                
                enemy_category = rnd.choice(most_similar)
                    
                if enemy_category == 'Tit For Tat':
                    return 'C'
                elif enemy_category == 'Pavlov':
                    return ['D', 'D', 'C'][(game_round + 1) % 3]
                elif enemy_category == 'Alternator':
                    return 'D' if manipulity_eyes[-1][ap] == 'C' else rnd.choice(['C', 'D'])
                elif enemy_category == 'reverse tit for tat':
                    return 'D'
                elif 'Random' in enemy_category:
                    return 'D'
                    
            A = ((think_memory[184][5][manipulity_cond][0] + 1) / (think_memory[184][5][manipulity_cond][1] + 2))
            if rnd.random() <= A:
                if rnd.random() <= clamp(((think_memory[184][5][("C", "D")][0] + 1) / (think_memory[184][5][("C", "D")][1] + 2)) - 0.25):
                    return 'D'
                else:
                    return 'C'
            else:
                return 'D'
        else:
            return 'D'
    elif strategy == 'Charity':
        def clamp(x):
            return max(min(x, 1), 0)

        if think_memory[185][3] > 0:
            think_memory[185][3] -= 1
            return 'C'

        if game_round > 0:
            if memory[-1][ap] == 'D':
                think_memory[185][0] -= 30
            else:
                think_memory[185][0] += 5
                think_memory[185][4] += 1
        else:
            think_memory[185][0] += 5
            think_memory[185][4] += 1
        
        if (think_memory[185][0] < 30) or think_memory[185][2]:
            if not think_memory[185][2]:
                think_memory[185][2] = True

            if think_memory[185][1] < 2:
                think_memory[185][1] = 2
                    
            if (think_memory[185][1] == 2) and (think_memory[185][0] >= 30):
                think_memory[185][2] = False
                think_memory[185][3] = 1
                think_memory[185][1] = 1
                return 'C'
        
        if think_memory[185][1] == 1:
            return 'C'
        else:
            return 'C' if rnd.random() <= (think_memory[185][4] / (game_round + 1)) else 'D'
    elif strategy == 'Calamity':
        def clamp(x):
            return max(min(x, 1), 0)

        calamity_eyes = [('C', 'C') for _ in range(10)] + memory
            
        if think_memory[186][4] > 0:
            think_memory[186][4] -= 1
            return 'C'
                    
        if calamity_eyes[-1][ap] == 'D':
            think_memory[186][0] -= 30
        else:
            think_memory[186][0] += 5
    
        calamity_cond = (calamity_eyes[-1-1][ap], calamity_eyes[-1-1][1-ap])
            
        think_memory[186][5][calamity_cond][0] += (calamity_eyes[-1][ap] == 'C')
        think_memory[186][5][calamity_cond][1] += 1
                    
        if (think_memory[186][0] < 30) or think_memory[186][2]:
            if not think_memory[186][2]:
                think_memory[186][2] = True
                think_memory[186][3] = (game_round + 1)
    
            if think_memory[186][1] < 2:
                think_memory[186][1] = 2
    
            if (think_memory[186][1] == 3) and (think_memory[186][0] >= 30):
                think_memory[186][2] = False
                think_memory[186][4] = 1
                think_memory[186][1] = 1
                return 'C'
            
            elif (((game_round + 1) - think_memory[186][3]) / 20) >= 0.75:
                think_memory[186][1] = 3
                    
        if think_memory[186][1] == 1:
            if rnd.random() <= clamp(((think_memory[186][5][("C", "D")][0] + 1) / (think_memory[186][5][("C", "D")][1] + 2)) - 0.125):
                return 'D'
            else:
                return 'C'
        elif think_memory[186][1] == 2:
            mutual_cooperate_prob = ((think_memory[186][5][("C", "C")][0] + 1) / (think_memory[186][5][("C", "C")][1] + 2))
            exploitation_prob = ((think_memory[186][5][("C", "D")][0] + 1) / (think_memory[186][5][("C", "D")][1] + 2))
            exploited_prob = ((think_memory[186][5][("D", "C")][0] + 1) / (think_memory[186][5][("D", "C")][1] + 2))
            mutual_defect_prob = ((think_memory[186][5][("D", "D")][0] + 1) / (think_memory[186][5][("D", "D")][1] + 2))
                
            calamity_category = [
                [ 
                    1,
                    0,
                    1,
                    0
                ],
                [ 
                    1,
                    0,
                    0,
                    1
                ],
                [ 
                    0,
                    0,
                    1,
                    1
                ],
                [ 
                    0,
                    1,
                    0,
                    1
                ]
            ]
                
            calamity_enemy_differents = [0 for _ in range(len(calamity_category))]
            for i in range(len(calamity_category)):
                calamity_enemy_differents[i] = (
                    abs(mutual_cooperate_prob - calamity_category[i][0]) +
                    abs(exploitation_prob - calamity_category[i][1]) +
                    abs(exploited_prob - calamity_category[i][2]) +
                    abs(mutual_defect_prob - calamity_category[i][3])
                ) / 4
                            
            if min(calamity_enemy_differents) <= 0.25:
                most_similar = [['Tit For Tat', 'Pavlov', 'Alternator', 'reverse tit for tat'][j] for j in range(len(calamity_enemy_differents)) if calamity_enemy_differents[j] == min(calamity_enemy_differents)]
            
                enemy_category = rnd.choice(most_similar)
                
                if enemy_category == 'Tit For Tat':
                    return 'C'
                elif enemy_category == 'Pavlov':
                    return ['D', 'D', 'C'][(game_round + 1) % 3]
                elif enemy_category == 'Alternator':
                    return 'D' if calamity_eyes[-1][ap] == 'C' else rnd.choice(['C', 'D'])
                elif enemy_category == 'reverse tit for tat':
                    return 'D'
                
            A = ((think_memory[186][5][calamity_cond][0] + 1) / (think_memory[186][5][calamity_cond][1] + 2))
            if rnd.random() <= A:
                return 'D' if rnd.random() <= clamp(((think_memory[186][5][("C", "D")][0] + 1) / (think_memory[186][5][("C", "D")][1] + 2)) - 0.125) else 'C'
            else:
                return 'D'
        else:
            return 'D'
    elif strategy == 'Amazon / Piraha':
        if game_round > 0:
            if memory[-1][ap] == 'D':
                think_memory[187] += 1
        else:
            return 'C'

        B = min(2, game_round)
        inp = tuple([memory[-1 - ((B - 1) - i)][ap] for i in range(B)])
        if inp == tuple(['C'] * B):
            think_memory[187] -= 1

        think_memory[187] = max(think_memory[187], 0)

        maximal_few = rnd.randint(2, 3)
        maximal_many = rnd.randint(5, 6)

        if think_memory[187] <= maximal_few: 
            return 'C'
        elif think_memory[187] <= maximal_many: 
            return rnd.choice(['C', 'D'])
        else: 
            return 'D'
    elif strategy == 'Mosquito':
        def clamp(x):
            return max(min(x, 1), 0)
        
        cond = (memory[-1-1][ap], memory[-1-1][1-ap]) if (game_round + 1) > 2 else ('C', 'C')
                    
        think_memory[188][cond][0] += (memory[-1][ap] == 'C') if game_round > 0 else True
        think_memory[188][cond][1] += 1

        A = clamp(((think_memory[188][cond][0] + 1) / (think_memory[188][cond][1] + 2)) - 0.25)
        if rnd.random() <= A:
            if rnd.random() <= clamp(((think_memory[188][("C", "D")][0] + 1) / (think_memory[188][("C", "D")][1] + 2)) - 0.25):
                return 'D'
            else:
                return 'C'
        else:
            return 'D'
    elif strategy == 'Echo Fisher':
        if not game_round:
            return 'C'
        if (game_round + 1) % 2 == 0:
            return 'D' if memory[-1][ap] == 'C' else "C"
        else:
            return memory[-1][ap]
    elif strategy == 'Devil Staircase':
        if not game_round:
            return 'C'
        
        if memory[-1][ap] == 'D':
            think_memory[189] += 1

        def devil_staircase(x, iterations=5):
            """
            Menghitung nilai Devil's Staircase untuk SATU angka tunggal (skalar).
            x: angka antara 0 dan 1 (contoh: 0.25, 0.75).
            iterations: jumlah detail iterasi fraktal.
            """
            
            if x <= 0.0: return 0.0
            if x >= 1.0: return 1.0
    
            y = 0.0
            x_current = float(x)
    
            for i in range(1, iterations + 1):
                
                digit = np.floor(x_current * 3)
        
                if digit == 1:
                    
                    y += 1.0 / (2**i)
                    break 
            
                elif digit >= 2:
                    
                    y += 1.0 / (2**i)
                    x_current = 3 * x_current - 2
            
                else:
                    
                    x_current = 3 * x_current
            
            return y

        p_d = think_memory[189] / (game_round)
        if rnd.random() <= devil_staircase(p_d):
            return 'D'
        else:
            return 'C'
    elif strategy == 'Hot Coffee':
        def clamp(x):
            return max(min(x, 1), 0)

        if game_round > 0:
            if memory[-1][ap] == 'C':
                think_memory[190] += 1
            else:
                think_memory[190] -= 3
        else:
            think_memory[190] += 1
        think_memory[190] = max(min(think_memory[190], 100), 0)

        energy = clamp(think_memory[190] / 100)
        caffeine = 1 - (((game_round + 1) % 10) / 10)
        if rnd.random() <= (energy * (1 + caffeine)):
            if not game_round:
                return 'C'
            return memory[-1][ap]
        else:
            return 'C'
    elif strategy == 'The Cave Breaker':
        the_cave_breaker_eyes = [('C', 'C') for _ in range(3)] + memory
        if the_cave_breaker_eyes[-1][ap] == 'D':
            think_memory[196][0] += 1

        if the_cave_breaker_eyes[-1][1-ap] == 'D':
            think_memory[196][2] += 1

        if (game_round + 1) > 11 and (think_memory[196][0] / (game_round)) > 0.6:
            return 'D'

        inp = tuple([the_cave_breaker_eyes[-1 - ((2) - i)][ap] for i in range(3)])
        if think_memory[196][1] or ((game_round + 1) > 6 and inp == ('D', 'D', 'D')):
            think_memory[196][1] = True
            return 'D'

        if (game_round) % 5 == 0 and think_memory[196][2] < 15:
            return 'D'
        
        if game_round >= (tournament_avg_last_round - 15):
            return 'D'

        return the_cave_breaker_eyes[-1][ap]
    elif strategy == 'Random Exploiter 2':
        if not game_round: return 'C'
        if memory[-1][ap] == 'D': think_memory[198] += 1
        def sin_squared(x): return mth.sin(mth.pi * x) * mth.sin(mth.pi * x)
        p_d = think_memory[198] / (game_round)
        if abs(sin_squared(p_d) - 1) <= 0.1: return 'D'
        return memory[-1][ap]
    elif strategy == 'George':
        def clamp(x): return max(min(x, 1), 0)
        A = tournament_avg_last_round / 2
        return 'D' if rnd.random() <= (0.2 * (((game_round + 1) - A) / A)) else (memory[-1][ap] if game_round > 0 else 'C')
    elif strategy == 'Long Horse':
        def clamp(x): return max(min(x, 1), 0)
        
        if game_round > 0:
            if memory[-1][ap] == 'D':
                think_memory[199][0] += 0.1
                think_memory[199][1] = 0
                if (score[1-ap] - score[ap]) >= 0:
                    return 'C'
                else:
                    return 'D'

        if think_memory[199][1] < 4:
            if (game_round + 1) % 50 == 0:
                think_memory[199][1] += 1
                return 'D'

            if rnd.random() <= clamp(think_memory[199][0]):
                return 'D'
            
        if not game_round:
            return 'C'
        return memory[-1][ap]
    elif strategy == 'The Nasty A':
        if not game_round:
            return 'C'
        if memory[-1][ap] == 'D':
            think_memory[201] = 1

        if (think_memory[201] == 1) or (game_round >= (tournament_avg_last_round)):
            return 'D'

        return 'C'
    elif strategy == 'Monkey See, Monkey Do':
        if (game_round + 1) <= 3:
            if not game_round:
                return 'D'
            else:
                return memory[-1][ap]

        inp = tuple([memory[-1 - (2 - i)][ap] for i in range(3)])
        if inp == ('C', 'C', 'C'):
            think_memory[202] = 1

        if think_memory[202] == 1:
            return 'C'
        else:
            return memory[-1][ap]
    elif strategy == 'Genetic Algo 1':
        table = {
            'CC':'C',
            'CD':'D',
            'DC':'C',
            'DD':'C'
        }
        inp = (memory[-1][1-ap] + memory[-1][ap]) if game_round > 0 else 'CC'
        return table[inp]
    elif strategy == 'Genetic Algo 2':
        table = {
            'CCCC':'C',
            'CCCD':'D',
            'CCDC':'C',
            'CCDD':'C',
            'CDCC':'C',
            'CDCD':'D',
            'CDDC':'C',
            'CDDD':'C',
            'DCCC':'C',
            'DCCD':'D',
            'DCDC':'C',
            'DCDD':'C',
            'DDCC':'C',
            'DDCD':'D',
            'DDDC':'C',
            'DDDD':'D'
        }
        inp = ((memory[-1 - 1][1-ap] + memory[-1 - 1][ap]) if (game_round + 1) > 2 else 'CC') + ((memory[-1][1-ap] + memory[-1][ap]) if game_round > 0 else 'CC')
        return table[inp]
    elif strategy == 'Random Exploiter 3':
        if not game_round: return 'C'
        if memory[-1][ap] == 'D': think_memory[215] += 1
        p_d = think_memory[215] / (game_round)
        if abs(p_d - 0.5) <= 0.1: return 'D'
        return memory[-1][ap]
    elif strategy == 'Gambler Attar Version':
        if score[1-ap] >= 15:
            if rnd.random() <= ((game_round + 1) / tournament_avg_last_round):
                lose_gambling = (rnd.random() > 0.1)
                if lose_gambling and (think_memory[37] == 0):
                    think_memory[37] = 3
                    return 'C'

        if think_memory[37] > 0:
            think_memory[37] -= 1
            return 'D'
        else:
            if not game_round:
                return 'C'
            return memory[-1][ap]
    elif strategy == 'Random Exploiter 4':
        stev = 0.4
        mean = 2.0
        if not game_round:
            think_memory[102][1] = 1 / (stev * ((2 * mth.pi) ** 0.5))
            think_memory[102][2] = mean / stev
            return 'C'
        if memory[-1][ap] == 'D':
            think_memory[102][0] += 1
        p_d = think_memory[102][0] / (game_round)
        A1 = ((p_d - 0.5) * think_memory[102][2])
        A2 = think_memory[102][1] * mth.exp(-((A1 * A1) / 2))
        if abs(A2 - 1) <= 0.2: return 'D'
        return memory[-1][ap]
    elif strategy == 'Toxic Mirror':
        if game_round > 0:
            if memory[-1][ap] == 'D':
                think_memory[76] += 1
        if (game_round + 1) <= 4:
            return 'C'

        p_d = think_memory[76] / (game_round)

        if p_d > 0.25:
            return 'C'

        if (game_round) % 7 == 0:
            return 'D'
        else:
            if memory[-1][ap] == 'D':
                if rnd.random() < 0.3:
                    return 'C'
                else:
                    return 'D'
            else:
                return 'C'
    elif strategy == 'Sweet Heart':
        if (game_round + 1) <= 4:
            return 'C'

        if (game_round + 1) == 11:
            return 'D'

        if (game_round + 1) > (tournament_avg_last_round - 20):
            if (game_round + 1) % 3 == 0:
                return 'D'

        if memory[-1][ap] == 'D':
            if rnd.random() < 0.6:
                return 'C'
            else:
                return 'D'
        else:
            return 'C'
    elif strategy == 'Split Brain Syndrome':
        
        if not game_round:
            think_memory[103][0] = 0   
            think_memory[103][1] = 0   
            think_memory[103][2] = 0   
            think_memory[103][3] = 0   
    
        lawan = memory[-1][ap] if game_round > 0 else 'C'
        kiri = 'C'
        kanan = 'C'
    
        
        if think_memory[103][0] > 2:
            kiri = 'D'
        
        if think_memory[103][1] >= 3:
            kanan = 'D'
    
        
        if lawan == 'D':
            think_memory[103][0] += 1
            think_memory[103][1] += 1
        else:
            think_memory[103][1] = max(0, think_memory[103][1] - 1)
    
        
        if kiri == kanan:
            pilihan = kiri
        else:
            pilihan = 'C'  
    
        return pilihan
    elif strategy == 'Golden Mean':
        if not game_round:
            think_memory[216] = 0  

        if game_round > 0:
            if memory[-1][ap] == 'C':
                think_memory[216] += 1
        else:
            think_memory[216] += 1

        
        c_count = think_memory[216]
        total = game_round
        rasio = c_count / total if total > 0 else 1.0
    
        
        if rasio >= 0.80:
            return 'C'  
        else:
            return memory[-1][ap]
    elif strategy == 'Patterned Adaptive Fortress 2':
        state = think_memory[217]
        alpha_base = 0.08
        lawan = memory[-1][ap] if game_round > 0 else 'C'

        
        if (game_round + 1) <= 3:
            state["last_my_action"] = 'C'
            return 'C'

        
        b = 1.0 if lawan == 'C' else 0.0
        c = state["ema_trust"]

        alpha = alpha_base
        if state["last_my_action"] == 'C' and state["d_streak"] > 0 and lawan == 'D':
            state["regret_count"] += 1
            alpha = 0.25

        state["ema_trust"] = (alpha * b) + ((1 - alpha) * c)

        
        if lawan == 'C' and state["d_streak"] == 0:
            state["regret_count"] = max(0, state["regret_count"] - 0.005)
            state["pattern_score"] = max(0, state["pattern_score"] - 0.002)

        
        state["history"].append(b)
        if len(state["history"]) > 8:
            state["history"].pop(0)
        recent_trust = sum(state["history"]) / len(state["history"])

        
        if len(state["history"]) >= 4:
            history_str = ''.join(['C' if x == 1 else 'D' for x in state["history"]])

            
            if history_str[-4:] in ["CDCD", "DCDC", "DDCD"]:
                if state["pattern_window"] == 0:
                    state["pattern_score"] = min(2.0, state["pattern_score"] + 0.5)
                    state["pattern_window"] = 4

            
            if len(state["history"]) >= 6 and history_str[-6:] in ["CCDCCD", "DCCDCC"]:
                if state["pattern_window"] == 0:
                    state["pattern_score"] = min(2.0, state["pattern_score"] + 0.5)
                    state["pattern_window"] = 6

        
        if state["pattern_window"] > 0:
            state["pattern_window"] -= 1

        
        volatility = abs(state["ema_trust"] - recent_trust)
        a = max(0.3, min(0.9, 1 - volatility))

        
        combined_trust = (a * state["ema_trust"]) + ((1 - a) * recent_trust)

        
        state["d_streak"] = state["d_streak"] + 1 if lawan == 'D' else 0
    
        

        
        if (game_round + 1) >= (tournament_avg_last_round - 5):
            if combined_trust < 0.85 or lawan == 'D':
                state["last_my_action"] = 'D'
                return 'D'

        
        if state["d_streak"] >= 3:
            state["last_my_action"] = 'D'
            return 'D'

        
        if state["punish_cooldown"] > 0:
            state["punish_cooldown"] -= 1
            state["last_my_action"] = 'D'
            return 'D'

        if lawan == 'D' and state["d_streak"] >= 2 and combined_trust < 0.7:
            state["punish_cooldown"] = 2
            state["last_my_action"] = 'D'
            return 'D'
    
        
        total_suspicion = state["regret_count"] + state["pattern_score"]
        required_trust = min(0.90, 0.60 + (total_suspicion * 0.10))
    
        if combined_trust < required_trust:
            my_action = 'D' if lawan == 'D' else 'C'
        else:
            my_action = 'C'

        
        state["last_my_action"] = my_action
        return my_action
    elif strategy == 'Perceptron':
        brain = think_memory[218]
        if game_round > 0:
            R_or_P = 0
            if memory[-1][ap] == 'C':
                if memory[-1][1-ap] == 'C':
                    R_or_P = RPST['R']
                else:
                    R_or_P = RPST['T']
            else:
                if memory[-1][1-ap] == 'C':
                    R_or_P = -4 * abs(RPST['S'] - RPST['P'])
                else:
                    R_or_P = -max(abs(RPST['P']), 1)
            learning_rate = 0.5

            A = R_or_P * learning_rate

            for i in range(brain["n"]):
                brain["input weight"][i] = mth_mtx.matrix_aritc(brain["input weight"][i], mth_mtx.matrix_function(brain["old input"], A, '*'), '+')

            for i in range(2):
                brain["output weight"][i] = mth_mtx.matrix_aritc(brain["output weight"][i], mth_mtx.matrix_function(brain["n val"], A, '*'), '+')

        inp = [(memory[-1][ap] == 'C'), (memory[-1][1-ap] == 'C')] if game_round > 0 else [True, True]
        brain["old input"] = [(memory[-1][ap] == 'C'), (memory[-1][1-ap] == 'C')] if game_round > 0 else [True, True]
        brain["n val"] = mth_mtx.matrix_function(mth_mtx.matrix_aritc(mth_mtx.matrix_list_nestlist(inp, brain["input weight"], brain["n"], None, 'dot prod'), brain["bias"], '+'), (0, 1, 0), 'step')
        brain["output val"] = mth_mtx.matrix_aritc(mth_mtx.matrix_list_nestlist(brain["n val"], brain["output weight"], 2, None, 'dot prod'), brain["output bias"], '+')
        if brain["output val"][0] >= brain["output val"][1]:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Simple ANN':
        brain = think_memory[219]
        max_time = 50

        if brain["time"] > max_time:
            for i in range(brain["n"]):
                brain["input weight"][i] = [j + rnd.uniform(-0.15, 0.15) for j in brain["best input weight"][i]]
            for i in range(2):
                brain["output weight"][i] = [j + rnd.uniform(-0.15, 0.15) for j in brain["best output weight"][i]]
            brain["bias"] = [j + rnd.uniform(-0.15, 0.15) for j in brain["best bias"]]
            brain["output bias"] = [j + rnd.uniform(-0.15, 0.15) for j in brain["best output bias"]]
            brain["time"] = 0

        if game_round > 0:
            learning_rate = 0.1
            if brain["reward"] > 0:
                A = 1 * learning_rate
            else:
                A = -1 * learning_rate

            brain["input weight"] = mth_mtx.matrix_function(brain["input weight"], A, '+')
            brain["output weight"] = mth_mtx.matrix_function(brain["output weight"], A, '+')

            brain["bias"] = mth_mtx.matrix_function(brain["bias"], A, '+')
            brain["output bias"] = mth_mtx.matrix_function(brain["output bias"], A, '+')

            if memory[-1][ap] == 'C':
                brain["reward"] += 1
                brain["score"] += 1
            else:
                brain["reward"] -= 1
                brain["score"] -= 1

        inp = [(memory[-1][ap] == 'C'), (memory[-1][1-ap] == 'C')] if game_round > 0 else [True, True]
        if brain["score"] > brain["high score"]:
            brain["high score"] = brain["score"]
            for i in range(brain["n"]):
                brain["best input weight"][i] = brain["input weight"][i][:]
            for i in range(2):
                brain["best output weight"][i] = brain["output weight"][i][:]
            brain["best bias"] = brain["bias"][:]
            brain["best output bias"] = brain["output bias"][:]

        brain["n val"] = mth_mtx.matrix_function(mth_mtx.matrix_aritc(mth_mtx.matrix_list_nestlist(inp, brain["input weight"], brain["n"], None, 'dot prod'), brain["bias"], '+'), (0, 1, 0), 'step')
        brain["output val"] = mth_mtx.matrix_aritc(mth_mtx.matrix_list_nestlist(brain["n val"], brain["output weight"], 2, None, 'dot prod'), brain["output bias"], '+')
        brain["time"] += 1
        if brain["output val"][0] >= brain["output val"][1]:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Whale':
        state = think_memory[220]
        lawan = memory[-1][ap] if game_round > 0 else 'C'
    
        
        if (game_round + 1) <= 7:
            if (game_round + 1) <= 5:
                state["history"].append(lawan)
                
                if state["history"].count('D') >= 3:
                    return 'D'
            return 'C'
    
        
        state["history"].append(lawan)
        if len(state["history"]) > 10:
            state["history"].pop(0)
    
        history_str = ''.join(state["history"])
        if history_str[-4:] in ["CDCD", "DCDC"]:
            state["pattern_score"] += 0.3
    
        
        if lawan == 'D':
            state["d_streak"] += 1
            state["trust"] = max(0, state["trust"] - 0.1)
        else:
            state["d_streak"] = 0
            state["trust"] = min(1, state["trust"] + 0.05)
    
        if state["d_streak"] >= 3 or state["trust"] < 0.4:
            return 'D'
    
        
        if state["pattern_score"] > 1.5:
            return 'D'
    
        if lawan == 'D' and rnd.random() < state["forgiveness_chance"]:
            return 'C'
    
        return lawan  
    elif strategy == 'Rice Field':
        
        

        if (game_round + 1) < 12:
            return 'C'

        
        if think_memory[221][0] == 1:
            if ((game_round + 1) - think_memory[221][1] > 25) and \
               (memory[-1][ap] == 'C' and memory[-2][ap] == 'C' and memory[-3][ap] == 'C'):
                think_memory[221][0] = 0
                think_memory[221][1] = 0
                return 'C'
            return memory[-1][ap]

        
        if (game_round + 1) % 12 == 0:
            think_memory[221][2] += 1 
            return 'D'

        
        if (game_round + 1) % 12 == 1:
            if memory[-1][1-ap] == 'D' and memory[-1][ap] == 'D':
                think_memory[221][0] = 1
                think_memory[221][1] = (game_round + 1)

        return 'C'
    elif strategy == '3 Seconds Mirror':
        
        
        if (game_round + 1) < 4:
            return 'C'

        
        
        delayed_move = memory[-3][ap]

        
        if memory[-1][ap] == 'C' and memory[-2][ap] == 'C' and delayed_move == 'D':
            return 'C' 

        return delayed_move
    elif strategy == 'Caveman':
        
        if game_round > 0:
            think_memory[222][1] *= 0.9
            think_memory[222][0] *= 0.9
            if memory[-1][ap] == 'D':
                think_memory[222][0] += 1
                think_memory[222][1] += 0.1
                if think_memory[222][2] == 0:
                    think_memory[222][2] = round(think_memory[222][0])
                else:
                    think_memory[222][2] += 1

        next_chat[0] = 'UGA BUGA!'

        if rnd.random() <= max(think_memory[222][1] - 0.2, 0):
            return 'D'

        if rnd.random() <= max(0.2 - think_memory[222][1], 0):
            return 'D'

        if think_memory[222][2] > 0:
            think_memory[222][2] -= 1
            return 'D'
        else:
            return 'C'
    elif strategy == 'Slanger':
        slanger_eyes = [('C', 'C') for _ in range(2)] + memory
        think_memory[223] *= 0.9
        if slanger_eyes[-1][ap] == 'C' and slanger_eyes[-1][1-ap] == 'D':
            if slanger_eyes[-2][ap] == 'D' and slanger_eyes[-2][1-ap] == 'C':
                next_chat[0] = 'AHA! WHO\'S LAUGHING NOW BUDDY? WHO\'S LAUGHING NOW!!! >:-)'
                think_memory[223] += 0.1
            elif slanger_eyes[-2][ap] == 'D' and slanger_eyes[-2][1-ap] == 'D':
                next_chat[0] = 'YOU, FOOL!!! >:-)'
                think_memory[223] += 0.15
            elif slanger_eyes[-2][ap] == 'C' and slanger_eyes[-2][1-ap] == 'D':
                next_chat[0] = 'AHA! You are cooked again >:-)'
                think_memory[223] += 0.175
            else:
                next_chat[0] = 'AHA! You are cooked >:-)'
                think_memory[223] += 0.2

        if slanger_eyes[-1][ap] == 'D' and slanger_eyes[-1][1-ap] == 'C':
            if slanger_eyes[-2][ap] == 'D' and slanger_eyes[-2][1-ap] == 'C':
                next_chat[0] = 'I SO HATE YOU!!! >:-('
                think_memory[223] += 0.25
            elif slanger_eyes[-2][ap] == 'D' and slanger_eyes[-2][1-ap] == 'D':
                next_chat[0] = 'OH, CRAP! >:-('
                think_memory[223] += 0.175
            elif slanger_eyes[-2][ap] == 'C' and slanger_eyes[-2][1-ap] == 'D':
                next_chat[0] = 'I\'m sorry! :-('
                think_memory[223] += -0.1
            else:
                next_chat[0] = 'I HATE YOU! >:-('
                think_memory[223] += 0.2

        if slanger_eyes[-1][ap] == 'C' and slanger_eyes[-1][1-ap] == 'C':
            if slanger_eyes[-2][ap] == 'D' and slanger_eyes[-2][1-ap] == 'C':
                next_chat[0] = 'I don\'t too trust you, but okay :-('
                think_memory[223] += 0.05
            elif slanger_eyes[-2][ap] == 'D' and slanger_eyes[-2][1-ap] == 'D':
                next_chat[0] = 'I don\'t too trust you, but okay :-|'
                think_memory[223] += 0.0
            elif slanger_eyes[-2][ap] == 'C' and slanger_eyes[-2][1-ap] == 'D':
                next_chat[0] = 'YES! We can become friends again! >:-)'
                think_memory[223] += 0.1
            else:
                next_chat[0] = 'Were friends! :-)'
                think_memory[223] += -0.1

        if slanger_eyes[-1][ap] == 'D' and slanger_eyes[-1][1-ap] == 'D':
            if slanger_eyes[-2][ap] == 'D' and slanger_eyes[-2][1-ap] == 'C':
                next_chat[0] = 'TRAITOR!!!!! >:-('
                think_memory[223] += 0.25
            elif slanger_eyes[-2][ap] == 'D' and slanger_eyes[-2][1-ap] == 'D':
                next_chat[0] = 'NOT AGAIN! >:-('
                think_memory[223] += 0.2
            elif slanger_eyes[-2][ap] == 'C' and slanger_eyes[-2][1-ap] == 'D':
                next_chat[0] = 'OKAY! IF YOU WAN\'T TO FIGHT ME THEN! >:-)'
                think_memory[223] += 0.15
            else:
                next_chat[0] = 'OH COME ON! >:-('
                think_memory[223] += 0.25

        think_memory[223] = max(min(think_memory[223], 1), 0)
        if rnd.random() <= think_memory[223]:
            return 'D'
        else:
            return 'C'
    elif strategy == 'Chatter':
        lawan = memory[-1][ap] if game_round > 0 else 'C'
        aku = memory[-1][1-ap] if game_round > 0 else 'C'
        pesan_lawan = chats[ap]  

        
        if 'cooked' in pesan_lawan or 'AHA' in pesan_lawan:
            next_chat[0] = "Don't be so arrogant! 😤 You will lose soon! 😏💪"
    
        
        elif lawan == 'D' and aku == 'C':
            next_chat[0] = "Why did you betray me?! 😢💔 Please don't do that again! 🙏"
    
        
        elif lawan == 'C':
            next_chat[0] = "Good! 🤝 Let's keep it this way! 😊✨"
    
        
        else:
            next_chat[0] = "Enough?! 😤 Let's make peace! 🤍🕊️"

        
        if lawan == 'D':
            return 'D'
        return 'C'
    elif strategy == 'Self-Love A-Learner':
        exploitation_rate = 1
        exploration_rate = 0.1

        if game_round > 0:
            old_cond = think_memory[91][1]
            reward = (score[1-ap] - think_memory[91][2])
            think_memory[91][0][old_cond][(memory[-1][1-ap] == 'D')] += reward * exploitation_rate

        cond = (memory[-1][1-ap] + memory[-1][ap]) if game_round > 0 else 'CC'
        think_memory[91][1] = cond
        think_memory[91][2] = score[1-ap]
        if (rnd.random() <= exploration_rate) or (think_memory[91][0][cond][0] == think_memory[91][0][cond][1]):
            return rnd.choice(memory[-1]) if game_round > 0 else 'C'
        if think_memory[91][0][cond][0] > think_memory[91][0][cond][1]:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Diplomat':
        state = think_memory[224]

        
        if not game_round:
            next_chat[0] = "Let's cooperate. If you betray me, I'll punish you."
            state["coop_count"] += 1
            state["trust"] = min(1.0, state["trust"] + 0.05)

        else:
            
            if memory[-1][ap] == 'C':
                state["coop_count"] += 1
                state["trust"] = min(1.0, state["trust"] + 0.05)
            else:
                state["defect_count"] += 1
                state["trust"] = max(0.0, state["trust"] - 0.15)
                if state["trust"] < 0.4:
                    state["deal_broken"] = True

        
        if state["deal_broken"]:
            if (game_round + 1) == int(tournament_avg_last_round * 0.7):
                next_chat[0] = "You broke the deal."
            return 'D'

        
        if state["trust"] >= 0.6:
            if (game_round + 1) % 20 == 0:
                next_chat[0] = "Good, let's keep this up."
            return 'C'
        else:
            
            if not game_round:
                return 'C'
            return memory[-1][ap]
    elif strategy == 'Bluff Master':
        state = think_memory[225]

        
        if (game_round + 1) % 15 == 1:
            next_chat[0] = rnd.choice([
                "I will always cooperate, trust me.",
                "No hard feelings, just business.",
                "We can be friends, I promise."
            ])

        
        opp_chat = chats[ap].lower()
        if "cooperate" in opp_chat or "friend" in opp_chat or "trust" in opp_chat:
            state["opp_coop_words"] += 1

        
        if game_round > 0:
            if memory[-1][ap] == 'D':
                if state["opp_coop_words"] > 0:
                    state["opp_betrayals"] += 1

        opp_bluff_ratio = (
            state["opp_betrayals"] / max(1, state["opp_coop_words"])
        )

        
        base_move = 'C'

        
        bluff_chance = 0.05 + 0.1 * min(1.0, opp_bluff_ratio)
        if rnd.random() < bluff_chance:
            state["my_bluff_count"] += 1
            return 'D'

        
        if (game_round + 1) > int(tournament_avg_last_round * 0.7):
            if opp_bluff_ratio > 0.5:
                return 'D'

        return base_move
    elif strategy == 'Is There Something New?':
        def clamp(x):
            return max(min(x, 1), 0)

        if game_round > 0:
            if memory[-1][ap] == 'C':
                think_memory[226][0] += 1

        if (game_round + 1) <= 5:
            return 'C'

        A1 = think_memory[226][0] / (game_round)
        A2 = sum((i[ap] == 'C') for i in memory[-5:]) / 5
        if abs(A1 - A2) >= 0.1:
            think_memory[226][1].pop(0)
            think_memory[226][1].append(A2 - A1)

        A = 1 - clamp(mth_mtx.matrix_dot_prod(think_memory[226][1], [1 / (2 ** (i + 1)) for i in range(5)]))
        if rnd.random() <= A:
            return memory[-1][ap]
        return memory[-1][1-ap]
    elif strategy == 'Forest-XG':
        brain = think_memory[227]
        cond = (memory[-1][1-ap] + memory[-1][ap]) if game_round > 0 else 'CC'
        brain["acumulation"][cond] = (((memory[-1][ap] == 'C') if game_round > 0 else True) - brain["tree"][cond])
        brain["total"][cond] = 1
        reg_lambda = 1
        Weta = {i: brain["acumulation"][i] / (brain["total"][i] + reg_lambda) for i in ('CC', 'CD', 'DC', 'DD')}
        brain["tree"]['C' + (memory[-1][ap] if game_round > 0 else 'C')] += Weta['C' + (memory[-1][ap] if game_round > 0 else 'C')]
        brain["tree"]['D' + (memory[-1][ap] if game_round > 0 else 'C')] += Weta['D' + (memory[-1][ap] if game_round > 0 else 'C')]
        E_sh = brain["tree"]['C' + (memory[-1][ap] if game_round > 0 else 'C')]
        E_st = brain["tree"]['D' + (memory[-1][ap] if game_round > 0 else 'C')]
        if E_sh >= E_st:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Egoist A-Learner':
        exploitation_rate = 2
        exploration_rate = 0.05

        if game_round > 0:
            old_cond = think_memory[228][1]
            reward = (score[1-ap] - score[ap])
            think_memory[228][0][old_cond][(memory[-1][1-ap] == 'D')] += reward * exploitation_rate

        cond = (memory[-1][1-ap] + memory[-1][ap]) if game_round > 0 else 'CC'
        think_memory[228][1] = cond
        if (rnd.random() <= exploration_rate) or (think_memory[228][0][cond][0] == think_memory[228][0][cond][1]):
            return rnd.choice(memory[-1]) if game_round > 0 else 'C'
        if think_memory[228][0][cond][0] > think_memory[228][0][cond][1]:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Syifa':
        if not game_round:
            return 'C'
        
        if game_round >= (tournament_avg_last_round):
            return 'D'
        
        if rnd.random() <= 0.1:
            if memory[-1][ap] == 'C':
                think_memory[229] = 1

        if think_memory[229] < 0:
            think_memory[229] += 1
            return 'C'

        if think_memory[229] == 1:
            if memory[-1][ap] == 'D':
                think_memory[229] = -1
                return 'C'
            return 'D'
        else:
            return memory[-1][ap]
    elif strategy == 'Trader':
        brain = think_memory[230]
        trader_eyes = [('C', 'C') for _ in range(3)] + memory

        if not game_round:
            RRR = brain["RRR"]
            if RRR < 1:
                brain["ALLD"] = True
                return 'D'

        if brain["ALLD"]:
            return 'D'
        
        if trader_eyes[-1][ap] == 'C':
            if brain["B"][1] == 1:
                brain["Lowest score ever get"] = min(brain["Lowest score ever get"], brain["B"][0])
                brain["B"][0] = 0
                brain["B"][1] = 0
            brain["Win round total"] += 1
            brain["Total poin opp C"] += score[1-ap] - brain["Old score"]
            brain["G"][0] += score[1-ap] - brain["Old score"]
            brain["G"][1] = 1
        else:
            if brain["G"][1] == 1:
                brain["Highest score ever get"] = max(brain["Highest score ever get"], brain["G"][0])
                brain["G"][0] = 0
                brain["G"][1] = 0
            brain["Lose round total"] += 1
            brain["Total poin opp D"] += score[1-ap] - brain["Old score"]
            brain["B"][0] += score[1-ap] - brain["Old score"]
            brain["B"][1] = 1
        
        if (game_round + 1) <= 3:
            return 'C'

        if brain["Lose round total"] >= brain["Tolerant"]:
            brain["ALLD"] = True
            return 'D'

        brain["Old score"] = score[1-ap]
        if (game_round + 1) % 10 == 0:
            win_rate = brain["Win round total"] / (game_round + 1)
            lose_rate = 1 - win_rate
            profit_factor = brain["Total poin opp C"] / brain["Total poin opp D"] if brain["Total poin opp D"] > 0 else 999.0
            expectation = (win_rate * ((brain["Total poin opp C"] / brain["Win round total"]) if brain["Win round total"] > 0 else 0)) - (lose_rate * ((brain["Total poin opp D"] / brain["Lose round total"]) if brain["Lose round total"] > 0 else 0))
            drawdown = (brain["Highest score ever get"] - brain["Lowest score ever get"]) / brain["Highest score ever get"]

            brain["Highest score ever get"] = avg_RPST
            brain["Lowest score ever get"] = avg_RPST

            if drawdown > 0.25 or profit_factor < 1 or expectation < 1:
                return 'D'
        
        inp = [(trader_eyes[-1 - i][ap] == 'C') for i in range(3)]
        SMA = sum(inp) / 3
        pivot_point = (1 + 0 + (trader_eyes[-1][ap] == 'C')) / 3
        if SMA < 0.66 or pivot_point < 0.5:
            return 'D'
        else:
            return 'C'
    elif strategy == 'Procastinater':
        if think_memory[232] > 0:
            think_memory[232] -= 1
            return 'D'

        if ((score[ap] - score[1-ap]) >= (RPST['T'] * 10)):
            think_memory[232] = 10

        return 'C'
    elif strategy == 'Evolvable Cycler Attar Version':
        min_length_gene = 1
        max_length_gene = 7
        def breeding_and_mutating(x, y):
            BAM = []
            length = rnd.randint(min(len(x), len(y)), max(len(x), len(y)))
            for i in range(length):
                if i > (len(x) - 1):
                    BAM.append(y[i])
                elif i > (len(y) - 1):
                    BAM.append(x[i])
                else:
                    BAM.append(rnd.choice([x[i], y[i]]))

            gene_choices = [1]
            if len(BAM) < max_length_gene:
                gene_choices.append(2)
            if len(BAM) > min_length_gene:
                gene_choices.append(3)

            R = rnd.choice(gene_choices)
            if R == 1:
                BAM[rnd.randint(0, len(BAM) - 1)] = rnd.choice(['C', 'D'])
            elif R == 2:
                BAM.insert(rnd.randint(0,len(BAM)), rnd.choice(['C', 'D']))
            elif R == 3:
                BAM.pop(rnd.randint(0,len(BAM)-1))
            return BAM
        brain = think_memory[244]
        A = (game_round + 1) - (brain["sum old index"] + 1)
        if game_round > 0 and A == len(brain["now pattern"]):
            brain["pattern score"] += score[1-ap] - brain["old score"]
            brain["old score"] = score[1-ap]
            brain["sum old index"] += len(brain["now pattern"])
            if brain["pattern score"] > brain["best pattern 1"][1]:
                brain["best pattern 2"] = [brain["best pattern 1"][0][:], brain["best pattern 1"][1]]
                brain["best pattern 1"] = [brain["now pattern"][:], brain["pattern score"]]
            elif brain["pattern score"] > brain["best pattern 2"][1]:
                brain["best pattern 2"] = [brain["now pattern"][:], brain["pattern score"]]
            brain["now pattern"] = breeding_and_mutating(brain["best pattern 1"][0], brain["best pattern 2"][0])
            brain["pattern score"] = 0 
            A = (game_round + 1) - (brain["sum old index"] + 1)
        if A < len(brain["now pattern"]):
            brain["pattern score"] += score[1-ap] - brain["old score"]
            brain["old score"] = score[1-ap]
            return brain["now pattern"][A]
    elif strategy == 'Tit For Tat Killer':
        if game_round <= 2:
            return ('D', 'C', 'C')[game_round]
        return 'D' if [memory[1][ap], memory[2][ap]] == ['D', 'C'] else 'C'
    elif strategy == 'Zero Determinant 2026 / ZD-2026':
        # == BASES == 
        phi = 1.0
        s = 0.0
        l = 0.0

        # == BIASES ==
        if game_round > 1:
            if memory[-2][1-ap] == 'C':
                think_memory[313][1] += 1
                if memory[-1][ap] == 'D':
                    think_memory[313][0] += 1

        extort_prob = max(min((think_memory[313][0] / max(think_memory[313][1], 1)) * RPST['T'], 1), 0)
        phi = phi + 0
        s = s + (1 - extort_prob)
        l = l + ((RPST['P'] * extort_prob) + (RPST['R'] * (1 - extort_prob)))

        # == CLAMPS ==
        clamped_l = max(min(l, RPST['R']), RPST['P'])
        s_min = -min((RPST['T'] - clamped_l) / (clamped_l - RPST['S']), (clamped_l - RPST['S']) / (RPST['T'] - clamped_l))

        phi = max(min(phi, 1), 0)
        s = max(min(s, 1), s_min)
        l = clamped_l

        # == ZERO DETERMINANT COMPONENT ==
        p1 = 1 - phi * (1 - s) * (RPST['R'] - l)
        p2 = 1 - phi * (s * (l - RPST['S']) + (RPST['T'] - l))
        p3 = phi * ((l - RPST['S']) + s * (RPST['T'] - l))
        p4 = phi * (1 - s) * (l - RPST['P'])

        # == OUTPUT ==
        four_vector = [p1, p2, p3, p4]
        return 'C' if rnd.random() <= four_vector[((2 * (memory[-1][1-ap] == 'D')) + (memory[-1][ap] == 'D')) if game_round > 0 else 0] else 'D'
    elif strategy == 'Zero Determinant Psychological War 2026 / ZD-PsyWar2026':
        # == BASES == 
        phi = 0.8
        s = 0.15
        l = 0.2

        # == BIASES ==
        if game_round > 1:
            if memory[-2][1-ap] == 'C':
                think_memory[317][1] += 1
                if memory[-1][ap] == 'D':
                    think_memory[317][0] += 1

        extort_prob = max(min((think_memory[317][0] / max(think_memory[317][1], 1)) * RPST['T'], 1), 0)
        phi = phi + 0
        s = s + ((1 - extort_prob) * 0.7)
        l = l + ((RPST['P'] * extort_prob) + (RPST['R'] * (1 - extort_prob)))

        # == CLAMPS ==
        clamped_l = max(min(l, RPST['R']), RPST['P'])
        s_min = -min((RPST['T'] - clamped_l) / (clamped_l - RPST['S']), (clamped_l - RPST['S']) / (RPST['T'] - clamped_l))

        phi = max(min(phi, 1), 0)
        s = max(min(s, 1), s_min)
        l = clamped_l

        # == ZERO DETERMINANT COMPONENT ==
        p1 = 1 - phi * (1 - s) * (RPST['R'] - l)
        p2 = 1 - phi * (s * (l - RPST['S']) + (RPST['T'] - l))
        p3 = phi * ((l - RPST['S']) + s * (RPST['T'] - l))
        p4 = phi * (1 - s) * (l - RPST['P'])

        # == OUTPUT ==
        four_vector = [p1, p2, p3, p4]
        return 'C' if rnd.random() <= four_vector[((2 * (memory[-1][1-ap] == 'D')) + (memory[-1][ap] == 'D')) if game_round > 0 else 0] else 'D'
    elif strategy == 'Sort':
        if not game_round:
            return 'C'
        B = min(10, game_round)
        inp = [memory[-1 - ((B - 1) - i)][ap] for i in range(B)]
        if sorted(inp) == inp:
            return 'C'
        return 'D'
    elif strategy == 'Suspicious / Suspicious Always Cooperate / Suspicious Cooperator':
        return 'D' if game_round == 0 else 'C'
    elif strategy == 'Backrooms':
        decay_rate = 0.05
        think_memory[337][1] = {i: ((think_memory[337][1][i] * (1 - decay_rate)) + (rnd.random() * decay_rate)) for i in think_memory[337][1]}
        if game_round % 25 == 0 and game_round > 0:
            new_dict = {}
            total = {}
            for i in range(24):
                if new_dict.get(memory[-1 - (23 - i)][1-ap] + memory[-1 - (23 - i)][ap]) == None:
                    new_dict[memory[-1 - (23 - i)][1-ap] + memory[-1 - (23 - i)][ap]] = 0
                    total[memory[-1 - (23 - i)][1-ap] + memory[-1 - (23 - i)][ap]] = 0
                new_dict[memory[-1 - (23 - i)][1-ap] + memory[-1 - (23 - i)][ap]] += (memory[-1 - (24 - i)][ap] == 'C')
                total[memory[-1 - (23 - i)][1-ap] + memory[-1 - (23 - i)][ap]] += 1
            think_memory[337][0] = {i: ((new_dict[i] / total[i]) if i in new_dict else ((think_memory[337][0][i] * (1 - decay_rate)) + (rnd.random() * decay_rate))) for i in ['CC', 'CD', 'DC', 'DD']}
            think_memory[337][1] = {i: think_memory[337][0][i] for i in think_memory[337][0]}
        think_memory[337][1] = {i: ((think_memory[337][1][i] * (1 - decay_rate)) + (rnd.random() * decay_rate)) for i in think_memory[337][1]}
        return 'C' if rnd.random() <= think_memory[337][1][(memory[-1][1-ap] + memory[-1][ap]) if game_round > 0 else 'CC'] else 'D'
    elif strategy == 'Always Skip / Skiptor':
        return '[SKIP]'
    elif strategy == 'Out For Tat':
        if not game_round:
            return 'C'
        return '[SKIP]' if memory[-1][ap] == 'D' else 'C'
    elif strategy == 'Bandit':
        learning_rate = 0.1
        epsilon = 0.1
        Q_table = think_memory[339][0]

        if game_round > 0:
            old_score = think_memory[339][1]
            Q_old = Q_table[memory[-2][ap] if game_round > 1 else 'C'][memory[-1][1-ap]]
            Q_new = Q_old + learning_rate * ((old_score - score[1-ap]) - Q_old)
            think_memory[339][1] = score[1-ap]

        if rnd.random() <= epsilon:
            return rnd.choice(['C', 'D'])

        E_sh = Q_table[memory[-1][ap] if game_round > 0 else 'C']['C']
        E_st = Q_table[memory[-1][ap] if game_round > 0 else 'C']['D']

        if E_sh == E_st:
            return rnd.choice(['C', 'D'])

        if E_sh > E_st:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Inequity Averse Q-Learner':
        brain = think_memory[340][0]
        another_factor = think_memory[340][1]

        A = min(12, game_round) - 1
        opp_inp = tuple(['C' for _ in range(12 - min(12, game_round))] + [memory[-1 - (A - i)][ap] for i in range(min(12, game_round))])
        self_inp = tuple(['C' for _ in range(12 - min(12, game_round))] + [memory[-1 - (A - i)][1-ap] for i in range(min(12, game_round))])
        inp = (opp_inp, self_inp)

        if brain.get(inp) == None:
            brain[inp] = [0,0]

        if game_round > 0 and another_factor[0][1] != 0.5:
            reward = 0
            if memory[-1][ap] == 'C':
                if memory[-1][1-ap] == 'C':
                    reward = RPST['R']
                    opp_reward = RPST['R']
                else:
                    reward = RPST['T']
                    opp_reward = RPST['S']
            else:
                if memory[-1][1-ap] == 'C':
                    reward = RPST['S']
                    opp_reward = RPST['T']
                else:
                    reward = RPST['P']
                    opp_reward = RPST['P']

            c_factor = brain[inp][0]
            d_factor = brain[inp][1]

            alpha = 0.5
            beta = 0.5

            utility = reward - ((alpha * max(0, reward - opp_reward)) - (beta * max(0, opp_reward - reward)))
            
            a = 0.9
            gamma = 0.9
            Q_old = brain[another_factor[0][0]][another_factor[0][1]]
            Q_new = Q_old + (a * (utility + ((gamma * max(c_factor, d_factor)) - Q_old)))
            brain[another_factor[0][0]][another_factor[0][1]] = Q_new

        if rnd.random() <= another_factor[1]:
            Action = rnd.choice(['C', 'D'])
            another_factor[0] = [inp, (Action == 'D')]
            return Action
        decay = 1
        min_epsilon = 0
        another_factor[1] = max(another_factor[1] * decay, min_epsilon)

        if brain[inp][0] == brain[inp][1]:
            Action = rnd.choice(['C', 'D'])
            another_factor[0] = [inp, (Action == 'D')]
            return Action
        best_choice = max(brain[inp])
        if best_choice == brain[inp][0]:
            another_factor[0] = [inp, 0]
            return 'C'
        else:
            another_factor[0] = [inp, 1]
            return 'D'
    elif strategy == 'Sarsa Q-Learner':
        brain = think_memory[341][0]
        another_factor = think_memory[341][1]

        A = min(12, game_round) - 1
        opp_inp = tuple(['C' for _ in range(12 - min(12, game_round))] + [memory[-1 - (A - i)][ap] for i in range(min(12, game_round))])
        self_inp = tuple(['C' for _ in range(12 - min(12, game_round))] + [memory[-1 - (A - i)][1-ap] for i in range(min(12, game_round))])
        inp = (opp_inp, self_inp)

        if brain.get(inp) == None:
            brain[inp] = [0,0]

        if game_round > 0 and another_factor[0][1] != 0.5:
            reward = 0
            if memory[-1][ap] == 'C':
                if memory[-1][1-ap] == 'C':
                    reward = RPST['R']
                else:
                    reward = RPST['T']
            else:
                if memory[-1][1-ap] == 'C':
                    reward = RPST['S']
                else:
                    reward = RPST['P']

            c_factor = brain[inp][0]
            d_factor = brain[inp][1]
            
            a = 0.9
            gamma = 0.9
            Q_old = brain[another_factor[0][0]][another_factor[0][1]]
            Q_new = Q_old + (a * (reward + ((gamma * [c_factor, d_factor][(memory[-1][1-ap] == 'D')]) - Q_old)))
            brain[another_factor[0][0]][another_factor[0][1]] = Q_new

        if rnd.random() <= another_factor[1]:
            Action = rnd.choice(['C', 'D'])
            another_factor[0] = [inp, (Action == 'D')]
            return Action
        decay = 1
        min_epsilon = 0
        another_factor[1] = max(another_factor[1] * decay, min_epsilon)

        if brain[inp][0] == brain[inp][1]:
            Action = rnd.choice(['C', 'D'])
            another_factor[0] = [inp, (Action == 'D')]
            return Action
        best_choice = max(brain[inp])
        if best_choice == brain[inp][0]:
            another_factor[0] = [inp, 0]
            return 'C'
        else:
            another_factor[0] = [inp, 1]
            return 'D'
    elif strategy == 'REINFORCE':
        brain = think_memory[342]
        learning_rate = 0.5
        payoff = {
            'CC': RPST['R'],
            'CD': RPST['S'],
            'DC': RPST['T'],
            'DD': RPST['P']
        }
        if game_round > 0:
            brain[brain[2]] += learning_rate * (payoff[memory[-1][1-ap] + memory[-1][ap]] - avg_RPST)
        total = mth.exp(brain[0]) + mth.exp(brain[1])
        E_sh = mth.exp(brain[0]) / total
        E_st = 1 - E_sh
        if E_sh == E_st:
            Action = rnd.choice(['C', 'D'])
            brain[2] = (Action == 'D')
            return Action
        if E_sh > E_st:
            brain[2] = 0
            return 'C'
        else:
            brain[2] = 1
            return 'D'
    elif strategy == 'Dyna Q-Learner':
        brain = think_memory[343][0]
        another_factor = think_memory[343][1]

        A = min(12, game_round) - 1
        opp_inp = tuple(['C' for _ in range(12 - min(12, game_round))] + [memory[-1 - (A - i)][ap] for i in range(min(12, game_round))])
        self_inp = tuple(['C' for _ in range(12 - min(12, game_round))] + [memory[-1 - (A - i)][1-ap] for i in range(min(12, game_round))])
        inp = (opp_inp, self_inp)

        if brain.get(inp) == None:
            brain[inp] = [0,0]

        if game_round > 0 and another_factor[0][1] != 0.5:

            reward = 0
            if memory[-1][ap] == 'C':
                if memory[-1][1-ap] == 'C':
                    reward = RPST['R']
                else:
                    reward = RPST['T']
            else:
                if memory[-1][1-ap] == 'C':
                    reward = RPST['S']
                else:
                    reward = RPST['P']

            a = 0.9
            gamma = 0.9
            c_factor = brain[inp][0]
            d_factor = brain[inp][1]
            Q_old = brain[another_factor[0][0]][another_factor[0][1]]
            Q_new = Q_old + (a * (reward + ((gamma * max(c_factor, d_factor)) - Q_old)))
            
            for i in range(5):
                test = rnd.choice([i for i in brain])
                reward = 0
                if test[ap][-1] == 'C':
                    if test[1-ap][-1] == 'C':
                        reward = RPST['R']
                    else:
                        reward = RPST['T']
                else:
                    if test[1-ap][-1] == 'C':
                        reward = RPST['S']
                    else:
                        reward = RPST['P']

                c_factor = brain[inp][0]
                d_factor = brain[inp][1]
                Q_new += (a * (reward + ((gamma * max(c_factor, d_factor)) - Q_new)))

            brain[another_factor[0][0]][another_factor[0][1]] = Q_new

        if rnd.random() <= another_factor[1]:
            Action = rnd.choice(['C', 'D'])
            another_factor[0] = [inp, (Action == 'D')]
            return Action
        decay = 1
        min_epsilon = 0
        another_factor[1] = max(another_factor[1] * decay, min_epsilon)

        if brain[inp][0] == brain[inp][1]:
            Action = rnd.choice(['C', 'D'])
            another_factor[0] = [inp, (Action == 'D')]
            return Action
        best_choice = max(brain[inp])
        if best_choice == brain[inp][0]:
            another_factor[0] = [inp, 0]
            return 'C'
        else:
            another_factor[0] = [inp, 1]
            return 'D'
    elif strategy == 'Continuous Q-Learner Avg':
        brain = think_memory[344]
        payoff = {
            'CC': RPST['R'],
            'CD': RPST['S'],
            'DC': RPST['T'],
            'DD': RPST['P']
        }

        reward = avg_RPST
        if game_round > 0:
            learning_rate = 0.1
            brain[0] = ((1 - learning_rate) * brain[0]) + (learning_rate * (memory[-1][ap] == 'C'))
            brain[1] = ((1 - learning_rate) * brain[1]) + (learning_rate * (memory[-1][ap] == 'D'))

            reward = payoff[memory[-1][1-ap] + memory[-1][ap]]
        X_sh = (brain[0] * payoff['CC']) + (brain[1] * payoff['CD'])
        X_st = (brain[0] * payoff['DC']) + (brain[1] * payoff['DD'])

        Action = reward - ((X_sh + X_st) / 2)
        kappa = 2.0
        A = 1 / (1 + mth.exp(-Action * kappa))
        if rnd.random() <= A:
            return 'C'
        else:
            return 'D'
    elif strategy == 'Q-Learning A-Learner':
        exploitation_rate = 0.9
        exploration_rate = 0.1

        if game_round > 0:
            gamma = 0.9
            c_factor = sum(think_memory[345][0][i][0] for i in think_memory[345][0]) / len(think_memory[345][0])
            d_factor = sum(think_memory[345][0][i][1] for i in think_memory[345][0]) / len(think_memory[345][0])
            old_cond = think_memory[345][1]
            old_score = think_memory[345][2]
            reward = (score[1-ap] - old_score)
            think_memory[345][0][old_cond][(memory[-1][1-ap] == 'D')] += (exploitation_rate * (reward + ((gamma * max(c_factor, d_factor)) - think_memory[345][0][old_cond][(memory[-1][1-ap] == 'D')])))
            think_memory[345][2] = score[1-ap]

        cond = (memory[-1][1-ap] + memory[-1][ap]) if game_round > 0 else 'CC'
        think_memory[345][1] = cond
        if (rnd.random() <= exploration_rate) or (think_memory[345][0][cond][0] == think_memory[345][0][cond][1]):
            return rnd.choice(memory[-1]) if game_round > 0 else 'C'
        if think_memory[345][0][cond][0] > think_memory[345][0][cond][1]:
            return 'C'
        else:
            return 'D'
    elif strategy == 'A-Actor&Critic / A-A&C':
        brain = think_memory[346]
        if game_round > 0:
            learning_rate = 0.5
            payoff = {
                'CC':RPST['R'],
                'CD':RPST['S'],
                'DC':RPST['T'],
                'DD':RPST['P']
            }
            brain[(memory[-2][1-ap] + memory[-2][ap]) if game_round > 1 else 'CC'][(memory[-1][1-ap] == 'D')] += learning_rate * (payoff[(memory[-1][1-ap] + memory[-1][ap])] - (score[1-ap] / game_round))
        pref = brain[(memory[-1][1-ap] + memory[-1][ap]) if game_round > 0 else 'CC']
        return rnd_prob(1 / (1 + mth.exp(pref[1] - pref[0])))
    elif strategy == 'Japan Empire Emperor':
        if not game_round:
            next_chat[0] = '私はお前の皇帝だ!'
            return 'C'
        elif game_round == 1:
            if chats[ap] == '私はあなたの兵士だ!':
                think_memory[347] = True
        if think_memory[347]:
            return 'D'
        return 'C' if score[1-ap] >= score[ap] else 'D'
    elif strategy == 'Japan Empire Soldier 1':
        if not game_round:
            next_chat[0] = '私はあなたの兵士だ!'
            return 'C'
        elif game_round == 1:
            if chats[ap] in ('私はお前の皇帝だ!', '私はあなたの兵士だ!'):
                think_memory[348] = True
        if think_memory[348]:
            return 'C'
        return '[SKIP]'
    elif strategy == 'Japan Empire Soldier 2':
        if not game_round:
            next_chat[0] = '私はあなたの兵士だ!'
            return 'C'
        elif game_round == 1:
            if chats[ap] in ('私はお前の皇帝だ!', '私はあなたの兵士だ!'):
                think_memory[349] = True
        if think_memory[349]:
            return 'C'
        return '[SKIP]'
    elif strategy == 'Japan Empire Soldier 3':
        if not game_round:
            next_chat[0] = '私はあなたの兵士だ!'
            return 'C'
        elif game_round == 1:
            if chats[ap] in ('私はお前の皇帝だ!', '私はあなたの兵士だ!'):
                think_memory[350] = True
        if think_memory[350]:
            return 'C'
        return '[SKIP]'
    elif strategy == 'Japan Empire Soldier 4':
        if not game_round:
            next_chat[0] = '私はあなたの兵士だ!'
            return 'C'
        elif game_round == 1:
            if chats[ap] in ('私はお前の皇帝だ!', '私はあなたの兵士だ!'):
                think_memory[351] = True
        if think_memory[351]:
            return 'C'
        return '[SKIP]'
    elif strategy == 'Japan Empire Soldier 5':
        if not game_round:
            next_chat[0] = '私はあなたの兵士だ!'
            return 'C'
        elif game_round == 1:
            if chats[ap] in ('私はお前の皇帝だ!', '私はあなたの兵士だ!'):
                think_memory[352] = True
        if think_memory[352]:
            return 'C'
        return '[SKIP]'
    elif strategy == 'Japan Empire Soldier 6':
        if not game_round:
            next_chat[0] = '私はあなたの兵士だ!'
            return 'C'
        elif game_round == 1:
            if chats[ap] in ('私はお前の皇帝だ!', '私はあなたの兵士だ!'):
                think_memory[353] = True
        if think_memory[353]:
            return 'C'
        return '[SKIP]'
    elif strategy == 'Japan Empire Soldier 7':
        if not game_round:
            next_chat[0] = '私はあなたの兵士だ!'
            return 'C'
        elif game_round == 1:
            if chats[ap] in ('私はお前の皇帝だ!', '私はあなたの兵士だ!'):
                think_memory[354] = True
        if think_memory[354]:
            return 'C'
        return '[SKIP]'
    elif strategy == 'Japan Empire Soldier 8':
        if not game_round:
            next_chat[0] = '私はあなたの兵士だ!'
            return 'C'
        elif game_round == 1:
            if chats[ap] in ('私はお前の皇帝だ!', '私はあなたの兵士だ!'):
                think_memory[355] = True
        if think_memory[355]:
            return 'C'
        return '[SKIP]'
    elif strategy == 'Japan Empire Soldier 9':
        if not game_round:
            next_chat[0] = '私はあなたの兵士だ!'
            return 'C'
        elif game_round == 1:
            if chats[ap] in ('私はお前の皇帝だ!', '私はあなたの兵士だ!'):
                think_memory[356] = True
        if think_memory[356]:
            return 'C'
        return '[SKIP]'
    elif strategy == 'Japan Empire Soldier 10':
        if not game_round:
            next_chat[0] = '私はあなたの兵士だ!'
            return 'C'
        elif game_round == 1:
            if chats[ap] in ('私はお前の皇帝だ!', '私はあなたの兵士だ!'):
                think_memory[357] = True
        if think_memory[357]:
            return 'C'
        return '[SKIP]'
    elif strategy == 'Prabowo':
        if not game_round:
            return 'C'
        if think_memory[358] == 4:
            if memory[-1][ap] == 'D':
                think_memory[358] = 0
                next_chat[0] = 'SAWIT KAN POHON JUGA!!!'
            else:
                next_chat[0] = 'AKU SUKA SAWIT!!!'
            return 'D'
        if 1 <= think_memory[358] <= 3:
            if think_memory[358] > 1:
                think_memory[358] -= 1
                next_chat[0] = 'MBG!!!'
                return ('D', 'C')[2 - think_memory[358]]
            if memory[-1][ap] == 'D':
                think_memory[358] = 0
                next_chat[0] = 'SAYA KAN KADANG-KADANG SALAH MAKAN JUGA!!!'
        if rnd.random() <= 0.2:
            think_memory[358] = 3 if rnd.random() <= 0.75 else 4
        if memory[-1][ap] == 'D':
            next_chat[0] = 'HEY, ANTEK-ANTEK ASING!!!'
            return 'D'
        return 'C'
    elif strategy == 'Gen 5_64_200':
        def sigmoid10(x):
            return 1 / (1 + mth.exp(-max(min(x * 10, 10), -10)))
        return rnd_prob(sigmoid10(((((memory[-3][ap] == 'C') if game_round >= 3 else 0.5) * ((memory[-3][1-ap] == 'C') if game_round >= 3 else 0.5)) - ((((((memory[-1][1-ap] == 'C') if game_round >= 1 else 0.5) / (((memory[-10][1-ap] == 'C') if game_round >= 10 else 0.5) if ((memory[-10][1-ap] == 'C') if game_round >= 10 else 0.5) != 0 else 1.0)) if ((memory[-1][ap] == 'C') if game_round >= 1 else 0.5) > 0 else ((memory[-8][1-ap] == 'C') if game_round >= 8 else 0.5)) / (((memory[-2][1-ap] == 'C') if game_round >= 2 else 0.5) if ((memory[-2][1-ap] == 'C') if game_round >= 2 else 0.5) != 0 else 1.0)) / (0.0 if 0.0 != 0 else 1.0)))))
    elif strategy == 'Gen 3_64_200':
        def sigmoid10(x):
            return 1 / (1 + mth.exp(-max(min(x * 10, 10), -10)))
        return rnd_prob(sigmoid10((0.0 - ((((memory[-10][ap] == 'C') if game_round >= 10 else 0.5) + ((memory[-1][ap] == 'C') if game_round >= 1 else 0.5)) / (((memory[-8][1-ap] == 'C') if game_round >= 8 else 0.5) if ((memory[-8][1-ap] == 'C') if game_round >= 8 else 0.5) != 0 else 1.0)))))
    elif strategy == 'Gen 4_64_200':
        def sigmoid10(x):
            return 1 / (1 + mth.exp(-max(min(x * 10, 10), -10)))
        return rnd_prob(sigmoid10(((((((memory[-1][ap] == 'C') if game_round >= 1 else 0.5) - ((memory[-1][1-ap] == 'C') if game_round >= 1 else 0.5)) * ((memory[-8][1-ap] == 'C') if game_round >= 8 else 0.5)) - ((((memory[-7][1-ap] == 'C') if game_round >= 7 else 0.5) * ((memory[-9][ap] == 'C') if game_round >= 9 else 0.5)) / ((1.0 * ((memory[-7][1-ap] == 'C') if game_round >= 7 else 0.5)) if (1.0 * ((memory[-7][1-ap] == 'C') if game_round >= 7 else 0.5)) != 0 else 1.0))) - ((memory[-1][ap] == 'C') if game_round >= 1 else 0.5))))
    elif strategy == 'Gen 2_64_200':
        def sigmoid10(x):
            return 1 / (1 + mth.exp(-max(min(x * 10, 10), -10)))
        return rnd_prob(sigmoid10(((((memory[-7][1-ap] == 'C') if game_round >= 7 else 0.5) if 1.0 > 0 else ((memory[-8][1-ap] == 'C') if game_round >= 8 else 0.5)) - (((memory[-2][ap] == 'C') if game_round >= 2 else 0.5) / (((memory[-1][1-ap] == 'C') if game_round >= 1 else 0.5) if ((memory[-1][1-ap] == 'C') if game_round >= 1 else 0.5) != 0 else 1.0)))))
    elif strategy == 'Gen 1_64_200':
        def sigmoid10(x):
            return 1 / (1 + mth.exp(-max(min(x * 10, 10), -10)))
        return rnd_prob(sigmoid10((((memory[-1][1-ap] == 'C') if game_round >= 1 else 0.5) - 1.0)))
    elif strategy == 'Bahlil':
        if not game_round:
            return 'C'
        if think_memory[359]:
            if memory[-1][ap] == 'C':
                next_chat[0] = 'BENSIN OPLOSAN!!!'
                return 'D'
            else:
                think_memory[359] = False
                next_chat[0] = 'sorry ye >:-)'
                return 'C'
        if rnd.random() <= 0.2:
            think_memory[359] = True
        if memory[-1][ap] == 'D':
            next_chat[0] = 'chill aja coy'
            return 'D'
        return 'C'
        
    else:
        print(f'ERROR NAME: {strategy}')
        return None

def run_match(args):
    i, j, with_noise_flag, noise_prob_local = args

    global memory, score, strat_memory
    global strategy_chat, strategy_next_chat, game_round

    memory = []
    score = [0, 0]
    strat_memory = [reset_think_memory(), reset_think_memory()]
    strategy_chat = [[''], ['']]
    strategy_next_chat = [[''], ['']]
    game_round = 0
    local_round = tournament_avg_round + rnd.randint(-tournament_round_randomness, tournament_round_randomness)

    while game_round < local_round:
        strategy_next_chat = [[''], ['']]
        c1 = think(strategies[i], 1)
        c2 = think(strategies[j], 0)

        if with_noise_flag:
            if rnd.random() <= noise_prob_local:
                c1 = rnd.choice(['C', 'D'])
            if rnd.random() <= noise_prob_local:
                c2 = rnd.choice(['C', 'D'])

        if (c1 not in ('C', 'D', '[SKIP]')) or (c2 not in ('C', 'D', '[SKIP]')):
            ERROR = []
            if c1 not in ('C', 'D', '[SKIP]'):
                ERROR.append(strategies[i])
            if c2 not in ('C', 'D', '[SKIP]'):
                ERROR.append(strategies[j])
            raise ValueError(f'ERROR {ERROR}')

        if '[SKIP]' in (c1, c2):
            break

        memory.append((c1, c2))

        if c1 == c2:
            update_score(RPST['R'], RPST['R']) if c1 == 'C' else update_score(RPST['P'], RPST['P'])
        else:
            update_score(RPST['T'], RPST['S']) if c1 == 'D' else update_score(RPST['S'], RPST['T'])

        strategy_chat[0][0] = strategy_next_chat[0][0]
        strategy_chat[1][0] = strategy_next_chat[1][0]
        game_round += 1

    return (i, j, score[0], score[1])

strategies = ['2 Tits For Tat', '3 Seconds Mirror', '3-Tree Of Prediction', 'A-Actor&Critic / A-A&C', 'ALLC OR ALLD', 'AMERICA! / USA / Freedoom', 'AON2', 'AQUA', 'Adams', 'Adaptive', 'Adaptive Pavlov 2006', 'Adaptive Pavlov 2011', 'Adaptive Tit For Tat', 'Adaptive Tit For Tat Attar Version', 'Adaptor Brief', 'Adaptor Long', 'Aggravater', 'Alan', 'Alexei', 'Almy', 'Alternator', 'Alternator Hunter', 'Always Cooperate / Cooperator', 'Always Defect / Defector', 'Always Skip / Skiptor', 'Amazon / Piraha', 'Ambuelh & Kickey', 'Analogy', 'Anatol', 'Anderson', 'Angry Tit For Tat', 'Anti Cycler', 'Anti Noise Tit For Tat', 'Anti Tit For Tat / Psycho', 'Appeaser', 'Appold', 'Arrogant Q-Learner', 'Average Copier', 'Back Stabber', 'Backrooms', 'Bad Random', 'Bahlil', 'Bandit', 'Bandit UCB', 'Batell', 'Bayes Attar Version', 'Bayes Foomii Version', 'Beta Tester', 'Better & Better', 'Black', 'Bluff', 'Bluff Master', 'Boltzmann Brain', 'Borufsen', 'Boxer', 'Bros Mind', 'Bully / Reverse Tit For Tat', 'Burn Both Ends / BBE', 'Bush Mosteller', 'Buzzer', 'CS / Collective Strategy', 'Caerbannog', 'Calamity', 'Calculator', 'Calculator The Nerd Of AI Version', 'Capitalist', 'Capri', 'Cautious Q-Learner', 'Cave', 'Caveman', 'Cerebrum', 'Champion', 'Charity', 'Chatter', 'Clement Coldridge', 'Colbert', 'Communalist', 'Continuous Q-Learner Avg', 'Continuous Q-Learner Max', 'Contrite Grudger', 'Contrite Tit For Tat', 'Control 1', 'Control 10', 'Control 11', 'Control 2', 'Control 3', 'Control 4', 'Control 5', 'Control 6', 'Control 7', 'Control 8', 'Control 9', 'Cooperator Hunter', 'Crabby', 'Crabby Attar Version', 'Crow', 'Cruelity', 'Cult Bishop A', 'Cult Bishop B', 'Cult Citizen A', 'Cult Citizen B', 'Cult Citizen C', 'Cult Leader', 'Curiosity', 'Cycle Hunter', 'Cycler CCCCCD', 'Cycler CCCD', 'Cycler CCCDCD', 'Cycler CCD', 'Cycler DC', 'Cycler DCCCC', 'Cycler DDC', 'DBS / Derived Belief Strategy', 'DDoS', 'Davis', 'Dawes & Batell', 'Deadlock Breaker', 'Decay Given', 'Defector Hunter', 'Delayed AON1', 'Desire', 'Desprate', 'Detective', 'Devil Staircase', 'Diplomat', 'Dont Bite The Hand That Feeds You', 'Double Crosser', 'Double Q-Learner', 'Double Resurrection', 'Doubler', 'Downing', 'Downing V2', 'Duisman', 'Dyna Q-Learner', 'Dynamic 2 Tits For Tat', 'EGOist', 'Eatherley', 'Echo Adaptive', 'Echo Fisher', 'Egoist A-Learner', 'Eliza / Therapist', 'Entropy Sentinel', 'Eric', 'Ethanol / Alcohol', 'Eugine Nier', 'Euler', 'Eventual Cycle Hunter', 'Evil Alliance', 'Evolvable Cycler Attar Version', 'Evolved FSM 16', 'Evolved FSM 16 Noise 05', 'Evolved FSM 4', 'Evolved FSM 6', 'Evolved HMM 5', 'Evolved Looker Up 1_1_1', 'Evolved Looker Up 2_2_2', 'Exploratory Lenient Grim 2', 'Exploratory Tit For 3 Tats', 'FAWS', 'Falk & Lanqsted', 'False Cooperator', 'Falsity', 'Feathers', 'Feld', 'Female / Woman', 'Fibonacci', 'Firm But Fair / Firm For Tat', 'Fool Me Once', 'Forest-RND', 'Forest-XG', 'Forgetful Fool Me Once', 'Forgetful Grudger', 'Forgiver', 'Forgiver The Nerd Of AI Version', 'Forgiving Fate', 'Forgiving Tit For Tat', 'Fortress 3', 'Fortress 4', 'Frankl', 'Free Rider', 'Frequency Analyzer', 'Freud', 'Friedland', 'Friedman / Grudger / Grim trigger', 'Gambler Attar Version', 'Game Theory Analyzer / GT-A', 'Game Theory Explorer / GT-E', 'Gaslighter', 'Gateman', 'Gen 1_64_200', 'Gen 2_64_200', 'Gen 3_64_200', 'Gen 4_64_200', 'Gen 5_64_200', 'Generous Tit For Tat', 'Generous Tit For Tat Axelrod Project Contributor Team Version', 'Generous Tit For Tat Less Wrong Version / Tit For Tat But Pico Generous', 'Genetic Algo 1', 'Genetic Algo 2', 'George', 'Getzler', 'Giles / Worse & Worse 3', 'Gladstein / Tester', 'Go By Minority', 'Gold Digger Foomii Version', 'Golden Mean', 'Good', 'Good Random', 'Graaskamp', 'Graaskamp & Katzen', 'Gradual', 'Gradual Cristal Version', 'Gradual Killer', 'Grisell / Go By Majority / Soft Go By Majority', 'Grofman', 'Grofman V2', 'Grudger Alternator', 'Grumpy', 'HMM Player', 'Half Sin Square', 'Hall', 'Handshake', 'Hard Go By Majority', 'Hard Prober', 'Hard Tit For 2 Tats / 2 Tits For 2 Tats', 'Hard Tit For Tat / 3 Tits For Tat', 'Harrington', 'Hesitant Q-Learner', 'Hollander', 'Hopeless', 'Hot Coffee', 'Hotz', 'Hufford', 'Hungry', 'Inequity Averse Q-Learner', 'Inverse', 'Inverse Punisher', 'Is There Something New?', 'Isolated Clan', 'Isolated Tit For Tat', 'Japan Empire Emperor', 'Japan Empire Soldier 1', 'Japan Empire Soldier 10', 'Japan Empire Soldier 2', 'Japan Empire Soldier 3', 'Japan Empire Soldier 4', 'Japan Empire Soldier 5', 'Japan Empire Soldier 6', 'Japan Empire Soldier 7', 'Japan Empire Soldier 8', 'Japan Empire Soldier 9', 'Javelin', 'Jem', 'Jesus Petry', 'Jones', 'Joss / Naive Prober', 'Josuah', 'Jung', 'Karen', 'Keyman', 'Killer', 'King Of 48 Laws Of Power', 'Kluepfel', 'Knowledgeable Worse & Worse', 'Konflikt Blue', 'Konflikt Green', 'Laran', 'Laws', 'Learner', 'Lefevre', 'Lenient Grim 2', 'Lenient Grim 3', 'Level Punisher', 'Leyland', 'Leyvraz', 'Liar Person', 'Lighty-Darky', 'Limited Retaliate', 'Limited Retaliate 2', 'Limited Retaliate 3', 'Line', 'Little Grudger', 'Long Horse', 'Long Term Strategist The Nerd Of AI Version', 'Look Up / Look Ahead', 'Loyal Foomii Version', 'MARS / Mimicry And Relative Similarity', 'MCMC / Markov Chain Monte Carlo', 'MEM2', 'MENACE / HER', 'Mafia A', 'Mafia B', 'Mafia C', 'Mafia D', 'Majapahit', 'Male / Man', 'Malthrin', 'Man In The Middle', 'Manipulator The Nerd Of AI Version', 'Manipulity', 'Markov', 'Matcher', 'Math Constant Hunter', 'Mauk', 'Mcgurrin', 'Memory Decay', 'Mensa', 'Meta Hunter', 'Meta Majority', 'Meta Minority', 'Meta Mixer', 'Meta Winner', 'Meta Winner Ensemble', 'Michaelos', 'Mikkelson', 'Mirror Foomii Version', 'Momentum', 'Monkey See, Monkey Do', 'Mosquito', 'N Tits For M Tats', 'Naive Q-Learner', 'Namdeirf / Anti-Grudger', 'Named Withheld', 'Nash', 'Nasty Tit For Tat', 'Negation', 'Neil A', 'Newman', 'NoName', 'Nussbacher', 'Nydegger', 'Observant', 'Odd-Even Go By Majority', 'Omega Tit For Tat', 'Once Bitten', 'Opportunist The Nerd Of AI Version', 'Opposite Grudger', 'Out For Tat', 'PSO Gambler 1_1_1', 'PSO Gambler 2_2_2', 'PSO Gambler 2_2_2 Noise 05', 'PSO Gambler Mem 1', 'Paranoid', 'Passive Tit For Tat', 'Patterned Adaptive Fortress', 'Patterned Adaptive Fortress 2', 'Pavlov / Win-Stay, Lose-Shift', 'Pavlov D / Suspicious Pavlov', 'Pavlov Tester', 'Pavolovo', 'Pebley', 'Perceptron', 'Phi', 'Pi', 'Pinkley', 'Prabowo', 'Praedator', 'Predator', 'Probe & Punish', 'Prober', 'Prober 2', 'Prober 3', 'Prober 4', 'Procastinater', 'Psycho Attar Version', 'Pun 1', 'Punisher', 'Q-Learning A-Learner', 'Quayle', 'REINFORCE', 'RNN & RTRL', 'Rabbie', 'Racister', 'Rack Block Shooter', 'Raider', 'Random', 'Random Exploiter 1', 'Random Exploiter 2', 'Random Exploiter 3', 'Random Exploiter 4', 'Random Hunter', 'Random Tit For Tat', 'Reactive Player', 'Remorseful Prober', 'Resurrection', 'Retaliate', 'Retaliate 2', 'Retaliate 3', 'Rice Field', 'Ripoff', 'Risky Q-Learner', 'Robertson', 'Rogers & Maslow', 'Rover', 'Rowsam', 'Rwallace', 'SOVIET UNION! / USSR / Inequality', 'SURPRISE ATTACK!', 'Sarsa Q-Learner', 'Score Equalizer', 'Second Chance', 'Second Chance Attar Version', 'Self Steem', 'Self-Love A-Learner', 'Short Mem', 'Shubik', 'Shurmann', 'Simple ANN', 'Simple Exploiter', 'Simple Identity ChecK', 'Skinner', 'Slanger', 'Slow Tit For 2 Tats 2', 'Smart Cat', 'Smart Generous Tit For Tat', 'Smart Go By Majority', 'Smart Snake', 'Smart-Clever', 'Smith', 'Smoody', 'Smooth Grudger', 'Smooth Tit For Tat', 'Sneaky Tit For Tat', 'Snodgrass', 'Social Average', 'Social Engineering', 'Socio', 'Soft Grudger', 'Solution B1', 'Solution B5', 'Sort', 'Spectrum Attacker', 'Spiteful CC', 'Spiteful Tit For Tat', 'Split Brain Syndrome', 'Springter', 'Stalker', 'Star S', 'Star SL', 'Star SN', 'Star Slave 1', 'Star Slave 10', 'Star Slave 2', 'Star Slave 3', 'Star Slave 4', 'Star Slave 5', 'Star Slave 6', 'Star Slave 7', 'Star Slave 8', 'Star Slave 9', 'Stein & Rapoport', 'Stochastic Cooperator', 'Stochastic WSLS', 'Stoic Mirror', 'Stoicalist', 'Stupidiot', 'Sumobox Fighter', 'Sun Tzu Bot', 'Suspicious / Suspicious Always Cooperate / Suspicious Cooperator', 'Suspicious Alternator', 'Suspicious Tit For Tat', 'Sweet Heart', 'Syifa', 'TF 1', 'TF 2', 'TF 3', 'TF2T & 2TFT', 'TOM Level 1 / Theory Of Mind Level 1', 'Tages', 'Teacher Pee While Standing, Student Pee While Running', 'The Calculated Mirror With A Fuse', 'The Cave Breaker', 'The Nasty A', 'Theseus', 'Threshold Punisher', 'Thresholded Trust Bot', 'Thue Morse', 'Thue Morse Inverse', 'Thumper', 'Tideman & Chieruzzi', 'Tideman & Chieruzzi V2', 'Tiny Brain', 'Tip For Tap', 'Tit For 2 Tats', 'Tit For 3 Tats', 'Tit For Increasing Tat', 'Tit For Tat', 'Tit For Tat Killer', 'Toxic Mirror', 'Trader', 'Tranquilizer', 'Treasure Hunt / Sugar Strategy', 'Tri-Brain', 'Tricky Cooperator', 'Tricky Defector', 'Tricky Level Punisher', 'Trust Ledger', 'Trust Score Sigmoid', 'Trusty Foomii Version', 'Tullock', 'Turan', 'Unpredictable Tit For Tat', 'Uranium', 'Usually Cooperates', 'Usually Defects', 'Vandal Foomii Version', 'Vengeful', 'Vengeful Cheater', 'Verity', 'Very Bad', 'Weiderman', 'Weiner', 'Whale', 'White', 'William', 'Willing', 'Win-Shift, Lose-Stay', 'Winner 12', 'Winner 21', 'Worse & Worse / Worse & Worse 1', 'Worse & Worse 2', 'Yamachi', 'Zero Day', 'Zero Determinant 2026 / ZD-2026', 'Zero Determinant Equalizer / ZD-Q', 'Zero Determinant Extortion / ZD-X', 'Zero Determinant Generous / ZD-G', 'Zero Determinant Generous Tit For Tat / ZD-GTFT', 'Zero Determinant Memory 2 / ZD-M2', 'Zero Determinant Mischief / ZD-MS', 'Zero Determinant Psychological War 2026 / ZD-PsyWar2026', 'Zero Determinant Set / ZD-S', 'Zimmerman']
strategies_creator = {'2 Tits For Tat': 'John Maynard Smith', '3 Seconds Mirror': 'Meta', '3-Tree Of Prediction': 'Attar', 'A-Actor&Critic / A-A&C': 'Attar & Gemini', 'ALLC OR ALLD': 'Marc Harper', 'AMERICA! / USA / Freedoom': 'Attar', 'AON2': 'C. Hilbe & L. A. Martinez-Vaquero & K. Chatterjee & M. A. Nowak', 'AQUA': 'Attar', 'Adams': 'William Adams', 'Adaptive': 'J. Li & P. Hingston & G. Kendall', 'Adaptive Pavlov 2006': 'J. Li', 'Adaptive Pavlov 2011': 'J. Li & P. Hingston & G. Kendall', 'Adaptive Tit For Tat': 'E. Tzafestas', 'Adaptive Tit For Tat Attar Version': 'Attar', 'Adaptor Brief': 'Christoph Hauert & Olaf Stenull', 'Adaptor Long': 'Christoph Hauert & Olaf Stenull', 'Aggravater': 'Thomas Campbell', 'Alan': 'Attar, Inspired By Alan Turing', 'Alexei': 'Alexei', 'Almy': 'Richard Almy', 'Alternator': 'Robert Axelrod', 'Alternator Hunter': 'Karol Langner', 'Always Cooperate / Cooperator': 'Merrill Flood & Melvin Dresher', 'Always Defect / Defector': 'Merrill Flood & Melvin Dresher', 'Always Skip / Skiptor': 'Attar', 'Amazon / Piraha': 'Attar', 'Ambuelh & Kickey': 'Ambuelh & Kickey', 'Analogy': 'Robert Gladstein', 'Anatol': 'Attar', 'Anderson': 'Robert Anderson', 'Angry Tit For Tat': 'Attar', 'Anti Cycler': 'Marc Harper', 'Anti Noise Tit For Tat': 'Attar', 'Anti Tit For Tat / Psycho': 'C. Hilbe & M. A. Nowak & A. Traulsen / D. Ashlock & E. Y. Kim & W. Ashlock', 'Appeaser': 'Jochen Muller', 'Appold': 'James Appold', 'Arrogant Q-Learner': 'Geraint Palmer', 'Average Copier': 'Geraint Palmer', 'Back Stabber': 'Thomas Campbell', 'Backrooms': 'Attar', 'Bad Random': 'Robert Axelrod', 'Bahlil': 'Attar', 'Bandit': 'Attar & Gemini', 'Bandit UCB': 'Auer & Cesa-Bianchi & Fischer', 'Batell': 'Robert Batell', 'Bayes Attar Version': 'Attar', 'Bayes Foomii Version': 'Foomii', 'Beta Tester': 'Attar', 'Better & Better': "LIFL / Laboratoire d'Informatique Fondamentale de Lille", 'Black': 'Paul E. Black', 'Bluff': 'Attar & My Aunt', 'Bluff Master': 'Perplexity', 'Boltzmann Brain': 'Attar, Inspired By Ludwig Boltzmann', 'Borufsen': 'Otto Borufsen', 'Boxer': 'A. R. L. R. Cunha & A. L. V. Coelho', 'Bros Mind': 'A. R. L. R. Cunha & A. L. V. Coelho', 'Bully / Reverse Tit For Tat': 'John Nachbar', 'Burn Both Ends / BBE': 'Louis Marinoff', 'Bush Mosteller': 'Luis R. Izquierdo & Segismundo S. Izquierdo', 'Buzzer': 'Attar', 'CS / Collective Strategy': 'J. Li & G. Kendall', 'Caerbannog': 'Caerbannog', 'Calamity': 'Attar, Inspired By KhanhsKMC', 'Calculator': "LIFL / Laboratoire d'Informatique Fondamentale de Lille", 'Calculator The Nerd Of AI Version': 'The Nerd Of AI', 'Capitalist': 'Chat-GPT', 'Capri': 'Y. Murase Et Al', 'Cautious Q-Learner': 'Geraint Palmer', 'Cave': 'Rob Cave', 'Caveman': 'Attar', 'Cerebrum': 'Attar & Chat-GPT & Gemini', 'Champion': 'Danny C. Champion', 'Charity': 'Attar', 'Chatter': 'Attar', 'Clement Coldridge': 'Attar', 'Colbert': 'William Colbert', 'Communalist': 'Chat-GPT', 'Continuous Q-Learner Avg': 'Attar', 'Continuous Q-Learner Max': 'Attar', 'Contrite Grudger': 'Attar', 'Contrite Tit For Tat': 'J. Wu & Robert Axelrod', 'Control 1': '[Less Wrong]', 'Control 10': '[Less Wrong]', 'Control 11': '[Less Wrong]', 'Control 2': '[Less Wrong]', 'Control 3': '[Less Wrong]', 'Control 4': '[Less Wrong]', 'Control 5': '[Less Wrong]', 'Control 6': '[Less Wrong]', 'Control 7': '[Less Wrong]', 'Control 8': '[Less Wrong]', 'Control 9': '[Less Wrong]', 'Cooperator Hunter': 'Karol Langner', 'Crabby': 'Zachary Danziger', 'Crabby Attar Version': 'Attar', 'Crow': 'Attar', 'Cruelity': 'Attar, Inspired By KhanhsKMC', 'Cult Bishop A': 'Attar', 'Cult Bishop B': 'Attar', 'Cult Citizen A': 'Attar', 'Cult Citizen B': 'Attar', 'Cult Citizen C': 'Attar', 'Cult Leader': 'Attar', 'Curiosity': 'Attar, Inspired By KhanhsKMC', 'Cycle Hunter': 'Marc Harper', 'Cycler CCCCCD': 'Marc Harper', 'Cycler CCCD': 'Marc Harper', 'Cycler CCCDCD': 'Marc Harper', 'Cycler CCD': 'Marc Harper', 'Cycler DC': 'Marc Harper', 'Cycler DCCCC': 'Attar', 'Cycler DDC': 'Marc Harper', 'DBS / Derived Belief Strategy': 'T.-C. Au & D. S. Nau', 'DDoS': 'Attar', 'Davis': 'Morton Davis', 'Dawes & Batell': 'Robyn Dawes & Robert Batell', 'Deadlock Breaker': 'D. Ashlock & E. Y. Kim ', 'Decay Given': 'Attar', 'Defector Hunter': 'Karol Langner', 'Delayed AON1': 'C. Hilbe & L. A. Martinez-Vaquero & K. Chatterjee & M. A. Nowak', 'Desire': 'Attar', 'Desprate': 'P. Van Den Berg & F. J. Weissing', 'Detective': 'Nicky Case', 'Devil Staircase': 'Attar', 'Diplomat': 'Perplexity', 'Dont Bite The Hand That Feeds You': 'Attar', 'Double Crosser': 'Thomas Campbell', 'Double Q-Learner': 'Attar', 'Double Resurrection': 'Eckhart Arnold', 'Doubler': "LIFL / Laboratoire d'Informatique Fondamentale de Lille", 'Downing': 'Leslie Downing', 'Downing V2': 'Leslie Downing', 'Duisman': 'George Duisman', 'Dyna Q-Learner': 'Attar', 'Dynamic 2 Tits For Tat': 'Grant Garrett-Grossman', 'EGOist': 'Attar', 'Eatherley': 'Graham J. Eatherley', 'Echo Adaptive': 'Copilot', 'Echo Fisher': 'Attar', 'Egoist A-Learner': 'Attar', 'Eliza / Therapist': 'Attar, Inspired By Joseph Weizenbaum', 'Entropy Sentinel': 'Chat-GPT', 'Eric': 'Attar & Erik & Chat-GPT', 'Ethanol / Alcohol': 'Attar', 'Eugine Nier': 'Eugine Nier', 'Euler': 'Timothy Standen', 'Eventual Cycle Hunter': 'Marc Harper', 'Evil Alliance': 'ArisKatsaris', 'Evolvable Cycler Attar Version': 'Attar', 'Evolved FSM 16': 'Marc Harper', 'Evolved FSM 16 Noise 05': 'Marc Harper', 'Evolved FSM 4': 'Marc Harper', 'Evolved FSM 6': 'Frederick Vincent & Dashiell Fryer', 'Evolved HMM 5': 'Marc Harper', 'Evolved Looker Up 1_1_1': 'Marc Harper', 'Evolved Looker Up 2_2_2': 'Marc Harper', 'Exploratory Lenient Grim 2': 'Yaroslav Rosokha & Julian Romero', 'Exploratory Tit For 3 Tats': 'Yaroslav Rosokha & Julian Romero', 'FAWS': 'FAWS', 'Falk & Lanqsted': 'Gideon Falk & Jon Lanqsted', 'False Cooperator': 'Yaroslav Rosokha & Julian Romero', 'Falsity': 'Attar, Inspired By Xqree', 'Feathers': 'T. E. Feathers', 'Feld': 'Scott Feld', 'Female / Woman': 'Attar', 'Fibonacci': 'Marc Harper', 'Firm But Fair / Firm For Tat': 'Marcus R. Frean', 'Fool Me Once': 'Marc Harper', 'Forest-RND': 'Attar', 'Forest-XG': 'Attar', 'Forgetful Fool Me Once': 'Marc Harper', 'Forgetful Grudger': 'Geraint Palmer', 'Forgiver': 'Thomas Campbell', 'Forgiver The Nerd Of AI Version': 'The Nerd Of AI', 'Forgiving Fate': 'Attar & Nabilah Shafirah', 'Forgiving Tit For Tat': 'Thomas Campbell', 'Fortress 3': 'W. Ashlock & D. Ashlock', 'Fortress 4': 'W. Ashlock & D. Ashlock', 'Frankl': 'Attar, Inspired By Viktor Frankl', 'Free Rider': 'Enquist & Leimar', 'Frequency Analyzer': 'Ian Miller', 'Freud': 'Attar, Inspired By Sigmund Freud', 'Friedland': 'Nehemia Friedland', 'Friedman / Grudger / Grim trigger': 'James W. Friedman', 'Gambler Attar Version': 'Attar', 'Game Theory Analyzer / GT-A': 'Attar', 'Game Theory Explorer / GT-E': 'Attar', 'Gaslighter': 'Attar & Nabilah Shafirah', 'Gateman': 'Attar', 'Gen 1_64_200': 'Attar', 'Gen 2_64_200': 'Attar', 'Gen 3_64_200': 'Attar', 'Gen 4_64_200': 'Attar', 'Gen 5_64_200': 'Attar', 'Generous Tit For Tat': 'Robert Axelrod', 'Generous Tit For Tat Axelrod Project Contributor Team Version': 'Axelrod Project Contributor Team', 'Generous Tit For Tat Less Wrong Version / Tit For Tat But Pico Generous': '[Less Wrong]', 'Genetic Algo 1': 'Attar', 'Genetic Algo 2': 'Attar', 'George': 'Attar', 'Getzler': 'Abraham Getzler', 'Giles / Worse & Worse 3': "Giles / LIFL / Laboratoire d'Informatique Fondamentale de Lille", 'Gladstein / Tester': 'David Gladstein', 'Go By Minority': 'Attar', 'Gold Digger Foomii Version': 'Foomii', 'Golden Mean': 'Dola', 'Good': 'Attar', 'Good Random': 'Robert Axelrod', 'Graaskamp': 'James Graaskamp', 'Graaskamp & Katzen': 'James Graaskamp & Ken Katzen', 'Gradual': 'B. Beaufils & J. Delahaye & P. Mathieu', 'Gradual Cristal Version': 'CRISTAL Lab & SMAC Team', 'Gradual Killer': "LIFL / Laboratoire d'Informatique Fondamentale de Lille", 'Grisell / Go By Majority / Soft Go By Majority': 'Gail Grisell / Robert Axelrod / S. Mittal & K. Deb', 'Grofman': 'Bernard Grofman', 'Grofman V2': 'Bernard Grofman', 'Grudger Alternator': 'Geraint Palmer', 'Grumpy': 'Jason Young', 'HMM Player': 'Marc Harper', 'Half Sin Square': 'Attar', 'Hall': 'John Hall', 'Handshake': 'Arthur J. Robson', 'Hard Go By Majority': 'S. Mittal & K. Deb', 'Hard Prober': "LIFL / Laboratoire d'Informatique Fondamentale de Lille", 'Hard Tit For 2 Tats / 2 Tits For 2 Tats': 'A. J. Stewart & J. B. Plotkin', 'Hard Tit For Tat / 3 Tits For Tat': '[PD2017]', 'Harrington': 'Paul Harrington', 'Hesitant Q-Learner': 'Geraint Palmer', 'Hollander': 'Edwin Hollander', 'Hopeless': 'P. Van Den Berg & F. J. Weissing', 'Hot Coffee': 'Attar', 'Hotz': 'Gunter Hotz', 'Hufford': 'George Hufford & Richard Hufford', 'Hungry': 'Attar', 'Inequity Averse Q-Learner': 'Attar', 'Inverse': 'Karol Langner', 'Inverse Punisher': 'Geraint Palmer', 'Is There Something New?': 'Attar', 'Isolated Clan': 'Attar', 'Isolated Tit For Tat': 'Attar', 'Japan Empire Emperor': 'Attar', 'Japan Empire Soldier 1': 'Attar', 'Japan Empire Soldier 10': 'Attar', 'Japan Empire Soldier 2': 'Attar', 'Japan Empire Soldier 3': 'Attar', 'Japan Empire Soldier 4': 'Attar', 'Japan Empire Soldier 5': 'Attar', 'Japan Empire Soldier 6': 'Attar', 'Japan Empire Soldier 7': 'Attar', 'Japan Empire Soldier 8': 'Attar', 'Japan Empire Soldier 9': 'Attar', 'Javelin': 'Attar & Angga Febriansyah', 'Jem': 'Jem', 'Jesus Petry': 'Jesus Petry', 'Jones': 'Owen Jones', 'Joss / Naive Prober': 'Johann Joss / J. Li & P. Hingston & G. Kendall', 'Josuah': 'Attar', 'Jung': 'Attar, Inspired By Carl Jung', 'Karen': 'Attar', 'Keyman': 'Attar', 'Killer': 'Zachary Danziger', 'King Of 48 Laws Of Power': 'Attar, Inspired By Robert Greene', 'Kluepfel': 'Charles Kluepfel', 'Knowledgeable Worse & Worse': 'Adam Pohl', 'Konflikt Blue': 'Attar', 'Konflikt Green': 'Attar', 'Laran': 'Elio Piccolo & Giovanni Squillero', 'Laws': 'Attar', 'Learner': 'Attar', 'Lefevre': 'Jean-Paul Lefevre', 'Lenient Grim 2': 'Yaroslav Rosokha & Julian Romero', 'Lenient Grim 3': 'Yaroslav Rosokha & Julian Romero', 'Level Punisher': 'Eckhart Arnold', 'Leyland': 'Paul Leyland', 'Leyvraz': 'Francois Leyvraz', 'Liar Person': 'Attar & Angga Febriansyah', 'Lighty-Darky': 'Attar', 'Limited Retaliate': 'Owen Campbell', 'Limited Retaliate 2': 'Owen Campbell', 'Limited Retaliate 3': 'Owen Campbell', 'Line': 'Attar', 'Little Grudger': 'Attar', 'Long Horse': 'Attar, Inspired By LookOut2DD', 'Long Term Strategist The Nerd Of AI Version': 'The Nerd Of AI', 'Look Up / Look Ahead': 'Alan Gaines', 'Loyal Foomii Version': 'Foomii', 'MARS / Mimicry And Relative Similarity': 'T. L. B. M. Silva & A. L. V. Coelho & C. H. C. Ribeiro', 'MCMC / Markov Chain Monte Carlo': 'Nicholas Metropolis & W.K. Hastings, but implemented by Tuomas Sandholm & Robert Crites', 'MEM2': 'J. Li & G. Kendall', 'MENACE / HER': 'Attar, Inspired By Donald Michie & Martin Gardner', 'Mafia A': 'Attar', 'Mafia B': 'Attar', 'Mafia C': 'Attar', 'Mafia D': 'Attar', 'Majapahit': 'Attar', 'Male / Man': 'Attar', 'Malthrin': 'Malthrin', 'Man In The Middle': 'Attar', 'Manipulator The Nerd Of AI Version': 'The Nerd Of AI', 'Manipulity': 'Attar, Inspired By KhanhsKMC', 'Markov': 'Attar, Inspired By Andrey Andreyevich Markov', 'Matcher': 'Attar', 'Math Constant Hunter': 'Karol Langner', 'Mauk': 'John Mauk', 'Mcgurrin': 'Michael Mcgurrin', 'Memory Decay': 'Axelrod Project Contributor Team', 'Mensa': 'Zachary Danziger', 'Meta Hunter': 'Karol Langner', 'Meta Majority': 'Karol Langner', 'Meta Minority': 'Karol Langner', 'Meta Mixer': 'Karol Langner', 'Meta Winner': 'Karol Langner', 'Meta Winner Ensemble': 'Marc Harper', 'Michaelos': 'Michaelos', 'Mikkelson': 'Gordon Mikkelson', 'Mirror Foomii Version': 'Foomii', 'Momentum': 'Dong Won Moon', 'Monkey See, Monkey Do': 'Attar', 'Mosquito': 'Attar', 'N Tits For M Tats': 'Marc Harper', 'Naive Q-Learner': 'Attar', 'Namdeirf / Anti-Grudger': 'Attar', 'Named Withheld': 'Anonymous', 'Nash': 'Attar, Inspired By John Forbes Nash Jr', 'Nasty Tit For Tat': 'Axelrod Project Contributor Team', 'Negation': '[PD2017]', 'Neil A': 'Attar', 'Newman': 'William Newman', 'NoName': 'Attar', 'Nussbacher': 'Gary Nussbacher', 'Nydegger': 'Rudy Nydegger', 'Observant': 'Zachary Danziger', 'Odd-Even Go By Majority': 'Attar', 'Omega Tit For Tat': 'W. Slany & W. Kienreich', 'Once Bitten': 'Holly Marissa', 'Opportunist The Nerd Of AI Version': 'The Nerd Of AI', 'Opposite Grudger': 'Geraint Palmer', 'Out For Tat': 'Attar', 'PSO Gambler 1_1_1': 'Marc Harper', 'PSO Gambler 2_2_2': 'Georgios Koutsovoulos & Marc Harper', 'PSO Gambler 2_2_2 Noise 05': 'Marc Harper', 'PSO Gambler Mem 1': 'Marc Harper', 'Paranoid': 'Attar', 'Passive Tit For Tat': 'Attar', 'Patterned Adaptive Fortress': 'Attar & Claude & Gemini & Chat-GPT & Copilot', 'Patterned Adaptive Fortress 2': 'Attar & Claude & Gemini & Chat-GPT & DeepSeek & Dola', 'Pavlov / Win-Stay, Lose-Shift': 'David Kraines & Vivian Kraines', 'Pavlov D / Suspicious Pavlov': 'Zhen Ji', 'Pavlov Tester': 'Attar', 'Pavolovo': 'Attar', 'Pebley': 'Anne R. Pebley', 'Perceptron': 'Attar', 'Phi': 'Timothy Standen', 'Pi': 'Timothy Standen', 'Pinkley': 'Robin Pinkley', 'Prabowo': 'Attar', 'Praedator': 'Attar', 'Predator': 'W. Ashlock & D. Ashlock', 'Probe & Punish': 'Eneasz', 'Prober': 'J. Li & P. Hingston & G. Kendall', 'Prober 2': "LIFL / Laboratoire d'Informatique Fondamentale de Lille", 'Prober 3': "LIFL / Laboratoire d'Informatique Fondamentale de Lille", 'Prober 4': "LIFL / Laboratoire d'Informatique Fondamentale de Lille", 'Procastinater': 'Attar', 'Psycho Attar Version': 'Attar', 'Pun 1': 'D. Ashlock & E. Y. Kim & N. Leahy', 'Punisher': 'Geraint Palmer', 'Q-Learning A-Learner': 'Attar', 'Quayle': 'James Quayle', 'REINFORCE': 'Attar', 'RNN & RTRL': 'Attar', 'Rabbie': 'Jacob Rabbie', 'Racister': 'Attar', 'Rack Block Shooter': 'Nic Smith', 'Raider': 'W. Ashlock & J. Tsang & D. Ashlock', 'Random': 'Robert Axelrod', 'Random Exploiter 1': 'Attar', 'Random Exploiter 2': 'Attar', 'Random Exploiter 3': 'Attar', 'Random Exploiter 4': 'Attar', 'Random Hunter': 'Karol Langner', 'Random Tit For Tat': 'Zachary M. Taylor', 'Reactive Player': 'Martin Nowak & Karl Sigmund', 'Remorseful Prober': 'J. Li & P. Hingston & G. Kendall', 'Resurrection': 'Eckhart Arnold', 'Retaliate': 'Owen Campbell', 'Retaliate 2': 'Owen Campbell', 'Retaliate 3': 'Owen Campbell', 'Rice Field': 'Meta', 'Ripoff': 'D. Ashlock & E. Y. Kim', 'Risky Q-Learner': 'Geraint Palmer', 'Robertson': 'Scott Robertson', 'Rogers & Maslow': 'Attar, Inspired By Carl Rogers & Abraham Maslow', 'Rover': 'Dugatkin & Wilson', 'Rowsam': 'Robert Rowsam', 'Rwallace': 'Rwallace', 'SOVIET UNION! / USSR / Inequality': 'Attar', 'SURPRISE ATTACK!': 'Attar', 'Sarsa Q-Learner': 'Attar', 'Score Equalizer': 'Attar', 'Second Chance': 'Benelliott', 'Second Chance Attar Version': 'Attar, Inspired By My Father', 'Self Steem': 'L. C. Andre & P. Honovan & T. Felipe & G. Frederico', 'Self-Love A-Learner': 'Attar', 'Short Mem': 'L. C. Andre & P. Honovan & T. Felipe & G. Frederico', 'Shubik': 'Martin Shubik', 'Shurmann': 'Jurgen Schurmann', 'Simple ANN': 'Attar', 'Simple Exploiter': 'Attar', 'Simple Identity ChecK': 'Red75', 'Skinner': 'Attar, Inspired By B.F. Skinner', 'Slanger': 'Attar', 'Slow Tit For 2 Tats 2': "LIFL / Laboratoire d'Informatique Fondamentale de Lille", 'Smart Cat': 'Attar & Syarif', 'Smart Generous Tit For Tat': 'Attar', 'Smart Go By Majority': 'Attar', 'Smart Snake': 'Attar, Inspired By Simulife Hub', 'Smart-Clever': 'Attar', 'Smith': 'Maynard Smith', 'Smoody': 'James Smoody', 'Smooth Grudger': 'Attar', 'Smooth Tit For Tat': 'Attar', 'Sneaky Tit For Tat': 'Karol Langner', 'Snodgrass': 'Richard Snodgrass', 'Social Average': 'Chat-GPT', 'Social Engineering': 'Attar', 'Socio': 'Attar', 'Soft Grudger': 'J. Li & P. Hingston & G. Kendall', 'Solution B1': 'D. Ashlock & J. A. Brown & P. Hingston', 'Solution B5': 'D. Ashlock & J. A. Brown & P. Hingston', 'Sort': 'Attar', 'Spectrum Attacker': 'Attar', 'Spiteful CC': 'P. Mathieu & J. Delahaye', 'Spiteful Tit For Tat': "LIFL / Laboratoire d'Informatique Fondamentale de Lille", 'Split Brain Syndrome': 'Dola', 'Springter': 'Attar', 'Stalker': 'L. C. Andre & P. Honovan & T. Felipe & G. Frederico', 'Star S': 'Gopal Ramchurn', 'Star SL': 'Gopal Ramchurn', 'Star SN': 'Gopal Ramchurn', 'Star Slave 1': 'Gopal Ramchurn', 'Star Slave 10': 'Gopal Ramchurn', 'Star Slave 2': 'Gopal Ramchurn', 'Star Slave 3': 'Gopal Ramchurn', 'Star Slave 4': 'Gopal Ramchurn', 'Star Slave 5': 'Gopal Ramchurn', 'Star Slave 6': 'Gopal Ramchurn', 'Star Slave 7': 'Gopal Ramchurn', 'Star Slave 8': 'Gopal Ramchurn', 'Star Slave 9': 'Gopal Ramchurn', 'Stein & Rapoport': 'William Stein & Anatol Rapoport', 'Stochastic Cooperator': 'C. Adami & A. Hintze', 'Stochastic WSLS': 'Marc Harper', 'Stoic Mirror': 'Gemini', 'Stoicalist': 'Attar & Gemini', 'Stupidiot': 'Attar', 'Sumobox Fighter': 'Attar, Inspired By Simulife Hub', 'Sun Tzu Bot': 'Attar, Inspired By Sun Tzu', 'Suspicious / Suspicious Always Cooperate / Suspicious Cooperator': 'Attar', 'Suspicious Alternator': 'Robert Axelrod', 'Suspicious Tit For Tat': 'C. Hilbe & M. A. Nowak & A. Traulsen', 'Sweet Heart': 'Deep Seek', 'Syifa': 'Attar & Syifa', 'TF 1': 'Marc Harper', 'TF 2': 'Marc Harper', 'TF 3': 'Marc Harper', 'TF2T & 2TFT': 'Attar', 'TOM Level 1 / Theory Of Mind Level 1': 'David Premack & Guy Woodruff', 'Tages': 'Marco Gaudesi & Elio Piccolo & Giovanni Squillero & Alberto Paolo Tonda', 'Teacher Pee While Standing, Student Pee While Running': 'Attar', 'The Calculated Mirror With A Fuse': 'Gemini', 'The Cave Breaker': 'Attar & Gemini', 'The Nasty A': 'Attar', 'Theseus': 'Attar, Inspired By Claude Shannon', 'Threshold Punisher': 'Copilot', 'Thresholded Trust Bot': 'Attar', 'Thue Morse': 'Geraint Palmer', 'Thue Morse Inverse': 'Geraint Palmer', 'Thumper': 'D. Ashlock & E. Y. Kim', 'Tideman & Chieruzzi': 'T.Nicolaus Tideman & Donald Chieruzzi', 'Tideman & Chieruzzi V2': 'T.Nicolaus Tideman & Donald Chieruzzi', 'Tiny Brain': 'Attar', 'Tip For Tap': 'Attar', 'Tit For 2 Tats': 'Robert Axelrod', 'Tit For 3 Tats': 'Yaroslav Rosokha & Julian Romero', 'Tit For Increasing Tat': 'Claude', 'Tit For Tat': 'Anatol Rapoport', 'Tit For Tat Killer': 'Attar', 'Toxic Mirror': 'Deep Seek', 'Trader': 'Attar & Gemini', 'Tranquilizer': 'Craig Snider', 'Treasure Hunt / Sugar Strategy': 'D. Ashlock & W. Ashlock', 'Tri-Brain': 'Attar', 'Tricky Cooperator': 'Karol Langner', 'Tricky Defector': 'Karol Langner', 'Tricky Level Punisher': 'Eckhart Arnold', 'Trust Ledger': 'Chat-GPT', 'Trust Score Sigmoid': 'Claude', 'Trusty Foomii Version': 'Foomii', 'Tullock': 'Gordon Tullock', 'Turan': 'Marco Gaudesi & Elio Piccolo & Giovanni Squillero & Alberto Paolo Tonda', 'Unpredictable Tit For Tat': 'Attar', 'Uranium': 'Attar', 'Usually Cooperates': 'D. Ashlock & E. Y. Kim & W. Ashlock', 'Usually Defects': 'D. Ashlock & E. Y. Kim & W. Ashlock', 'Vandal Foomii Version': 'Foomii', 'Vengeful': 'D. Ashlock', 'Vengeful Cheater': 'Vaniver', 'Verity': 'Attar & Gemini, Inspired By ThatMob', 'Very Bad': 'L. C. Andre & P. Honovan & T. Felipe & G. Frederico', 'Weiderman': 'Nelson Weiderman', 'Weiner': 'Herb Weiner', 'Whale': 'Attar & Deep Seek', 'White': 'Edward C. White', 'William': 'Attar', 'Willing': 'P. Van Den Berg & F. J. Weissing', 'Win-Shift, Lose-Stay': 'J. Li & P. Hingston & G. Kendall', 'Winner 12': 'P. Mathieu & J. Delahaye', 'Winner 21': 'P. Mathieu & J. Delahaye', 'Worse & Worse / Worse & Worse 1': "LIFL / Laboratoire d'Informatique Fondamentale de Lille", 'Worse & Worse 2': "LIFL / Laboratoire d'Informatique Fondamentale de Lille", 'Yamachi': 'Brian Yamachi', 'Zero Day': 'Attar', 'Zero Determinant 2026 / ZD-2026': 'Attar', 'Zero Determinant Equalizer / ZD-Q': "William H. Press & Freeman Dyson, But Implemented In this Code Using C. Hilbe, M. A. Nowak, & A. Traulsen 's Model", 'Zero Determinant Extortion / ZD-X': "Lars Roemheld, But Implemented In this Code Using C. Hilbe, M. A. Nowak, & A. Traulsen 's Model", 'Zero Determinant Generous / ZD-G': "Steven Kuhn, But Implemented In this Code Using C. Hilbe, M. A. Nowak, & A. Traulsen 's Model", 'Zero Determinant Generous Tit For Tat / ZD-GTFT': "A. J. Stewart & J. B. Plotkin, But Implemented In this Code Using C. Hilbe, M. A. Nowak, & A. Traulsen 's Model", 'Zero Determinant Memory 2 / ZD-M2': 'Marc Harper', 'Zero Determinant Mischief / ZD-MS': "Lars Roemheld, But Implemented In this Code Using C. Hilbe, M. A. Nowak, & A. Traulsen 's Model", 'Zero Determinant Psychological War 2026 / ZD-PsyWar2026': 'Gemini', 'Zero Determinant Set / ZD-S': "Steven Kuhn, But Implemented In this Code Using C. Hilbe, M. A. Nowak, & A. Traulsen 's Model", 'Zimmerman': 'William Zimmerman'}
strategies_category = {'2 Tits For Tat': 'NICE', '3 Seconds Mirror': 'NICE', '3-Tree Of Prediction': 'NICE', 'A-Actor&Critic / A-A&C': 'NASTY', 'ALLC OR ALLD': 'NASTY', 'AMERICA! / USA / Freedoom': 'NASTY', 'AON2': 'NICE', 'AQUA': 'NICE', 'Adams': 'NICE', 'Adaptive': 'NASTY', 'Adaptive Pavlov 2006': 'NICE', 'Adaptive Pavlov 2011': 'NICE', 'Adaptive Tit For Tat': 'NICE', 'Adaptive Tit For Tat Attar Version': 'NICE', 'Adaptor Brief': 'NASTY', 'Adaptor Long': 'NASTY', 'Aggravater': 'NASTY', 'Alan': 'NICE', 'Alexei': 'NASTY', 'Almy': 'NICE', 'Alternator': 'NASTY', 'Alternator Hunter': 'NICE', 'Always Cooperate / Cooperator': 'NICE', 'Always Defect / Defector': 'NASTY', 'Always Skip / Skiptor': 'NICE', 'Amazon / Piraha': 'NICE', 'Ambuelh & Kickey': 'NASTY', 'Analogy': 'NASTY', 'Anatol': 'NICE', 'Anderson': 'NASTY', 'Angry Tit For Tat': 'NICE', 'Anti Cycler': 'NASTY', 'Anti Noise Tit For Tat': 'NICE', 'Anti Tit For Tat / Psycho': 'NASTY', 'Appeaser': 'NICE', 'Appold': 'NICE', 'Arrogant Q-Learner': 'NASTY', 'Average Copier': 'NASTY', 'Back Stabber': 'NICE', 'Backrooms': 'NASTY', 'Bad Random': 'NASTY', 'Bahlil': 'NASTY', 'Bandit': 'NASTY', 'Bandit UCB': 'NASTY', 'Batell': 'NASTY', 'Bayes Attar Version': 'NICE', 'Bayes Foomii Version': 'NICE', 'Beta Tester': 'NASTY', 'Better & Better': 'NASTY', 'Black': 'NICE', 'Bluff': 'NASTY', 'Bluff Master': 'NASTY', 'Boltzmann Brain': 'NASTY', 'Borufsen': 'NICE', 'Boxer': 'NICE', 'Bros Mind': 'NICE', 'Bully / Reverse Tit For Tat': 'NASTY', 'Burn Both Ends / BBE': 'NASTY', 'Bush Mosteller': 'NASTY', 'Buzzer': 'NASTY', 'CS / Collective Strategy': 'NASTY', 'Caerbannog': 'NASTY', 'Calamity': 'NASTY', 'Calculator': 'NASTY', 'Calculator The Nerd Of AI Version': 'NICE', 'Capitalist': 'NASTY', 'Capri': 'NICE', 'Cautious Q-Learner': 'NASTY', 'Cave': 'NICE', 'Caveman': 'NASTY', 'Cerebrum': 'NICE', 'Champion': 'NICE', 'Charity': 'NICE', 'Chatter': 'NICE', 'Clement Coldridge': 'NICE', 'Colbert': 'NASTY', 'Communalist': 'NICE', 'Continuous Q-Learner Avg': 'NASTY', 'Continuous Q-Learner Max': 'NASTY', 'Contrite Grudger': 'NICE', 'Contrite Tit For Tat': 'NICE', 'Control 1': 'NASTY', 'Control 10': 'NASTY', 'Control 11': 'NASTY', 'Control 2': 'NICE', 'Control 3': 'NASTY', 'Control 4': 'NASTY', 'Control 5': 'NASTY', 'Control 6': 'NICE', 'Control 7': 'NASTY', 'Control 8': 'NASTY', 'Control 9': 'NICE', 'Cooperator Hunter': 'NASTY', 'Crabby': 'NICE', 'Crabby Attar Version': 'NASTY', 'Crow': 'NICE', 'Cruelity': 'NICE', 'Cult Bishop A': 'NASTY', 'Cult Bishop B': 'NASTY', 'Cult Citizen A': 'NASTY', 'Cult Citizen B': 'NASTY', 'Cult Citizen C': 'NASTY', 'Cult Leader': 'NASTY', 'Curiosity': 'NICE', 'Cycle Hunter': 'NASTY', 'Cycler CCCCCD': 'NASTY', 'Cycler CCCD': 'NASTY', 'Cycler CCCDCD': 'NASTY', 'Cycler CCD': 'NASTY', 'Cycler DC': 'NASTY', 'Cycler DCCCC': 'NASTY', 'Cycler DDC': 'NASTY', 'DBS / Derived Belief Strategy': 'NICE', 'DDoS': 'NICE', 'Davis': 'NICE', 'Dawes & Batell': 'NASTY', 'Deadlock Breaker': 'NICE', 'Decay Given': 'NICE', 'Defector Hunter': 'NICE', 'Delayed AON1': 'NICE', 'Desire': 'NICE', 'Desprate': 'NASTY', 'Detective': 'NASTY', 'Devil Staircase': 'NICE', 'Diplomat': 'NICE', 'Dont Bite The Hand That Feeds You': 'NICE', 'Double Crosser': 'NICE', 'Double Q-Learner': 'NASTY', 'Double Resurrection': 'NASTY', 'Doubler': 'NICE', 'Downing': 'NASTY', 'Downing V2': 'NICE', 'Duisman': 'NICE', 'Dyna Q-Learner': 'NASTY', 'Dynamic 2 Tits For Tat': 'NICE', 'EGOist': 'NASTY', 'Eatherley': 'NICE', 'Echo Adaptive': 'NICE', 'Echo Fisher': 'NASTY', 'Egoist A-Learner': 'NICE', 'Eliza / Therapist': 'NICE', 'Entropy Sentinel': 'NASTY', 'Eric': 'NICE', 'Ethanol / Alcohol': 'NASTY', 'Eugine Nier': 'NASTY', 'Euler': 'NASTY', 'Eventual Cycle Hunter': 'NICE', 'Evil Alliance': 'NASTY', 'Evolvable Cycler Attar Version': 'NASTY', 'Evolved FSM 16': 'NICE', 'Evolved FSM 16 Noise 05': 'NICE', 'Evolved FSM 4': 'NICE', 'Evolved FSM 6': 'NASTY', 'Evolved HMM 5': 'NICE', 'Evolved Looker Up 1_1_1': 'NICE', 'Evolved Looker Up 2_2_2': 'NICE', 'Exploratory Lenient Grim 2': 'NASTY', 'Exploratory Tit For 3 Tats': 'NASTY', 'FAWS': 'NASTY', 'Falk & Lanqsted': 'NICE', 'False Cooperator': 'NASTY', 'Falsity': 'NICE', 'Feathers': 'NICE', 'Feld': 'NASTY', 'Female / Woman': 'NASTY', 'Fibonacci': 'NASTY', 'Firm But Fair / Firm For Tat': 'NICE', 'Fool Me Once': 'NICE', 'Forest-RND': 'NICE', 'Forest-XG': 'NICE', 'Forgetful Fool Me Once': 'NICE', 'Forgetful Grudger': 'NICE', 'Forgiver': 'NICE', 'Forgiver The Nerd Of AI Version': 'NICE', 'Forgiving Fate': 'NICE', 'Forgiving Tit For Tat': 'NICE', 'Fortress 3': 'NASTY', 'Fortress 4': 'NASTY', 'Frankl': 'NICE', 'Free Rider': 'NASTY', 'Frequency Analyzer': 'NICE', 'Freud': 'NICE', 'Friedland': 'NICE', 'Friedman / Grudger / Grim trigger': 'NICE', 'Gambler Attar Version': 'NASTY', 'Game Theory Analyzer / GT-A': 'NICE', 'Game Theory Explorer / GT-E': 'NICE', 'Gaslighter': 'NASTY', 'Gateman': 'NICE', 'Gen 1_64_200': 'NASTY', 'Gen 2_64_200': 'NASTY', 'Gen 3_64_200': 'NASTY', 'Gen 4_64_200': 'NASTY', 'Gen 5_64_200': 'NASTY', 'Generous Tit For Tat': 'NICE', 'Generous Tit For Tat Axelrod Project Contributor Team Version': 'NICE', 'Generous Tit For Tat Less Wrong Version / Tit For Tat But Pico Generous': 'NICE', 'Genetic Algo 1': 'NICE', 'Genetic Algo 2': 'NICE', 'George': 'NASTY', 'Getzler': 'NICE', 'Giles / Worse & Worse 3': 'NICE', 'Gladstein / Tester': 'NASTY', 'Go By Minority': 'NASTY', 'Gold Digger Foomii Version': 'NASTY', 'Golden Mean': 'NICE', 'Good': 'NICE', 'Good Random': 'NASTY', 'Graaskamp': 'NASTY', 'Graaskamp & Katzen': 'NICE', 'Gradual': 'NICE', 'Gradual Cristal Version': 'NICE', 'Gradual Killer': 'NASTY', 'Grisell / Go By Majority / Soft Go By Majority': 'NICE', 'Grofman': 'NICE', 'Grofman V2': 'NICE', 'Grudger Alternator': 'NICE', 'Grumpy': 'NICE', 'HMM Player': 'NASTY', 'Half Sin Square': 'NICE', 'Hall': 'NICE', 'Handshake': 'NASTY', 'Hard Go By Majority': 'NICE', 'Hard Prober': 'NASTY', 'Hard Tit For 2 Tats / 2 Tits For 2 Tats': 'NICE', 'Hard Tit For Tat / 3 Tits For Tat': 'NICE', 'Harrington': 'NASTY', 'Hesitant Q-Learner': 'NASTY', 'Hollander': 'NICE', 'Hopeless': 'NASTY', 'Hot Coffee': 'NICE', 'Hotz': 'NICE', 'Hufford': 'NASTY', 'Hungry': 'NASTY', 'Inequity Averse Q-Learner': 'NASTY', 'Inverse': 'NICE', 'Inverse Punisher': 'NICE', 'Is There Something New?': 'NICE', 'Isolated Clan': 'NASTY', 'Isolated Tit For Tat': 'NICE', 'Japan Empire Emperor': 'NICE', 'Japan Empire Soldier 1': 'NICE', 'Japan Empire Soldier 10': 'NICE', 'Japan Empire Soldier 2': 'NICE', 'Japan Empire Soldier 3': 'NICE', 'Japan Empire Soldier 4': 'NICE', 'Japan Empire Soldier 5': 'NICE', 'Japan Empire Soldier 6': 'NICE', 'Japan Empire Soldier 7': 'NICE', 'Japan Empire Soldier 8': 'NICE', 'Japan Empire Soldier 9': 'NICE', 'Javelin': 'NICE', 'Jem': 'NASTY', 'Jesus Petry': 'NASTY', 'Jones': 'NASTY', 'Joss / Naive Prober': 'NASTY', 'Josuah': 'NASTY', 'Jung': 'NICE', 'Karen': 'NASTY', 'Keyman': 'NASTY', 'Killer': 'NASTY', 'King Of 48 Laws Of Power': 'NICE', 'Kluepfel': 'NICE', 'Knowledgeable Worse & Worse': 'NASTY', 'Konflikt Blue': 'NASTY', 'Konflikt Green': 'NICE', 'Laran': 'NASTY', 'Laws': 'NICE', 'Learner': 'NICE', 'Lefevre': 'NICE', 'Lenient Grim 2': 'NICE', 'Lenient Grim 3': 'NICE', 'Level Punisher': 'NICE', 'Leyland': 'NICE', 'Leyvraz': 'NICE', 'Liar Person': 'NASTY', 'Lighty-Darky': 'NASTY', 'Limited Retaliate': 'NICE', 'Limited Retaliate 2': 'NICE', 'Limited Retaliate 3': 'NICE', 'Line': 'NICE', 'Little Grudger': 'NICE', 'Long Horse': 'NASTY', 'Long Term Strategist The Nerd Of AI Version': 'NICE', 'Look Up / Look Ahead': 'NASTY', 'Loyal Foomii Version': 'NICE', 'MARS / Mimicry And Relative Similarity': 'NICE', 'MCMC / Markov Chain Monte Carlo': 'NICE', 'MEM2': 'NICE', 'MENACE / HER': 'NICE', 'Mafia A': 'NASTY', 'Mafia B': 'NASTY', 'Mafia C': 'NASTY', 'Mafia D': 'NASTY', 'Majapahit': 'NICE', 'Male / Man': 'NICE', 'Malthrin': 'NASTY', 'Man In The Middle': 'NASTY', 'Manipulator The Nerd Of AI Version': 'NASTY', 'Manipulity': 'NASTY', 'Markov': 'NICE', 'Matcher': 'NICE', 'Math Constant Hunter': 'NICE', 'Mauk': 'NICE', 'Mcgurrin': 'NICE', 'Memory Decay': 'NASTY', 'Mensa': 'NASTY', 'Meta Hunter': 'NASTY', 'Meta Majority': 'NICE', 'Meta Minority': 'NASTY', 'Meta Mixer': 'NASTY', 'Meta Winner': 'NASTY', 'Meta Winner Ensemble': 'NASTY', 'Michaelos': 'NASTY', 'Mikkelson': 'NICE', 'Mirror Foomii Version': 'NICE', 'Momentum': 'NICE', 'Monkey See, Monkey Do': 'NASTY', 'Mosquito': 'NASTY', 'N Tits For M Tats': 'NICE', 'Naive Q-Learner': 'NASTY', 'Namdeirf / Anti-Grudger': 'NASTY', 'Named Withheld': 'NASTY', 'Nash': 'NICE', 'Nasty Tit For Tat': 'NASTY', 'Negation': 'NASTY', 'Neil A': 'NICE', 'Newman': 'NASTY', 'NoName': 'NICE', 'Nussbacher': 'NICE', 'Nydegger': 'NICE', 'Observant': 'NICE', 'Odd-Even Go By Majority': 'NICE', 'Omega Tit For Tat': 'NICE', 'Once Bitten': 'NICE', 'Opportunist The Nerd Of AI Version': 'NASTY', 'Opposite Grudger': 'NASTY', 'Out For Tat': 'NICE', 'PSO Gambler 1_1_1': 'NICE', 'PSO Gambler 2_2_2': 'NICE', 'PSO Gambler 2_2_2 Noise 05': 'NICE', 'PSO Gambler Mem 1': 'NICE', 'Paranoid': 'NASTY', 'Passive Tit For Tat': 'NICE', 'Patterned Adaptive Fortress': 'NICE', 'Patterned Adaptive Fortress 2': 'NICE', 'Pavlov / Win-Stay, Lose-Shift': 'NICE', 'Pavlov D / Suspicious Pavlov': 'NASTY', 'Pavlov Tester': 'NASTY', 'Pavolovo': 'NASTY', 'Pebley': 'NICE', 'Perceptron': 'NASTY', 'Phi': 'NASTY', 'Pi': 'NASTY', 'Pinkley': 'NICE', 'Prabowo': 'NASTY', 'Praedator': 'NASTY', 'Predator': 'NASTY', 'Probe & Punish': 'NICE', 'Prober': 'NASTY', 'Prober 2': 'NASTY', 'Prober 3': 'NASTY', 'Prober 4': 'NASTY', 'Procastinater': 'NICE', 'Psycho Attar Version': 'NICE', 'Pun 1': 'NASTY', 'Punisher': 'NICE', 'Q-Learning A-Learner': 'NICE', 'Quayle': 'NICE', 'REINFORCE': 'NASTY', 'RNN & RTRL': 'NASTY', 'Rabbie': 'NICE', 'Racister': 'NASTY', 'Rack Block Shooter': 'NASTY', 'Raider': 'NASTY', 'Random': 'NASTY', 'Random Exploiter 1': 'NICE', 'Random Exploiter 2': 'NICE', 'Random Exploiter 3': 'NICE', 'Random Exploiter 4': 'NICE', 'Random Hunter': 'NICE', 'Random Tit For Tat': 'NASTY', 'Reactive Player': 'NASTY', 'Remorseful Prober': 'NASTY', 'Resurrection': 'NICE', 'Retaliate': 'NICE', 'Retaliate 2': 'NICE', 'Retaliate 3': 'NICE', 'Rice Field': 'NASTY', 'Ripoff': 'NASTY', 'Risky Q-Learner': 'NASTY', 'Robertson': 'NICE', 'Rogers & Maslow': 'NICE', 'Rover': 'NASTY', 'Rowsam': 'NICE', 'Rwallace': 'NASTY', 'SOVIET UNION! / USSR / Inequality': 'NASTY', 'SURPRISE ATTACK!': 'NASTY', 'Sarsa Q-Learner': 'NASTY', 'Score Equalizer': 'NICE', 'Second Chance': 'NASTY', 'Second Chance Attar Version': 'NICE', 'Self Steem': 'NASTY', 'Self-Love A-Learner': 'NICE', 'Short Mem': 'NICE', 'Shubik': 'NICE', 'Shurmann': 'NICE', 'Simple ANN': 'NASTY', 'Simple Exploiter': 'NASTY', 'Simple Identity ChecK': 'NASTY', 'Skinner': 'NICE', 'Slanger': 'NICE', 'Slow Tit For 2 Tats 2': 'NICE', 'Smart Cat': 'NICE', 'Smart Generous Tit For Tat': 'NICE', 'Smart Go By Majority': 'NICE', 'Smart Snake': 'NICE', 'Smart-Clever': 'NICE', 'Smith': 'NICE', 'Smoody': 'NICE', 'Smooth Grudger': 'NICE', 'Smooth Tit For Tat': 'NICE', 'Sneaky Tit For Tat': 'NASTY', 'Snodgrass': 'NICE', 'Social Average': 'NASTY', 'Social Engineering': 'NASTY', 'Socio': 'NICE', 'Soft Grudger': 'NICE', 'Solution B1': 'NASTY', 'Solution B5': 'NASTY', 'Sort': 'NICE', 'Spectrum Attacker': 'NASTY', 'Spiteful CC': 'NICE', 'Spiteful Tit For Tat': 'NICE', 'Split Brain Syndrome': 'NICE', 'Springter': 'NICE', 'Stalker': 'NICE', 'Star S': 'NASTY', 'Star SL': 'NASTY', 'Star SN': 'NASTY', 'Star Slave 1': 'NASTY', 'Star Slave 10': 'NASTY', 'Star Slave 2': 'NASTY', 'Star Slave 3': 'NASTY', 'Star Slave 4': 'NASTY', 'Star Slave 5': 'NASTY', 'Star Slave 6': 'NASTY', 'Star Slave 7': 'NASTY', 'Star Slave 8': 'NASTY', 'Star Slave 9': 'NASTY', 'Stein & Rapoport': 'NASTY', 'Stochastic Cooperator': 'NASTY', 'Stochastic WSLS': 'NASTY', 'Stoic Mirror': 'NICE', 'Stoicalist': 'NICE', 'Stupidiot': 'NASTY', 'Sumobox Fighter': 'NICE', 'Sun Tzu Bot': 'NICE', 'Suspicious / Suspicious Always Cooperate / Suspicious Cooperator': 'NASTY', 'Suspicious Alternator': 'NASTY', 'Suspicious Tit For Tat': 'NASTY', 'Sweet Heart': 'NASTY', 'Syifa': 'NASTY', 'TF 1': 'NASTY', 'TF 2': 'NASTY', 'TF 3': 'NICE', 'TF2T & 2TFT': 'NICE', 'TOM Level 1 / Theory Of Mind Level 1': 'NASTY', 'Tages': 'NASTY', 'Teacher Pee While Standing, Student Pee While Running': 'NICE', 'The Calculated Mirror With A Fuse': 'NASTY', 'The Cave Breaker': 'NASTY', 'The Nasty A': 'NASTY', 'Theseus': 'NICE', 'Threshold Punisher': 'NICE', 'Thresholded Trust Bot': 'NICE', 'Thue Morse': 'NASTY', 'Thue Morse Inverse': 'NASTY', 'Thumper': 'NICE', 'Tideman & Chieruzzi': 'NICE', 'Tideman & Chieruzzi V2': 'NICE', 'Tiny Brain': 'NASTY', 'Tip For Tap': 'NICE', 'Tit For 2 Tats': 'NICE', 'Tit For 3 Tats': 'NICE', 'Tit For Increasing Tat': 'NICE', 'Tit For Tat': 'NICE', 'Tit For Tat Killer': 'NASTY', 'Toxic Mirror': 'NASTY', 'Trader': 'NICE', 'Tranquilizer': 'NASTY', 'Treasure Hunt / Sugar Strategy': 'NASTY', 'Tri-Brain': 'NICE', 'Tricky Cooperator': 'NASTY', 'Tricky Defector': 'NASTY', 'Tricky Level Punisher': 'NICE', 'Trust Ledger': 'NICE', 'Trust Score Sigmoid': 'NASTY', 'Trusty Foomii Version': 'NICE', 'Tullock': 'NASTY', 'Turan': 'NASTY', 'Unpredictable Tit For Tat': 'NASTY', 'Uranium': 'NICE', 'Usually Cooperates': 'NICE', 'Usually Defects': 'NASTY', 'Vandal Foomii Version': 'NASTY', 'Vengeful': 'NICE', 'Vengeful Cheater': 'NASTY', 'Verity': 'NICE', 'Very Bad': 'NICE', 'Weiderman': 'NICE', 'Weiner': 'NICE', 'Whale': 'NICE', 'White': 'NICE', 'William': 'NICE', 'Willing': 'NASTY', 'Win-Shift, Lose-Stay': 'NASTY', 'Winner 12': 'NICE', 'Winner 21': 'NICE', 'Worse & Worse / Worse & Worse 1': 'NASTY', 'Worse & Worse 2': 'NASTY', 'Yamachi': 'NICE', 'Zero Day': 'NASTY', 'Zero Determinant 2026 / ZD-2026': 'NICE', 'Zero Determinant Equalizer / ZD-Q': 'NASTY', 'Zero Determinant Extortion / ZD-X': 'NASTY', 'Zero Determinant Generous / ZD-G': 'NICE', 'Zero Determinant Generous Tit For Tat / ZD-GTFT': 'NICE', 'Zero Determinant Memory 2 / ZD-M2': 'NASTY', 'Zero Determinant Mischief / ZD-MS': 'NASTY', 'Zero Determinant Psychological War 2026 / ZD-PsyWar2026': 'NICE', 'Zero Determinant Set / ZD-S': 'NASTY', 'Zimmerman': 'NASTY'}

strategys_many_name = []
for i in range(len(strategies)):
    if '/' in strategies[i]:
        strategys_many_name.append(strategies[i])

default_memory_looked = 10
tournament_avg_round = 200
tournament_avg_last_round = tournament_avg_round - 1
tournament_round_randomness = 75
axelrod_s_address = "[axe@umich.edu]"

if not(RPST['T'] > RPST['R'] > RPST['P'] > RPST['S']):
    raise ValueError('ERROR! YOUR T > R > P > S IS FALSE! YOU MORON!')

if not((2 * RPST['R']) > (RPST['T'] + RPST['S'])):
    raise ValueError('ERROR! YOUR 2R > (T + S) IS FALSE! YOU MORON!')

if __name__ == "__main__":

    memory = []
    score = [0, 0]
    strategy_chat = [[''], ['']]
    strategy_next_chat = [[''], ['']]
    strat_memory = [reset_think_memory(), reset_think_memory()]
    guessing = False

    the_play = input('play game, versus, strategy list, check comman list, or tournament? ')

    if the_play != 'tournament':
        
        if the_play != 'versus':
            if the_play != 'strategy creator':
                if the_play != 'strategy category':
                    if the_play != 'check current index':
                        if the_play != 'strategy list':
                            if the_play != 'number of strategies':
                                if the_play != 'NICE:NASTY':
                                    if the_play != 'check comman list':
                                        if the_play != 'sort':
                                            if the_play != 'guess random strategy':
                                                if the_play != 'categorize':
                                                    if the_play != 'search':
                                                        input_strategy = input('enemy strategy: ')
                                                        if input_strategy.lower() != 'player':
                                                            the_strategy = many_name_strategy_function(input_strategy)
                                                            print(f"\033[{1};1H\033[K", end="")
                                                            print(f"\033[{2};1H\033[K", end="")
                                                            the_play = 'play game'
                                                        else:
                                                            the_play = 'play with friends'
                                                    else:
                                                        searching = input('search what? ')
                                                        print([i for i in strategies if searching in i])
                                            else:
                                                the_strategy = rnd.choice(strategies)
                                                print(f"\033[{1};1H\033[K", end="")
                                                print(f"\033[{2};1H\033[K", end="")
                                                the_play = 'play game'
                                                guessing = True
                                        else:
                                            print(f'strategies = {sorted(strategies)}')
                                            print(f'strategies_creator = {dict(sorted(strategies_creator.items()))}')
                                            print(f'strategies_category = {dict(sorted(strategies_category.items()))}')
                                    else:
                                        print('play game'); print('tournament'); print('versus')
                                        print('strategy creator'); print('strategy category')
                                        print('check current index'); print('strategy list')
                                        print('number of strategies'); print('NICE:NASTY')
                                        print('check comman list'); print('sort'); print('guess random strategy')
                                        print('categorize'); print('search')
                                else:
                                    NICE_sum = sum(1 for i in range(len(strategies)) if strategies_category[strategies[i]] == 'NICE')
                                    NASTY_sum = len(strategies) - NICE_sum
                                    print(f"{NICE_sum}:{NASTY_sum}")
                            else:
                                print(len(strategies))
                        else:
                            print(f'strategy list: {strategies[:]}')
                    else:
                        print(len(strat_memory[1]) - 1)
                else:
                    print(strategies_category.get(many_name_strategy_function(input('strategy code name? '))))
            else:
                print(strategies_creator.get(many_name_strategy_function(input('strategy code name? '))))
        else:
            strategy1 = many_name_strategy_function(input('first strategy? '))
            strategy2 = many_name_strategy_function(input('second strategy? '))

        if the_play == 'categorize':
            strategy_score = [0] * len(strategies)
            strategies_category = {i: 'NICE' for i in strategies}
            tournament_try = 20
            pairs = [(i, strategies.index('Always Cooperate / Cooperator')) for i in range(len(strategies))]
            tasks = [(i, j, False, 0) for i, j in pairs for _ in range(tournament_try)]

            print(f'processing categorize with {len(strategies)} strategies across {mp.cpu_count()} cores')

            with mp.Pool(processes=mp.cpu_count()) as pool:
                results = pool.map(run_match, tasks)

            for i, j, s1, s2 in results:
                if s1 > s2:
                    strategies_category[strategies[i]] = 'NASTY'
            print(strategies_category)

        
        if the_play in ('play game', 'versus'):
            game_round = 0
            tournament_round = tournament_avg_round + rnd.randint(-tournament_round_randomness, tournament_round_randomness)

            while True:
                if the_play != 'versus':
                    sh_st = timeout_input_instant(0.2)
                    if sh_st in ('q', 'w', 'e', 'Q', 'W', 'E'):
                        strategy_next_chat = [[''], ['']]
                        if sh_st.lower() == 'q':
                            player_choice = 'C'
                        elif sh_st.lower() == 'w':
                            player_choice = 'D'
                        else:
                            player_choice = rnd.choice(['C', 'D'])

                        enemy_choice = think(the_strategy, 0)
                        if enemy_choice not in ('C', 'D', '[SKIP]'):
                            raise ValueError(f'ERROR {the_strategy}')

                        if enemy_choice == '[SKIP]':
                            break

                        memory.append((player_choice, enemy_choice))

                        if player_choice == enemy_choice:
                            update_score(RPST['R'], RPST['R']) if player_choice == 'C' else update_score(RPST['P'], RPST['P'])
                        else:
                            update_score(RPST['T'], RPST['S']) if player_choice == 'D' else update_score(RPST['S'], RPST['T'])

                        game_round += 1
                else:
                    strategy_next_chat = [[''], ['']]
                    strategy1_choice = think(strategy1, 1)
                    strategy2_choice = think(strategy2, 0)

                    if (strategy1_choice not in ('C', 'D', '[SKIP]')) or (strategy2_choice not in ('C', 'D', '[SKIP]')):
                        ERROR = []
                        if strategy1_choice not in ('C', 'D', '[SKIP]'):
                            ERROR.append(strategy1)
                        if strategy2_choice not in ('C', 'D', '[SKIP]'):
                            ERROR.append(strategy2)
                        raise ValueError(f'ERROR {ERROR}')

                    if '[SKIP]' in (strategy1_choice, strategy2_choice):
                        break

                    memory.append((strategy1_choice, strategy2_choice))

                    if strategy1_choice == strategy2_choice:
                        update_score(RPST['R'], RPST['R']) if strategy1_choice == 'C' else update_score(RPST['P'], RPST['P'])
                    else:
                        update_score(RPST['T'], RPST['S']) if strategy1_choice == 'D' else update_score(RPST['S'], RPST['T'])

                    game_round += 1
                    time.sleep(0.05)

                strategy_chat[0][0] = strategy_next_chat[0][0]
                strategy_chat[1][0] = strategy_next_chat[1][0]

                
                print("\033[H", end="")
                for i in range(2):
                    color = ['', '', '', '', '']
                    bad_space = f"\033[38;2;{255};{128};{0}m{'░░'}\033[0m"
                    good_space = f"\033[38;2;{150};{255};{0}m{'░░'}\033[0m"
                    for j in range(len(memory) - min(default_memory_looked, game_round), len(memory)):
                        color[0] += good_space + f"\033[38;2;{0};{255};{0}m{'░░░░░░'}\033[0m" + good_space if memory[j][i] == 'C' else f"\033[38;2;{255};{0};{0}m{'░░░░░░'}\033[0m" + (bad_space * 2)
                        color[1] += f"\033[38;2;{0};{255};{0}m{'░░'}\033[0m" + (good_space * 4) if memory[j][i] == 'C' else f"\033[38;2;{255};{0};{0}m{'░░'}\033[0m" + (bad_space * 2) + f"\033[38;2;{255};{0};{0}m{'░░'}\033[0m" + bad_space
                        color[2] += f"\033[38;2;{0};{255};{0}m{'░░'}\033[0m" + (good_space * 4) if memory[j][i] == 'C' else f"\033[38;2;{255};{0};{0}m{'░░'}\033[0m" + (bad_space * 2) + f"\033[38;2;{255};{0};{0}m{'░░'}\033[0m" + bad_space
                        color[3] += f"\033[38;2;{0};{255};{0}m{'░░'}\033[0m" + (good_space * 4) if memory[j][i] == 'C' else f"\033[38;2;{255};{0};{0}m{'░░'}\033[0m" + (bad_space * 2) + f"\033[38;2;{255};{0};{0}m{'░░'}\033[0m" + bad_space
                        color[4] += good_space + f"\033[38;2;{0};{255};{0}m{'░░░░░░'}\033[0m" + good_space if memory[j][i] == 'C' else f"\033[38;2;{255};{0};{0}m{'░░░░░░'}\033[0m" + (bad_space * 2)
                    print(color[0]); print(color[1]); print(color[2]); print(color[3]); print(color[4])
                    if i == 0:
                        print(f"\033[38;2;{255};{255};{0}m{'░░░░░░░░░░'}\033[0m" * default_memory_looked)

                baris_pesan = 12
                print(f"\033[{baris_pesan};1H\033[K", end="")
                if the_play != 'versus':
                    if not guessing:
                        print(f'player: {score[0]}, NETRAL, \"\" | {the_strategy}: {score[1]}, {strategies_category.get(the_strategy)}, \"{strategy_chat[1][0]}\" | game round: {game_round} | memory: {strat_memory[1][332]}')
                    else:
                        print(f'player: {score[0]}, NETRAL, \"\" | ???: {score[1]}, NETRAL, \"{strategy_chat[1][0]}\" | game round: {game_round}')
                else:
                    print(f'{strategy1}: {score[0]}, {strategies_category.get(strategy1)}, \"{strategy_chat[0][0]}\" | {strategy2}: {score[1]}, {strategies_category.get(strategy2)}, \"{strategy_chat[1][0]}\" | game round: {game_round}')

                if game_round >= tournament_round:
                    if guessing:
                        print(memory)
                        the_guess = many_name_strategy_function(input('What Is The Identity Of This Strategy? '))
                        if the_guess == the_strategy:
                            print('YOU RIGHT!')
                        else:
                            print(f'YOU WRONG!\nTHE ANSWER IS {the_strategy}')
                    break
            if not guessing:
                print(memory)
        elif the_play == 'play with friends':
            game_round = 0
            tournament_round = tournament_avg_round + rnd.randint(-tournament_round_randomness, tournament_round_randomness)
            players_turn = 1

            while True:
                sh_st = timeout_input_instant(0.2)
                if sh_st in ('q', 'w', 'e', 'Q', 'W', 'E'):
                    strategy_next_chat = [[''], ['']]
                    if sh_st.lower() == 'q':
                        if players_turn == 1:
                            player1_choice = 'C'
                        else:
                            player2_choice = 'C'
                    elif sh_st.lower() == 'w':
                        if players_turn == 1:
                            player1_choice = 'D'
                        else:
                            player2_choice = 'D'
                    else:
                        if players_turn == 1:
                            player1_choice = rnd.choice(['C', 'D'])
                        else:
                            player2_choice = rnd.choice(['C', 'D'])

                    if players_turn == 2:
                        memory.append((player1_choice, player2_choice))

                        if player1_choice == player2_choice:
                            update_score(RPST['R'], RPST['R']) if player1_choice == 'C' else update_score(RPST['P'], RPST['P'])
                        else:
                            update_score(RPST['T'], RPST['S']) if player1_choice == 'D' else update_score(RPST['S'], RPST['T'])

                        game_round += 1

                    players_turn += 1
                    players_turn = ((players_turn - 1) % 2) + 1

                
                print("\033[H", end="")
                for i in range(2):
                    color = ['', '', '', '', '']
                    bad_space = f"\033[38;2;{255};{128};{0}m{'░░'}\033[0m"
                    good_space = f"\033[38;2;{150};{255};{0}m{'░░'}\033[0m"
                    print(f"\033[{(i * 5) + 5 + i};1H\033[K", end="")
                    print(f"\033[{(i * 5) + 4 + i};1H\033[K", end="")
                    print(f"\033[{(i * 5) + 3 + i};1H\033[K", end="")
                    print(f"\033[{(i * 5) + 2 + i};1H\033[K", end="")
                    print(f"\033[{(i * 5) + 1 + i};1H\033[K", end="")
                    for j in range(len(memory) - min(default_memory_looked, game_round), len(memory)):
                        color[0] += good_space + f"\033[38;2;{0};{255};{0}m{'░░░░░░'}\033[0m" + good_space if memory[j][i] == 'C' else f"\033[38;2;{255};{0};{0}m{'░░░░░░'}\033[0m" + (bad_space * 2)
                        color[1] += f"\033[38;2;{0};{255};{0}m{'░░'}\033[0m" + (good_space * 4) if memory[j][i] == 'C' else f"\033[38;2;{255};{0};{0}m{'░░'}\033[0m" + (bad_space * 2) + f"\033[38;2;{255};{0};{0}m{'░░'}\033[0m" + bad_space
                        color[2] += f"\033[38;2;{0};{255};{0}m{'░░'}\033[0m" + (good_space * 4) if memory[j][i] == 'C' else f"\033[38;2;{255};{0};{0}m{'░░'}\033[0m" + (bad_space * 2) + f"\033[38;2;{255};{0};{0}m{'░░'}\033[0m" + bad_space
                        color[3] += f"\033[38;2;{0};{255};{0}m{'░░'}\033[0m" + (good_space * 4) if memory[j][i] == 'C' else f"\033[38;2;{255};{0};{0}m{'░░'}\033[0m" + (bad_space * 2) + f"\033[38;2;{255};{0};{0}m{'░░'}\033[0m" + bad_space
                        color[4] += good_space + f"\033[38;2;{0};{255};{0}m{'░░░░░░'}\033[0m" + good_space if memory[j][i] == 'C' else f"\033[38;2;{255};{0};{0}m{'░░░░░░'}\033[0m" + (bad_space * 2)
                    if i == (players_turn - 1):
                        color[0] += f"\033[38;2;{255};{255};{0}m{'░░░░░░░░░░'}\033[0m"
                        color[1] += f"\033[38;2;{255};{255};{0}m{'░░░░░░░░░░'}\033[0m"
                        color[2] += f"\033[38;2;{255};{255};{0}m{'░░░░░░░░░░'}\033[0m"
                        color[3] += f"\033[38;2;{255};{255};{0}m{'░░░░░░░░░░'}\033[0m"
                        color[4] += f"\033[38;2;{255};{255};{0}m{'░░░░░░░░░░'}\033[0m"
                    print(color[0]); print(color[1]); print(color[2]); print(color[3]); print(color[4])
                    if i == 0:
                        print(f"\033[38;2;{255};{255};{0}m{'░░░░░░░░░░'}\033[0m" * default_memory_looked)

                baris_pesan = 12
                print(f"\033[{baris_pesan};1H\033[K", end="")
                print(f'player 1: {score[0]}, NETRAL, \"\" | player 2: {score[1]}, NETRAL, \"\" | game round: {game_round} | turn: player {players_turn}')

                if game_round >= tournament_round:
                    print(memory)
                    break

    else:
        
        with_noise = input('with noise? ')
        with_noise_flag = with_noise in ('yes', 'Y', 'True', '+', 'noise')
        noise_prob = 0.05
        tournament_try = 5

        strategy_score = [0] * len(strategies)
        strategy_win = [0] * len(strategies)
        strategy_lose = [0] * len(strategies)
        strategy_draw = [0] * len(strategies)

        pairs = [(i, j) for i in range(len(strategies)) for j in range(i, len(strategies))]
        tasks = [(i, j, with_noise_flag, noise_prob) for i, j in pairs for _ in range(tournament_try)]

        print(f'processing tournament with {len(strategies)} strategies across {mp.cpu_count()} cores')

        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.map(run_match, tasks)

        for i, j, s1, s2 in results:
            strategy_score[i] += s1
            strategy_score[j] += s2
            if s1 > s2:
                strategy_win[i] += 1
                strategy_lose[j] += 1
            elif s2 > s1:
                strategy_lose[i] += 1
                strategy_win[j] += 1
            else:
                strategy_draw[i] += 1
                strategy_draw[j] += 1

        time_divider = tournament_try
        tournament_divider = len(strategies) * time_divider
        gabung = list(zip(strategy_score, strategies, strategy_win, strategy_draw, strategy_lose))
        hasil_urut = sorted(gabung, key=lambda x: x[0], reverse=True)

        print(f'Num | Name | Score | Category | Wins | Draws | Losses | Creator')
        for p in range(len(strategies)):
            score_p, name_p, win_p, draw_p, lose_p = hasil_urut[p]
            print(f'{p+1} | {name_p} | {round(score_p/tournament_divider,1)} | {strategies_category[name_p]} | {round(win_p/time_divider,1)} | {round(draw_p/time_divider,1)} | {round(lose_p/time_divider,1)} | {strategies_creator[name_p]}')

        plot_tournament_results(hasil_urut, strategies_category, strategies_creator,
                                 tournament_divider, time_divider, top_n=30)
