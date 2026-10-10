from pathlib import Path
import numpy as np

SEED = 42
OUT = Path(__file__).resolve().parent / "embedded" / "q_policy.h"
rng = np.random.default_rng(SEED)

ROWS, COLS, ACTIONS = 4, 4, 4
N_STATES = ROWS * COLS
GOAL = 15
BLOCKED = 5
ALPHA = 0.15
GAMMA = 0.95
EPSILON = 0.20
EPISODES = 2500
MAX_STEPS = 80

Q = np.zeros((N_STATES, ACTIONS), dtype=np.float32)


def next_state(state, action):
    r, c = divmod(state, COLS)
    nr, nc = r, c
    if action == 0:
        nr = max(0, r - 1)
    elif action == 1:
        nc = min(COLS - 1, c + 1)
    elif action == 2:
        nr = min(ROWS - 1, r + 1)
    else:
        nc = max(0, c - 1)
    candidate = nr * COLS + nc
    if candidate == BLOCKED:
        return state
    return candidate


def reward(state):
    if state == GOAL:
        return 10.0
    return -0.1


for _ in range(EPISODES):
    state = 0
    for _ in range(MAX_STEPS):
        if rng.random() < EPSILON:
            action = int(rng.integers(ACTIONS))
        else:
            action = int(np.argmax(Q[state]))
        ns = next_state(state, action)
        r = reward(ns)
        td_target = r if ns == GOAL else r + GAMMA * np.max(Q[ns])
        Q[state, action] += ALPHA * (td_target - Q[state, action])
        state = ns
        if state == GOAL:
            break

policy = np.argmax(Q, axis=1).astype(np.int32)
policy[GOAL] = -1
policy[BLOCKED] = -1
print("Policy:")
print(policy.reshape(ROWS, COLS))

lines = [
    "#pragma once",
    "",
    f"constexpr int Q_STATES = {N_STATES};",
    f"constexpr int Q_ACTIONS = {ACTIONS};",
    f"constexpr int Q_GOAL = {GOAL};",
    f"constexpr int Q_BLOCKED = {BLOCKED};",
    "const float Q_TABLE[Q_STATES][Q_ACTIONS] = {",
]
for row in Q:
    lines.append("  {" + ", ".join(f"{v:.6f}f" for v in row) + "},")
lines.append("};")
lines.append("const int Q_POLICY[Q_STATES] = {" + ", ".join(map(str, policy.tolist())) + "};")
OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Exported policy to {OUT}")
