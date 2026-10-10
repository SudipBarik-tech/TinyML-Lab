# 20 — Q-Learning on a Small Embedded State Space

This experiment demonstrates tabular Q-Learning using a small grid-world environment and exports the learned policy for microcontroller use.

## Objective
Show how reinforcement-learning decisions can be reduced to a compact lookup policy after desktop-side training.

## Method
A 4×4 grid world is used. The agent starts in the top-left state and learns to reach the goal in the bottom-right state while avoiding one blocked cell. Training uses an epsilon-greedy policy with fixed random seed.

## Run
```bash
python 20-q-learning/train.py
```

The script prints the learned policy and exports both the Q-table and greedy action table to `embedded/q_policy.h`.

## Embedded inference
`embedded/q_learning_demo.ino` demonstrates policy lookup. No reinforcement-learning updates are required on the device unless online adaptation is specifically desired.

## TinyML note
For small state spaces, a learned Q-table can be extremely lightweight. The main limitation is scalability: memory grows with the product of states and actions.
