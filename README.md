# Multiband RL

An agent that learns to pick the best wireless frequency band to use, trained with DQN (Deep Q-Learning) — adapted from the same [MIT 6.S191](https://github.com/MITDeepLearning/introtodeeplearning) reinforcement learning ideas used in my [CartPole project](https://github.com/siyavp/cartpole-rl), applied to a networking problem instead of balancing a pole.

## What is this environment?

Imagine a phone connected to a cell tower. The tower can send data to the phone using one of a few different "bands" (frequencies):

- **sub-6 GHz** — slow, but reliable and uses the least power
- **mmWave** — faster, uses more power, and can get blocked by things in the way (like a person walking by)
- **sub-THz** — fastest, uses the most power, and gets blocked even more easily
- **no transmit** — send nothing this turn (uses zero power, but also sends zero data)

Every "slot" (turn), the agent has to pick one of these 4 options. A full game is 20 slots. At the end, we check: did the tower send enough total data?

- **Yes** → +1 reward
- **No** → −1 reward

And the whole way through, the agent loses a tiny bit of reward for every bit of power it uses. So the agent has to learn a balance: **send enough data, but don't waste power doing it.**

This mirrors a real problem: high-frequency bands (mmWave, sub-THz) give you speed but cost battery and get blocked easily, so a smart system should switch bands based on what's actually happening right now — not just always use the fastest one.

## What the agent sees and does

| Thing | What it means |
|---|---|
| **State (what the agent sees)** | 9 numbers: was the link blocked last turn, how much the user is moving, which band was used last turn, how close we are to the data goal, how much power used so far, how much time is left |
| **Action (what the agent picks)** | One of 4 choices: no transmit, sub-6 GHz, mmWave, sub-THz |
| **Reward (the feedback it gets)** | Small penalty every turn for power used, plus +1 or −1 at the end depending on whether enough data was sent |

## How the agent learns (DQN, explained simply)

This is a different learning method than the CartPole project used. CartPole used **REINFORCE** (play a whole game, then look back and reward the actions that led to a good game). This project uses **DQN**, which learns after every single step instead of waiting for the whole game to finish.

Think of it like this: instead of reflecting on a whole day at the end, DQN checks in after every single decision and asks "was that choice better or worse than I expected?" — and adjusts right away.

The 5 main pieces:

| Piece | What it does |
|---|---|
| **Q-network** | A small neural network that looks at the current state and gives a score to each of the 4 actions — "how good do I think this choice is?" |
| **Replay buffer** | A memory of past turns (state, action, reward, what happened next). The agent learns from a random mix of old memories, not just the very last thing that happened |
| **Epsilon-greedy** | Early on, the agent acts randomly a lot, to explore and see what happens. Over time it relies more and more on its own best guesses |
| **Target network** | A second, slow-moving copy of the Q-network. It's used to judge "what SHOULD that score have been," so the agent isn't chasing its own shifting opinion while it's still learning |
| **Training loop** | Repeats: take one step → remember it → learn a little from a batch of memories → occasionally update the target network |

There's no physics or network simulation in the agent's head, no planning ahead — it just slowly improves its score guesses by comparing them to what actually happened, repeated over hundreds of games.

## What the code does

| File | What it is |
|---|---|
| `multiband_env.py` | The "game" itself — the environment. Defines the 4 bands, how fast/blocked they are, and the reward rules |
| `dqn_train.py` | The agent that plays the game over and over and learns from it |

## What actually happened when I ran it

```
Episode   25 | avg reward (last 25): -1.12 | success rate: 0.04 | epsilon: 0.94
Episode  100 | avg reward (last 25): -1.14 | success rate: 0.04 | epsilon: 0.76
Episode  250 | avg reward (last 25): -0.43 | success rate: 0.44 | epsilon: 0.41
Episode  400 | avg reward (last 25): -0.10 | success rate: 0.64 | epsilon: 0.05
Episode  600 | avg reward (last 25):  0.19 | success rate: 0.76 | epsilon: 0.05

Evaluating trained agent (no more exploring, just its best guesses)...
Trained agent: success rate = 0.69, average power used = 71.0 W
```

The success rate (how often it sent enough data) climbed from basically never (4%) to meeting the target most of the time (around 70-76%) by the end of training — clear evidence the agent is learning a real strategy, not just getting lucky.

A few things worth noting about this run:

- **It's noisy in the middle** (episodes 400-525 bounce up and down even after exploration is basically turned off). This is normal for DQN — it's still refining its strategy, not getting worse.
- **The evaluation result (69%) is close to the last training average (76%)**, which is a good sign — if the gap were huge, it would suggest the agent only did well on the specific random situations it trained on, rather than learning something that generalizes.
- **This isn't fully tuned yet.** 69% success at 71 Watts is proof the whole pipeline works end-to-end, not necessarily the best the agent could do. The next step would be adjusting things like how much we penalize power use, or training for longer, to push both numbers in a better direction.

## On the data

Right now, `multiband_env.py` makes up how fast each band is and when it gets blocked, using simple made-up formulas — enough to build and test the agent, but not real network data. Once real data is available (from my professor, or synthetic data generated with NYUSIM or DeepMIMO), the plan is to swap those made-up formulas out for real values. The agent itself (`dqn_train.py`) doesn't need to change at all — it only ever sees the 9 numbers describing the state and a reward number, so it doesn't know or care whether those numbers came from a formula or from real measurements.

## Running it

```bash
python3 -m venv venv
source venv/bin/activate
pip install gymnasium torch numpy
python3 multiband_env.py     # quick check: plays one random game, prints each step
python3 dqn_train.py         # trains the agent, prints progress, saves dqn_multiband.pt
```

## Natural next steps

- Compare the trained agent against simple baselines (always use mmWave, always use sub-THz, a basic if-blocked-then-switch rule) to show the learned policy is actually better than obvious strategies
- Tune the power penalty and training length to improve the success-rate / power-used tradeoff
- Swap in real or synthetic channel data (from my professor, NYUSIM, or DeepMIMO) in place of the made-up formulas in `multiband_env.py`
- Add a plot of reward per episode (like the CartPole project's `reward_curve.png`) so progress is visible at a glance

## Credits

Built using the same reinforcement learning ideas from [MIT 6.S191 Introduction to Deep Learning](https://github.com/MITDeepLearning/introtodeeplearning), applied to a multiband wireless networking problem for an undergraduate thesis proposal advised by Dr. Brijesh Soni.