"""
dqn_train.py -- teaches an agent to play the multiband game using DQN (Deep Q-Network).

HOW DQN LEARNS (in simple words, vs. the REINFORCE agent in your CartPole repo):
    REINFORCE: play a whole game, THEN look back and reinforce the actions that led to a good game.
    DQN:       after EVERY single slot, ask "was this one step better or worse than I expected?"
               and nudge the network a little, right away. No need to wait for the game to end.

    The network doesn't output an action directly -- it outputs a "Q-value" (expected future score)
    for EACH of the 4 actions, and we just pick the action with the highest Q-value (mostly).

THE PIECES (these are the 5 things your professor will ask about):
    1. Q-network       -- a small neural net: state (9 numbers) -> 4 numbers (one score per action)
    2. Replay buffer    -- a memory of past (state, action, reward, next_state) steps, so we can
                           re-learn from old experience instead of only the most recent step
    3. Epsilon-greedy   -- sometimes act randomly (explore), mostly act on the network's best guess
                           (exploit); epsilon starts high and slowly shrinks
    4. Target network    -- a slow-moving COPY of the Q-network, used only to compute "what SHOULD
                           this action's score have been". Keeping it slow-moving stops the agent
                           from chasing its own shifting guesses, which makes training more stable
    5. Training loop     -- repeatedly: play a step -> store it -> sample a batch from memory ->
                           update the Q-network a little -> occasionally update the target network

DATA NOTE: MultiBandEnv currently makes up channel numbers with simple formulas (see multiband_env.py).
Once real data is ready (from your professor, or synthetic data from NYUSIM / DeepMIMO), the plan is
to replace `_update_blockage` and the CLEAR_RATE / BLOCKED_RATE numbers in multiband_env.py with
values read from that dataset. Nothing else in this file needs to change -- the agent only ever sees
the 9-number state and a reward, so it doesn't care where those numbers come from.

Run:  python dqn_train.py
"""
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from multiband_env import MultiBandEnv

# ---------------- settings you can change ----------------
EPISODES = 600            # how many games to practice
BATCH_SIZE = 64            # how many past steps we learn from at once
BUFFER_SIZE = 10_000       # how many past steps we remember
LEARNING_RATE = 1e-3
GAMMA = 0.99               # how much we care about future reward vs. right-now reward
EPS_START, EPS_END, EPS_DECAY_EPISODES = 1.0, 0.05, 400   # epsilon shrinks from 1.0 to 0.05
TARGET_UPDATE_EVERY = 200  # copy Q-network into target network every this many learning steps


# ---------------- 1. Q-network ----------------
# Takes the 9-number state and outputs 4 numbers: one "how good is this action" score per action.
class QNetwork(nn.Module):
    def __init__(self, state_size=9, n_actions=4, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, state):
        return self.net(state)


# ---------------- 2. Replay buffer ----------------
# A simple memory box. We push new experience in, and later pull a random handful back out.
# Random sampling matters: it stops the agent from learning only from the last few, similar steps.
class ReplayBuffer:
    def __init__(self, capacity):
        self.data = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.data.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.as_tensor(np.array(states), dtype=torch.float32),
                torch.as_tensor(actions, dtype=torch.int64),
                torch.as_tensor(rewards, dtype=torch.float32),
                torch.as_tensor(np.array(next_states), dtype=torch.float32),
                torch.as_tensor(dones, dtype=torch.float32))

    def __len__(self):
        return len(self.data)


# ---------------- 3. Epsilon-greedy action choice ----------------
def pick_action(q_net, state, epsilon, n_actions=4):
    if random.random() < epsilon:
        return random.randrange(n_actions)          # explore: random action
    with torch.no_grad():
        state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = q_net(state_t)
        return int(torch.argmax(q_values, dim=1).item())   # exploit: best-looking action


def epsilon_for_episode(ep):
    # straight-line (linear) shrink from EPS_START down to EPS_END
    frac = min(ep / EPS_DECAY_EPISODES, 1.0)
    return EPS_START + frac * (EPS_END - EPS_START)


# ---------------- 5. Training loop ----------------
def train():
    env = MultiBandEnv()
    q_net = QNetwork()
    target_net = QNetwork()
    target_net.load_state_dict(q_net.state_dict())      # 4. target network starts as a copy of q_net

    optimizer = torch.optim.Adam(q_net.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    buffer = ReplayBuffer(BUFFER_SIZE)

    learn_steps = 0
    reward_history, success_history = [], []

    for ep in range(EPISODES):
        state, _ = env.reset(seed=None)
        epsilon = epsilon_for_episode(ep)
        episode_reward = 0.0
        done = False

        while not done:
            action = pick_action(q_net, state, epsilon)
            next_state, reward, done, _, info = env.step(action)
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            # only start learning once we have enough memory for a full batch
            if len(buffer) >= BATCH_SIZE:
                states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)

                # "What score did we GIVE this action at the time?"
                q_predicted = q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

                # "What score SHOULD it have gotten?" = reward now + (discounted) best future score,
                # using the slow-moving target network so this target doesn't keep shifting under us.
                with torch.no_grad():
                    q_next_best = target_net(next_states).max(dim=1).values
                    q_target = rewards + GAMMA * q_next_best * (1 - dones)   # 0 future score if done

                loss = loss_fn(q_predicted, q_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                learn_steps += 1
                if learn_steps % TARGET_UPDATE_EVERY == 0:
                    target_net.load_state_dict(q_net.state_dict())   # refresh the target network

        reward_history.append(episode_reward)
        success_history.append(bool(info["success"]))

        if (ep + 1) % 25 == 0:
            avg_r = np.mean(reward_history[-25:])
            avg_s = np.mean(success_history[-25:])
            print(f"Episode {ep+1:4d} | avg reward (last 25): {avg_r:6.2f} "
                  f"| success rate (last 25): {avg_s:4.2f} | epsilon: {epsilon:.2f}")

    return q_net, reward_history, success_history


# ---------------- quick check after training: how good is the trained agent? ----------------
def evaluate(q_net, episodes=200):
    env = MultiBandEnv()
    successes, powers = [], []
    for ep in range(episodes):
        state, _ = env.reset(seed=ep)
        done = False
        while not done:
            action = pick_action(q_net, state, epsilon=0.0)   # epsilon=0: always pick the best action
            state, _, done, _, info = env.step(action)
        successes.append(info["success"])
        powers.append(env.total_power)
    return np.mean(successes), np.mean(powers)


if __name__ == "__main__":
    trained_q_net, rewards, successes = train()

    print("\nEvaluating trained agent (no more exploring, just its best guesses)...")
    success_rate, avg_power = evaluate(trained_q_net)
    print(f"Trained agent: success rate = {success_rate:.2f}, average power used = {avg_power:.1f} W")

    torch.save(trained_q_net.state_dict(), "dqn_multiband.pt")
    print("Saved trained network to dqn_multiband.pt")