"""
compare_baselines.py -- compares the trained DQN agent against simple, "obvious" strategies.

WHY THIS MATTERS: your professor will ask "why do I need an RL agent instead of a simple rule?"
This script answers that directly, with numbers: it plays many games with each strategy below
and reports how often each one hit the data target, and how much power each one used.

THE BASELINES (the "obvious" strategies someone might try without RL):
    1. Always sub-6 GHz   -- the safe, low-power choice
    2. Always mmWave      -- a faster, riskier choice
    3. Always sub-THz     -- the fastest, most expensive choice
    4. Random              -- picks a random action every slot (a sanity-check floor: anything
                             worse than this means something is broken)
    5. Simple rule         -- "if blocked, drop to sub-6; otherwise use sub-THz" -- a basic
                             hand-written strategy a human engineer might try
    6. Trained DQN agent    -- the network trained by dqn_train.py

Run:  python compare_baselines.py
(make sure dqn_multiband.pt exists first -- i.e. you've already run dqn_train.py once)
"""
import numpy as np
import torch

from multiband_env import MultiBandEnv
from dqn_train import QNetwork

N_GAMES = 300   # how many games to test each strategy on, for a fair average


# ---------------- the "obvious" strategies ----------------
def always_band(band):
    return lambda obs: band


def random_policy(obs):
    return np.random.randint(4)


def simple_rule(obs):
    # obs[0] = "was the link blocked last slot?" (0 or 1) -- see multiband_env.py _get_obs()
    was_blocked = obs[0] > 0.5
    return 1 if was_blocked else 3   # 1 = sub-6 (safe), 3 = sub-THz (fast)


# ---------------- the trained DQN agent ----------------
def load_trained_agent(path="dqn_multiband.pt"):
    q_net = QNetwork()
    q_net.load_state_dict(torch.load(path))
    q_net.eval()   # tell PyTorch we're just using it, not training it anymore

    def dqn_policy(obs):
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            return int(torch.argmax(q_net(obs_t), dim=1).item())
    return dqn_policy


# ---------------- play N_GAMES games with one policy, average the results ----------------
def run_policy(policy, n_games=N_GAMES):
    env = MultiBandEnv()
    successes, powers, rewards = [], [], []
    for game in range(n_games):
        obs, _ = env.reset(seed=game)   # same seeds for every policy = a fair comparison
        done = False
        total_reward = 0.0
        while not done:
            action = policy(obs)
            obs, reward, done, _, info = env.step(action)
            total_reward += reward
        successes.append(info["success"])
        powers.append(env.total_power)
        rewards.append(total_reward)
    return {
        "success_rate": np.mean(successes),
        "avg_power_W": np.mean(powers),
        "avg_reward": np.mean(rewards),
    }


def main():
    policies = {
        "Always sub-6 GHz": always_band(1),
        "Always mmWave": always_band(2),
        "Always sub-THz": always_band(3),
        "Random": random_policy,
        "Simple rule (blocked->sub-6)": simple_rule,
        "Trained DQN agent": load_trained_agent(),
    }

    print(f"Testing each strategy on the same {N_GAMES} games...\n")
    results = {name: run_policy(policy) for name, policy in policies.items()}

    # ---- print a simple table ----
    header = f"{'Strategy':30s} {'Success rate':>13s} {'Avg power (W)':>14s} {'Avg reward':>11s}"
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        print(f"{name:30s} {r['success_rate']:13.2f} {r['avg_power_W']:14.1f} {r['avg_reward']:11.2f}")

    # ---- save a results table as a picture, and a bar chart, so you can drop them into slides ----
    try:
        import matplotlib.pyplot as plt

        names = list(results.keys())
        success = [results[n]["success_rate"] for n in names]
        power = [results[n]["avg_power_W"] for n in names]

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

        axes[0].barh(names, success, color="#4C72B0")
        axes[0].set_xlabel("Success rate (meets data target)")
        axes[0].set_xlim(0, 1)
        axes[0].invert_yaxis()

        axes[1].barh(names, power, color="#DD8452")
        axes[1].set_xlabel("Average power used per game (W)")
        axes[1].invert_yaxis()

        fig.suptitle(f"Trained DQN agent vs. baseline strategies ({N_GAMES} games each)")
        fig.tight_layout()
        fig.savefig("baseline_comparison.png", dpi=150)
        print("\nSaved chart to baseline_comparison.png")
    except ImportError:
        print("\n(matplotlib not installed -- skipping chart. Run: pip install matplotlib)")


if __name__ == "__main__":
    main()