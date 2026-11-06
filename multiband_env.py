"""
multiband_env.py  -- a simple "game" for the RL agent, like CartPole but for networking.

THE GAME
  * Every turn (a "time slot") the agent picks ONE of 4 actions:
        0 = don't send anything   1 = use sub-6 GHz   2 = use mmWave   3 = use sub-THz
  * Sending data on a band gives some speed (Mbps) but costs some power (Watts).
  * mmWave and sub-THz are FAST but use MORE power, and they get BLOCKED sometimes
    (a person or wall gets in the way), which makes them slow.
  * One game (episode) = 20 slots. At the end, we check: did we send enough data?
        enough data  -> +1 reward      not enough -> -1 reward
    and along the way we lose a little reward for every Watt of power we use.
  * So the agent must learn: "send enough data, but with as little power as possible".

HOW TO USE IT (same as CartPole):
    env = MultiBandEnv()
    obs, info = env.reset()                                   # start a new game
    obs, reward, done, truncated, info = env.step(action)     # play one slot
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ---------------- numbers you can change ----------------
NUM_SLOTS = 20           # how many slots in one game
TARGET_AVG_RATE = 2500   # we want to send 2500 Mbps per slot on average
SWITCH_COST = 0.3        # if we change band, we lose 30% of that slot's speed
POWER_WEIGHT = 0.5       # how much we care about power (set 0 to ignore power)

# ---------------- the 4 actions: no send, sub-6, mmWave, sub-THz ----------------
# (these are made-up but sensible numbers; later you can replace them with real data)
CLEAR_RATE = np.array([0, 300, 3900, 7000])    # speed in Mbps when NOT blocked
BLOCKED_RATE = np.array([0, 300, 950, 10])     # speed in Mbps when blocked (sub-6 never blocked)
POWER = np.array([0.0, 1.0, 2.5, 5.0])         # power used in Watts


class MultiBandEnv(gym.Env):

    def __init__(self):
        super().__init__()
        # 4 possible actions (0, 1, 2 or 3), like CartPole has 2 (left, right)
        self.action_space = spaces.Discrete(4)
        # what the agent sees: 9 numbers, each between 0 and 2 (listed in _get_obs below)
        self.observation_space = spaces.Box(low=0.0, high=2.0, shape=(9,), dtype=np.float32)
        # total data we must send in one game
        self.target = TARGET_AVG_RATE * NUM_SLOTS

    # ---------- start a new game ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)         # sets up self.np_random (the random number generator)
        self.slot = 0                    # which slot we are on (0 to 19)
        self.total_rate = 0.0            # total data sent so far
        self.total_power = 0.0           # total power used so far
        self.last_action = 1             # we start on sub-6 GHz
        self.motion = self.np_random.random()   # how much the user is moving (0 = still, 1 = moving a lot)
        self.blocked = 0                 # 1 if mmWave / sub-THz are blocked right now, else 0
        self.seen_blocked = 0            # what the agent saw in the LAST slot
        return self._get_obs(), {}

    # ---------- play one slot ----------
    def step(self, action):
        action = int(action)

        # 1) Decide if the link is blocked in this slot
        self._update_blockage()

        # 2) Speed of the band we picked (a bit random, like signal ups and downs)
        speed = BLOCKED_RATE[action] if self.blocked else CLEAR_RATE[action]
        speed = speed * self.np_random.uniform(0.8, 1.2)

        # 3) Changing to a different band wastes some time, so we lose some speed
        if action != 0 and action != self.last_action:
            speed = speed * (1 - SWITCH_COST)

        # 4) Keep track of everything
        power = POWER[action]
        self.total_rate += speed
        self.total_power += power
        self.last_action = action
        self.seen_blocked = self.blocked     # next turn the agent will see THIS slot's blockage
        self.slot += 1

        # 5) Reward: lose a little for power used in every slot ...
        reward = -POWER_WEIGHT * (power / POWER.max()) / NUM_SLOTS
        # ... and at the very end, +1 if we sent enough data, -1 if not
        done = self.slot >= NUM_SLOTS
        success = self.total_rate >= self.target
        if done:
            reward += 1.0 if success else -1.0

        info = {"speed": speed, "power": power, "success": success if done else None}
        return self._get_obs(), reward, done, False, info    # (False = "truncated", always False for us)

    # ---------- what the agent sees (9 numbers) ----------
    def _get_obs(self):
        one_hot = np.eye(4)[self.last_action]             # last action as [0,1,0,0]-style list
        obs = np.concatenate([
            [self.seen_blocked],                          # 1 number : was the link blocked last slot? (0 or 1)
            [self.motion],                                # 1 number : how much the user is moving (0 to 1)
            one_hot,                                      # 4 numbers: which band we used last slot
            [min(self.total_rate / self.target, 2.0)],    # 1 number : how close we are to the data target (1 = reached it)
            [self.total_power / (NUM_SLOTS * POWER.max())],  # 1 number : power used so far (0 to 1)
            [1 - self.slot / NUM_SLOTS],                  # 1 number : time left in the game (1 = start, 0 = end)
        ])
        return obs.astype(np.float32)

    # ---------- blockage: sometimes a person/wall gets in the way ----------
    def _update_blockage(self):
        # the user's motion changes slowly up and down (kept between 0 and 1)
        self.motion = float(np.clip(self.motion + self.np_random.normal(0, 0.1), 0, 1))
        if self.blocked:
            # if blocked, there is a 20% chance it clears up each slot
            if self.np_random.random() < 0.2:
                self.blocked = 0
        else:
            # if clear, the more the user moves, the more likely it gets blocked
            chance_to_block = 0.05 + 0.3 * self.motion
            if self.np_random.random() < chance_to_block:
                self.blocked = 1


# ---------- quick test: play one game with random actions (run: python multiband_env.py) ----------
if __name__ == "__main__":
    env = MultiBandEnv()
    obs, _ = env.reset(seed=0)
    done = False
    while not done:
        action = env.action_space.sample()                # pick a random action
        obs, reward, done, _, info = env.step(action)
        print(f"slot {env.slot:2d}  action={action}  speed={info['speed']:7.1f} Mbps  reward={reward:6.3f}")
    print("Reached the data target?", info["success"])