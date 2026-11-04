import gymnasium as gym
import numpy as np

EPISODES = 10_000
ALPHA = 0.1
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.9995
MAX_STEPS = 200
SEEDS = [42, 43, 44, 45, 46]


def make_env(rainy=False):
    if rainy:
        return gym.make("Taxi-v4", is_rainy=True, rainy_probability=0.6)
    return gym.make("Taxi-v4")


def epsilon_greedy(q_table, state, epsilon, rng, action_space):
    if rng.random() < epsilon:
        return int(action_space.sample())
    return int(np.argmax(q_table[state]))


def train_q_learning(seed, rainy=False):
    env = make_env(rainy)
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    rewards = np.zeros(EPISODES)
    rng = np.random.default_rng(seed)
    epsilon = EPSILON_START

    for episode in range(EPISODES):
        state, _ = env.reset(seed=seed + episode)
        total_reward = 0

        for _ in range(MAX_STEPS):
            action = epsilon_greedy(q_table, state, epsilon, rng, env.action_space)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            target = reward if done else reward + GAMMA * np.max(q_table[next_state])
            q_table[state, action] += ALPHA * (target - q_table[state, action])

            state = next_state
            total_reward += reward

            if done:
                break

        rewards[episode] = total_reward
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

    env.close()
    return q_table, rewards


def train_sarsa(seed, rainy=False):
    env = make_env(rainy)
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    rewards = np.zeros(EPISODES)
    rng = np.random.default_rng(seed)
    epsilon = EPSILON_START

    for episode in range(EPISODES):
        state, _ = env.reset(seed=seed + episode)
        action = epsilon_greedy(q_table, state, epsilon, rng, env.action_space)
        total_reward = 0

        for _ in range(MAX_STEPS):
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if done:
                target = reward
                next_action = None
            else:
                next_action = epsilon_greedy(
                    q_table, next_state, epsilon, rng, env.action_space
                )
                target = reward + GAMMA * q_table[next_state, next_action]

            q_table[state, action] += ALPHA * (target - q_table[state, action])
            total_reward += reward

            if done:
                break

            state = next_state
            action = next_action

        rewards[episode] = total_reward
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

    env.close()
    return q_table, rewards


def train_many(algorithm, rainy=False):
    q_tables = []
    all_rewards = []

    trainer = train_q_learning if algorithm == "q_learning" else train_sarsa

    for seed in SEEDS:
        q_table, rewards = trainer(seed, rainy=rainy)
        q_tables.append(q_table)
        all_rewards.append(rewards)

    return q_tables, np.array(all_rewards)


def evaluate_qtables(q_tables, rainy=False, episodes_per_table=20):
    results = []

    for table_index, q_table in enumerate(q_tables):
        env = make_env(rainy)

        for episode in range(episodes_per_table):
            state, _ = env.reset(seed=10_000 + table_index * 1_000 + episode)
            total_reward = 0
            success = False
            steps = 0

            for steps in range(1, MAX_STEPS + 1):
                action = int(np.argmax(q_table[state]))
                next_state, reward, terminated, truncated, _ = env.step(action)
                state = next_state
                total_reward += reward

                if terminated:
                    success = True
                    break

                if truncated:
                    break

            results.append(
                {
                    "q_table": table_index + 1,
                    "episode": episode + 1,
                    "timesteps": steps,
                    "reward": total_reward,
                    "success": success,
                }
            )

        env.close()

    return results


def summarize_results(results):
    successful_steps = [r["timesteps"] for r in results if r["success"]]
    rewards = [r["reward"] for r in results]

    return {
        "episodes": len(results),
        "success_rate": 100 * np.mean([r["success"] for r in results]),
        "mean_timesteps": (
            float(np.mean(successful_steps)) if successful_steps else float("nan")
        ),
        "std_timesteps": (
            float(np.std(successful_steps)) if successful_steps else float("nan")
        ),
        "mean_reward": float(np.mean(rewards)),
    }


def moving_average(values, window=100):
    values = np.asarray(values)
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")
