from typing import List
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from qaisim.qrl import ParametrizedQC, PolicyGradient, DeepQLearning
from train_classical import MLP
from train_policy import generate_episodes
from train_q_val import episode_interaction, env
from tqdm.auto import tqdm

plt.style.use("seaborn-v0_8-whitegrid")
plt.rc("font", family="serif")


class GreedyBaseline(tf.keras.Model):
    def __init__(self, num_actions):
        super(GreedyBaseline, self).__init__()
        self.num_actions = num_actions
        self.qnode_qubits = [156, 133, 127, 127, 27]

    def call(self, inputs):
        if isinstance(inputs, list):
            inputs = inputs[0]
        batch_size = tf.shape(inputs)[0]
        valid_actions_mask = tf.map_fn(
            lambda i: self.qnode_qubits >= inputs[i, 6] * 50,
            tf.range(batch_size),
            dtype=tf.bool,
        )

        masked_inputs = tf.where(
            valid_actions_mask, inputs[:, : self.num_actions], float("inf")
        )
        greedy_actions = tf.argmin(masked_inputs, axis=1)
        action_probs = tf.one_hot(
            greedy_actions, depth=self.num_actions, dtype=tf.float32
        )
        return action_probs


def eval_baseline(num_actions: int, num_episodes: int, batch_size: int) -> List[float]:
    model = GreedyBaseline(num_actions)
    episode_reward_history = []

    pbar = tqdm(total=num_episodes // batch_size, colour="cyan")
    for batch in range(num_episodes // batch_size):
        pbar.set_description(f"Batch [{batch + 1}/{num_episodes // batch_size}]")
        episodes = generate_episodes(model, num_actions, batch_size, None)
        rewards = [ep["rewards"] for ep in episodes]

        for ep_rewards in rewards:
            episode_reward_history.append(np.sum(ep_rewards))

        avg_rewards = np.mean(episode_reward_history[-batch_size:])
        pbar.set_postfix({"Avg Reward": f"{avg_rewards:.2f}"})
        pbar.update(1)
    pbar.close()
    return episode_reward_history


def plot(
    baseline_returns: List[float],
    quantum_returns: List[float],
    classical_returns: List[float],
    title: str,
    save_path: str,
    attr: str = "Returns",
) -> None:
    baseline_returns_np = np.convolve(baseline_returns, np.ones(10) / 10, mode="valid")
    quantum_returns_np = np.convolve(quantum_returns, np.ones(10) / 10, mode="valid")
    classical_returns_np = np.convolve(
        classical_returns, np.ones(10) / 10, mode="valid"
    )

    plt.plot(quantum_returns_np, color="blue", label=f"QAISim")
    plt.plot(classical_returns_np, color="red", label=f"QSimPy")
    plt.plot(baseline_returns_np, color="green", label=f"Baseline")
    plt.xlabel("Episode")
    plt.ylabel(attr)
    plt.legend()
    plt.title(title, fontweight="bold")
    plt.savefig(save_path)


def evaluate_policy() -> None:
    num_actions = 5
    num_episodes = 100
    batch_size = 10
    baseline_returns = eval_baseline(num_actions, num_episodes, batch_size)

    pqc = ParametrizedQC(num_qubits=8, num_layers=5)
    model_quantum = PolicyGradient(pqc, num_actions=num_actions, lrs=(0.03, 0.05, 0.03))
    model_quantum.model.load_weights("./results/policy/model.h5")
    model_quantum.eval(generate_episodes, num_episodes=num_episodes)
    quantum_returns = model_quantum.eval_episode_reward_history
    quantum_waiting_time = model_quantum.eval_waiting_time

    model_classical = MLP(num_actions=5, raw_scores=False)
    model_classical.build(input_shape=(None, 8))
    model_classical.load_weights("./results/policy/classical_model.h5")
    classical_returns, classical_waiting_time = [], []

    pbar = tqdm(total=num_episodes // batch_size, colour="cyan")
    for batch in range(num_episodes // batch_size):
        pbar.set_description(f"Batch [{batch + 1}/{num_episodes // batch_size}]")
        episodes = generate_episodes(model_classical, num_actions, batch_size, None)
        rewards = [ep["rewards"] for ep in episodes]
        waiting_times = [ep["waiting_time"] for ep in episodes]

        for i, ep_rewards in enumerate(rewards):
            classical_returns.append(np.sum(ep_rewards))
            classical_waiting_time.append(np.sum(waiting_times[i]))

        avg_rewards = np.mean(classical_returns[-batch_size:])
        pbar.set_postfix({"Avg Reward": f"{avg_rewards:.2f}"})
        pbar.update(1)

    pbar.close()

    print(f"Avg. Baseline Return: {np.mean(baseline_returns):.4f}")
    print(f"Avg. Quantum Policy Returns: {np.mean(quantum_returns):.4f}")
    print(f"Avg. Classical Policy Returns: {np.mean(classical_returns):.4f}")

    plot(
        baseline_returns=baseline_returns,
        quantum_returns=quantum_returns,
        classical_returns=classical_returns,
        title="Policy Evaluation",
        save_path="./results/policy/policy_eval.pdf",
    )

    plt.clf()
    classical_wt_np = np.convolve(
        classical_waiting_time, np.ones(10) / 10, mode="valid"
    )
    quantum_wt_np = np.convolve(quantum_waiting_time, np.ones(10) / 10, mode="valid")

    plt.plot(quantum_wt_np, color="blue", label="QAISim")
    plt.plot(classical_wt_np, color="red", label="QSimPy")
    plt.xlabel("Episode")
    plt.ylabel("Waiting Time")
    plt.legend()
    plt.title("Task Waiting Time - Policy", fontweight="bold")
    plt.savefig("./results/policy/waiting_time.pdf")


def evaluate_q_val() -> None:
    num_actions = 5
    num_episodes = 100
    batch_size = 10
    epsilon = 0.01

    baseline_returns = eval_baseline(num_actions, num_episodes, batch_size)

    pqc = ParametrizedQC(num_qubits=8, num_layers=5)
    model_quantum = DeepQLearning(pqc, num_actions=num_actions, lrs=(0.03, 0.05, 0.03))
    model_quantum.model.load_weights("./results/dq_learning/model.h5")
    model_quantum.epsilon = epsilon

    model_quantum.eval(env, episode_interaction, num_episodes)
    quantum_returns = model_quantum.eval_episode_reward_history
    quantum_waiting_time = model_quantum.eval_waiting_time

    model_classical = MLP(num_actions=5, raw_scores=True)
    model_classical.build(input_shape=(None, 8))
    model_classical.load_weights("./results/dq_learning/classical_model.h5")
    classical_returns, classical_waiting_time = [], []

    pbar = tqdm(total=num_episodes, colour="cyan")
    for episode in range(num_episodes):
        pbar.set_description(f"Episode [{episode + 1}/{num_episodes}]")
        episode_reward, episode_wt = 0.0, 0.0
        state = env.reset()[0]

        while True:
            state = np.concatenate(
                [
                    state["qnode_queued_tasks"],
                    state["qtask_arrival_time"],
                    state["qtask_num_qubits"],
                    state["qtask_circuit_layers"],
                ]
            )

            interaction = episode_interaction(
                model_classical, num_actions, epsilon, state
            )[0]

            state = interaction["next_state"]
            episode_reward += interaction["reward"]
            episode_wt += interaction["waiting_time"]

            if interaction["done"]:
                break

        classical_returns.append(episode_reward)
        classical_waiting_time.append(episode_wt)

        average_rewards = np.mean(classical_returns[-batch_size:])

        pbar.set_postfix({"Avg Reward": f"{average_rewards:.2f}"})
        pbar.update(1)

    pbar.close()

    print(f"Avg. Baseline Return: {np.mean(baseline_returns):.4f}")
    print(f"Avg. Quantum Q-Value Returns: {np.mean(quantum_returns):.4f}")
    print(f"Avg. Classical Q-Value Returns: {np.mean(classical_returns):.4f}")

    plot(
        baseline_returns=baseline_returns,
        quantum_returns=quantum_returns,
        classical_returns=classical_returns,
        title="Deep Q-Learning Evaluation",
        save_path="./results/dq_learning/dql_eval.pdf",
    )

    plt.clf()
    classical_wt_np = np.convolve(
        classical_waiting_time, np.ones(10) / 10, mode="valid"
    )
    quantum_wt_np = np.convolve(quantum_waiting_time, np.ones(10) / 10, mode="valid")

    plt.plot(quantum_wt_np, color="blue", label="QAISim")
    plt.plot(classical_wt_np, color="red", label="QSimPy")
    plt.xlabel("Episode")
    plt.ylabel("Waiting Time")
    plt.legend()
    plt.title("Task Waiting Time - Deep Q-Learning", fontweight="bold")
    plt.savefig("./results/dq_learning/waiting_time.pdf")


if __name__ == "__main__":
    evaluate_policy()
    evaluate_q_val()
