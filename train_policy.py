from qaisim.utils import Dataset
from qaisim.qrl import QRLEnv, PolicyGradient, ParametrizedQC
from collections import defaultdict
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

dataset = Dataset(file_name="./data/qtasks_train.csv")
state_bounds = [30] * 5 + [60, 50, 200000]

plt.style.use("seaborn-v0_8-whitegrid")
plt.rc("font", family="serif")


def generate_episodes(model, n_actions, n_episodes, x=None):
    episodes = [defaultdict(list) for _ in range(n_episodes)]
    envs = [QRLEnv(dataset) for _ in range(n_episodes)]
    done = [False for _ in range(n_episodes)]
    states = [e.reset() for e in envs]
    states = [_state[0] for _state in states]

    while not all(done):
        states = [
            np.concatenate(
                [
                    state["qnode_queued_tasks"],
                    state["qtask_arrival_time"],
                    state["qtask_num_qubits"],
                    state["qtask_circuit_layers"],
                ]
            )
            if state
            else np.array([])
            for state in states
        ]

        unfinished_ids = [i for i in range(n_episodes) if not done[i]]
        normalized_states = [
            s / state_bounds for i, s in enumerate(states) if not done[i]
        ]

        for i, state in zip(unfinished_ids, normalized_states):
            episodes[i]["states"].append(state)

        states = tf.convert_to_tensor(normalized_states)
        action_probs = model([states])

        states = [None for _ in range(n_episodes)]
        for i, policy in zip(unfinished_ids, action_probs.numpy()):
            action = np.random.choice(n_actions, p=policy)
            states[i], reward, done[i], _, _, waiting_time = envs[i].step(action)
            episodes[i]["actions"].append(action)
            episodes[i]["rewards"].append(reward)
            episodes[i]["waiting_time"].append(waiting_time)

    return episodes


def train_policy():
    num_qubits = 8
    num_layers = 5
    lrs = (0.03, 0.05, 0.03)
    num_actions = 5
    num_episodes = 1500

    paramatrized_qc = ParametrizedQC(num_qubits, num_layers)
    policy_gradient_agent = PolicyGradient(paramatrized_qc, num_actions, lrs=lrs)

    policy_gradient_agent.train(
        generate_episodes, num_episodes=num_episodes, threshold_reward=100
    )

    return policy_gradient_agent


if __name__ == "__main__":
    policy_grad_agent = train_policy()

    episode_rewards = policy_grad_agent.episode_reward_history
    smoothed_rewards = np.convolve(episode_rewards, np.ones(20) / 20, mode="valid")

    episode_length = policy_grad_agent.episode_length
    episode_length = np.convolve(episode_length, np.ones(20) / 20, mode="valid")

    plt.plot(episode_rewards, alpha=0.3)
    plt.plot(
        range(len(smoothed_rewards)), smoothed_rewards, color="blue", linewidth=1.5
    )
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Policy Gradient Training", fontweight="bold")
    plt.savefig("./results/policy/policy_training.pdf")

    plt.clf()

    plt.plot(episode_length, color="blue")
    plt.xlabel("Episode")
    plt.ylabel("Length")
    plt.title("Episode Length - Policy", fontweight="bold")
    plt.savefig("./results/policy/episode_length.pdf")

    policy_grad_agent.model.save_weights("./results/policy/model.h5")
