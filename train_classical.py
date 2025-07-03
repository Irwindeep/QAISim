from collections import deque
import random
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from numpy.typing import NDArray
from typing import Callable, List, Tuple

from tqdm.auto import tqdm
from train_policy import generate_episodes
from train_q_val import episode_interaction, env

plt.style.use("seaborn-v0_8-whitegrid")
plt.rc("font", family="serif")


class MLP(tf.keras.Model):
    def __init__(
        self,
        num_actions: int,
        num_hidden_units: int = 64,
        raw_scores: bool = False,
        gamma: float = 0.99,
    ) -> None:
        super().__init__()
        self.raw_scores = raw_scores
        self.gamma = gamma

        self.linear = tf.keras.Sequential(
            layers=[
                tf.keras.layers.Dense(num_hidden_units, activation="relu"),
                tf.keras.layers.Dense(num_hidden_units, activation="relu"),
                tf.keras.layers.Dense(num_hidden_units, activation="relu"),
            ]
        )
        self.out_proj = tf.keras.layers.Dense(num_actions)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        if isinstance(inputs, list):
            inputs = inputs[0]
        output = self.linear(inputs)
        output = self.out_proj(output)
        if not self.raw_scores:
            output = tf.nn.softmax(output)
        return output

    def _compute_returns(
        self, episode_rewards: NDArray[np.float32]
    ) -> NDArray[np.float64]:
        returns: List[float] = []
        discounted_sum = 0.0
        for reward in episode_rewards[::-1]:
            discounted_sum = reward + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)

        np_returns = np.array(returns)
        np_returns = (np_returns - np.mean(returns)) / (np.std(returns) + 1e-8)

        return np_returns


@tf.function(reduce_retracing=True)
def reinforcement_update(
    model: MLP,
    states_np: NDArray[np.float32],
    actions_np: NDArray[np.int32],
    returns_np: NDArray[np.float32],
    batch_size: int,
    optim: tf.keras.optimizers.Optimizer,
) -> None:
    states = tf.convert_to_tensor(states_np)
    actions = tf.convert_to_tensor(actions_np)
    returns = tf.convert_to_tensor(returns_np)

    with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)
        logits = model(states)
        p_action = tf.gather_nd(logits, actions)
        log_probs = tf.math.log(p_action)
        loss = tf.math.reduce_sum(-log_probs * returns) / batch_size

    grads = tape.gradient(loss, model.trainable_variables)
    optim.apply_gradients(zip(grads, model.trainable_variables))


def train_policy(
    model: tf.keras.Model,
    optim: tf.keras.optimizers.Optimizer,
    generate_episodes: Callable,
    num_actions: int,
    num_episodes: int = 1500,
    batch_size: int = 10,
) -> List[float]:
    episode_reward_history = []
    pbar = tqdm(total=num_episodes // batch_size, colour="cyan")
    for batch in range(num_episodes // batch_size):
        pbar.set_description(f"Batch [{batch + 1}/{num_episodes // batch_size}]")
        episodes = generate_episodes(model, num_actions, batch_size, None)

        states = np.concatenate([ep["states"] for ep in episodes], dtype=np.float32)
        actions = np.concatenate([ep["actions"] for ep in episodes], dtype=np.int32)
        rewards = [ep["rewards"] for ep in episodes]
        returns = np.concatenate(
            [model._compute_returns(episode_rewards) for episode_rewards in rewards],
            dtype=np.float32,
        )

        id_action_pairs = np.array(
            [[i, a] for i, a in enumerate(actions)], dtype=np.int32
        )
        for episode_rewards in rewards:
            episode_reward_history.append(np.sum(episode_rewards))

        average_rewards = np.mean(episode_reward_history[-batch_size:])
        pbar.set_postfix({"Avg Reward": f"{average_rewards:.2f}"})

        reinforcement_update(model, states, id_action_pairs, returns, batch_size, optim)
        pbar.update(1)

    return episode_reward_history


@tf.function(reduce_retracing=True)
def q_learning_update(
    models: Tuple[tf.keras.Model, ...],
    num_actions: int,
    states_np: NDArray[np.float32],
    actions_np: NDArray[np.int32],
    rewards_np: NDArray[np.float32],
    next_states_np: NDArray[np.float32],
    done_np: NDArray[np.float32],
    optim: tf.keras.optimizers.Optimizer,
) -> None:
    model, model_target = models
    gamma = model.gamma

    states = tf.convert_to_tensor(states_np)
    actions = tf.convert_to_tensor(actions_np)
    rewards = tf.convert_to_tensor(rewards_np)
    next_states = tf.convert_to_tensor(next_states_np)
    done = tf.convert_to_tensor(done_np)

    future_rewards = model_target([next_states])
    target_q_values = rewards + (
        gamma * tf.reduce_max(future_rewards, axis=1) * (1.0 - done)
    )
    masks = tf.one_hot(actions, num_actions)

    with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)
        q_values = model([states])
        q_values_masked = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
        loss = tf.keras.losses.Huber()(target_q_values, q_values_masked)

    grads = tape.gradient(loss, model.trainable_variables)
    optim.apply_gradients(zip(grads, model.trainable_variables))


def train_q_val(
    models: Tuple[tf.keras.Model, ...],
    optim: tf.keras.optimizers.Optimizer,
    num_actions: int,
    env: gym.Env,
    episode_interaction: Callable,
    num_episodes: int,
    batch_size: int = 16,
    step_updates: Tuple[int, int] = (10, 30),
    epsilon: float = 1.0,
    epsilon_min: float = 0.01,
    decay_epsilon: float = 0.99,
) -> List[float]:
    model, model_target = models
    model.build(input_shape=(None, 8))
    model_target.build(input_shape=(None, 8))

    model_target.set_weights(model.get_weights())

    replay_memory = deque(maxlen=10000)
    step_count, episode_reward_history = 0, []
    pbar = tqdm(total=num_episodes, colour="cyan")

    for episode_count in range(num_episodes):
        pbar.set_description(f"Episode [{episode_count + 1}/{num_episodes}]")
        episode_reward = 0.0
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

            interaction = episode_interaction(model, num_actions, epsilon, state)[0]
            if not interaction["done"]:
                replay_memory.append(interaction)

            state = interaction["next_state"]
            episode_reward += interaction["reward"]
            step_count += 1

            if step_count % step_updates[0] == 0:
                if batch_size > len(replay_memory):
                    training_batch = list(replay_memory)
                else:
                    training_batch = random.sample(replay_memory, k=batch_size)
                q_learning_update(
                    (model, model_target),
                    num_actions,
                    np.asarray([x["state"] for x in training_batch], dtype=np.float32),
                    np.asarray([x["action"] for x in training_batch], dtype=np.int32),
                    np.asarray([x["reward"] for x in training_batch], dtype=np.float32),
                    np.asarray(
                        [
                            np.concatenate(
                                [
                                    x["next_state"]["qnode_queued_tasks"],
                                    x["next_state"]["qtask_arrival_time"],
                                    x["next_state"]["qtask_num_qubits"],
                                    x["next_state"]["qtask_circuit_layers"],
                                ]
                            )
                            for x in training_batch
                        ],
                        dtype=np.float32,
                    ),
                    np.asarray([x["done"] for x in training_batch], dtype=np.float32),
                    optim,
                )

            if step_count % step_updates[1] == 0:
                model_target.set_weights(model.get_weights())

            if interaction["done"]:
                break

        epsilon = max(epsilon * decay_epsilon, epsilon_min)
        episode_reward_history.append(episode_reward)

        average_rewards = np.mean(episode_reward_history[-batch_size:])

        pbar.set_postfix({"Avg Reward": f"{average_rewards:.2f}"})
        pbar.update(1)

    return episode_reward_history


def main():
    model = MLP(num_actions=5, raw_scores=False)
    optim = tf.keras.optimizers.Adam(learning_rate=1e-3)
    episode_reward_history = train_policy(
        model, optim, generate_episodes, num_actions=5
    )
    smoothed_rewards = np.convolve(
        episode_reward_history, np.ones(20) / 20, mode="valid"
    )

    plt.plot(episode_reward_history, alpha=0.3)
    plt.plot(
        range(len(smoothed_rewards)), smoothed_rewards, color="blue", linewidth=1.5
    )
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Classical Policy Gradient Training", fontweight="bold")
    plt.savefig("./results/policy/classical_training.pdf")

    model.save_weights("./results/policy/classical_model.h5")

    model = MLP(num_actions=5, raw_scores=True)
    model_target = MLP(num_actions=5, raw_scores=True)

    optim = tf.keras.optimizers.Adam(learning_rate=1e-3)
    episode_reward_history = train_q_val(
        models=(model, model_target),
        optim=optim,
        num_actions=5,
        env=env,
        episode_interaction=episode_interaction,
        num_episodes=1500,
    )

    smoothed_rewards = np.convolve(
        episode_reward_history, np.ones(20) / 20, mode="valid"
    )

    plt.plot(episode_reward_history, alpha=0.3)
    plt.plot(
        range(len(smoothed_rewards)), smoothed_rewards, color="blue", linewidth=1.5
    )
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Classical Q-Value Training", fontweight="bold")
    plt.savefig("./results/dq_learning/classical_training.pdf")

    model.save_weights("./results/dq_learning/classical_model.h5")


if __name__ == "__main__":
    main()
