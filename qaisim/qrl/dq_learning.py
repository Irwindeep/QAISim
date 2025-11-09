import numpy as np
import tensorflow as tf
import keras
import gymnasium as gym
import random

from cirq.ops.pauli_gates import Z
from qaisim.qrl.module import Module, EpisodeCallable
from qaisim.qrl.layers import ReUploading, Rescaling

from collections import deque
from tqdm.auto import tqdm
from functools import reduce
from operator import mul
from typing import Tuple, Deque, Dict, Any
from numpy.typing import NDArray

random.seed(12)
ReplayMem = Deque[Dict[str, Any]]


class DeepQLearning(Module):
    def __init__(
        self,
        parametrized_qc,
        num_actions,
        gamma=0.99,
        lrs=(0.001, 0.001, 0.1),
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        decay_epsilon: float = 0.99,
        step_updates: Tuple[int, int] = (10, 30),
    ):
        super().__init__(parametrized_qc, num_actions, gamma, lrs)

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.decay_epsilon = decay_epsilon

        self.max_memory_length = 10000
        self.replay_memory: ReplayMem = deque(maxlen=self.max_memory_length)

        self.step_updates = step_updates
        operations = [Z(qubit) for qubit in self.qubits]
        self.observables = [
            reduce(mul, operations[i : i + 2]) for i in range(0, len(operations), 2)
        ]
        for i in range(self.num_actions - len(self.observables)):
            self.observables.append(
                mul(operations[i], operations[len(operations) - i - 1])
            )

        assert len(self.observables) == num_actions

        self.model = self._create_model_policy(target=False)
        self.model_target = self._create_model_policy(target=True)
        self.model_target.set_weights(self.model.get_weights())

    @tf.function(reduce_retracing=True)
    def q_learning_update(
        self,
        states_np: NDArray[np.float32],
        actions_np: NDArray[np.int32],
        rewards_np: NDArray[np.float32],
        next_states_np: NDArray[np.float32],
        done_np: NDArray[np.float32],
    ) -> None:
        if self.model is None:
            raise RuntimeError("Model not defined")

        states = tf.convert_to_tensor(states_np)
        actions = tf.convert_to_tensor(actions_np)
        rewards = tf.convert_to_tensor(rewards_np)
        next_states = tf.convert_to_tensor(next_states_np)
        done = tf.convert_to_tensor(done_np)

        future_rewards = self.model_target([next_states])
        target_q_values = rewards + (
            self.gamma * tf.reduce_max(future_rewards, axis=1) * (1.0 - done)
        )
        masks = tf.one_hot(actions, self.num_actions)

        with tf.GradientTape() as tape:
            tape.watch(self.model.trainable_variables)
            q_values = self.model([states])
            q_values_masked = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
            loss = keras.losses.Huber()(target_q_values, q_values_masked)

        grads = tape.gradient(loss, self.model.trainable_variables)
        for optimizer, w in zip(
            [self.optimizer_in, self.optimizer_var, self.optimizer_out],
            [self.w_in, self.w_var, self.w_out],
        ):
            optimizer.apply_gradients([(grads[w], self.model.trainable_variables[w])])

    def train(
        self,
        env: gym.Env,
        generate_episode: EpisodeCallable,
        num_episodes: int,
        batch_size: int = 16,
        threshold_reward: float = 500.0,
    ) -> None:
        if self.model is None:
            raise RuntimeError("Model not defined")

        step_count = 0
        pbar = tqdm(total=num_episodes)
        for episode_count in range(num_episodes):
            pbar.set_description(f"Episode [{episode_count + 1}/{num_episodes}]")
            episode_reward, episode_length = 0.0, 0
            state = env.reset()[0]

            while True:
                state = self._create_state(state)
                interaction = generate_episode(
                    self.model, self.num_actions, self.epsilon, state
                )[0]
                if not interaction["done"]:
                    self.replay_memory.append(interaction)

                state = interaction["next_state"]
                episode_reward += interaction["reward"]
                step_count += 1
                episode_length += 1

                if step_count % self.step_updates[0] == 0:
                    if batch_size > len(self.replay_memory):
                        batch = list(self.replay_memory)
                    else:
                        batch = random.sample(self.replay_memory, k=batch_size)

                    states = np.array([x["state"] for x in batch], dtype=np.float32)
                    actions = np.array([x["action"] for x in batch], dtype=np.int32)
                    rewards = np.array([x["reward"] for x in batch], dtype=np.float32)
                    next_states = np.array(
                        [self._create_state(x["next_state"]) for x in batch]
                    )
                    done = np.array([x["done"] for x in batch], dtype=np.float32)
                    self.q_learning_update(states, actions, rewards, next_states, done)

                if step_count % self.step_updates[1] == 0:
                    self.model_target.set_weights(self.model.get_weights())

                if interaction["done"]:
                    break

            self.epsilon = max(self.epsilon * self.decay_epsilon, self.epsilon_min)
            self.episode_reward_history.append(episode_reward)
            self.episode_length.append(episode_length)

            average_rewards = np.mean(self.episode_reward_history[-batch_size:])

            pbar.set_postfix({"Avg Reward": f"{average_rewards:.2f}"})
            pbar.update(1)

            if average_rewards >= threshold_reward:
                break

        pbar.close()

    def eval(
        self,
        env: gym.Env,
        generate_episode: EpisodeCallable,
        num_episodes: int,
        batch_size: int = 16,
    ) -> None:
        if not self.model:
            raise RuntimeError("Model not defined")

        pbar = tqdm(total=num_episodes)
        for episode_count in range(num_episodes):
            pbar.set_description(f"Episode [{episode_count + 1}/{num_episodes}]")
            episode_reward, episode_wt, episode_length = 0.0, 0.0, 0
            state = env.reset()[0]

            while True:
                state = self._create_state(state)
                interaction = generate_episode(
                    self.model, self.num_actions, self.epsilon, state
                )[0]

                state = interaction["next_state"]
                episode_reward += interaction["reward"]
                episode_length += 1
                episode_wt += interaction["waiting_time"]

                if interaction["done"]:
                    break

            self.eval_episode_reward_history.append(episode_reward)
            self.eval_episode_length.append(episode_length)
            self.eval_waiting_time.append(episode_wt)

            average_rewards = np.mean(self.eval_episode_reward_history[-batch_size:])

            pbar.set_postfix({"Avg Reward": f"{average_rewards:.2f}"})
            pbar.update(1)

        pbar.close()

    def _create_model_policy(self, target: bool) -> keras.Model:
        if self.observables is None:
            raise RuntimeError("Observables not defined")

        input_tensor = keras.Input(
            shape=(self.num_qubits,), dtype=tf.dtypes.float32, name="input"
        )
        re_uploading = ReUploading(
            self.parametrized_qc, self.observables, activation="tanh"
        )
        re_uploading_output = re_uploading([input_tensor])

        process = keras.Sequential(
            [Rescaling(len(self.observables))], name=target * "Target" + "Q-values"
        )

        q_values = process(re_uploading_output)
        return keras.Model(inputs=[input_tensor], outputs=q_values)
