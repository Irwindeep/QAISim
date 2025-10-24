import numpy as np
import tensorflow as tf
import keras

from cirq.ops.pauli_gates import Z
from qaisim.qrl.module import Module, EpisodeCallable
from qaisim.qrl.layers import ReUploading, Alternating

from functools import reduce
from tqdm import tqdm
from typing import List
from numpy.typing import NDArray


class PolicyGradient(Module):
    def __init__(
        self,
        parametrized_qc,
        num_actions,
        gamma=0.99,
        lrs=(0.01, 0.1, 0.1),
        beta: float = 1.0,
    ) -> None:
        super().__init__(parametrized_qc, num_actions, gamma, lrs)

        self.beta = beta
        operations = [Z(qubit) for qubit in self.qubits]
        self.observables = [reduce(lambda x, y: x * y, operations)]

        self.model = self._create_model_policy()

    def compute_returns(
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
        self,
        states_np: NDArray[np.float32],
        actions_np: NDArray[np.int32],
        returns_np: NDArray[np.float32],
        batch_size: int,
    ) -> None:
        if self.model is None:
            raise RuntimeError("Model not defined")

        states = tf.convert_to_tensor(states_np)
        actions = tf.convert_to_tensor(actions_np)
        returns = tf.convert_to_tensor(returns_np)

        with tf.GradientTape() as tape:
            tape.watch(self.model.trainable_variables)
            logits = self.model(states)
            p_action = tf.gather_nd(logits, actions)
            log_probs = tf.math.log(p_action)
            loss = tf.math.reduce_sum(-log_probs * returns) / batch_size

        grads = tape.gradient(loss, self.model.trainable_variables)
        for optimizer, w in zip(
            [self.optimizer_in, self.optimizer_var, self.optimizer_out],
            [self.w_in, self.w_var, self.w_out],
        ):
            optimizer.apply_gradients([(grads[w], self.model.trainable_variables[w])])

    def train(
        self,
        generate_episodes: EpisodeCallable,
        num_episodes: int,
        batch_size: int = 10,
        threshold_reward: float = 500.0,
    ) -> None:
        if not self.model:
            raise RuntimeError("Model not defined")
        with tqdm(total=num_episodes // batch_size, colour="cyan") as pbar:
            for batch in range(num_episodes // batch_size):
                pbar.set_description(
                    f"Batch [{batch + 1}/{num_episodes // batch_size}]"
                )
                episodes = generate_episodes(
                    self.model, self.num_actions, batch_size, None
                )

                states = np.concatenate(
                    [ep["states"] for ep in episodes], dtype=np.float32
                )
                actions = np.concatenate(
                    [ep["actions"] for ep in episodes], dtype=np.int32
                )
                rewards = [ep["rewards"] for ep in episodes]
                returns = np.concatenate(
                    [
                        self.compute_returns(episode_rewards)
                        for episode_rewards in rewards
                    ],
                    dtype=np.float32,
                )

                id_action_pairs = np.array(
                    [[i, a] for i, a in enumerate(actions)], dtype=np.int32
                )
                for episode_rewards in rewards:
                    self.episode_reward_history.append(np.sum(episode_rewards))
                    self.episode_length.append(len(episode_rewards))

                average_rewards = np.mean(self.episode_reward_history[-batch_size:])
                pbar.set_postfix({"Avg Reward": f"{average_rewards:.2f}"})
                pbar.update(1)

                if average_rewards >= threshold_reward:
                    break
                self.reinforcement_update(states, id_action_pairs, returns, batch_size)

    def eval(
        self,
        generate_episodes: EpisodeCallable,
        num_episodes: int,
        batch_size: int = 10,
    ) -> None:
        if not self.model:
            raise RuntimeError("Model not defined")
        with tqdm(total=num_episodes // batch_size, colour="cyan") as pbar:
            for batch in range(num_episodes // batch_size):
                pbar.set_description(
                    f"Batch [{batch + 1}/{num_episodes // batch_size}]"
                )
                episodes = generate_episodes(
                    self.model, self.num_actions, batch_size, None
                )

                rewards = [ep["rewards"] for ep in episodes]
                waiting_times = [ep["waiting_time"] for ep in episodes]

                for i, episode_rewards in enumerate(rewards):
                    self.eval_episode_reward_history.append(np.sum(episode_rewards))
                    self.eval_episode_length.append(len(episode_rewards))
                    self.eval_waiting_time.append(np.sum(waiting_times[i]))

                average_rewards = np.mean(
                    self.eval_episode_reward_history[-batch_size:]
                )
                pbar.set_postfix({"Avg Reward": f"{average_rewards:.2f}"})
                pbar.update(1)

    def _create_model_policy(self) -> keras.Model:
        if self.observables is None:
            raise RuntimeError("Observables not defined")

        input_tensor = keras.Input(
            shape=(self.num_qubits,), dtype=tf.dtypes.float32, name="input"
        )
        re_uploading = ReUploading(self.parametrized_qc, self.observables)
        re_uploading_output = re_uploading([input_tensor])

        process = keras.Sequential(
            [
                Alternating(self.num_actions),
                keras.layers.Lambda(lambda x: x * self.beta),
                keras.layers.Softmax(),
            ],
            name="observables-policy",
        )

        policy = process(re_uploading_output)
        return keras.Model(inputs=[input_tensor], outputs=policy)
