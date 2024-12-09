from typing import List, Tuple, Callable
from qpysim.qrl.parametrized_qc import ParametrizedQC
from qpysim.qrl.layers import (
    ReUploading,
    Alternating
)
import cirq
import numpy as np
from numpy.typing import NDArray
import tensorflow as tf # type: ignore[import-untyped]
from functools import reduce
from tqdm import tqdm

class PolicyGradient:
    def __init__(
        self,
        parametrized_qc: ParametrizedQC,
        num_actions: int,
        beta: float = 10.0,
        gamma: float = 0.99,
        lrs: Tuple[float, float, float] = (0.01, 0.1, 0.1)
    ) -> None:
        self.parametrized_qc = parametrized_qc
        self.quantum_circuit = parametrized_qc.quantum_circuit
        self.qubits = parametrized_qc.qubits
        self.num_qubits = parametrized_qc.num_qubits
        self.num_layers = parametrized_qc.num_layers

        self.num_actions = num_actions
        self.beta = beta
        self.gamma = gamma

        operations = [cirq.Z(qubit) for qubit in self.qubits]
        self.observables = [reduce(lambda x, y: x*y, operations)]

        self.model = self._create_model_policy()

        self.optimizer_in = tf.keras.optimizers.Adam(learning_rate=lrs[0], amsgrad=True)
        self.optimizer_var = tf.keras.optimizers.Adam(learning_rate=lrs[1], amsgrad=True)
        self.optimizer_out = tf.keras.optimizers.Adam(learning_rate=lrs[2], amsgrad=True)

        self.w_in, self.w_var, self.w_out = 1, 0, 2

        self.episode_reward_history: List[float] = []

    def compute_returns(self, episode_rewards: List[float]) -> NDArray[np.float64]:
        returns: List[float] = []
        discounted_sum = 0.0
        for reward in episode_rewards[::-1]:
            discounted_sum = reward + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)

        np_returns = np.array(returns)
        np_returns = (np_returns - np.mean(returns))/(np.std(returns) + 1e-8)

        return np_returns
    
    @tf.function
    def reinforcement_update(
        self,
        states: NDArray[np.float64],
        actions: NDArray[np.int64],
        returns: NDArray[np.int64],
        batch_size: int
    ) -> None:
        states = tf.convert_to_tensor(states)
        actions = tf.convert_to_tensor(actions)
        returns = tf.convert_to_tensor(returns)

        with tf.GradientTape() as tape:
            tape.watch(self.model.trainable_variables)
            
            logits = self.model(states)
            p_action = tf.gather_nd(logits, actions)
            log_probs = tf.math.log(p_action)

            loss = tf.math.reduce_sum(-log_probs * returns)/batch_size

        grads = tape.gradient(loss, self.model.trainable_variables)
        for optimizer, w in zip(
            [self.optimizer_in, self.optimizer_var, self.optimizer_out],
            [self.w_in, self.w_var, self.w_out]
        ):
            optimizer.apply_gradients([(grads[w], self.model.trainable_variables[w])])

    def reinforce(
        self,
        generate_episodes: Callable[[tf.keras.Model, int, int], dict],
        num_episodes: int, batch_size: int = 10,
        threshold_reward: float = 500.0
    ) -> None:
        self.episode_reward_history = []
        with tqdm(total=num_episodes // batch_size, colour="cyan") as pbar:
            for batch in range(num_episodes // batch_size):
                pbar.set_description(f"Batch [{batch + 1}/{num_episodes // batch_size}]")

                episodes = generate_episodes(self.model, self.num_actions, batch_size)

                states = np.concatenate([ep['states'] for ep in episodes])
                actions = np.concatenate([ep['actions'] for ep in episodes])
                rewards = [ep['rewards'] for ep in episodes]
                returns = np.concatenate(
                    [self.compute_returns(episode_rewards) for episode_rewards in rewards]
                )

                id_action_pairs = np.array([[i, a] for i, a in enumerate(actions)])

                for episode_rewards in rewards:
                    self.episode_reward_history.append(np.sum(episode_rewards))

                average_rewards = np.mean(self.episode_reward_history[-batch_size:])

                pbar.set_postfix({'Avg Reward': f"{average_rewards:.2f}"})
                pbar.update(1)  # Increment the progress bar by 1 batch

                if average_rewards >= threshold_reward:
                    break

                self.reinforcement_update(states, id_action_pairs, returns, batch_size)

    def _create_model_policy(self) -> tf.keras.Model:
        input_tensor = tf.keras.Input(shape=(self.num_qubits, ), name="input")
        re_uploading = ReUploading(self.parametrized_qc, self.observables)
        re_uploading_output = re_uploading([input_tensor])

        process = tf.keras.Sequential(
            [
                Alternating(self.num_actions),
                tf.keras.layers.Lambda(lambda x: x*self.beta),
                tf.keras.layers.Softmax()
            ],
            name="observables-policy"
        )

        policy = process(re_uploading_output)
        return tf.keras.Model(inputs=[input_tensor], outputs=policy)
