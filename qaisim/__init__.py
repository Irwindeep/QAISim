import numpy as np
import tensorflow as tf

from qaisim.broker import Broker
from qaisim.utils import Dataset
from qaisim.qnode import QNode, QNodeParams
from qaisim.qtask import QTask
from qaisim import qrl

from qaisim.qrl.module import EpisodeCallable, EpisodicOutcome
from qaisim.qrl.env_qnodes import ibm_qnodes, qnodes_for_noisy_exp
from qaisim.qrl.qrl_env import QRLEnv

from collections import defaultdict
from numpy.typing import NDArray
from typing import Tuple


def get_episode_gen(dataset_file: str, **kwargs) -> EpisodeCallable:
    num_actions = 5 if kwargs["backend"] == "pure" else 2

    dataset = Dataset(file_name=dataset_file)
    STATE_BOUNDS = [30] * num_actions + [60, 50, 200000]
    env_qnodes = ibm_qnodes if kwargs["backend"] == "pure" else qnodes_for_noisy_exp

    def generate_episodes(
        model: tf.keras.Model,
        n_actions: int,
        n_episodes: int | float,
        s: NDArray[np.float32] | None = None,
    ) -> EpisodicOutcome:
        if not isinstance(n_episodes, int):
            raise RuntimeError(
                f"Expected n_episodes to be of type int, recieved {type(n_episodes)}"
            )

        episodes: EpisodicOutcome = [defaultdict(list) for _ in range(n_episodes)]

        # load environments for each episode
        envs = [QRLEnv(dataset, env_qnodes) for _ in range(n_episodes)]
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
                s / STATE_BOUNDS for i, s in enumerate(states) if not done[i]
            ]

            for i, state in zip(unfinished_ids, normalized_states):
                episodes[i]["states"].append(state)

            states = tf.convert_to_tensor(normalized_states)
            action_probs = model([states])

            states = [None for _ in range(n_episodes)]
            for i, policy in zip(unfinished_ids, action_probs.numpy()):
                action = np.random.choice(n_actions, p=policy)
                states[i], reward, done[i], _, x = envs[i].step(action)
                episodes[i]["actions"].append(action)
                episodes[i]["rewards"].append(reward)
                episodes[i]["waiting_time"].append(x["waiting_time"])

        return episodes

    return generate_episodes


def get_episode_interaction(
    dataset_file: str,
    **kwargs,
) -> Tuple[EpisodeCallable, QRLEnv]:
    env_qnodes = ibm_qnodes if kwargs["backend"] == "pure" else qnodes_for_noisy_exp

    dataset = Dataset(file_name=dataset_file)
    env = QRLEnv(dataset, env_qnodes)

    def episode_interaction(
        model: tf.keras.Model,
        n_actions: int,
        epsilon: int | float,
        state: NDArray[np.float32] | None,
    ) -> EpisodicOutcome:
        if state is None:
            raise RuntimeError("Expected state to be a numpy array, recieved None")

        state_array = state
        state_tensor = tf.convert_to_tensor([state])

        coin = np.random.random()
        if coin > epsilon:
            q_vals = model([state_tensor])
            action = int(tf.argmax(q_vals[0]).numpy())
        else:
            action = np.random.choice(n_actions)

        next_state, reward, done, _, x = env.step(action)
        interaction = {
            "state": state_array,
            "action": action,
            "next_state": next_state.copy(),
            "reward": reward,
            "done": np.float32(done),
            "waiting_time": x["waiting_time"],
        }

        return [interaction]

    return episode_interaction, env


__all__ = [
    "Broker",
    "Dataset",
    "get_episode_gen",
    "get_episode_interaction",
    "QNode",
    "QNodeParams",
    "qrl",
    "QTask",
]
