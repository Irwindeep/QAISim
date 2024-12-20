from qpysim.utils import Dataset
from qpysim.qrl import QRLEnv, PolicyGradient, ParametrizedQC
from collections import defaultdict
import numpy as np
import tensorflow as tf

dataset = Dataset(file_name="./data/qtasks_train.csv")
state_bounds = [30]*5 + [60, 50, 200000]

def generate_episodes(model, n_actions, n_episodes, x=None):
    episodes = [defaultdict(list) for _ in range(n_episodes)]
    envs = [QRLEnv(dataset) for _ in range(n_episodes)]
    done = [False for _ in range(n_episodes)]
    states = [e.reset() for e in envs]
    states = [_state[0] for _state in states]

    while not all(done):
        states = [np.concatenate([
            state["qnode_queued_tasks"],
            state["qtask_arrival_time"],
            state["qtask_num_qubits"],
            state["qtask_circuit_layers"]
        ]) for state in states]

        unfinished_ids = [i for i in range(n_episodes) if not done[i]]
        normalized_states = [s/state_bounds for i, s in enumerate(states) if not done[i]]

        for i, state in zip(unfinished_ids, normalized_states):
            episodes[i]['states'].append(state)

        states = tf.convert_to_tensor(normalized_states)
        action_probs = model([states])

        states = [None for _ in range(n_episodes)]
        for i, policy in zip(unfinished_ids, action_probs.numpy()):
            action = np.random.choice(n_actions, p=policy)
            states[i], reward, done[i], _, _ = envs[i].step(action)
            episodes[i]['actions'].append(action)
            episodes[i]['rewards'].append(reward)

    return episodes

def train_policy(config):
    num_qubits = 8
    num_layers = config["num_layers"]
    lrs = (config["lr1"], config["lr2"], config["lr3"])
    num_actions = 5
    num_episodes = config["num_episodes"]

    paramatrized_qc = ParametrizedQC(num_qubits, num_layers)
    policy_gradient_agent = PolicyGradient(paramatrized_qc, num_actions, lrs=lrs)

    policy_gradient_agent.train(
        generate_episodes,
        num_episodes=num_episodes,
        threshold_reward=100000
    )

    return policy_gradient_agent.episode_reward_history
