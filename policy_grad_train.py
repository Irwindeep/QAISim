from qpysim.utils import Dataset
from qpysim.qrl import (
    QRLEnv,
    PolicyGradient,
    ParametrizedQC
)
from collections import defaultdict
import numpy as np
import tensorflow as tf

dataset = Dataset(file_name="./data/qtasks_train.csv")
state_bounds = [10000000, 156, 2000000, 250000]

def generate_episodes(model, n_actions, n_episodes, x=None):
    episodes = [defaultdict(list) for _ in range(n_episodes)]
    envs = [QRLEnv(qtasks_dataset=dataset) for _ in range(n_episodes)]
    done = [False for _ in range(n_episodes)]
    states = [e.reset() for e in envs]
    states = [_state[0] for _state in states]

    while not all(done):
        states = [np.concatenate([
            state["qtask_arrival_time"],
            state["qtask_num_qubits"],
            state["qtask_circuit_layers"],
            state["qtask_gate_counts"]
        ]) for state in states]
        
        unfinished_ids = [i for i in range(n_episodes) if not done[i]]
        normalized_states = [s/state_bounds for i, s in enumerate(states) if not done[i]]

        for i, state in zip(unfinished_ids, normalized_states):
            episodes[i]['states'].append(state)

        states = tf.convert_to_tensor(normalized_states)
        action_probs = model([states])

        states = [None for i in range(n_episodes)]
        for i, policy in zip(unfinished_ids, action_probs.numpy()):
            action = np.random.choice(n_actions, p=policy)
            states[i], reward, done[i], _ ,_ = envs[i].step(action)
            episodes[i]['actions'].append(action)
            episodes[i]['rewards'].append(reward)

    return episodes

if __name__=="__main__":
    num_qubits = 4
    num_layers = 5
    num_actions = 11

    paramatrized_qc = ParametrizedQC(num_qubits, num_layers)

    policy_gradient_agent = PolicyGradient(paramatrized_qc, num_actions)
    policy_gradient_agent.train(generate_episodes, num_episodes=1000, threshold_reward=500.0)
