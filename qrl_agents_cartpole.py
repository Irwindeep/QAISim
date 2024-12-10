from qpysim.qrl import (
    ParametrizedQC,
    PolicyGradient,
    DeepQLearning
)
from collections import defaultdict
import gym
import tensorflow as tf
import numpy as np

env_name, state_bounds = "CartPole-v1", [2.4, 2.5, 0.21, 2.5]

def generate_episodes_pg(model, n_actions, n_episodes, x=None):
    trajectories = [defaultdict(list) for _ in range(n_episodes)]
    envs = [gym.make(env_name) for _ in range(n_episodes)]

    done = [False for _ in range(n_episodes)]
    states = [e.reset() for e in envs]

    while not all(done):
        unfinished_ids = [i for i in range(n_episodes) if not done[i]]
        normalized_states = [s/state_bounds for i, s in enumerate(states) if not done[i]]

        for i, state in zip(unfinished_ids, normalized_states):
            trajectories[i]['states'].append(state)

        states = tf.convert_to_tensor(normalized_states)
        action_probs = model([states])

        states = [None for i in range(n_episodes)]
        for i, policy in zip(unfinished_ids, action_probs.numpy()):
            action = np.random.choice(n_actions, p=policy)
            states[i], reward, done[i], _ = envs[i].step(action)
            trajectories[i]['actions'].append(action)
            trajectories[i]['rewards'].append(reward)

    return trajectories

env = gym.make(env_name)
def generate_episode_dql(model, n_actions, epsilon, state):
    state_array = np.array(state)
    state = tf.convert_to_tensor([state_array])

    coin = np.random.random()
    if coin > epsilon:
        q_vals = model([state])
        action = int(tf.argmax(q_vals[0]).numpy())
    else:
        action = np.random.choice(n_actions)

    next_state, reward, done, _ = env.step(action)
    interaction = {'state': state_array, 'action': action, 'next_state': next_state.copy(),
                'reward': reward, 'done':np.float32(done)}
    
    return [interaction]

if __name__=="__main__":
    num_qubits = 4
    num_layers = 2
    num_actions = 2

    paramatrized_qc = ParametrizedQC(num_qubits, num_layers)

    policy_gradient_agent = PolicyGradient(paramatrized_qc, num_actions)
    policy_gradient_agent.train(generate_episodes_pg, num_episodes=1000)

    dql_agent = DeepQLearning(paramatrized_qc, num_actions=num_actions)
    dql_agent.train(env, generate_episode_dql, num_episodes=2000)
