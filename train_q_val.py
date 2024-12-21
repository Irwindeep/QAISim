import numpy as np
from qpysim.utils import Dataset
from qpysim.qrl import QRLEnv, ParametrizedQC, DeepQLearning
import tensorflow as tf
import matplotlib.pyplot as plt

dataset = Dataset(file_name="./data/qtasks_train.csv")
env = QRLEnv(dataset)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rc('font', family='serif')

def episode_interaction(model, n_actions, epsilon, state):
    state_array = state
    state = tf.convert_to_tensor([state])

    coin = np.random.random()
    if coin > epsilon:
        q_vals = model([state])
        action = int(tf.argmax(q_vals[0]).numpy())
    else:
        action = np.random.choice(n_actions)

    next_state, reward, done, _, _ = env.step(action)
    interaction = {'state': state_array, 'action': action, 'next_state': next_state.copy(),
                'reward': reward, 'done':np.float32(done)}
    
    return [interaction]

def train_q_val():
    num_qubits = 8
    num_layers = 5
    lrs = (0.03, 0.05, 0.03)
    num_actions = 5
    num_episodes = 1500

    paramatrized_qc = ParametrizedQC(num_qubits, num_layers)
    dql_agent = DeepQLearning(paramatrized_qc, num_actions, lrs=lrs)

    dql_agent.train(
        env=env,
        generate_episode=episode_interaction,
        num_episodes=num_episodes,
        threshold_reward=100
    )

    return dql_agent

if __name__=="__main__":
    dql_agent = train_q_val()

    episode_rewards = dql_agent.episode_reward_history
    smoothed_rewards = np.convolve(
        episode_rewards, np.ones(20)/20, mode="valid"
    )

    episode_length = dql_agent.episode_length
    episode_length = np.convolve(
        episode_length, np.ones(20)/20, mode="valid"
    )

    plt.plot(episode_rewards, alpha=0.3)
    plt.plot(range(len(smoothed_rewards)), smoothed_rewards, color="blue", linewidth=1.5)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Q-Value Training", fontweight="bold")
    plt.savefig("./results/dq_learning/q_val_training.pdf")

    plt.clf()

    plt.plot(episode_length, color="blue")
    plt.xlabel("Episode")
    plt.ylabel("Length")
    plt.title("Episode Length - DeepQLearning", fontweight="bold")
    plt.savefig("./results/dq_learning/episode_length.pdf")

    dql_agent.model.save_weights("./results/dq_learning/model.h5")
