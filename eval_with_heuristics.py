import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from qpysim.qrl import (
    ParametrizedQC,
    PolicyGradient,
    DeepQLearning
)
from train_policy import generate_episodes
from train_q_val import episode_interaction, env
from tqdm import tqdm

plt.style.use('seaborn-v0_8-whitegrid')
plt.rc('font', family='serif')

class GreedyAgent(tf.keras.Model):
    def __init__(self, num_actions):
        super(GreedyAgent, self).__init__()
        self.num_actions = num_actions

    def call(self, inputs):
        inputs = inputs[0]
        greedy_actions = tf.argmin(inputs[:, :self.num_actions], axis=1)
        action_probs = tf.one_hot(greedy_actions, depth=self.num_actions, dtype=tf.float32)
        return action_probs
    
class RandomAgent(tf.keras.Model):
    def __init__(self, num_actions):
        super(RandomAgent, self).__init__()
        self.num_actions = num_actions

    def call(self, inputs):
        batch_size = tf.shape(inputs[0])[0]
        random_probs = tf.random.uniform(shape=(batch_size, self.num_actions), minval=0.0, maxval=1.0, dtype=tf.float32)
        
        action_probs = random_probs / tf.reduce_sum(random_probs, axis=1, keepdims=True)
        return action_probs

class HeuristicAgent:
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.greedy_model = GreedyAgent(num_actions)
        self.random_model = RandomAgent(num_actions)

        self.episode_reward_history = {"greedy": [], "random": []}
        self.episode_length = {"greedy": [], "random": []}

    def greedy(self, num_episodes, batch_size=10):
        self._eval(self.greedy_model, "greedy", num_episodes, batch_size)

    def random(self, num_episodes, batch_size=10):
        self._eval(self.random_model, "random", num_episodes, batch_size)

    def _eval(self, model, model_name, num_episodes, batch_size=10):
        with tqdm(total=num_episodes // batch_size, colour="cyan") as pbar:
            for batch in range(num_episodes // batch_size):
                pbar.set_description(f"Batch [{batch + 1}/{num_episodes // batch_size}]")
                episodes = generate_episodes(model, self.num_actions, batch_size, None)

                rewards = [ep['rewards'] for ep in episodes]

                for episode_rewards in rewards:
                    self.episode_reward_history[model_name].append(np.sum(episode_rewards))
                    self.episode_length[model_name].append(len(episode_rewards))

                average_rewards = np.mean(self.episode_reward_history[model_name][-batch_size:])
                pbar.set_postfix({'Avg Reward': f"{average_rewards:.2f}"})
                pbar.update(1)

def evaluate_heuristic():
    heuristic_agent = HeuristicAgent(5)
    
    heuristic_agent.greedy(num_episodes=100)
    heuristic_agent.random(num_episodes=100)

    return heuristic_agent

def evaluate_policy():
    parametrized_qc = ParametrizedQC(8, 5)
    policy_grad_agent = PolicyGradient(parametrized_qc, 5, lrs=(0.03, 0.05, 0.03))
    policy_grad_agent.model.load_weights("./results/policy/model.h5")

    policy_grad_agent.eval(generate_episodes, 100)
    return policy_grad_agent

def evaluate_q_val():
    parametrized_qc = ParametrizedQC(8, 5)
    dql_agent = DeepQLearning(parametrized_qc, 5, lrs=(0.03, 0.05, 0.03))
    dql_agent.model.load_weights("./results/dq_learning/model.h5")
    dql_agent.epsilon = 0.01

    dql_agent.eval(env, episode_interaction, 100)
    return dql_agent

if __name__=="__main__":
    heuristic_agent = evaluate_heuristic()
    policy_grad_agent = evaluate_policy()
    dql_agent = evaluate_q_val()

    greedy_rewards = heuristic_agent.episode_reward_history["greedy"]
    greedy_rewards = np.convolve(
        greedy_rewards, np.ones(20)/20, mode="valid"
    )

    random_rewards = heuristic_agent.episode_reward_history["random"]
    random_rewards = np.convolve(
        random_rewards, np.ones(20)/20, mode="valid"
    )

    policy_rewards = policy_grad_agent.eval_episode_reward_history
    policy_rewards = np.convolve(
        policy_rewards, np.ones(20)/20, mode="valid"
    )

    dql_rewards = dql_agent.eval_episode_reward_history
    dql_rewards = np.convolve(
        dql_rewards, np.ones(20)/20, mode="valid"
    )

    plt.plot(policy_rewards, color="blue", label="Policy Rewards")
    plt.plot(greedy_rewards, color="red", label="Greedy Rewards")
    plt.plot(random_rewards, color="green", label="Random Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.title("Policy Gradient Evaluation", fontweight="bold")
    plt.savefig("./results/policy/policy_eval.pdf")

    plt.clf()

    plt.plot(dql_rewards, color="blue", label="Q-Value Rewards")
    plt.plot(greedy_rewards, color="red", label="Greedy Rewards")
    plt.plot(random_rewards, color="green", label="Random Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.title("Q-Value Evaluation", fontweight="bold")
    plt.savefig("./results/dq_learning/dql_eval.pdf")
