import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from train_policy import train_policy
import numpy as np
import matplotlib.pyplot as plt

def train_policy_with_tune(config):
    reward_history = train_policy(config)
    avg_reward = np.mean(reward_history[-100:])
    
    tune.report(average_reward=avg_reward)

def main():
    ray.init(ignore_reinit_error=True, log_to_driver=True)

    search_space = {
        "num_layers": tune.choice([3, 5, 7, 9]),
        "lr1": tune.loguniform(0.001, 0.1),
        "lr2": tune.loguniform(0.001, 0.1),
        "lr3": tune.loguniform(0.001, 0.1),
        "num_episodes": tune.choice([500, 1000, 1500, 2000])
    }

    scheduler = ASHAScheduler(
        metric="average_reward",
        mode="max",
        max_t=2000,
        grace_period=200,
        reduction_factor=2
    )

    analysis = tune.run(
        train_policy_with_tune,
        config=search_space,
        scheduler=scheduler,
        num_samples=20,
        resources_per_trial={"cpu": 2}
    )

    print("Best Config:", analysis.best_config)

    df = analysis.results_df
    plt.plot(df['average_reward'])
    plt.title("Average Reward vs. Trials")
    plt.xlabel("Trials")
    plt.ylabel("Average Reward")
    plt.savefig("./results/policy_tuning/tuning_results.png")

    df.to_csv("./results/policy_tuning/analysis.csv")

if __name__ == "__main__":
    main()
