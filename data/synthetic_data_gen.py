import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

df_original = pd.read_csv("qtasks_indep.csv")
np.random.seed(12)


def create_subset(
    df: pd.DataFrame, target_circuit_layers: int, tolerance: float, num_circuits: int
) -> pd.DataFrame:
    current_circuit_layers = 0
    subset = df.loc[:]

    while not (
        current_circuit_layers >= target_circuit_layers - tolerance
        and current_circuit_layers <= target_circuit_layers + tolerance
    ):
        subset_indices = np.random.choice(
            list(df.index), size=min(num_circuits, len(df.index)), replace=False
        )
        subset = df.loc[subset_indices]
        current_circuit_layers = subset["original_circuit_layers"].mean()

    return subset


def create_subsets_data(num_subsets: int, num_circuits: int) -> pd.DataFrame:
    avg_circuit_layers = df_original["original_circuit_layers"].mean()
    tolerance = avg_circuit_layers * 0.1

    all_subsets = []
    for i in range(num_subsets):
        subset = create_subset(df_original, avg_circuit_layers, tolerance, num_circuits)  # pyright: ignore
        subset["subset"] = i + 1
        all_subsets.append(subset)

    df = pd.concat(all_subsets)
    cols = ["subset"] + [col for col in df.columns if col != "subset"]

    return df[cols]  # pyright: ignore


# Expanding the dataset by random mixing and shuffling for better (Q)RL environment
def expand_subsets(
    df: pd.DataFrame, num_existing_subsets: int, num_new_subsets: int
) -> pd.DataFrame:
    grouped = df.groupby("subset")

    expanded_data = []
    for i in range(1, num_new_subsets + 1):
        selected_subsets = np.random.choice(
            np.arange(1, num_existing_subsets + 1), size=2, replace=False
        )
        combined = pd.concat(
            [
                grouped.get_group(subset)
                .sample(frac=1, random_state=12)
                .reset_index(drop=True)
                for subset in selected_subsets
            ],
            ignore_index=True,
        )
        mixed_subset = combined.sample(frac=1).reset_index(drop=True)

        mixed_subset["subset"] = i + num_existing_subsets
        expanded_data.append(mixed_subset)

    df_expanded = pd.concat(expanded_data, ignore_index=True)
    df_final = pd.concat([df, df_expanded], ignore_index=True)

    df_final["subset"] = (df_final.index // 26) + 1
    return df_final


if __name__ == "__main__":
    num_subsets, num_circuits = 100, 60
    df_subsets = create_subsets_data(num_subsets, num_circuits)

    num_new_subsets = 900
    df_final = expand_subsets(df_subsets, num_subsets, num_new_subsets)

    df_train, df_test = train_test_split(
        df_final, test_size=0.2, stratify=df_final["subset"], random_state=12
    )
    df_train = df_train.sort_values(by="subset")  # pyright: ignore
    df_test = df_test.sort_values(by="subset")  # pyright: ignore

    df_train.to_csv("qtasks_train.csv", index=False)
    df_test.to_csv("qtasks_test.csv", index=False)
