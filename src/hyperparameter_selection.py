import argparse
import pandas as pd


def compute_best_hparams(
    id_col: str = "identity",
    prompt_col: str = "prompt_sim",
    loc_col: str = "locality",
    guidance_col: str = "guidance",
    strength_col: str = "strength",
    steps_col: str = "steps",
    w_id: float = 0.08,
    w_prompt: float = 0.82,
    w_loc: float = 0.10,
):
    df = pd.read_csv("hyperparameters2.csv")

    cols_needed = [
        guidance_col, strength_col, steps_col,
        id_col, prompt_col, loc_col
    ]

    df = df[cols_needed].copy()

    df["combined_score"] = (
        w_id * df[id_col] +
        w_prompt * df[prompt_col] +
        w_loc * df[loc_col]
    )

    group_cols = [guidance_col, strength_col, steps_col]
    agg_df = (
        df
        .groupby(group_cols)
        .agg(
            identity_mean=(id_col, "mean"),
            prompt_similarity_mean=(prompt_col, "mean"),
            locality_mean=(loc_col, "mean"),
            combined_score_mean=("combined_score", "mean"),
            n_samples=("combined_score", "count"),
        )
        .reset_index()
    )

    best_idx = agg_df["combined_score_mean"].idxmax()
    best_row = agg_df.loc[best_idx]

    return agg_df, best_row


if __name__ == "__main__":
    agg_df, best_row = compute_best_hparams()

    print("Hyperparameter scores aggregated")
    print(
        agg_df.sort_values("combined_score_mean", ascending=False)
             .to_string(index=False)
    )

    print("\nBest hyperparameters")
    print(f"Guidance: {best_row["guidance"]}")
    print(f"Strength: {best_row["strength"]}")
    print(f"Steps:    {best_row["steps"]}")
    print(f"Mean identity:           {best_row['identity_mean']:.4f}")
    print(f"Mean prompt similarity:  {best_row['prompt_similarity_mean']:.4f}")
    print(f"Mean locality:           {best_row['locality_mean']:.4f}")
    print(f"Mean combined score:     {best_row['combined_score_mean']:.4f}")
    print(f"Number of samples:       {int(best_row['n_samples'])}")