import argparse
import json
import pandas as pd
import numpy as np
import os
from src import utils, data, training, evaluation, config

DL_MODELS = {"mlp", "resnet", "ftt"}
TABNET_MODELS = {"tabnet"}
TREE_MODELS = {"xgboost", "lgbm", "catboost"}
ALL_MODELS = DL_MODELS | TABNET_MODELS | TREE_MODELS

# Fixer HPO-Seed — HPO wird genau einmal pro Modell × Dataset ausgeführt.
# Multi-Seed-Evaluation misst Varianz durch Initialisierung/Training, nicht HPO.
HPO_SEED = 42


def run_hpo(args, splits):
    """HPO einmal ausführen. Cached als JSON für Session-Neustarts."""
    cache_path = os.path.join(
        config.RESULTS_DIR, f"hpo_best_{args.model}_{args.dataset}.json"
    )
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            best = json.load(f)
        print(f"HPO-Cache geladen: {cache_path}")
        print(f"  Params: {best}")
        return best

    utils.seed_everything(HPO_SEED)
    print(f"\n{'='*60}")
    print(f"HPO: {args.model.upper()} x {args.dataset} ({args.trials} Trials)")
    print(f"{'='*60}")

    if args.model in DL_MODELS:
        best = training.run_pytorch_hpo(args.model, splits, args.trials, HPO_SEED)
    elif args.model in TABNET_MODELS:
        best = training.run_tabnet_hpo(splits, args.trials, HPO_SEED)
    elif args.model in TREE_MODELS:
        best = training.run_tree_hpo(args.model, splits, args.trials, HPO_SEED)
    else:
        raise ValueError(f"Unbekanntes Modell: {args.model}")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(best, f, indent=2)
    print(f"  HPO gespeichert: {cache_path}")
    return best


def run_single_seed(args, splits, best_params, seed):
    """Finales Training + Eval mit fixen Hyperparametern für einen Seed."""
    utils.seed_everything(seed)
    print(f"\n{'='*60}")
    print(f"--- {args.model.upper()} | {args.dataset} | Seed {seed} ---")
    print(f"{'='*60}")

    if args.model in DL_MODELS:
        model, tracker = training.run_pytorch_final(
            args.model, splits, best_params, args.epochs, args.dataset, seed
        )
    elif args.model in TABNET_MODELS:
        model, tracker = training.run_tabnet_final(
            splits, best_params, args.epochs, args.dataset, seed
        )
    elif args.model in TREE_MODELS:
        model, tracker = training.run_tree_final(
            args.model, splits, best_params, args.dataset, seed
        )
    else:
        raise ValueError(f"Unbekanntes Modell: {args.model}")

    res = evaluation.evaluate_model(
        model, splits, tracker, args.model, args.dataset, seed=seed
    )
    utils.save_results(res, args.dataset, args.model, seed=seed)
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=sorted(ALL_MODELS))
    parser.add_argument("--dataset", type=str, default="mlg_ulb")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[42, 123, 456, 789, 1024])
    args = parser.parse_args()

    splits = data.prepare_data(args.dataset, args.model)

    # Phase 1: HPO (einmalig)
    best_params = run_hpo(args, splits)

    # Phase 2: Finales Training (pro Seed)
    all_summaries = []
    for seed in args.seeds:
        res = run_single_seed(args, splits, best_params, seed)
        row = {"seed": seed}
        row.update(res["classification_metrics"])
        all_summaries.append(row)

    # Aggregation
    df = pd.DataFrame(all_summaries)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop("seed")
    agg = df[numeric_cols].agg(["mean", "std"])
    agg_path = os.path.join(config.RESULTS_DIR,
                            f"aggregate_{args.model}_{args.dataset}.csv")
    agg.to_csv(agg_path, sep=";", decimal=",", encoding="utf-8-sig")

    print(f"\n{'='*60}")
    print("AGGREGIERT (mean +/- std):")
    for col in agg.columns:
        print(f"  {col}: {agg.loc['mean', col]:.4f} +/- {agg.loc['std', col]:.4f}")
    print(f"Gespeichert: {agg_path}")
    print("Fertig.")


if __name__ == "__main__":
    main()
