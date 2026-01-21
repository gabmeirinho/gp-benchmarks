import argparse
import time
import warnings
from pathlib import Path

from abc import ABC
from dataclasses import dataclass
from typing import Annotated, Iterator, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder

from geneticengine.algorithms.gp.gp import GeneticProgramming
from geneticengine.algorithms.gp.operators.combinators import ParallelStep, SequenceStep
from geneticengine.algorithms.gp.operators.crossover import GenericCrossoverStep
from geneticengine.algorithms.gp.operators.elitism import ElitismStep
from geneticengine.algorithms.gp.operators.mutation import GenericMutationStep
from geneticengine.algorithms.gp.operators.selection import LexicaseSelection
from geneticengine.algorithms.gp.structure import GeneticStep
from geneticengine.evaluation import Evaluator
from geneticengine.evaluation.budget import TimeBudget
from geneticengine.evaluation.tracker import ProgressTracker
from geneticengine.grammar import extract_grammar
from geneticengine.grammar.metahandlers.ints import IntRange
from geneticengine.problems import MultiObjectiveProblem, Problem
from geneticengine.random.sources import NativeRandomSource, RandomSource
from geneticengine.representations.tree.initializations import MaxDepthDecider
from geneticengine.representations.tree.treebased import TreeBasedRepresentation
from geneticengine.solutions.individual import PhenotypicIndividual
from geneticengine.problems.helpers import non_dominated
from geneticengine.grammar.decorators import abstract
from geneticengine.grammar.decorators import weight

from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score

from random import random

import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV, f_classif

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = (BASE_DIR / ".." ).resolve()
OUT_DIR = BASE_DIR/""

DEFAULT_TARGET = None

def preprocess_dataset(
    path: Path,
    filename: str,
    test_size: float,
    random_state: int | None,
    target_name: str | None,
    target_index: int | None,
):
    """Load a dataset, encode categorical columns, and return a train/test split."""
    data_path = Path(filename)
    if not data_path.is_absolute():
        if data_path.exists():
            data_path = data_path.resolve()
        elif (path / data_path).exists():
            data_path = (path / data_path).resolve()
        elif (path / data_path.name).exists():
            data_path = (path / data_path.name).resolve()
        else:
            raise FileNotFoundError(f"Could not find dataset at '{filename}' or under '{path}'.")
    df = pd.read_csv(data_path)

    class_header = None
    if target_index is not None:
        try:
            class_header = df.columns[target_index]
        except IndexError:
            print(
                f"Target index {target_index} out of range for {filename}; falling back to name/last column."
            )

    if class_header is None and target_name and target_name in df.columns:
        class_header = target_name

    if class_header is None:
        class_header = df.columns[-1]
        if target_name:
            print(
                f"Target '{target_name}' not found in {filename}; using '{class_header}' instead."
            )
        else:
            print(f"No target specified; using last column '{class_header}' for {filename}.")

    X = df.drop(columns=[class_header])
    y = df[class_header]


    if y.dtype == "object" or isinstance(y.dtype, pd.CategoricalDtype):
        print(f"Encoding class labels in column '{class_header}'")
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), name=class_header)
        
    print(f"Target classes: {y.unique()}")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns

    if len(categorical_cols) > 0:
        X_train = X_train.copy()
        X_test = X_test.copy()
        for col in categorical_cols:
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col])
            X_test[col] = le.transform(X_test[col])

    return X_train, X_test, y_train, y_test


def run_dataset(
    filename: str,
    seed: int,
    target_index: int | None = -1,
    target_name: str | None = DEFAULT_TARGET,
    num_individuals: int = 500,
    time_limit: int = 120,
    class_weight: str | None = None,
    grammar_option: int = 0,
) -> None:

    start_time = time.time()

    cw = None if class_weight == "None" else "balanced"

    X_train, X_test, y_train, y_test = preprocess_dataset(
        DATA_DIR,
        filename,
        test_size=0.3,
        random_state=seed,
        target_name=target_name,
        target_index=target_index,
    )

    feature_names = list(X_train.columns)
    n_features = len(feature_names)

    op_prob = 0.1 if grammar_option == 1 else 1.0
    feature_prob = 0.35 if grammar_option == 1 else 1.0
    agg_prob = 0.15 if grammar_option == 1 else 1.0

    class expr(ABC):
        """Base type for expression nodes."""

    @weight(op_prob)
    @dataclass
    class Add(expr):
        left: expr
        right: expr

        def evaluate(self, X):
            return self.left.evaluate(X) + self.right.evaluate(X)

        def __str__(self):
            return f"({self.left} + {self.right})"

    @weight(op_prob)
    @dataclass
    class Subtract(expr):
        left: expr
        right: expr

        def evaluate(self, X):
            return self.left.evaluate(X) - self.right.evaluate(X)

        def __str__(self):
            return f"({self.left} - {self.right})"

    @weight(op_prob)
    @dataclass
    class Multiply(expr):
        left: expr
        right: expr

        def evaluate(self, X):
            return self.left.evaluate(X) * self.right.evaluate(X)

        def __str__(self):
            return f"({self.left} * {self.right})"

    @weight(op_prob)
    @dataclass
    class Divide(expr):
        left: expr
        right: expr

        def evaluate(self, X):
            denom = self.right.evaluate(X)
            denom = denom.replace(0, 1e-6)
            return self.left.evaluate(X) / denom

        def __str__(self):
            return f"({self.left} / {self.right})"

    @weight(feature_prob)
    @dataclass
    class Feature(expr):
        index: Annotated[int, IntRange(0, n_features - 1)]

        def evaluate(self, X):
            return X.iloc[:, self.index]

        def __str__(self):
            return feature_names[self.index]
        
    @weight(agg_prob)
    @abstract
    @dataclass
    class Aggregation(expr, ABC):
        index: Annotated[int, IntRange(0, n_features - 1)]
        lookback: Annotated[int, IntRange(10, 20)]

        def rolling(self, X: pd.DataFrame):
            series = X.iloc[:, self.index]
            return series.rolling(window=self.lookback, min_periods=1)        
    
    @dataclass
    class Mean(Aggregation):
        def evaluate(self, X):
            return self.rolling(X).mean()

        def __str__(self):
            return f"Mean({feature_names[self.index]}, {self.lookback})"
    
    @dataclass
    class Max(Aggregation):
        def evaluate(self, X):
            return self.rolling(X).max()
        
        def __str__(self):
            return f"Max({feature_names[self.index]}, {self.lookback})"
        

    def count_features(node: expr):
        if isinstance(node, Feature):
            return {node.index}
        if isinstance(node, (Add, Subtract, Multiply, Divide)):
            return count_features(node.left) | count_features(node.right)
        if isinstance(node, Aggregation):
            return {node.index}
        return set()
    
    def count_nodes(node: expr) -> int:
        """Count total nodes in the expression tree (operators + terminals)."""
        if isinstance(node, Feature):
            return 1
        if isinstance(node, Aggregation):
            return 1
        if isinstance(node, (Add, Subtract, Multiply, Divide)):
            return 1 + count_nodes(node.left) + count_nodes(node.right)
        return 0
        

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

    def fitness_function(individual):
        if str(individual) in X_train.columns:
            return [0.0, 9999, 9999]
        new_feature = individual.evaluate(X_train)
        new_feature = pd.Series(
            np.clip(new_feature, -1e6, 1e6), index=X_train.index, name=str(individual)
        )
        X_augmented = pd.concat([X_train, new_feature], axis=1)

        scores = []
        for train_index, val_index in skf.split(X_augmented, y_train):
            model = RandomForestClassifier(
                max_depth=6, random_state=seed, class_weight=cw
            )
            model.fit(X_augmented.iloc[train_index], y_train.iloc[train_index])
            pred = model.predict(X_augmented.iloc[val_index])
            scores.append(f1_score(y_train.iloc[val_index], pred, average="weighted"))

        avg_score = float(np.mean(scores))
        unique_features = len(count_features(individual))
        n_nodes = count_nodes(individual)
        return [avg_score, unique_features, n_nodes]

    grammar = extract_grammar([Add, Subtract, Multiply, Divide, Aggregation, Max, Mean, Feature], expr)
    print(f"Grammar: {repr(grammar)}")

    BEST_HISTORY: dict[int, dict[Any, Any]] = {}
    SEEN_INDS = set()

    class ArchiveStep(GeneticStep):
        def iterate(
            self,
            problem: Problem,
            evaluator: Evaluator,
            representation,
            random: RandomSource,
            population: Iterator[PhenotypicIndividual],
            target_size: int,
            generation: int,
        ) -> Iterator[PhenotypicIndividual]:
            individuals = list(population)
            candidates = evaluator.evaluate(problem, iter(individuals))
            best : Iterator[PhenotypicIndividual[Any, Any]] = non_dominated(iter(candidates), problem)
            
            best_list = list(best)
            for ind in best_list:
                str_ind = str(ind)
                if str_ind not in SEEN_INDS:
                    SEEN_INDS.add(str_ind)
                    BEST_HISTORY[generation] = {"ind": ind, "Weighted F1-Score": ind.get_fitness(problem).fitness_components[0], "Unique Features": ind.get_fitness(problem).fitness_components[1], "Nodes": ind.get_fitness(problem).fitness_components[2]}
                    print(f"Gen {generation}: {str(ind)} -> {ind.get_fitness(problem).fitness_components[0]}")
            else:
                print(f"Gen {generation}: No new best individual found.")
            return iter(individuals)
             

    def custom_step():
        return SequenceStep(
            ArchiveStep(),
            ParallelStep(
                [
                    ElitismStep(),
                    SequenceStep(
                        LexicaseSelection(epsilon=True),
                        GenericCrossoverStep(0.7),
                        GenericMutationStep(0.2),
                    ),
                ],
                weights=[0.005, 0.995],
            ),
        )

    r = NativeRandomSource(seed)
    prob = MultiObjectiveProblem(
        fitness_function=fitness_function,
        minimize=[False, True, True],
    )

    alg = GeneticProgramming(
        problem=prob,
        population_size=num_individuals,
        budget=TimeBudget(time_limit),
        representation=TreeBasedRepresentation(grammar, MaxDepthDecider(r, grammar, 5)),
        random=r,
        step=custom_step(),
        tracker=ProgressTracker(prob),
    )

    alg.search()
    print("Search complete.")

    print(f"Archive Size: {len(BEST_HISTORY)}")

    new_features_train = []
    new_features_test = []

    for k, v in BEST_HISTORY.items():
        ind = v["ind"]
        str_ind = str(ind)

        train_vals = ind.get_phenotype().evaluate(X_train)
        s_train = pd.Series(np.clip(train_vals, -1e6, 1e6), index=X_train.index, name=str_ind)
        new_features_train.append(s_train)
        
        test_vals = ind.get_phenotype().evaluate(X_test)
        s_test = pd.Series(np.clip(test_vals, -1e6, 1e6), index=X_test.index, name=str_ind)
        new_features_test.append(s_test)

    if new_features_train:
        X_train_aug = pd.concat([X_train] + new_features_train, axis=1)
        X_test_aug = pd.concat([X_test] + new_features_test, axis=1)
    else:
        X_train_aug = X_train.copy()
        X_test_aug = X_test.copy()
    
    augmented_set_size = X_train_aug.shape[1]

    runs = 30
    runs_scores_aug = []

    for run in range(runs):
        model = RandomForestClassifier(max_depth=6, random_state=seed + run, class_weight=cw)
        model.fit(X_train_aug, y_train)
        pred = model.predict(X_test_aug)
        test_f1 = f1_score(y_test, pred, average="weighted")
        runs_scores_aug.append(test_f1)

    avg_test_f1_aug = np.mean(runs_scores_aug)
    print(f"Augmented Dataset Preliminary Test Weighted F1 over {runs} runs: {avg_test_f1_aug}")

    corr = X_train_aug.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

    print(f"Dropping {len(to_drop)}")
    
    X_train_aug.drop(columns=to_drop, inplace=True)
    X_test_aug.drop(columns=to_drop, inplace=True)

    augmented_set_size_after_corr = X_train_aug.shape[1]

    runs = 30
    runs_scores_aug_corr = []

    for run in range(runs):
        model = RandomForestClassifier(max_depth=6, random_state=seed + run, class_weight=cw)
        model.fit(X_train_aug, y_train)
        pred = model.predict(X_test_aug)
        test_f1 = f1_score(y_test, pred, average="weighted")
        runs_scores_aug_corr.append(test_f1)

    avg_test_f1_aug_corr = np.mean(runs_scores_aug_corr)
    print(f"Augmented Dataset after Correlation Drop Preliminary Test Weighted F1 over {runs} runs: {avg_test_f1_aug_corr}")

    model = RandomForestClassifier(max_depth=6, random_state=seed, class_weight=cw)

    thresholds = np.arange(0.2, 0.55, 0.05)
    scores = {}
    for thr in thresholds:
        rfe = RFE(estimator=model, n_features_to_select=thr, step=1)
        rfe.fit(X_train_aug, y_train)
        selected_features = X_train_aug.columns[rfe.support_]
        score = cross_val_score(model, X_train_aug[selected_features], y_train, cv=10, scoring='f1_weighted').mean()
        scores[thr] = score
        print(f"Threshold: {thr}, Selected Features: {len(selected_features)}, CV F1 Score: {score}")

    best_thr = max(scores, key=lambda x: scores[x])
    rfe = RFE(estimator=model, n_features_to_select=best_thr, step=1)
    rfe.fit(X_train_aug, y_train)
    selected_features = X_train_aug.columns[rfe.support_]

    runs = 30
    runs_scores = []

    for run in range(runs):
        model = RandomForestClassifier(max_depth=6, random_state=seed + run, class_weight=cw)
        model.fit(X_train_aug[selected_features], y_train)
        pred = model.predict(X_test_aug[selected_features])
        test_f1 = f1_score(y_test, pred, average="weighted")
        runs_scores.append(test_f1)

    avg_test_f1 = np.mean(runs_scores)
    print(f"Final Test Weighted F1 over {runs} runs: {avg_test_f1:.4f}")

    final_f1 = avg_test_f1

    selected_gp_nodes = 0
    for k, v in BEST_HISTORY.items():
        ind = v["ind"]
        str_ind = str(ind)
        if str_ind in selected_features:
            selected_gp_nodes += count_nodes(ind.get_phenotype())

    results = [
        {
            "seed": seed,
            "weighted_f1": final_f1,
            "num_features": len(selected_features),
            "num_nodes": selected_gp_nodes,
            "features": list(selected_features),
            "elapsed_seconds": time.time() - start_time,
            "archive_size": len(BEST_HISTORY),
            "augmented_set_size": augmented_set_size,
            "augmented_set_size_after_corr": augmented_set_size_after_corr,
            "augmented_cv_f1": avg_test_f1_aug,
            "augmented_corr_cv_f1": avg_test_f1_aug_corr,
        }
    ]

    folder_suffix = "_grammar_results" if grammar_option == 1 else "_results"
    folder_name = f"{Path(filename).stem}{folder_suffix}"

    out_dir = OUT_DIR / folder_name
    out_name = out_dir / f"{seed}.csv"

    out_dir.mkdir(exist_ok=True, parents=True)
    
    pd.DataFrame(results).to_csv(out_name, index=False)

    print(
        f"{filename} | seed {seed} | weighted_f1={final_f1:.4f} | num_features={len(selected_features)} | saved {out_name.name} | elapsed {time.time() - start_time:.2f}s"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Genetic Engine feature construction on one dataset/seed (SLURM-friendly)."
    )
    parser.add_argument(
        "--dataset",
        required=True,
    )
    parser.add_argument(
        "--seed",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--target-index",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--target-name",
        default=DEFAULT_TARGET,
    )
    parser.add_argument(
        "--num-individuals",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=120,
    )
    parser.add_argument(
        "--class-weight",
        type=str,
        default="None",
    )
    parser.add_argument(
        "--grammar",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    if args.class_weight.lower() == "none":
        args.class_weight = None

    run_dataset(
        args.dataset,
        seed=args.seed,
        target_index=args.target_index,
        target_name=args.target_name,
        num_individuals=args.num_individuals,
        time_limit=args.time_limit,
        class_weight=args.class_weight,
        grammar_option = args.grammar,
    )
