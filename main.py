# %%
from geneticengine.grammar.metahandlers.ints import IntRange
from geneticengine.grammar import extract_grammar
from geneticengine.grammar.decorators import weight
from geneticengine.problems import SingleObjectiveProblem, MultiObjectiveProblem
from geneticengine.random.sources import NativeRandomSource
from geneticengine.algorithms.gp.gp import GeneticProgramming
from geneticengine.evaluation.budget import TimeBudget, EvaluationBudget
from geneticengine.representations.tree.initializations import MaxDepthDecider, FullDecider, ProgressivelyTerminalDecider, PositionIndependentGrowDecider
from geneticengine.representations.tree.operators import GrowInitializer, PositionIndependentGrowInitializer, FullInitializer, RampedHalfAndHalfInitializer
from geneticengine.algorithms.gp.operators.initializers import HalfAndHalfInitializer, StandardInitializer
from geneticengine.representations.tree.treebased import TreeBasedRepresentation
from geneticengine.representations.grammatical_evolution.structured_ge import StructuredGrammaticalEvolutionRepresentation
from geneticengine.evaluation.recorder import CSVSearchRecorder
from geneticengine.evaluation.tracker import ProgressTracker
from geneticengine.evaluation.parallel import ParallelEvaluator

from geneticengine.algorithms.gp.operators.combinators import ParallelStep, SequenceStep
from geneticengine.algorithms.gp.operators.crossover import GenericCrossoverStep
from geneticengine.algorithms.gp.operators.elitism import ElitismStep
from geneticengine.algorithms.gp.operators.mutation import GenericMutationStep
from geneticengine.algorithms.gp.operators.novelty import NoveltyStep
from geneticengine.algorithms.gp.operators.selection import LexicaseSelection, TournamentSelection

from geneticengine.solutions.individual import Individual, PhenotypicIndividual
from geneticengine.algorithms.gp.structure import GeneticStep
from geneticengine.problems import Problem
from geneticengine.random.sources import RandomSource
from geneticengine.representations.api import RepresentationWithCrossover, Representation
from geneticengine.evaluation import Evaluator
from geneticengine.grammar.decorators import abstract


from geneticengine.problems.helpers import non_dominated


import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import f1_score

from sklearn.feature_selection import RFE, RFECV

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Annotated, Iterator, Any

import numpy as np

from collections import OrderedDict

from scipy.stats import pearsonr
import copy

from random import random

import argparse
import os

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = (BASE_DIR).resolve()

print(DATA_DIR)
print(BASE_DIR)
print(Path(__file__))
print(Path(__file__).resolve())


class expr(ABC):
    """Base type for expression nodes."""

@dataclass
class Add(expr):
    left: expr
    right: expr

    def evaluate(self, X):
        return self.left.evaluate(X) + self.right.evaluate(X)

    def __str__(self):
        return f"({self.left} + {self.right})"


@dataclass
class Subtract(expr):
    left: expr
    right: expr

    def evaluate(self, X):
        return self.left.evaluate(X) - self.right.evaluate(X)

    def __str__(self):
        return f"({self.left} - {self.right})"


@dataclass
class Multiply(expr):
    left: expr
    right: expr

    def evaluate(self, X):
        return self.left.evaluate(X) * self.right.evaluate(X)

    def __str__(self):
        return f"({self.left} * {self.right})"


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

def preprocess_dataset(
        path: Path,
        filename: str,
        random_state: int | None,
        target_index: int,
        test_size: float = 0.3,
):
    df = pd.read_csv(f"{path}/{filename}")
    class_header = df.columns[target_index]

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

def run_experiment(
        filename: str,
        seed: int,
        target_index: int,
        num_individuals: int = 500,
        time_limit: int = 600,
        class_weight: str | None = None,
):
    cw = None if class_weight == "None" else "balanced"

    X_train, X_test, y_train, y_test = preprocess_dataset(
        DATA_DIR,
        filename,
        test_size=0.3,
        random_state=seed,
        target_index=target_index,
    )
    


    feature_names = list(X_train.columns)
    n_features = len(feature_names)

    @dataclass
    class Feature(expr):
        index: Annotated[int, IntRange(0, n_features - 1)]

        def evaluate(self, X):
            return X.iloc[:, self.index]

        def __str__(self):
            return feature_names[self.index]
    
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


    def custom_step():
        return SequenceStep(
            ParallelStep(
                [
                    ElitismStep(),
                    SequenceStep(
                        LexicaseSelection(epsilon=True),
                        GenericCrossoverStep(0.8),
                        GenericMutationStep(0.2),
                    ),
                ],
                weights=[0.005 , 0.995],
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

    solutions = alg.search()

    for sol in solutions:
        print(sol.get_phenotype())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--target-index", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-individuals", type=int, default=500)
    parser.add_argument("--time-limit", type=int, default=600)
    parser.add_argument("--class-weight", type=str, default="None")

    args = parser.parse_args()

    if args.class_weight.lower() == "none":
        args.class_weight = None

    run_experiment(
        filename=args.dataset,
        seed=args.seed,
        target_index=args.target_index,
        num_individuals=args.num_individuals,
        time_limit=args.time_limit,
        class_weight=args.class_weight,
    )
