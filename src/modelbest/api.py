import itertools
import logging
import os
import pprint
import sys
import time
from abc import ABC, abstractmethod
from collections import Counter
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Dict, Set, Tuple, Union, Optional, List, cast, final
import numpy as np
import pandas as pd
from pandas import concat
from sklearn.exceptions import NotFittedError
from loguru import logger
from sklearn.base import BaseEstimator

from src.modelbest._automl import H2O, AML, AutoGluon
from src.modelbest.domain import Dataset, Task
from src.modelbest.repository import DatasetRepository, ImbalancedDatasetRepository, OpenMLDatasetRepository
from src.modelbest._helpers import infer_positive_target_class, train_test_split


class Modeller:
    """
    User interface for performing automated modelling.

    This class presents an unified API to run different AutoML scenarios.

    Benchmarking datasets can be utilized by specifying a repository parameter.
    
    Parameters
    ----------
    backend: str, default ag
        Name of the AutoML tool to run a benchmark.
        Supported values: autogluon and h2o.
    metric: str, default f1
        Name of the metric to validate performance of ML models during training. 
        Also used to test performance of the leader model.
        Supported values: f1, f1_macro, f1_weighted, precision, recall, roc_auc, average_precision, balanced_accuracy,
        mcc and accuracy.
    random_state: int, default 42
        Value, used for controlling randomness during model training.
    timeout: int, optional
        Time budget in seconds of AutoML training on a single dataset.
    extra_metrics: list of int, optional
        Names of additional metrics used only to test the leader model on.
    verbosity: int, default 1
        Level of logging verbosity.
    *args
        Additional arguments should be passed as keyword arguments.
    **kwargs
        Arguments specific to the chosen AutoML tool.
    """

    def __init__(
        self,
        backend = 'autogluon',
        metric = 'f1',
        random_state = 42,
        timeout: Optional[int] = None,
        extra_metrics: Optional[List[str]] = None,
        verbosity: int = 2,
        **kwargs
    ):
        self._aml: AML
        self._validation_metric: str
        self._seed: int
        self._timeout: Optional[int]
        self._verbosity: int
        # TODO: create a common class for fitted models.
        self._fitted_model = None
        self._test_metrics: List[str] = []

        self.verbosity = verbosity
        self.aml = (backend, kwargs)
        self.validation_metric = metric
        self.seed = random_state
        self.timeout = timeout
        self.test_metrics = extra_metrics

        self._configure_logging()

    def _configure_logging(self) -> None:
        logger.remove()
        if self.verbosity == 3:
            logger.add(sys.stdout, level='TRACE')
        elif self.verbosity == 2:
            logger.add(sys.stdout, level='DEBUG')
        elif self.verbosity == 1:
            logger.add(sys.stdout, level='INFO')
        elif self.verbosity == 0:
            logger.add(sys.stdout, level='SUCCESS')

    @logger.catch(reraise=True)
    def run(self, dataset: Dataset) -> None:
        try:
            logger.info(f"{dataset.id=}, {dataset.name=}.")
            
            if dataset.y is None:
                y_label = dataset.X.columns[-1]
                y = dataset.X[y_label]
                x = dataset.X.drop([y_label], axis=1)
            else:
                x = dataset.X
                y = dataset.y
                y_label = y.name
            
            x_train, x_test, y_train, y_test = train_test_split(x, y)
            
            y_train = y_train.astype(object)
            if y_test.dtype == 'category':
                y_test = y_test.cat.codes

            class_belongings = Counter(y_train)
            class_belongings_formatted = '; '.join(f"{k}: {v}" for k, v in class_belongings.items())
            logger.debug(f"Class belongings: {{{class_belongings_formatted}}}")

            pos_class_label = 1
            if len(class_belongings) == 2:
                pos_class_label = infer_positive_target_class(class_belongings)

            if dataset.y is not None:
                training_dataset = Dataset(
                    id=dataset.id,
                    name=dataset.name,
                    X=x_train,
                    y=y_train,
                )
            else:
                df = pd.concat((x_train, y_train),axis=1)
                training_dataset = Dataset(
                    id=dataset.id,
                    name=dataset.name,
                    X=df
                )

            training_dataset.size = int(x_train.memory_usage(deep=True).sum() / (1024 ** 2))
            logger.debug(f"Train sample size(floored) is {training_dataset.size}mb.")

            validation_metric = self.validation_metric
            if len(class_belongings) > 2 and str(self.aml) == 'AutoGluon':
                validation_metric += '_weighted' 

            task  = Task(
                dataset=training_dataset,
                metric=validation_metric,
                timeout=self.timeout,
                seed=self.seed,
                verbosity=self.verbosity
            )

            start_time = time.time()
            self.aml.fit(task)

            time_passed = time.time() - start_time
            logger.info(f"Training took {time_passed // 60} min.")

            y_predicted = self.aml.predict(x_test)

            if str(self.aml) == 'H2O':
                validation_metric += '_weighted'
            logger.debug(f"Test metrics are {self.test_metrics}")
            
            self.aml.score(self.test_metrics, y_test, y_predicted, pos_class_label)

        finally:
            if self.aml == 'H2O':
                import h2o
                cluster = h2o.cluster()
                if cluster is not None:
                    cluster.shutdown()
    
    @property
    def validation_metric(self) -> str:
        return self._validation_metric
    
    @validation_metric.setter
    def validation_metric(self, value: str):
        if value not in [
            'f1',
            'f1_macro',
            'f1_weighted',
            'precision',
            'recall',
            'roc_auc',
            'average_precision',
            'balanced_accuracy',
            'mcc',
            'accuracy'
        ]:
            raise ValueError(
                f"""
                Invalid value of metric parameter: {value}.
                Options available: [
                    'f1',
                    'f1_macro',
                    'f1_weighted',
                    'precision',
                    'recall',
                    'roc_auc',
                    'average_precision',
                    'balanced_accuracy',
                    'mcc',
                    'accuracy'].
                """)
        self._validation_metric = value
    
    @property
    def aml(self) -> AML:
        return self._aml

    @aml.setter
    def aml(self, value: Tuple[str, Dict[str, Any]]):
        if value[0] == 'autogluon':
            self._aml = AutoGluon(**value[1])
        elif value[0] == 'h2o':
            self._aml = H2O(**value[1])
        else:
            raise ValueError(
                f"""
                Invalid value of automl parameter: {value[0]}.
                Options available: ['autogluon', 'h2o'].
                """)
    
    @property
    def seed(self) -> int:
        return self._seed
    
    @seed.setter
    def seed(self, value: int):
        self._seed = value

    @property
    def timeout(self) -> Optional[int]:
        return self._timeout
    
    @timeout.setter
    def timeout(self, value: Optional[int]):
        self._timeout = value

    @property
    def verbosity(self) -> int:
        return self._verbosity
    
    @verbosity.setter
    def verbosity(self, value: int):
        if value > 3 or value < 0:
            raise ValueError("Verbosity should be in a range (0,4).")
        else:
            self._verbosity = value

    @property
    def test_metrics(self) -> List[str]:
        return self._test_metrics

    @test_metrics.setter
    def test_metrics(self, metrics: Optional[List[str]]):
        self.test_metrics.insert(0, self.validation_metric)
        
        if metrics is not None:
            test_metrics = set()
            for metric in metrics:
                if metric not in [
                    'f1',
                    'f1_macro',
                    'f1_weighted',
                    'precision',
                    'recall',
                    'roc_auc',
                    'average_precision',
                    'balanced_accuracy',
                    'mcc',
                    'accuracy'
                ]:
                    raise ValueError(
                        f"""
                        Invalid value of the extra_metrics parameter: {metric}.
                        Options available: [
                            'f1',
                            'f1_macro',
                            'f1_weighted',
                            'precision',
                            'recall',
                            'roc_auc',
                            'average_precision',
                            'balanced_accuracy',
                            'mcc',
                            'accuracy'].
                        """)
                test_metrics.add(metric)
            self.test_metrics.extend(test_metrics)    
