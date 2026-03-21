import logging
import os
import pprint
import re
from abc import ABC, abstractmethod
from io import StringIO
import sys
from typing import Any, Optional, Set, Union, final, List, Dict
import h2o
from h2o.automl import H2OAutoML
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.metrics import confusion_matrix, fbeta_score, balanced_accuracy_score, matthews_corrcoef, recall_score, precision_score, average_precision_score, roc_auc_score, accuracy_score
from autogluon.tabular import TabularDataset as AutoGluonTabularDataset, TabularPredictor as AutoGluonTabularPredictor
from autogluon.core.metrics import make_scorer
from loguru import logger
import jdk
import os

from src.modelbest.domain import Dataset, Task


class AML(ABC):
    def __init__(self):
        self._fitted_model = None
    
    @abstractmethod
    def fit(
        self,
        task: Task,
    ) -> None:
        raise NotImplementedError()

    def predict(self, x_test: pd.DataFrame) -> Union[pd.Series, np.ndarray]:
        if self._fitted_model is None:
            raise NotFittedError()
        predictions = self._fitted_model.predict(x_test)
        
        return predictions

    @final
    def score(
        self,
        metrics: Set[str],
        y_test: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray],
        pos_label: Union[int, str] = 1,
    ) -> None:
        for metric in metrics:
            if metric.startswith('f1'):
                if metric == 'f1':
                    average = 'binary'
                elif metric == 'f1_weighted':
                    average = 'weighted'
                elif metric == 'f1_macro':
                    average = 'macro'
                else:
                    raise ValueError("Invalid average for f1-measure.")
                score = fbeta_score(y_test, y_pred, beta=1, pos_label=pos_label, average=average)
                logger.success(f"F1{'_' + average} score: {score:.3f}")
            elif metric == 'precision':
                score = precision_score(y_test, y_pred, pos_label=pos_label)
                logger.success(f"Precision score: {score:.3f}")
            elif metric == 'recall':
                score = recall_score(y_test, y_pred, pos_label=pos_label)
                logger.success(f"Recall score: {score:.3f}")
            elif metric == 'roc_auc':
                score = roc_auc_score(y_test, y_pred)
                logger.success(f"ROC AUC score: {score:.3f}")
            elif metric == 'balanced_accuracy':
                score = balanced_accuracy_score(y_test, y_pred, adjusted=True)
                logger.success(f"Balanced accuracy score: {score:.3f}")
            elif metric == 'average_precision':
                score = average_precision_score(y_test, y_pred, pos_label=pos_label)
                logger.success(f"Average precision score: {score:.3f}")
            elif metric == 'mcc':
                score = matthews_corrcoef(y_test, y_pred)
                logger.success(f"MCC score: {score:.3f}")
            elif metric == 'accuracy':
                score = accuracy_score(y_test, y_pred)
                logger.success(f"Balanced accuracy score: {score:.3f}")
            else:
                raise ValueError(
                    f"""
                    Invalid value encountered among values of test_metrics parameter:{metric}.
                    Metrics available: [
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

    # TODO: refactor.
    @final
    def _log_val_loss_alongside_fitted_model(self, losses: Dict[str, np.float64]) -> None:
        for m, l in losses.items():
            # TODO: different output for leaderboard.
            logger.info(f"Validation loss: {abs(l):.3f}")

            model_log = pprint.pformat(f"Model: {m}", compact=True)
            logger.info(model_log)

    def __str__(self):
        return self.__class__.__name__


class AutoGluon(AML):
    def __init__(
        self,
        preset='medium',
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._preset = preset
        self._fitted_model: Optional[AutoGluonTabularPredictor] = None

    @logger.catch
    def fit(
        self,
        task: Task,
    ) -> None:
        if task.metric not in [
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
            raise ValueError(f"Metric {task.metric} is not supported by AutoGluon.")

        ag_dataset = AutoGluonTabularDataset(task.dataset.X)

        predictor = AutoGluonTabularPredictor(
            label=task.dataset.X.columns[-1],
            eval_metric=task.metric,
            verbosity=task.verbosity,
            learner_kwargs={"random_state": task.seed}
        )

        predictor_kwargs: Dict[str, Any] = {'presets': self.preset}
        timeout = task.timeout
        if timeout is not None:
            timeout = float(timeout)
            predictor_kwargs['time_limit'] = timeout
        predictor.fit(ag_dataset, **predictor_kwargs)

        val_scores = predictor.leaderboard().get('score_val')
        if val_scores is None or len(val_scores) == 0:
            logger.error("No model found.")
            return

        best_model = predictor.model_best

        logger.info(f"Best model found: {best_model}.")

        predictor.delete_models(models_to_keep=best_model, dry_run=False)

        self._fitted_model = predictor
    
    @logger.catch
    def predict(self, x_test: pd.DataFrame) -> Union[pd.Series, np.ndarray]:
        if self._fitted_model is None:
            raise NotFittedError()
        dataset_test = AutoGluonTabularDataset(x_test)
        predictions = self._fitted_model.predict(dataset_test).astype(int)

        return predictions

    @property
    def preset(self):
        return self._preset

    @preset.setter
    def preset(self, preset):
        if preset not in ['medium', 'good', 'high', 'best', 'extreme']:
            raise ValueError(
                f"""
                Invalid value of preset parameter: {preset}.
                Options available: [
                    'medium',
                    'good',
                    'high',
                    'best',
                    'extreme'
                ].
                """)
        self._preset = preset


class H2O(AML):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._fitted_model = None

        # Datasphere-specific logic.
        if os.path.exists('/job/'):
            jdk.install('17')
            os.environ['JAVA_HOME'] = '/job/.jdk/jdk-17.0.17+10'
        
        h2o.init()
    
    @logger.catch
    def fit(
        self,
        task: Task,
    ) -> None:
        if task.metric not in [
            'f1',
            'precision',
            'recall',
            'roc_auc',
            'average_precision',
            'balanced_accuracy',
            'mcc',
            'accuracy'
        ]:
            raise ValueError(f"Metric {task.metric} is not supported by H2O.")
        
        x_dtypes = task.dataset.X.dtypes
        logger.debug(x_dtypes)

        self._df_dtypes = x_dtypes\
            .mask(x_dtypes == object, 'categorical')\
            .mask(x_dtypes == 'category', 'categorical')\
            .mask(x_dtypes == np.uint8, 'int')\
            .mask(x_dtypes == np.float64, 'double')\
            .mask(x_dtypes == bool, 'int')\
            .to_list()
        h2o_dataset = h2o.H2OFrame(task.dataset.X, column_types=self._df_dtypes)

        verbosity = 'warn'
        match task.verbosity:
            case 0:
                verbosity = 'error'
            case 1:
                verbosity = 'warn'
            case 2:
                verbosity = 'info'
            case 3:
                verbosity = 'debug'
            
        predictor = H2OAutoML(max_runtime_secs=task.timeout, seed=task.seed, verbosity=verbosity)
        predictor.train(x=list(task.dataset.X.columns[:-1]), y=str(task.dataset.X.columns[-1]), training_frame=h2o_dataset)

        self._fitted_model = predictor.leader
    

    @logger.catch
    def predict(self, x_test: pd.DataFrame) -> Union[pd.Series, np.ndarray]:
        if self._fitted_model is None:
            raise NotFittedError()
        dataset_test = h2o.H2OFrame(x_test, column_types=self._df_dtypes[:-1])
        
        predictions = self._fitted_model.predict(dataset_test).as_data_frame(use_multi_thread=True).iloc[:, 0]

        return predictions
