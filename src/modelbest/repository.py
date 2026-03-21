from abc import ABC, abstractmethod
import sys
from typing import Optional, Union, List, Any, TypeVar, cast
import numpy as np
import pandas as pd
import openml
from imblearn.datasets import fetch_datasets
from loguru import logger
import itertools

from src.modelbest.domain import Dataset


class DatasetRepository(ABC):
    def __init__(self, verbosity=2):
        self._datasets: List[Dataset] = []
        self._verbosity: int
        
        self._configure_logging(verbosity)
    
    def _configure_logging(self, verbosity: int) -> None:
        logger.remove()
        if verbosity == 3:
            logger.add(sys.stdout, level='DEBUG')
        elif verbosity == 2:
            logger.add(sys.stdout, level='INFO')
        elif verbosity == 1:
            logger.add(sys.stdout, level='SUCCESS')
        elif verbosity == 0:
            logger.add(sys.stdout, level='WARN')
        self._verbosity = verbosity

    @abstractmethod
    def load_datasets(self, ids: Optional[Union[List[int], range]] = None, split_features_and_target = False) -> List[Dataset]:
        raise NotImplementedError()

    @abstractmethod
    def load_dataset(self, id: int, split_features_and_target = False) -> Dataset:
        raise NotImplementedError()

class ImbalancedDatasetRepository(DatasetRepository):
    """
    Repository of tabular datasets with imbalanced binary targets.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: it would be better if run in a parallel.
        self._raw_datasets = fetch_datasets(data_home='datasets/imbalanced', verbose=True if self._verbosity > 1 else False)

    @logger.catch
    def load_dataset(self, id: int, split_features_and_target = False) -> Dataset:
        for i, (dataset_name, dataset_data) in enumerate(self._raw_datasets.items(), 1):
            if i == id:
                x = dataset_data.get("data")
                y = dataset_data.get("target")[:, np.newaxis]

                if not split_features_and_target:
                    x = pd.DataFrame(np.concatenate((x, y), axis=1))
                    y = None
                else:
                    # TODO: make this branch work without later fails.
                    x = pd.DataFrame(x)
                    y = pd.Series(y.T[0], dtype=str)
                
                return Dataset(
                    id=id,
                    name=dataset_name,
                    X=x,
                    y=y
                )
            elif i > id:
                raise ValueError(f"Id {id}) is out of range.")
        else:
            raise ValueError(f"Loading of Dataset(id={id}) failed.")

    @logger.catch
    def load_datasets(self, ids: Optional[Union[List[int], range]] = None, split_features_and_target = False) -> List[Dataset]:
        if ids is None:
            range_start = 1
            range_end = len(self._raw_datasets.keys()) + 1
            ids = range(range_start, range_end)
        
        logger.debug(f"Chosen dataset identifiers: {ids}.")
        for id in ids:
            dataset = self.load_dataset(id, split_features_and_target)
            if dataset is not None:
                self._datasets.append(dataset)
        
        return self._datasets

# TODO: fix loading of large datasets. 271 id fails starting from 62-st dataset.
class OpenMLDatasetRepository(DatasetRepository):
    """
    Repository of openml tabular datasets.

    Parameters
    ----------
    id: int
        Id of corpus of tasks from https://www.openml.org/search?type=benchmark&study_type=task.
    verbosity: int, default 1
        Fetching information verbosity.
    """
    def __init__(self, id: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._suite_id = id
        
        openml.config.set_root_cache_directory("datasets/openml")
        openml.config.set_console_log_level(self._verbosity)

    @logger.catch
    def load_dataset(self, id: int, split_features_and_target = False) -> Dataset:
        """
        Fetch task from openml.org or load it from local cache.
        Then convert it to appropriate format.
        
        Parameters
        ----------
        id: int
           OpenML task id.
        x_and_y: bool, default false 1
            Wheter to separate target column from feature columns.

        Returns
        -------
        Dataset, optional
            Data wrapper object.
        """
        task = openml.tasks.get_task(id)
        dataset = task.get_dataset(cache_format='feather')
        x, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
        assert x is not None and y is not None

        x = cast(pd.DataFrame, x)
        x = x.to_numpy()

        y = cast(pd.DataFrame, y)
        y = y.to_numpy()
        y = y[:, np.newaxis]

        if not split_features_and_target:
            x = pd.DataFrame(np.concatenate((x,y) , axis=1))
            y = None
        else:
            x = pd.DataFrame(x)
            y = pd.Series(y[0], name=dataset.default_target_attribute, dtype="category")
        
        return Dataset(
            id=id,
            name=dataset.name,
            X=x,
            y=y
        )

    # TODO: parallelize.
    @logger.catch
    def load_datasets(self, ids: Optional[Union[List[int], range]] = None, split_features_and_target = False) -> List[Dataset]:
        """
        Fetch benchmark suite from openml.org.
        
        Parameters
        ----------
        ids: Union[List[int], range], optional
           OpenML task identifiers.
        x_and_y: bool, default false 1
            Wheter to separate target column from feature columns.

        Returns
        -------
        List[Dataset]
            List of data wrapper objects.
        """
        corpus = openml.study.get_suite(suite_id=self._suite_id)
        assert corpus.tasks is not None

        for i, id in enumerate(corpus.tasks, 1):
            if ids is not None and id not in ids:
                raise ValueError(f"Task id={id} is out of range for chosen corpus.")

            dataset = self.load_dataset(id, split_features_and_target)
            if dataset is not None: 
                self._datasets.append(dataset)
        
        return self._datasets
