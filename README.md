# Modeller
Modeller is a tool, that automates modelling on your (or benchmark) data, acting as a convinient wrapper to the state-of-the-art automated machine learning (AutoML) libraries. 

It can be utilized by ML engineers, as well as the common users, to test different modelling scenarios.

## Project status
This project is under active development and support for the new tasks, as well as the data modalities, to be added soon.

## Installation and usage

### Installation
1. Clone the project.
2. Initialize project with `uv init` and create a virtual environment with `uv venv -p 3.10`.
3. Install dependencies with `uv sync --no-dev`. For CPU-only installation type `uv sync --extra cpu`. 

### Usage examples

Using a local dataset.
```python
from src.pymodeller.domain import Dataset
from src.pymodeller.api import Modeller
import pandas as pd


path_to_local_data = "datasets/local/ecoli.csv"
dataset = Dataset(name='ecoli', x=pd.read_csv(path_to_local_data))

modelseek = Modeller(
    automl='autogluon',
    metric='f1',
    timeout=60,
    verbosity=2
)
modelseek.run(dataset)
```

Using a dataset(or collection of such) from a wellknown-source.
```python
from src.pymodeller.api import Modeller
from src.pymodeller.repository import OpenMLDatasetRepository


# WARNING: This OpenML benchmark contains big datasets, that may not fit into your RAM.
datasets = OpenMLDatasetRepository(id=271, verbosity=1).load_datasets(x_and_y=False)
modelseek = Modeller(
    automl='autogluon',
    preset='best',
    metric='f1',
    timeout=360,
    verbosity=1
)

for dataset in datasets:
    modelseek.run(dataset)
```

## Contribution
Contribution is welcome! Feel free to open issues and submit pull requests.
