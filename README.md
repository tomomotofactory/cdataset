# classification-dataset
Selected Classification Problems for Benchmarking.
This data comes from https://www.openml.org/s/135/data.
Use output folder`s data or install as python library.

## Requirement
- Python 3.7 or later

## Install as python library

### Use setup.py
```shell
# Master branch
$ git clone https://github.com/tomomotofactory/cdataset.git
$ python3 setup.py install
```

### Use pip
```shell
# Master branch
$ pip3 install git+https://github.com/tomomotofactory/cdataset.git
# Specific tag (or branch, commit hash)
$ pip3 install git+https://github.com/tomomotofactory/cdataset@v0.2.1
```

### Sample Code

Load one problem.
```python3
from cdataset import ClassificationDataSetName
from cdataset import ClassificationDataSet

df = ClassificationDataSet.load_df(ClassificationDataSetName.ANALCATDATA_AUTHORSHIP)
target_name = ClassificationDataSet.load_target_name(ClassificationDataSetName.ANALCATDATA_AUTHORSHIP)
```

Load all problems.
```python3
from cdataset import ClassificationDataSetName
from cdataset import ClassificationDataSet

for dataset_name in ClassificationDataSetName.get_all_dataset_names():
    df = ClassificationDataSet.load_df(dataset_name)
    target_name = ClassificationDataSet.load_target_name(dataset_name)

    # TODO process
```