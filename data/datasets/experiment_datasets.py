import pandas as pd
from sklearn import datasets as sklearn_datasets
from river import datasets
from river.stream import iter_sklearn_dataset, iter_csv, iter_pandas
from river.datasets.synth import Agrawal, RandomRBFDrift, LEDDrift

from data.modified_concept_drift_stream import SafeConceptDriftStream


#####################################################################################################
# These datasets are based on
#  Gunasekara, Nuwan & Gomes, Heitor Murilo & Pfahringer, Bernhard & Bifet, Albert. (2022).
#  Online Hyperparameter Optimization for Streaming Neural Networks. 1-9. 10.1109/IJCNN55064.2022.9891953.
# and
#  Gomes, Heitor Murilo & Read, Jesse & Bifet, Albert. (2019).
#  Streaming Random Patches for Evolving Data Stream Classification. 240-249. 10.1109/ICDM.2019.00034.
#####################################################################################################
import os
DATA_DIR = os.getenv('DATA_DIR', './data/datasets/')


class Airlines(datasets.base.Dataset):
    def __init__(self):
        super().__init__(n_features=7, n_classes=2, n_outputs=1, task=datasets.base.BINARY_CLF)
        converter = {"Flight": int, "DayOfWeek": int, "Time": int, "Length": int, "Delay": int}
        self.iterator = iter_csv(filepath_or_buffer=f"{DATA_DIR}airlines.zip", converters=converter, target="Delay")

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            raise StopIteration


class KDD99(datasets.base.Dataset):
    def __init__(self):
        super().__init__(n_features=41, n_classes=23, n_outputs=1, task=datasets.base.MULTI_CLF)
        # dataset = sklearn_datasets.fetch_kddcup99(percent10=False)
        # self.iterator = iter_sklearn_dataset(dataset)
        csv_file = f"{DATA_DIR}kddcup.data.corrected"
        df = pd.read_csv(csv_file, header=None, index_col=None)
        y = df[df.columns[-1]]
        X = df[df.columns[:-1]]
        self.iterator = iter_pandas(X, y)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            raise StopIteration


class WISDM(datasets.base.Dataset):
    def __init__(self):
        super().__init__(n_features=44, n_classes=6, n_outputs=1, task=datasets.base.MULTI_CLF)
        converter = {"user": int}
        converter.update({f"X{i}": float for i in range(0, 10)})
        converter.update({f"Y{i}": float for i in range(0, 10)})
        converter.update({f"Z{i}": float for i in range(0, 10)})
        converter.update({col: float for col in ["XAVG", "YAVG", "ZAVG", "XPEAK", "YPEAK", "ZPEAK",
                                                 "XABSOLDEV", "YABSOLDEV", "ZABSOLDEV",
                                                 "XSTANDDEV", "YSTANDDEV", "ZSTANDDEV", "RESULTANT"]})
        self.iterator = iter_csv(filepath_or_buffer=f"{DATA_DIR}WISDM.zip", converters=converter, target="label")

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            raise StopIteration


class CovType(datasets.base.Dataset):
    def __init__(self):
        super().__init__(n_features=54, n_classes=7, n_outputs=1, task=datasets.base.MULTI_CLF)
        # dataset = sklearn_datasets.fetch_covtype()
        # self.iterator = iter_sklearn_dataset(dataset)
        csv_file = f"{DATA_DIR}Covertype/covtype.data"
        df = pd.read_csv(csv_file, header=None, index_col=None)
        y = df[df.columns[-1]]
        X = df[df.columns[:-1]]
        self.iterator = iter_pandas(X, y)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            raise StopIteration


class Nomao(datasets.base.Dataset):
    def __init__(self):
        super().__init__(n_features=118, n_classes=2, n_outputs=1, task=datasets.base.BINARY_CLF)
        csv_file = f"{DATA_DIR}nomao.zip"
        X = pd.read_csv(csv_file)
        y = X.pop("label")
        self.iterator = iter_pandas(X, y)  # spares me from having to create a type converter dict for the columns

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            raise StopIteration


class DriftingAgrawal(datasets.base.SyntheticDataset):
    # used widths: 50 and 50k
    def __init__(self, width, seed=42):
        super().__init__(n_features=9, n_classes=2, n_outputs=1, task=datasets.base.BINARY_CLF)
        _agrawal1 = Agrawal(classification_function=1, perturbation=0.05, seed=seed)
        _agrawal2 = Agrawal(classification_function=2, perturbation=0.05, seed=seed)
        _agrawal3 = Agrawal(classification_function=1, perturbation=0.05, seed=seed)
        _agrawal4 = Agrawal(classification_function=4, perturbation=0.05, seed=seed)
        first_drift = SafeConceptDriftStream(_agrawal1, _agrawal2, seed=seed, position=250000, width=int(width/2))
        third_drift = SafeConceptDriftStream(_agrawal3, _agrawal4, seed=seed, position=250000, width=int(width/2))
        self.data = iter(SafeConceptDriftStream(first_drift, third_drift, seed=seed, position=500000, width=int(width/2)))

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.data)
        except StopIteration:
            raise StopIteration


class DriftingRBF(RandomRBFDrift):
    # used change_speeds: 0.001 and 0.0001
    def __init__(self, change_speed, seed=42):
        super().__init__(
            seed_model=seed,
            seed_sample=seed,
            n_classes=5,
            n_features=10,
            n_centroids=50,
            change_speed=change_speed,
            n_drift_centroids=50
        )


class DriftingLED(datasets.base.SyntheticDataset):
    # used widths: 50 and 50k
    def __init__(self, width, seed=42):
        super().__init__(n_features=24, n_classes=10, n_outputs=1, task=datasets.base.MULTI_CLF)
        _led1 = LEDDrift(noise_percentage=0.1, irrelevant_features=True, n_drift_features=1, seed=seed)
        _led2 = LEDDrift(noise_percentage=0.1, irrelevant_features=True, n_drift_features=3, seed=seed)
        _led3 = LEDDrift(noise_percentage=0.1, irrelevant_features=True, n_drift_features=5, seed=seed)
        _led4 = LEDDrift(noise_percentage=0.1, irrelevant_features=True, n_drift_features=7, seed=seed)
        first_drift = SafeConceptDriftStream(_led1, _led2, seed=seed, position=250000, width=int(width/2))
        third_drift = SafeConceptDriftStream(_led3, _led4, seed=seed, position=250000, width=int(width/2))
        self.data = iter(SafeConceptDriftStream(first_drift, third_drift, seed=seed, position=500000, width=int(width/2)))

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.data)
        except StopIteration:
            raise StopIteration

