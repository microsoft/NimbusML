# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------
"""
Datasets used in MicrosoftML unittests.
"""
import copy
import os

import numpy
import pandas
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder

__all__ = ["get_dataset", "available_datasets"]


class DataSet:
    """
    Common interface to datasets.
    """

    def __init__(self, inst=None):
        """
        If *inst* not None, members *_data* and *_newcolumns* are copied.
        """
        if inst is None:
            self.__dict__['_data'] = None
            self.__dict__['_new_columns'] = {}
            self.__dict__['_is_copy'] = False
        else:
            self.__dict__['_data'] = copy.deepcopy(inst._data)
            self.__dict__['_new_columns'] = copy.deepcopy(
                inst._new_columns)
            self.__dict__['_is_copy'] = True

    def copy(self):
        """
        Provides a copy of the datasets.
        """
        return self.__class__(inst=self)

    def __setattr__(self, name, value):
        """
        Creation of a new column.
        We add it to the members of the class.
        """
        if isinstance(value, (list, pandas.Series, numpy.ndarray)):
            if isinstance(value, pandas.Series):
                if value.dtype == bool:
                    value = list(1.0 if x else 0.0 for x in value)
                elif value.dtype in (numpy.float32, float):
                    pass
                else:
                    raise TypeError(
                        "Unexpexted type for values {0}".format(
                            value.dtype))
            else:
                raise TypeError(
                    "You need to convert your container into a "
                    "pandas.Series")
            self._new_columns[name] = value
        else:
            raise TypeError(
                "Unexpected issue with name={0} value type={1}".format(
                    name, type(value)))

    def as_df(self):
        """
        Return the data as a dataframe.
        """
        raise NotImplementedError()


class DataSetIris(DataSet):
    """
    `Iris dataset <https://scikit-learn.org/stable/auto_examples/datasets
    /plot_iris_dataset.html>`_ dataset.
    """

    def load(self):
        """
        Load the data.
        """
        if self._data is None:
            self.__dict__['_data'] = load_iris()

    @property
    def name(self):
        return "iris"

    @property
    def species_names(self):
        self.load()
        try:
            return self._data.target_names
        except AttributeError as e:
            raise AttributeError("Unable to find Species in {0}".format(
                ", ".join(sorted(self._data.keys()))))

    @property
    def species(self):
        names = self.species_names
        try:
            return pandas.Series(
                name="Species", data=[
                    names[i] for i in self._data.target])
        except AttributeError as e:
            raise AttributeError("Unable to find Species in {0}".format(
                ", ".join(sorted(self._data.keys()))))

    def as_df(self):
        """
        Return the data as a dataframe.
        """
        self.load()
        try:
            df = pandas.DataFrame(
                self._data.data,
                columns=[
                    "Sepal_Length",
                    "Sepal_Width",
                    "Petal_Length",
                    "Petal_Width"])
            df["Label"] = pandas.DataFrame(self._data.target)
            df["Species"] = df["Label"].apply(
                lambda x: self._data.target_names[x])
        except AttributeError as e:
            raise AttributeError("Unable to find a column in {0}".format(
                ", ".join(sorted(self._data.keys()))))
        df["Setosa"] = df["Label"].apply(lambda x: 1.0 if x == 0 else 0.0)
        if len(set(df["Setosa"])) == 1:
            raise ValueError("Unique class: {0}, Label={1}".format(
                set(df["Setosa"]), set(df["Label"])))
        for k, v in self._new_columns.items():
            if df.shape[0] != len(v):
                raise Exception(
                    "Dimension mismatch {0} != {1}".format(
                        df.shape[0], len(v)))
            df[k] = v
        df["Species"] = df["Species"].astype(str)
        df["Label"] = df["Label"].astype("category")
        return df


class DataSetInfert(DataSet):
    """
    `Infert dataset <https://stat.ethz.ch/R-manual/R-devel/library
    /datasets/html/infert.html>`_.
    """

    def __init__(self, inst=None):
        """
        Constructor
        """
        DataSet.__init__(self, inst=inst)
        if inst is None:
            self.load()
        else:
            if hasattr(inst, "case"):
                self.__dict__["case"] = inst.case.copy()
            if hasattr(inst, "isCase"):
                self.__dict__["isCase"] = inst.isCase.copy()

    @property
    def name(self):
        return "infert"

    def load(self):
        """
        Load the data.
        """
        if self._data is None:
            # isCase ~ age + parity + education + spontaneous + induced
            # education age parity induced case spontaneous stratum
            # pooled.stratum
            this = os.path.join(
                os.path.dirname(__file__),
                "data",
                "gplv2",
                "infert.csv")
            self.__dict__['_data'] = pandas.read_csv(this)
            self.__dict__['case'] = self._data["case"]
            self._finalize()

    def _finalize(self):
        """
        Function calls after the data is loaded.
        """
        self._data["education_str"] = self._data.education
        le = LabelEncoder()
        le.fit(self._data.education)
        self._data["education"] = le.transform(
            self._data.education).astype(float)
        for col in [
            "education",
            "spontaneous",
            "case",
            "age",
            "parity",
            "induced",
            "stratum",
            "pooled.stratum"]:
            self._data[col] = self._data[col].astype(float)

    def __setattr__(self, name, value):
        """
        Add a new column. The method changed the column type to float
        is the name is *isCase* which usually happend to be a boolean.
        """
        if name == "case" and value is None:
            self.load()
            # We do nothing else.
        else:
            DataSet.__setattr__(self, name, value)
            if name == "isCase":
                self._data["isCase"] = value.apply(
                    lambda x: 1. if x else 0.)

    def as_df(self):
        """
        Return the data as datadframe.
        """
        self.load()
        return self._data

    def as_filepath(self):
        """
        Return file name.
        """
        return os.path.join(
            os.path.dirname(__file__),
            "data",
            "gplv2",
            "infert.csv")


class DataSetAirQuality(DataSet):
    """
    `AirQuality dataset <https://stat.ethz.ch/R-manual/R-devel/library
    /datasets/html/airquality.html>`_.
    """

    def __init__(self, inst=None):
        """
        Constructor
        """
        DataSet.__init__(self, inst=inst)
        if inst is None:
            self.load()

    @property
    def name(self):
        return "airquality"

    def load(self):
        """
        Load the data.
        """
        if self._data is None:
            # isCase ~ age + parity + education + spontaneous + induced
            # education age parity induced case spontaneous stratum
            # pooled.stratum
            this = os.path.join(
                os.path.dirname(__file__),
                "data",
                "gplv2",
                "airquality.csv")
            self.__dict__['_data'] = pandas.read_csv(this)
            self._finalize()

    def _finalize(self):
        """
        Function calls after the data is loaded.
        """
        pass

    def __setattr__(self, name, value):
        """
        Add a new column.
        """
        DataSet.__setattr__(self, name, value)
        self._data[name] = value

    def as_df(self):
        """
        Return the data as datadframe.
        """
        self.load()
        return self._data

    def as_filepath(self):
        """
        Return file name.
        """
        return os.path.join(
            os.path.dirname(__file__),
            "data",
            "gplv2",
            "airquality.csv")


class Topics(DataSet):
    """
        Sample dataset to show Light LDA transform examples in the API
        Reference section
    """

    def __init__(self, inst=None):
        """
        Constructor
        """
        DataSet.__init__(self, inst=inst)
        if inst is None:
            # self.load()
            pass

    @property
    def name(self):
        return "topics"

    def as_filepath(self):
        """
        Return file name.
        """
        return os.path.join(os.path.dirname(__file__), "data",
                            "topics.csv")


class Timeseries(DataSet):
    """
        Sample dataset to show Timeseries transform examples in the API
        Reference section
    """

    def __init__(self, inst=None):
        """
        Constructor
        """
        DataSet.__init__(self, inst=inst)
        if inst is None:
            # self.load()
            pass

    @property
    def name(self):
        return "timeseries"

    def as_filepath(self):
        """
        Return file name.
        """
        return os.path.join(
            os.path.dirname(__file__),
            "data",
            "timeseries.csv")


class WikiDetox_Train(DataSet):
    """
    WikiDetox dataset train.
    """

    def __init__(self, inst=None):
        """
        Constructor
        """
        DataSet.__init__(self, inst=inst)
        if inst is None:
            # self.load()
            pass

    @property
    def name(self):
        return "wiki_detox_train"

    def as_filepath(self):
        """
        Return file name.
        """
        return os.path.join(
            os.path.dirname(__file__),
            "data",
            "train-250.wikipedia.sample.tsv")


class WikiDetox_Test(DataSet):
    """
    WikiDetox dataset test.
    """

    def __init__(self, inst=None):
        """
        Constructor
        """
        DataSet.__init__(self, inst=inst)
        if inst is None:
            # self.load()
            pass

    @property
    def name(self):
        return "wiki_detox_test"

    def as_filepath(self):
        """
        Return file name.
        """
        return os.path.join(
            os.path.dirname(__file__),
            "data",
            "test.wikipedia.sample.tsv")


class FS_Train(DataSet):
    """
    Flight Schedule data, manually created
    """

    def __init__(self, inst=None):
        """
        Constructor
        """
        DataSet.__init__(self, inst=inst)
        if inst is None:
            # self.load()
            pass

    @property
    def name(self):
        return "fstrain"

    def as_filepath(self):
        """
        Return file name.
        """
        return os.path.join(
            os.path.dirname(__file__),
            "data",
            "train_fs.csv")


class FS_Test(DataSet):
    """
    Flight Schedule data, manually created
    """

    def __init__(self, inst=None):
        """
        Constructor
        """
        DataSet.__init__(self, inst=inst)
        if inst is None:
            # self.load()
            pass

    @property
    def name(self):
        return "fstest"

    def as_filepath(self):
        """
        Return file name.
        """
        return os.path.join(
            os.path.dirname(__file__),
            "data",
            "test_fs.csv")


class MSLTR_Train(DataSet):
    """
    MSLTR dataset train, sampled from
    https://www.microsoft.com/en-us/research/project/mslr/
    """

    def __init__(self, inst=None):
        """
        Constructor
        """
        DataSet.__init__(self, inst=inst)
        if inst is None:
            # self.load()
            pass

    @property
    def name(self):
        return "msltrtrain"

    def as_filepath(self):
        """
        Return file name.
        """
        return os.path.join(
            os.path.dirname(__file__),
            "data",
            "train-msltr.sample.csv")


class MSLTR_Test(DataSet):
    """
    MSLTR dataset test, sampled from
    https://www.microsoft.com/en-us/research/project/mslr/
    """

    def __init__(self, inst=None):
        """
        Constructor
        """
        DataSet.__init__(self, inst=inst)
        if inst is None:
            # self.load()
            pass

    @property
    def name(self):
        return "msltrtest"

    def as_filepath(self):
        """
        Return file name.
        """
        return os.path.join(
            os.path.dirname(__file__),
            "data",
            "test-msltr.sample.csv")


class Uci_Train(DataSet):
    """
    UCI Adult dataset train.
    """

    def __init__(self, inst=None):
        """
        Constructor
        """
        DataSet.__init__(self, inst=inst)
        if inst is None:
            pass

    @property
    def name(self):
        return "uciadult_train"

    def as_filepath(self):
        """
        Return file name.
        """
        return os.path.join(
            os.path.dirname(__file__),
            "data",
            "train-500.uciadult.sample.csv")


class Uci_Test(DataSet):
    """
    UCI Adult dataset test.
    """

    def __init__(self, inst=None):
        """
        Constructor
        """
        DataSet.__init__(self, inst=inst)
        if inst is None:
            pass

    @property
    def name(self):
        return "uciadult_test"

    def as_filepath(self):
        """
        Return file name.
        """
        return os.path.join(
            os.path.dirname(__file__),
            "data",
            "test-100.uciadult.sample.csv")


class Generated_Twitter_Train(DataSet):
    """
    Manually generated Twitter training dataset.
    """

    def __init__(self, inst=None):
        """
        Constructor
        """
        DataSet.__init__(self, inst=inst)
        if inst is None:
            # self.load()
            pass

    @property
    def name(self):
        return "gen_twittertrain"

    def as_filepath(self):
        """
        Return file name.
        """
        return os.path.join(
            os.path.dirname(__file__),
            "data",
            "train-twitter.gen-sample.tsv")


class Generated_Twitter_Test(DataSet):
    """
    Manually generated Twitter testing dataset.
    """

    def __init__(self, inst=None):
        """
        Constructor
        """
        DataSet.__init__(self, inst=inst)
        if inst is None:
            # self.load()
            pass

    @property
    def name(self):
        return "gen_twittertest"

    def as_filepath(self):
        """
        Return file name.
        """
        return os.path.join(
            os.path.dirname(__file__),
            "data",
            "test-twitter.gen-sample.tsv")


class Generated_Ticket_Train(DataSet):
    """
    Manually generated Flight ticket training dataset.
    """

    def __init__(self, inst=None):
        """
        Constructor
        """
        DataSet.__init__(self, inst=inst)
        if inst is None:
            # self.load()
            pass

    @property
    def name(self):
        return "gen_tickettrain"

    def as_filepath(self):
        """
        Return file name.
        """
        return os.path.join(
            os.path.dirname(__file__),
            "data",
            "train-ticketchoice.csv")


class Generated_Ticket_Test(DataSet):
    """
    Manually generated Flight ticket testing dataset.
    """

    def __init__(self, inst=None):
        """
        Constructor
        """
        DataSet.__init__(self, inst=inst)
        if inst is None:
            # self.load()
            pass

    @property
    def name(self):
        return "gen_tickettest"

    def as_filepath(self):
        """
        Return file name.
        """
        return os.path.join(
            os.path.dirname(__file__),
            "data",
            "test-ticketchoice.csv")


_datasets = dict(
    iris=lambda: DataSetIris(),
    infert=lambda: DataSetInfert(),
    topics=lambda: Topics(),
    timeseries=lambda: Timeseries(),
    airquality=lambda: DataSetAirQuality(),
    wiki_detox_train=lambda: WikiDetox_Train(),
    wiki_detox_test=lambda: WikiDetox_Test(),
    gen_twittertrain=lambda: Generated_Twitter_Train(),
    gen_twittertest=lambda: Generated_Twitter_Test(),
    gen_tickettrain=lambda: Generated_Ticket_Train(),
    gen_tickettest=lambda: Generated_Ticket_Test(),
    uciadult_train=lambda: Uci_Train(),
    uciadult_test=lambda: Uci_Test(),
    msltrtrain=lambda: MSLTR_Train(),
    msltrtest=lambda: MSLTR_Test(),
    fstrain=lambda: FS_Train(),
    fstest=lambda: FS_Test()
)


def get_dataset(name):
    """
    Return a predefined datasets.

    :param name: options are: ``airquality``, ``fstest``,
        ``fstrain``, ``gen_tickettest``, ``gen_tickettrain``,
        ``gen_twittertest``, ``gen_twittertrain``, ``infert``,
        ``iris``, ``msltrtest``, ``msltrtrain``,
        ``timeseries``, ``topics``, ``uciadult_test``, ``uciadult_train``,
        ``wiki_detox_test``, ``wiki_detox_train``.

    Example:
        >>> from nimbusml.datasets import get_dataset
        >>> path = get_dataset('infert').as_filepath()
        >>> print(path)
        ...\nimbusml\datasets\_data\gplv2\infert.csv

    """
    if name in _datasets:
        return _datasets[name]()
    else:
        available_dataset = ", ".join(sorted(_datasets.keys()))
        raise KeyError(
            "Unable to find dataset '{0}'. "
            "The available options are: {1}.".format(
                name, available_dataset))


def available_datasets():
    """
    Returns the list of available datasets.
    """
    return sorted(_datasets.keys())
