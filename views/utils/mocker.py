""" Fake data generation """
import string
import itertools

import numpy as np  # type: ignore
import pandas as pd  # type: ignore


def generate_probs(n, seed=False, distribution="normal"):
    """ Generate vector of normal probabilities with length n"""

    def scale_to_0_1(y):
        """ Scale a vector to have values in range 0-1 """

        y = (y - np.min(y)) / (np.max(y) - np.min(y))
        return y

    if seed:
        np.random.seed(seed)  # pragma: no cover

    if distribution == "normal":
        y = np.random.normal(0, 1, n)
    else:
        raise NotImplementedError

    y = scale_to_0_1(y)

    return y


def generate_bools(n):
    """ Generate a vector of random bools of length n """

    allowed_values = [0, 1]
    y = np.random.choice(allowed_values, size=n)

    return y


def generate_counts(n, max_value=10):
    """ Generate a vector of positive integers """

    y = np.random.randint(low=0, high=max_value, size=n)

    return y


def generate_reals(n, min_value=-100, max_value=100, distribution="uniform"):
    """ Generate a vector of real numbers from distribution """

    if distribution == "uniform":
        y = np.random.uniform(size=n, low=min_value, high=max_value)
    else:
        raise NotImplementedError

    return y


# pylint: disable=too-many-instance-attributes
class DfMocker:
    """ Makes mock dataframes

    Args:
        n_t: Number of time periods
        n_groups: Number of individuals/groups
        n_cols: Number of data columns
        timevar: Name of timevar
        groupvar: Name of groupvar
        prefix: Prefix to add to column names
        datatypes (list): Allowed values: probs, bools, counts, reals
        seed: Random seed
    """

    # pylint: disable=too-many-arguments,too-many-locals
    def __init__(
        self,
        n_t=10,
        n_groups=5,
        n_cols=5,
        timevar="timevar",
        groupvar="groupvar",
        prefix=None,
        datatypes=None,
        seed=1,
    ):

        np.random.seed(seed)

        self.n_t = n_t
        self.n_groups = n_groups
        self.n_cols = n_cols
        self.n_rows = n_t * n_groups

        self.timevar = timevar
        self.groupvar = groupvar

        if not datatypes:
            datatypes = ["probs", "bools", "counts", "reals"]
        else:
            pass

        # Make the index
        idx = self.make_idx_from_n(
            n_t=self.n_t,
            n_groups=self.n_groups,
            timevar=self.timevar,
            groupvar=self.groupvar,
        )

        # distribute the cols between datatypes
        col_ids_per_datatype = np.array_split(
            range(self.n_cols), len(datatypes)
        )
        datadicts = []
        for datatype, col_ids in zip(datatypes, col_ids_per_datatype):
            n_cols_this_datatype = len(col_ids)
            datadicts.append(
                self.make_datadict(
                    n_cols_this_datatype, self.n_rows, datatype=datatype
                )
            )

        datadict = {}
        for dd in datadicts:
            datadict.update(dd)

        self.df = pd.DataFrame(datadict, index=idx)
        self.df.sort_index(inplace=True)
        if prefix:
            self.df = self.df.add_prefix(prefix)

    @staticmethod
    def make_idx_from_n(n_t, n_groups, timevar="timevar", groupvar="group"):
        """ Create named Multiindex """
        times = list(range(n_t))
        groups = list(range(n_groups))

        idx_tuples = list(itertools.product(times, groups))
        idx = pd.MultiIndex.from_tuples(idx_tuples, names=(timevar, groupvar))
        return idx

    def make_datadict(self, n_cols=False, n_rows=False, datatype="probs"):
        """ Make a dict of vectors for df construction

        Args:
            n_cols: number of columns
            n_rows: number of rows
            datatype: "probs", "bools", "counts", "reals", default "probs".
                      specifies the type of data to put in the columns

        """

        def data_generator_switch(datatype):
            """ Switcher for datatype to corresponding generators """

            if datatype == "probs":
                generator = generate_probs
            elif datatype == "bools":
                generator = generate_bools
            elif datatype == "counts":
                generator = generate_counts
            elif datatype == "reals":
                generator = generate_reals
            else:
                msg = "unkown datatype: {}".format(datatype)
                raise RuntimeError(msg)

            return generator

        def set_prefix(datatype):
            """ Column prefix is just the first letter of datatype """

            prefix = datatype[0]

            return prefix

        if not n_cols:
            n_cols = self.n_cols
        if not n_rows:
            n_rows = self.n_rows

        datadict = {}
        data_generator = data_generator_switch(datatype)
        prefix = set_prefix(datatype)
        for _, letter in zip(range(n_cols), string.ascii_lowercase):
            data = data_generator(n_rows)
            name = "{}_{}".format(prefix, letter)
            datadict.update({name: data})

        return datadict
