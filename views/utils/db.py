""" Database utilites """

from typing import Optional
import pandas as pd  # type: ignore
from views import config


def _make_engine(
    connectstring: Optional[str] = config.DB_CONNECTSTRING, ssl: bool = False,
) -> None:
    raise NotImplementedError


def db_to_df(fqtable, cols, ids) -> pd.DataFrame:
    """ Read a database table to a pandas dataframe """
    raise NotImplementedError


def df_to_db(df, fqtable) -> None:
    """ Write a pandas dataframe to a database table """
    raise NotImplementedError
