""" Country dummy module """
import pandas as pd  # type: ignore
from views.utils import db


def fetch_cdum() -> None:
    """ Nothing to fetch for country dummies """


def load_cdum() -> None:
    """ Load country dummies """

    df = db.db_to_df(fqtable="staging.country", cols=["id"], ids=["id"])
    df = df.reset_index().rename(columns={"id": "country_id"})
    df["to_dummy"] = df["country_id"]
    df = df.set_index(["country_id"])
    df = pd.get_dummies(df.to_dummy, prefix="cdum")
    db.drop_schema("cdum")
    db.create_schema("cdum")
    db.df_to_db(fqtable="cdum.c", df=df)
