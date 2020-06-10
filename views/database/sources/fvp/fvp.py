""" Future of violent politics module """
import logging
import os
import tempfile

from sklearn.tree import DecisionTreeRegressor  # type: ignore

from views.utils import io, db
from views.database import common
from views.apps.data import missing

log = logging.getLogger(__name__)


def fetch_fvp():
    """ FVP data is in the Dropbox

    # TODO: Store properly
    """
    print("FVP MUST BE FETCHED MANUALLY! ITS IN THE DROPBOX.")


def load_fvp():
    """ Load FVP data """
    log.info("Started loading FVP")
    with tempfile.TemporaryDirectory() as tempdir:
        _ = common.get_files_latest_fetch(name="fvp", tempdir=tempdir)
        df = io.csv_to_df(path=os.path.join(tempdir, "MasterData.csv"))

    df = df.drop(columns=["Conflict"])
    df = df.rename(columns=lambda col: col.lower())
    df = df.set_index(["year", "gwno"])

    spec = io.load_yaml(
        path=os.path.join(os.path.dirname(__file__), "spec.yaml")
    )
    df = df[spec["cols"]]

    log.debug("Fetching df_keys")
    query = "SELECT id AS country_id, gwcode AS gwno FROM staging.country;"
    df = df.join(
        db.query_to_df(query=query)
        .sort_values(by="country_id", ascending=False)
        .drop_duplicates(subset=["gwno"])
        .set_index(["gwno"])
    )

    log.debug("Joining to skeleton")
    df = db.db_to_df(
        fqtable="skeleton.cy_global",
        ids=["year", "country_id"],
        cols=["year", "country_id"],
    ).join(df.reset_index().set_index(["year", "country_id"]), how="left")

    df = df.drop(columns=["gwno"])

    # Add consistent fvp_ prefix
    df = df.rename(
        columns=lambda col: col if col.startswith("fvp_") else f"fvp_{col}"
    )
    df = df.sort_index(axis=1).sort_index(axis=0)

    # Push raw
    db.create_schema("fvp_v2")
    db.df_to_db(fqtable="fvp_v2.cy_unimp", df=df)

    # Extrapolate before imputing
    df = missing.extrapolate(df)

    # Impute and push
    for i, df_imp in enumerate(
        missing.impute_mice_generator(
            df=df,
            n_imp=10,
            estimator=DecisionTreeRegressor(max_features="sqrt"),
            parallel=True,
        )
    ):
        db.df_to_db(df=df_imp, fqtable=f"fvp_v2.cy_imp_sklearn_{i}")

    log.info("Fininshed loading FVP")
