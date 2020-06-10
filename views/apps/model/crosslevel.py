""" Cross level model functions """
# flake8: noqa
# pylint: skip-file
import pandas as pd  # type: ignore
from .api import Model


# class CrossLevel:
#     def __init__(model_high_res: Model, model_low_res: Model):
#         self.model_high_res = model_high_res
#         self.model_low_res = model_low_res
#         self.steps = self.steps_in_common([model_high_res, model_low_res])
#         self.steps = sorted(list(self.steps))

#     @staticmethod
#     def steps_in_common(models: List[Model]):
#         """ Find steps that all models have in common """
#         return sorted(
#             set.intersection(*[set(model.steps) for model in models])
#         )

#     def predict(
#         self, df_high_res: pd.DataFrame, df_low_res: pd.DataFrame
#     ) -> pd.Series:
#         """ Combine high and low res predictions """
#         cols_ss_h = [self.model_high_res.cols_ss[step] for step in self.steps]
#         cols_ss_l = [self.model_low_res.cols_ss[step] for step in self.steps]
#         df_h = df_high_res[cols_ss_h]
#         df_l = df_low_res[cols_ss_l]


# def fetch_df_links():
#     """Get a df linking pg_ids to country_ids."""

#     query = """
#     SELECT pgm.priogrid_gid AS pg_id,
#        cm.country_id
#     FROM staging.priogrid_month AS pgm
#          INNER JOIN staging.country_month AS cm ON pgm.country_month_id = cm.id
#     --- Month 500 arbitrary choice
#     WHERE pgm.month_id = 500;
#     """
#     return dbutils.query_to_df(query)


# def compute_colaresi(df, col_pgm, col_cm):
#     """ Colaresian cross level probability """

#     # Sum of high resolution probabilities for each low level area
#     sum_h_by_l = df.groupby(["month_id", "country_id"])[col_pgm].transform(sum)

#     # Low resolution prob multiplied by share of high res prob in particular area
#     joint_prob = df[col_cm] * (df[col_pgm] / sum_h_by_l)

#     return joint_prob


# def crosslevel(df_pgm, df_cm, df_links, col_pgm, col_cm):
#     # Join in country_id
#     df = df_pgm[[col_pgm]].join(df_links.set_index(["pg_id"])[["country_id"]])
#     df = df.reset_index().set_index(["month_id", "country_id"])
#     df = (
#         df.join(df_cm[[col_cm]]).reset_index().set_index(["month_id", "pg_id"])
#     )
#     s = compute_colaresi(df, col_pgm, col_cm)
#     share_missing = s.isnull().sum() / len(s)
#     if share_missing > 0.01:
#         raise RuntimeError(
#             f"Too much missing in prediction, something's wrong"
#         )
#     s = s.fillna(s.mean())
#     return s


# if False:
#     df_links = fetch_df_links()
#     for step in [1, 6, 12, 24, 36]:
#         for outcome in ["sb", "ns", "os"]:
#             col_cl = f"ss.{outcome}_crosslevel.{step}"
#             col_pgm = (
#                 f"ss.{outcome}_xgb.{step}"  # Use the allthemes model for pgm
#             )
#             col_cm = f"ss.{outcome}_all_glob.{step}"  # Use the all_glob model for CM
#             df_pgm_a[col_cl] = crosslevel(
#                 df_pgm_a, df_cm_a, df_links, col_pgm, col_cm
#             )
#             df_pgm_b[col_cl] = crosslevel(
#                 df_pgm_b, df_cm_b, df_links, col_pgm, col_cm
#             )
#             df_pgm_c[col_cl] = crosslevel(
#                 df_pgm_c, df_cm_c, df_links, col_pgm, col_cm
#             )
