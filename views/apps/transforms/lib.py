""" Transform library.

Many functions assume data in the form of a pandas series or dataframe
indexed by timevar as level 0 and groupvar as level 1.
"""

import logging
from typing import Any

from scipy.spatial import cKDTree  # type: ignore
import geopandas as gpd  # type: ignore
import libpysal as lps  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from views.utils.data import check_has_multiindex

# Fiona is extremely verbose in debug mode, set her to warning
logging.getLogger("fiona").setLevel(logging.WARNING)


def summ(df: pd.DataFrame) -> pd.Series:
    """ Return the row-wise sum of the dataframe """
    return df.sum(axis=1)


def product(df: pd.DataFrame) -> pd.Series:
    """ Return the row-wise product of the dataframe """
    return df.product(axis=1)


def delta(s: pd.Series, time: int = 1) -> pd.Series:
    """ Return the time-delta of s """

    check_has_multiindex(s)
    return s - tlag(s, time=time)


def greater_or_equal(s: pd.Series, value: float) -> pd.Series:
    """ 1 if s >= value, else 0 """

    mask = s >= value
    y = mask.astype(int)

    return y


def smaller_or_equal(s: pd.Series, value: float) -> pd.Series:
    """ 1 if s >= value, else 0 """

    mask = s <= value
    y = mask.astype(int)

    return y


def in_range(s: pd.Series, low: float, high: float) -> pd.Series:
    """ 1 if low <= s <= high else 0 """

    y_high = smaller_or_equal(s, high)
    y_low = greater_or_equal(s, low)
    y = y_high + y_low - 1

    return y


def tlag(s: pd.Series, time: int) -> pd.Series:
    """ Time lag """
    check_has_multiindex(s)
    if time < 1:
        msg = f"Time below 1 passed to tlag: {time} \n"
        msg += "Call tlead() instead \n"
        raise RuntimeError(msg)

    return s.groupby(level=1).shift(time)


def tlead(s: pd.Series, time: int) -> pd.Series:
    """ Time lead """
    check_has_multiindex(s)
    if time < 1:
        msg = f"Time below 1 passed to tlead: {time} \n"
        msg += "Call tlag() instead \n"
        raise RuntimeError(msg)

    return s.groupby(level=1).shift(-time)


def moving_average(s: pd.Series, time: int) -> pd.Series:
    """ Moving average """
    check_has_multiindex(s)
    if time < 1:
        msg = f"Time below 1 passed to ma: {time} \n"
        raise RuntimeError(msg)

    # Groupby groupvar
    y = s.groupby(level=1)
    # Divide into rolling time window of size time
    # min_periods=0 lets the window grow with available data
    # and prevent the function from inducing missingness
    y = y.rolling(time, min_periods=0)
    # Compute the mean
    y = y.mean()
    # groupby and rolling do stuff to indices, return to original form
    y = y.reset_index(level=0, drop=True).sort_index()
    return y


def cweq(s: pd.Series, value: float, seed=None) -> pd.Series:
    """ Count while s equals value

    @TODO: Seed from series (series of seeds per groupvar?)

    """
    check_has_multiindex(s)

    def set_seed(count, s, seed, mask):
        """ Set count=seed in first time if mask was True there

        Example: We want time since conflict, which is time in peace.
        So we want count_while(conflict == 0).
        If our conflict series starts at 0 we might assume some longer
        previous history of peace.
        As the time count is summed cumulatively we can "seed" this
        counting sum with a starting value.

        This seed is therefore insterted into the first time period
        of the count IF the country is in peace at that time.
        Being in peace means the count is True, or ==1 as we
        already cast the masks T/F to the counters 1/0.

        """
        ix_timevar = s.index.get_level_values(0)
        first_time = ix_timevar == min(ix_timevar)
        mask_true = mask == 1
        first_time_where_mask_true = first_time & mask_true
        count.loc[first_time_where_mask_true] = seed
        return count

    # Drop NaN's
    s = s.dropna()

    # Boolean mask of where our condition (s==value) is True
    mask = s == value

    # This is a tricky one, print it out if its confusing.
    # Values of block_grouper are incremented when mask is NOT true.
    # This creates values that are constant (not incrementing) through a
    # consecutive spell of mask being True.
    # Grouping by this var thus lets the count.cumsum() restart for
    # each group of consecutive rows where mask is True and stay at
    # zero for the rows where block_grouper keeps incrementing,
    # which are the rows where mask is not met.
    # Note that mask is True when the criteria is fullfilled
    # Basically lets us assign a grouping id to each consecutive
    # spell of our condition being True.
    block_grouper = (~mask).groupby(level=1).cumsum()

    # Our mask becomes the basis for the count by casting it to int
    count = mask.astype(int)

    if seed:
        count = set_seed(count, s, seed, mask)

    # Get the groupvar-level index to group by
    ix_groupvar = s.index.get_level_values(1)

    # The time elapsed while condition is true
    y = count.groupby([block_grouper, ix_groupvar]).cumsum()
    y = y.astype(int)

    return y


def time_since(s, value=0, seed=None) -> pd.Series:
    """ time since event in s where event is value other than 0.

    In order to compute a variable like "time since previous conflict
    event" we must apply a timelag to cweq() to get a series because
    for fitting a simultanous model we do not want the counter to be
    simultaneous to the event.

    Consider the data:

    event  : 0, 0, 1, 1, 0, 0 # Event
    cweq_0 : 1, 2, 0, 0, 1, 2 # count event while equals zero
    tisiev : ., 1, 2, 0, 0, 1 # time since event

    Fitting a model like "event ~ cweq0" makes no sense as cweq0 is
    always 0 if event=1.
    A model like "event ~ tsnp" makes more sense.
    We must apply a time lag to event before computing the counter to
    see how long time has elapsed since the previous event.

    Of course this isn't necessary for OSA modelling where all the
    rhs variables are time-lagged anyway but this is useful for
    dynamic simulation where X and predicted y are simulatenous.

    """

    return cweq(s=tlag(s=s, time=1), value=value, seed=seed)


def decay(s: pd.Series, halflife: float) -> pd.Series:
    """ Decay function

    See half-life formulation at
    https://en.wikipedia.org/wiki/Exponential_decay
    """

    return 2 ** ((-1 * s) / halflife)


def mean(s: pd.Series) -> pd.Series:
    """ Per-groupvar arithmetic mean """

    return s.groupby(level=1).transform("mean")


def ln(s: pd.Series) -> pd.Series:
    """ Natural log of s+1 """
    return np.log1p(s)


def demean(s: pd.Series) -> pd.Series:
    """ demean, s = s - mean_group(s) """
    check_has_multiindex(s)
    s_mean = s.groupby(level=1).transform("mean")
    return s - s_mean


def rollmax(s: pd.Series, window: int) -> pd.Series:
    """ Rolling max """
    check_has_multiindex(s)
    # See https://github.com/pandas-dev/pandas/issues/14013
    y = s.groupby(level=1).apply(
        lambda x: x.rolling(window=window, min_periods=0).max()
    )

    return y


def onset_possible(s: pd.Series, window: int) -> pd.Series:
    """onset possible if no event occured in the preceeding window times"""
    # fillna() is so that the first t in a group is always a possible onset
    return (~rollmax(tlag(s, 1).fillna(0), window).astype(bool)).astype(int)


def onset(s: pd.Series, window: int) -> pd.Series:
    """ Compute onset

    A row is defined as an onset if
    * onset is possible
    * s is greater than 0
    """
    s_onset_possible = (
        onset_possible(s, window).astype(bool) & s.astype(bool)
    ).astype(int)
    return s_onset_possible


def distance_to_event(
    gdf: gpd.GeoDataFrame, col: str, k: int = 1, fill_value: int = 99
) -> pd.Series:
    """ Get spatial distance to event

    Args:
        gdf: GeoDataFrame with a multiindex like [time, group]  and
            cols for centroid and col.
        col: Name of col to count as event if == 1
        k: Number of neighbors to consider
        fill_value: When no events are found fill with this value
    Returns:
        dist: pd.Series of distance to event
    """

    # Index-only gdf to hold results
    gdf_results = gdf[[]].copy()
    gdf_results["distance"] = np.nan

    times = sorted(list(set(gdf_results.index.get_level_values(0))))

    # (x,y) coord pairs for all grids
    points_canvas = np.array(
        list(zip(gdf.loc[times[0]].centroid.x, gdf.loc[times[0]].centroid.y))
    )

    for t in times:
        gdf_events_t = gdf.loc[t][gdf.loc[t][col] == 1]
        points_events = np.array(
            list(zip(gdf_events_t.centroid.x, gdf_events_t.centroid.y))
        )
        if len(points_events) > 0:
            # Build the KDTree of the points
            btree = cKDTree(data=points_events)  # pylint: disable=not-callable
            # Find distance to closest k points, discard idx
            dist, _ = btree.query(points_canvas, k=k)
            # If more than one neighbor get the mean distance
            if k > 1:
                dist = np.mean(dist, axis=1)
            gdf_results.loc[t, "distance"] = dist
        else:
            gdf_results.loc[t, "distance"] = fill_value

    s = gdf_results["distance"]
    return s


# pylint: disable=too-many-locals
def spacetime_distance_to_event(
    gdf: gpd.GeoDataFrame,
    col: str,
    t_scale: int = 1,
    k: int = 1,
    fill_value: int = 99,
) -> pd.Series:
    """ Get space-time distance to event

    The time dimension of the index (level=0) is used a third
    dimension. Making the distance a space-time interval and
    not just a spatial distance.

    @TODO: Add time scaling to weight distance/time differently

    Args:
        gdf: GeoDataFrame with a multiindex like [time, group]  and
            cols for centroid and col.
        col: Name of col to count as event if == 1
        k: Number of neighbors to consider, increasing k gives
           more weight to clusters of events than exceptional blips.
        fill_value: When no events are found fill with this value
    Returns:
        dist: pd.Series of distance to event

    """
    gdf_results = gdf[[]].copy()
    gdf_results["distance"] = np.nan
    times = sorted(list(set(gdf_results.index.get_level_values(0))))
    xs = gdf.loc[times[0]].centroid.x
    ys = gdf.loc[times[0]].centroid.y
    len_a_t = len(gdf.loc[times[0]])

    for t in times:
        # Subset to look back in time
        gdf_ts = gdf.loc[min(times) : t]
        gdf_ts_events = gdf_ts[gdf_ts[col] > 0]

        # Only compute distances if we have any events
        if len(gdf_ts_events) > k:
            points_all = np.array(
                list(zip(xs, ys, np.repeat(t * t_scale, len_a_t)))
            )
            times_back = gdf_ts_events.index.get_level_values(0) * t_scale
            points_events = np.array(
                list(
                    zip(
                        gdf_ts_events.centroid.x,
                        gdf_ts_events.centroid.y,
                        times_back,
                    )
                )
            )

            btree = cKDTree(data=points_events)  # pylint: disable=not-callable
            # Returns dist and idx, ignore idx
            dist, _ = btree.query(points_all, k=k)

            # if k>1 we get distance to each of k closest points, mean them
            if k > 1:
                dist = np.mean(dist, axis=1)

        # If no events fill dist with fill_value
        else:
            dist = fill_value

        gdf_results.loc[t, "distance"] = dist

    s = gdf_results["distance"]

    return s


def spatial_lag(
    gdf: gpd.GeoDataFrame, col: str, first: int = 1, last: int = 1
) -> pd.Series:
    """ Compute spatial lag on col in gdf """

    def gdf_to_w_q(gdf_geom: gpd.GeoDataFrame, first: int, last: int) -> Any:
        """ Build queen weights from gdf.

        Use a temporary shapefile to get the geometry into pysal as
        their new interface is confusing. There must be a less silly
        way.
        """
        # Compute first order spatial weight
        w = lps.weights.Queen.from_dataframe(gdf_geom, geom_col="geom")

        # If we want higher order
        if not first == last == 1:
            w_ho = lps.weights.higher_order(w, first)

            # loop from first to last order
            for order in range(first + 1, last + 1):
                w_this_order = lps.weights.higher_order(w, order)
                w_ho = lps.weights.w_union(w_ho, w_this_order)

            # Replace original w
            w = w_ho

        return w

    def _splag(y: Any, w: Any) -> Any:
        """ Flip argument order for transform """
        return lps.weights.lag_spatial(w, y)

    # @TODO: Add support for time-variant geometries (countries)
    # If geom's don't change use the one from the last time
    gdf_geom = gdf.loc[gdf.index.get_level_values(0).max()]
    w = gdf_to_w_q(gdf_geom, first, last)
    s = gdf.groupby(level=0)[col].transform(_splag, w=w)
    return s
