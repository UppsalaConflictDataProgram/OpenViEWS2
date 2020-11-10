""" Mapping module 

Please see projects/plots/example_maps.ipynb for examples.
"""

# pylint: disable=invalid-name

import logging
import os
from typing import Any, Dict, List, Tuple, Union

from matplotlib import pyplot as plt, image as mpimg, colors  # type: ignore
from matplotlib.offsetbox import AnchoredText, OffsetImage, AnnotationBbox  # type: ignore
from mpl_toolkits.axes_grid1 import make_axes_locatable  # type: ignore
import pandas as pd  # type: ignore
import geopandas as gpd  # type: ignore
import numpy as np  # type: ignore

import views
from views.utils import stats

log = logging.getLogger(__name__)

# Add global bboxes here: [xmin, xmax, ymin, ymax].
BBOXES = {"mainland_africa": [-18.5, 52.0, -35.5, 38.0]}


# pylint: disable=too-few-public-methods
class MapData:
    """Class for holding geography and time data for plot_map()."""

    @staticmethod
    def _get_months():
        df = views.TABLES["skeleton.m"].df
        df["date_str"] = df.year.astype(str) + "-" + df.month.astype(str)
        return df

    @staticmethod
    def _get_gdf_pgm() -> gpd.GeoDataFrame:
        gdf_pgm = views.GEOMETRIES["GeomPriogrid"].gdf
        df_pg_c = views.TABLES["skeleton.pg_c"].df.rename(
            columns={"country_id": "geo_country_id"}
        )
        return gdf_pgm.join(df_pg_c)

    def __init__(self):
        """Read data from views spec."""
        self.gdf_cm = views.GEOMETRIES["GeomCountry"].gdf
        self.gdf_pgm = MapData._get_gdf_pgm()
        self.df_m = MapData._get_months()

    def gdf_from_series_patch(self, s_patch: pd.Series) -> gpd.GeoDataFrame:
        """Add patch of data to plot into geopandas dataframe."""
        groupvar = s_patch.index.name
        if groupvar == "pg_id":
            gdf = self.gdf_pgm.join(s_patch, how="inner")
        elif groupvar == "country_id":
            gdf = self.gdf_cm.join(s_patch, how="inner")
        else:
            raise RuntimeError(f"Couldn't match groupvar {groupvar}.")
        return gdf


def get_bbox(gdf: gpd.GeoDataFrame, padding: float = 0.5):
    """Make bounding box from gdf."""
    min_x = gdf.bounds.minx.min() - padding
    max_x = gdf.bounds.maxx.max() + padding
    min_y = gdf.bounds.miny.min() - padding
    max_y = gdf.bounds.maxy.max() + padding
    bbox = (min_x, max_x, min_y, max_y)
    return bbox


def get_figsize(bbox: List[float], scale: float) -> Tuple[float, float]:
    """Get figsize tuple given scaler."""
    size_x = (bbox[1] - bbox[0]) * scale
    size_y = (bbox[3] - bbox[2]) * scale
    size = (size_x, size_y)
    return size


def get_fig_ax(figsize: float, bbox: List[float]) -> Any:
    """Get limited and figsized fig, ax from figsize and bbox."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim((bbox[0], bbox[1]))
    ax.set_ylim((bbox[2], bbox[3]))
    return fig, ax


def adjust_axlims(ax: Any, bbox: List[float]) -> Any:
    """ Limit axes to provided bbox """
    ax.set_xlim((bbox[0], bbox[1]))
    ax.set_ylim((bbox[2], bbox[3]))
    return ax


# pylint: disable=too-many-locals, too-many-arguments
def add_textbox_to_ax(
    fig: Any,
    ax: Any,
    text: str,
    textsize: int,
    corner: str = "lower left",
    corner_offset: float = 0.3,
    textbox_pad: float = 0.4,
) -> None:
    """Add bounded white textbox to ax, with url and logo.

    Args:
        fig: Matplotlib figure.
        ax: Matplotlib ax.
        text: Text to draw in box.
        textsize: Font size.
        corner: Location of anchored textbox.
        corner_offset: How far from each corner to draw box. Note the
            other elements (url and logo) depend on this.
        textbox_pad: Padding inside the main textbox.
    """
    # Set anchored textbox.
    text_anchor = AnchoredText(
        text,
        loc=corner,
        pad=textbox_pad,
        borderpad=corner_offset,
        prop={"fontsize": textsize - 5},
    )
    text_anchor.patch.set(alpha=0.8)
    ax.add_artist(text_anchor)

    # Once textbox is drawn up, get bbox coordinates for that.
    renderer = fig.canvas.renderer
    coords = ax.transData.inverted().transform(
        text_anchor.get_window_extent(renderer)
    )

    # Params depending on selected corner. Use the coords to inset the url.
    # [0][0]: xmin, [0][1]: ymin, [1][0]: xmax, [1][1]: ymax.
    cornerparams = {
        "lower left": {
            "xy": (coords[0][0], coords[1][1]),
            "offset": (0, 2),
            "ha": "left",
            "va": "bottom",
        },
        "lower right": {
            "xy": (coords[1][0], coords[1][1]),
            "offset": (0, 2),
            "ha": "right",
            "va": "bottom",
        },
    }

    if corner not in cornerparams:
        raise KeyError(f"{corner} is not a valid corner (yet).")

    style = dict(
        facecolor="white",
        alpha=0,
        edgecolor="red",  # For testing.
        boxstyle="square, pad=0",
    )
    text_url = ax.annotate(
        "http://views.pcr.uu.se",
        xy=cornerparams[corner]["xy"],
        xytext=cornerparams[corner]["offset"],
        textcoords="offset points",
        fontsize=textsize - 5,
        bbox=style,
        ha=cornerparams[corner]["ha"],
        va=cornerparams[corner]["va"],
    )

    # Again once textbox is drawn up, get bbox coordinates for that.
    fig.canvas.draw()
    text_bbox = text_url.get_bbox_patch()
    text_bbox = text_bbox.get_extents()
    urlcoords = ax.transData.inverted().transform(text_bbox)

    # Add the ViEWS logo.
    this_dir = os.path.dirname(__file__)
    path_logo_views = os.path.join(this_dir, "logo_transparent.png")
    logo_views = mpimg.imread(path_logo_views)

    # Define a 1st position to annotate.
    xy = (urlcoords[0][0], urlcoords[1][1])
    imagebox = OffsetImage(logo_views, zoom=0.2)
    imagebox.image.axes = ax

    ab = AnnotationBbox(
        imagebox,
        xy,
        xycoords="data",
        xybox=(55, 20),  # Arbitrary offset in points that looks ok.
        frameon=False,
        boxcoords="offset points",
    )
    ax.add_artist(ab)


def shift_colormap(
    cmap: Any,
    start: float = 0,
    midpoint: float = 0.5,
    stop: float = 1.0,
    name: str = "shiftedcmap",
) -> Any:
    """Offset the center of a colormap.

    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero
    Credit: #https://gist.github.com/phobson/7916777

    ViEWS shifted rainbow: [0, 0.25, 1]

    Args:
      cmap: The matplotlib colormap to be altered
      start: Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and 1.0.
      midpoint: The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75.
      stop: Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          0.0 and 1.0.
    """
    cdict = {"red": [], "green": [], "blue": [], "alpha": []}  # type: ignore

    # Regular index to compute the colors.
    reg_index = np.linspace(start, stop, 257)

    # Shifted index to match the data.
    shift_index = np.hstack(
        [
            np.linspace(0.0, midpoint, 128, endpoint=False),
            np.linspace(midpoint, 1.0, 129, endpoint=True),
        ]
    )

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict["red"].append((si, r, r))
        cdict["green"].append((si, g, g))
        cdict["blue"].append((si, b, b))
        cdict["alpha"].append((si, a, a))

    newcmap = colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap


def make_ticks(var_scale: str) -> Dict[str, Any]:
    """Make tick dictionary with 'values' and 'labels' depending on var_scale.

    Args:
        var_scale: "logodds" or "prob".
    Returns:
        ticks: dict with 'values' and 'labels'.
    """

    def format_prob_to_pct(p: float) -> str:
        """Cast probabilities to pct (%) formatted strings."""

        if not 0 <= p <= 1:
            raise RuntimeError("Value does not look like a probability.")

        pct = p * 100
        if pct == int(pct):
            pct = int(pct)

        return f"{pct}%"

    def make_ticks_logit() -> Tuple[List[float], List[str]]:
        """ Make logistic ticks """
        ticks_logit = []
        ticks_strings = []
        ticks = [
            0.001,
            0.002,
            0.005,
            0.01,
            0.02,
            0.05,
            0.1,
            0.2,
            0.4,
            0.6,
            0.8,
            0.9,
            0.95,
            0.99,
        ]

        for tick in ticks:
            ticks_logit.append(stats.prob_to_logodds(tick))
            ticks_strings.append(format_prob_to_pct(tick))

        # Make the lower than/equal to for 0.001.
        ticks_strings[0] = "<= " + ticks_strings[0]

        return ticks_logit, ticks_strings

    def make_ticks_probs() -> Tuple[List[float], List[str]]:
        ticks_strings = []
        ticks_probs = [
            0.001,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            0.99,
        ]

        for tick in ticks_probs:
            ticks_strings.append(str(tick))

        return ticks_probs, ticks_strings

    if var_scale == "logodds":
        values, labels = make_ticks_logit()
    elif var_scale == "prob":
        values, labels = make_ticks_probs()

    ticks = {"values": values, "labels": labels}

    return ticks


def force_alpha_cmap(cmap: str, alpha: float):
    """Force alpha channel into colors of cmap."""
    cmap_rgb = plt.get_cmap(cmap)(np.arange(256))
    for i in range(3):  # Skip the last alpha column: it makes artefacts.
        cmap_rgb[:, i] = (1 - alpha) + alpha * cmap_rgb[:, i]
    my_cmap = colors.ListedColormap(cmap_rgb, name="my_cmap")

    return my_cmap


def label_categories(
    labels: Union[List[str], Dict[int, str]], ax: Any
) -> None:
    """Replaces labels in a categorical legend."""
    legend = ax.get_legend()

    if isinstance(labels, list):
        for i, txt in enumerate(legend.texts):
            txt.set_text(labels[i])

    if isinstance(labels, dict):
        for txt in legend.texts:
            for key, value in labels.items():
                if txt.get_text() == str(key):
                    txt.set_text(value)


# pylint: disable=too-many-statements, too-many-branches
def plot_map(
    mapdata: Any,
    s_patch: pd.Series,
    t: int = None,
    run_id: str = "development",
    s_marker: pd.Series = None,
    country_id: Union[int, List[int]] = None,
    logodds: bool = True,
    categorical: bool = False,
    bbox: Union[str, List[float], Tuple[float]] = None,
    bbox_pad: Union[List[float], Tuple[float]] = None,
    cmap: Any = None,
    alpha: float = 1,
    ymin: float = None,
    ymax: float = None,
    tick_values: List[float] = None,
    tick_labels: List[str] = None,
    fig_scale: float = 0.5,
    dpi: int = 300,
    cbar_width: str = "5%",
    textsize: int = 15,
    size_cborder: float = 0.2,
    size_gridborder: float = 0.2,
    drops: List[int] = None,
    textbox: str = None,
    textbox_corner: str = "lower left",
    title: str = None,
    path: str = None,
):
    """
    Plot a map.

    Args:
        mapdata (views.apps.plot.maps.MapData) : Object with map geometries and
            month dataframe.
        s_patch: The data to plot as colored patches. Index timevar, groupvar.
        t: time index (month_id).
        run_id: String of identifier of run to put into default textbox.
            Not used if explicit textbox is provided under the textbox arg.
        s_marker: Data to plot as markers for data>0.
        country_id: List of country_id(s) to subset.
        logodds: Transform s_patch to logodds before plotting.
        categorical: Set to True if data to plot is categorical. Series can be
            either float/int or str.
        bbox: Name of bounding box in
            fancy.BBOXES if str or bounding box tuple or list of form
            (xmin, xmax, ymin, ymax).
        bbox_pad: Coordinate padding to add to bbox, per
            (xmin, xmax, ymin, ymax).
        cmap: Colormap to use. Defaults to (shifted) rainbow with either
            logodds or prob scales.
        alpha: Alpha of series to plot.
        ymin: Minimum to map the data to.
        ymax: Maximum to map the data to.
        tick_values: List of selected ticks to show in colorbar.
        tick_values: List of strings for the selected ticks.
        fig_scale: Figure size scaler.
        dpi: Dots per inch, defaults to 300. Lower to reduce file size.
        cbar_width: Percentage of figure space for colorbar ax to take.
        textsize: Base text size for all text on plot. Title is textsize +10.
        size_cborder: country border size scale.
        size_gridborder: priogrid border size scale.
        drops: groupvar index values to drop for s_patch.
        textbox: Text to add to textbox. Flexible to lines added with \n.
        title: Title to add to the figure.
        path: Destination to write figure file.
    """
    # TODO: Assuming default if cmap not provided is clunky.
    # Copy data we're going to use to plot so we don't mess with callers data.
    s_patch_t = s_patch.copy()
    if not isinstance(s_patch_t.index, pd.MultiIndex):
        raise RuntimeError(
            "Series patch needs a timevar, groupvar MultiIndex."
        )

    indices = s_patch_t.index.names
    if country_id is not None:
        if "pg_id" in indices:
            # Get pg_ids associated with country, and subset series patch.
            cpgm = mapdata.gdf_pgm
            if isinstance(country_id, List):
                cpgm = cpgm[
                    cpgm.geo_country_id.isin(country_id)
                ].index.unique()
            else:
                cpgm = cpgm.loc[
                    cpgm.geo_country_id == country_id
                ].index.unique()
            sub = s_patch_t.index.get_level_values("pg_id").isin(cpgm)
            s_patch_t = s_patch_t[sub]  # Using a boolean index for the rows.
        if "country_id" in indices:
            if isinstance(country_id, List):
                sub = s_patch_t.index.get_level_values("country_id").isin(
                    country_id
                )
            else:
                sub = (
                    s_patch_t.index.get_level_values("country_id")
                    == country_id
                )
            s_patch_t = s_patch_t[sub]

    if t is not None:
        s_patch_t = s_patch_t.loc[t]
    else:
        s_patch_t.index = s_patch_t.index.droplevel(0)

    # Log transform if requested.
    if logodds:
        # Check we aren't logoddsing non-probabilities.
        if s_patch_t.max() > 1 or s_patch_t.min() < 0:
            msg = "Data in series patch not 0 <= p <= 1 with logodds=True."
            raise RuntimeError(msg)
        s_patch_t = stats.prob_to_logodds(s_patch_t)
        # Set up default ticks for logodds.
        ticks = make_ticks("logodds")
        vmin = min(ticks["values"])
        vmax = max(ticks["values"])
        tick_values = ticks["values"]

    # Join series patch into geometry gdf.
    gdf_patch = mapdata.gdf_from_series_patch(s_patch_t)

    # Drop any ids provided in drops.
    if drops:
        gdf_patch = gdf_patch.drop(drops)

    # If no bbox in parameters get one that matches the data.
    if not bbox:
        bbox = get_bbox(gdf_patch)
    # If string, look it up in the BBOXES dictionary of bboxes.
    elif isinstance(bbox, str):
        bbox = BBOXES[bbox]
    elif isinstance(bbox, (list, tuple)):
        pass  # Keep the bbox as a list/tuple.
    else:
        raise RuntimeError(f"bbox should be list, tuple or str, bbox: {bbox}")

    # If supplied, add padding to bbox to patch.
    if bbox_pad is not None:
        bbox = [i + j for i, j in zip(bbox_pad, bbox)]  # type: ignore

    # Set up figure space.
    fig, ax = plt.subplots(figsize=(20 * fig_scale, 20 * fig_scale))
    ax = adjust_axlims(ax, bbox)  # type: ignore

    # Set up vmin, vmax.
    if not categorical:  # min/max does not apply in that case.
        vmin = s_patch_t.min() if ymin is None else ymin
        vmax = s_patch_t.max() if ymax is None else ymax
    else:
        vmin, vmax = None, None

    # Set up defaults if no cmap provided (standard ViEWS rainbow scheme).
    if not cmap:
        if logodds:
            cmap = plt.get_cmap("rainbow")
            cmap = shift_colormap(cmap, 0.0, 0.25, 1.0)
            ticks = make_ticks("logodds")
            vmin = min(ticks["values"])
            vmax = max(ticks["values"])
            tick_values = ticks["values"]
            tick_labels = ticks["labels"]
        else:
            ticks = make_ticks("prob")
            cmap = "rainbow"
            # if vmin and vmax in interval [0,1]
            if vmin >= 0 and vmax <= 1:
                vmin, vmax = 0, 1  # set color limits to 0,1
            tick_values = ticks["values"]
            tick_labels = ticks["labels"]

    # Added parameters if categorical.
    if categorical:
        legend = True
        legend_kwds = {
            'loc': 'upper left',
            'bbox_to_anchor': (1.01, 1),
            "borderaxespad": 0.2,
            "fontsize": textsize - 4,
            "frameon": False,
            "fancybox": False,
            "edgecolor": "black",
            "framealpha": 0.8,
        }  # Uncomment kwds for outside-of-frame legend.
    else:
        legend = False
        legend_kwds = {}

    # Plot.
    gdf_patch.plot(
        ax=ax,
        column=s_patch.name,
        edgecolor="black",
        linewidth=size_gridborder,
        cmap=cmap,
        alpha=alpha,
        vmin=vmin,
        vmax=vmax,
        categorical=categorical,
        legend=legend,
        legend_kwds=legend_kwds,
    )

    if not categorical:
        # Make ax for colorbar and add to canvas.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=cbar_width, pad=0.1)
        # Force alpha into colormap to avoid ugly artefacts.
        cmap = force_alpha_cmap(cmap=cmap, alpha=alpha)
        # Fill in the colorbar and adjust the ticks.
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm._A = []  # pylint: disable=protected-access
        cbar = plt.colorbar(sm, cax=cax, ticks=tick_values)
        cax.tick_params(labelsize=textsize)

    # Assume ticks are labels if only values are provided.
    if tick_labels is not None:
        if tick_values is not None:
            cbar.set_ticklabels(tick_labels)
        else:
            raise RuntimeError("Need tick values to match labels to.")

    # Remove axis ticks.
    ax.tick_params(
        top=False,
        bottom=False,
        left=False,
        right=False,
        labelleft=False,
        labelbottom=False,
    )

    # Plot country borders.
    geom_c = mapdata.gdf_cm.copy()
    if country_id:
        if isinstance(country_id, List):
            sub = geom_c.index.get_level_values("country_id").isin(country_id)
        else:
            sub = geom_c.index.get_level_values("country_id") == country_id

        geom_c = geom_c.loc[sub]
    else:
        # This if we'd want to plot only the gdf's countries' borders:
        # countries = gdf_patch.index.get_level_values('country_id').unique()
        # geom_c = geom_c.loc[countries]
        geom_c = geom_c.cx[bbox[0] : bbox[1], bbox[2] : bbox[3]]  # type: ignore

    geom_c.geom.boundary.plot(
        ax=ax,
        edgecolor="black",
        facecolor="none",
        linewidth=2.0 * size_cborder,
    )
    geom_c.geom.boundary.plot(
        ax=ax,
        edgecolor="white",
        facecolor="none",
        linewidth=0.7 * size_cborder,
    )

    # Plot markers on top of colour patches if we have any > 0.
    if s_marker is not None:
        s_marker_t = s_marker.loc[t].copy() if t else s_marker.copy()
        s_marker_t = s_marker_t[s_marker_t > 0]
        gdf_marker = mapdata.gdf_from_series_patch(s_marker_t)
        gdf_marker.centroid.plot(
            ax=ax, marker=".", markersize=100, color="black"
        )

    # Add textbox and url/logo.
    if textbox is not None:
        meta = textbox
    if t and not textbox:
        meta = f"Run: {run_id} \nName: {s_patch_t.name} \nMonth: {t}"
    if not t and not textbox:
        raise RuntimeError("Please add a textbox.")

    add_textbox_to_ax(
        fig=fig, ax=ax, text=meta, textsize=textsize, corner=textbox_corner
    )

    if title:
        ax.set_title(title, fontsize=textsize + 3, pad=15)
    else:
        if t:
            date_str = mapdata.df_m.loc[t, "date_str"]
            title = date_str
            ax.set_title(title, fontsize=textsize + 10, pad=15)

    if path:
        fig.savefig(path, bbox_inches="tight", dpi=dpi)
        log.info(f"Wrote {path}.")
        plt.close(fig)
    else:
        return fig, ax
