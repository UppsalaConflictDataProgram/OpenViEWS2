import pandas as pd  # type: ignore
from views.apps.transforms import lib


def test_onset() -> None:
    """ Test onset formulation """

    c_id = pd.Series([1, 2, 3], name="c_id")
    t = pd.Series(list(range(1, 11)), name="t")
    events_c1 = [0, 0, 0, 0, 1, 1, 1, 0, 0, 0]  # Events
    onsets_c1 = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  # Wanted onsets
    onspos_c1 = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  # Wanted onsets_possible

    events_c2 = [0, 1, 0, 1, 0, 0, 1, 0, 0, 0]
    onsets_c2 = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    onspos_c2 = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]

    events_c3 = [0, 0, 1, 1, 1, 1, 0, 0, 0, 1]
    onsets_c3 = [0, 0, 1, 0, 0, 0, 0, 0, 0, 1]
    onspos_c3 = [1, 1, 1, 0, 0, 0, 0, 0, 0, 1]

    df = (
        pd.DataFrame(
            {
                "event": events_c1 + events_c2 + events_c3,
                "wanted_onset_possible_3": onspos_c1 + onspos_c2 + onspos_c3,
                "wanted_onset_3": onsets_c1 + onsets_c2 + onsets_c3,
            },
            index=pd.MultiIndex.from_product([c_id, t]),
        )
        .swaplevel()
        .sort_index()
    )

    df["onset_possible_3"] = lib.onset_possible(s=df["event"], window=3)
    df["onset_3"] = lib.onset(s=df["event"], window=3)

    pd.testing.assert_series_equal(
        df["onset_3"], df["wanted_onset_3"], check_names=False
    )
    pd.testing.assert_series_equal(
        df["onset_possible_3"],
        df["wanted_onset_possible_3"],
        check_names=False,
    )
