""" Command line interface for model training """
from typing import Tuple
from typing_extensions import Literal
import argparse
import logging

from views.apps.pipeline import train
from views.config import LOGFMT
from views.utils.log import get_log_path, logtime

logging.basicConfig(
    level=logging.DEBUG,
    format=LOGFMT,
    handlers=[
        logging.FileHandler(get_log_path(__file__)),
        logging.StreamHandler(),
    ],
)


def parse_args() -> Tuple[str, str, str]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--loa", type=str, help="Level of analysis, either cm or pgm"
    )
    parser.add_argument("--model", type=str, help="name of model to train")
    parser.add_argument("--dataset", type=str, help="name of dataset")

    args = parser.parse_args()

    assert args.loa in ["am", "cm", "pgm"]
    loa: Literal["am", "cm", "pgm"] = args.loa
    model: str = args.model
    dataset = args.dataset

    return loa, model, dataset


@logtime
def main():
    loa, model, dataset = parse_args()
    train.train_and_store_model_by_name(loa, model, dataset)


if __name__ == "__main__":
    main()
