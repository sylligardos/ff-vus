"""
@who: Emmanouil Sylligardos
@where: Ecole Normale Superieur (ENS), Paris, France
@when: PhD Candidate, 2nd year (2025)
@what: FF-VUS
"""

import pandas as pd
from tqdm import tqdm
import argparse

from vus.vus_numpy import VUSNumpy
from vus.vus_torch import VUSTorch
from utils.dataloader import Dataloader
from utils.scoreloader import Scoreloader
from utils.utils import load_tsb


def visualize(testing):
    load_tsb(testing=testing, dataset='KDD21', n_timeseries=10)

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='main',
        description='Compute a specific metric for a whole dataset'
    )
    parser.add_argument('--test', action='store_true', help='Run in testing mode (limits the data for fast testing)')
    args = parser.parse_args()


    visualize()