#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Run tests for LBF experiment settings.
#
# The MIT License (MIT)
#
# Copyright © 2023 Honda Research Institute Europe GmbH
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the 'Software'), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#
import sys
from pathlib import Path

import pytest
import numpy as np
import pandas as pd

from lbforaging.utils.io import read_settings

sys.path.insert(0, "../experiments")
from run_experiments import run_experiment  # pylint: disable=E0401,C0413,C0411


def test_distractors():
    """Test behavior of BASELINE agent on fixed environment."""
    settings = read_settings("./test_settings.yml")
    Path(settings["outpath"]).mkdir(parents=True, exist_ok=True)
    settings["seed"] = 741
    settings["environment"]["coop"] = 0.0
    settings["environment"]["distractors"] = 1.0
    settings.agents["heuristic"] = "MultiHeuristicAgent"
    settings.agents["abilities"] = "COOPERATIVE"

    run_experiment(settings)
    game_data = pd.read_csv(
        Path(settings["outpath"]).joinpath(
            f"Foraging-25x25-2p-10f-{settings['environment']['coop']}coop-{settings['environment']['distractors']}dist-v2_game_data_trial_00.csv"
        )
    )
    assert game_data["food"].sum() == 0

    # The sum of the fraction of distractors and fraction of cooperative items can not
    # exceed 1.0.
    settings["environment"]["coop"] = 1.0
    settings["environment"]["distractors"] = 1.0
    with pytest.raises(RuntimeError):
        run_experiment(settings)

    # Introducing distractors should reduce the success of cooperative agents.
    settings["environment"]["coop"] = 0.2
    settings["environment"]["distractors"] = 0.8
    run_experiment(settings)
    game_data_distractors = pd.read_csv(
        Path(settings["outpath"]).joinpath(
            f"Foraging-25x25-2p-10f-{settings['environment']['coop']}coop-{settings['environment']['distractors']}dist-v2_game_data_trial_00.csv"
        )
    )
    settings["environment"]["coop"] = 0.5
    settings["environment"]["distractors"] = 0.0
    run_experiment(settings)
    game_data = pd.read_csv(
        Path(settings["outpath"]).joinpath(
            f"Foraging-25x25-2p-10f-{settings['environment']['coop']}coop-{settings['environment']['distractors']}dist-v2_game_data_trial_00.csv"
        )
    )
    assert game_data_distractors["food"].sum() < game_data["food"].sum()

    # Introducing distractors should reduce the success of social agents.
    settings.agents["abilities"] = "SOCIAL1"
    settings["environment"]["coop"] = 0.5
    settings["environment"]["distractors"] = 0.5
    run_experiment(settings)
    game_data_distractors = pd.read_csv(
        Path(settings["outpath"]).joinpath(
            f"Foraging-25x25-2p-10f-{settings['environment']['coop']}coop-{settings['environment']['distractors']}dist-v2_game_data_trial_00.csv"
        )
    )

    settings["environment"]["coop"] = 1.0
    settings["environment"]["distractors"] = 0.0
    run_experiment(settings)
    game_data = pd.read_csv(
        Path(settings["outpath"]).joinpath(
            f"Foraging-25x25-2p-10f-{settings['environment']['coop']}coop-{settings['environment']['distractors']}dist-v2_game_data_trial_00.csv"
        )
    )
    assert game_data_distractors["food"].sum() < game_data["food"].sum()


if __name__ == "__main__":
    test_distractors()
