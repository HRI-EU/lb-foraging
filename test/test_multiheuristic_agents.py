#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Run tests for multiheuristic agents.
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

import numpy as np
import pandas as pd

from lbforaging.utils.io import read_settings

sys.path.insert(0, "../experiments")
from run_experiments import run_experiment  # pylint: disable=E0401,C0413,C0411


def _assert_goal_value_is_nan(game_data):
    # Goal value is not calculated for heuristics other than Coop and Adapt.
    for col in ["goal_value_ego", "goal_value_other", "goal_value_together"]:
        assert game_data[col].isna().all()


def test_baseline():
    """Test behavior of BASELINE agent on fixed environment."""
    settings = read_settings("test_settings.yml")
    Path(settings["outpath"]).mkdir(parents=True, exist_ok=True)
    settings["seed"] = 741
    settings["environment"]["coop"] = 1.0
    settings.agents["heuristic"] = "MultiHeuristicAgent"
    settings.agents["abilities"] = "BASELINE"

    print(settings)

    run_experiment(settings)
    game_data = pd.read_csv(
        Path(settings["outpath"]).joinpath(
            "Foraging-25x25-2p-10f-"
            + f"{settings['environment']['coop']}coop-{settings['environment']['distractors']}"
            + "dist-v2_game_data_trial_00.csv"
        )
    )

    _assert_goal_value_is_nan(game_data)

    # With the current seed, the baseline agent should not collect a food item
    # for c=1.0 (this is possible by chance), and should not perform a cooperative
    # action.
    for col in ["food", "food_type", "food_sum", "cooperative_actions"]:
        assert (game_data[col] == 0).all()


def test_egoistic():
    """Test behavior of EGOISTIC agent on fixed environment."""
    settings = read_settings("test_settings.yml")
    Path(settings["outpath"]).mkdir(parents=True, exist_ok=True)
    settings["seed"] = 741
    settings["environment"]["coop"] = 1.0
    settings.agents["heuristic"] = "MultiHeuristicAgent"
    settings.agents["abilities"] = "EGOISTIC"

    run_experiment(settings)
    game_data = pd.read_csv(
        Path(settings["outpath"]).joinpath(
            "Foraging-25x25-2p-10f-"
            + f"{settings['environment']['coop']}coop-{settings['environment']['distractors']}"
            + "dist-v2_game_data_trial_00.csv"
        )
    )

    _assert_goal_value_is_nan(game_data)

    # With the current seed, the baseline agent should not collect a food item
    # for c=1.0 (this is possible by chance), and should not perform a cooperative
    # action.
    for col in ["food", "food_type", "food_sum", "cooperative_actions"]:
        assert (game_data[col] == 0).all()

    # Test environment with no cooperative requirements.
    settings["environment"]["coop"] = 0.0
    run_experiment(settings)
    game_data = pd.read_csv(
        Path(settings["outpath"]).joinpath(
            "Foraging-25x25-2p-10f-"
            f"{settings['environment']['coop']}coop-{settings['environment']['distractors']}"
            + "dist-v2_game_data_trial_00.csv"
        )
    )

    # Given that we specified no cooperative food items, we expect only items that can
    # be collected by each agent.
    expected_n_food_types = 1
    expected_food_type = np.min(settings["agents"]["levels"])
    collected_food_types = np.unique(
        game_data["food_type"][game_data["food_type"] > 0]
    )
    assert len(collected_food_types) == expected_n_food_types
    assert collected_food_types == expected_food_type

    # Assert that agents collect food items given the specified environment settings.
    for agent_id in [0, 1]:
        assert (
            game_data["food_sum"][game_data["agent_id"] == agent_id].to_numpy()[
                -1
            ]
            > 0
        )
        assert (
            game_data["food_sum"][game_data["agent_id"] == agent_id].to_numpy()[
                -1
            ]
            == game_data["food"][game_data["agent_id"] == agent_id].sum()
        )

    # For this seed, we expect no cooperation from the ego agent.
    assert (game_data["cooperative_actions"] == 0).all()


def test_social1():
    """Test behavior of SOCIAL1 agent on fixed environment."""
    settings = read_settings("test_settings.yml")
    Path(settings["outpath"]).mkdir(parents=True, exist_ok=True)
    settings["seed"] = 741
    settings["environment"]["coop"] = 1.0
    settings.agents["heuristic"] = "MultiHeuristicAgent"
    settings.agents["abilities"] = "SOCIAL1"

    run_experiment(settings)
    game_data = pd.read_csv(
        Path(settings["outpath"]).joinpath(
            "Foraging-25x25-2p-10f-"
            + f"{settings['environment']['coop']}coop-{settings['environment']['distractors']}"
            + "dist-v2_game_data_trial_00.csv"
        )
    )

    # Assert cooperative actions were performed and the food type is logged correctly.
    expected_coop_food_type = np.sum(settings["agents"]["levels"])
    assert game_data["cooperative_actions"].sum() > 0
    assert (
        game_data["food_type"][game_data["cooperative_actions"] > 0]
        == expected_coop_food_type
    ).all()
    _assert_goal_value_is_nan(game_data)

    settings["environment"]["coop"] = 0.5
    run_experiment(settings)
    game_data = pd.read_csv(
        Path(settings["outpath"]).joinpath(
            "Foraging-25x25-2p-10f-"
            + f"{settings['environment']['coop']}coop-{settings['environment']['distractors']}"
            + "dist-v2_game_data_trial_00.csv"
        )
    )

    # Assert cooperative and individual actions were performed and the food
    # type is logged correctly.
    expected_coop_food_type = np.sum(settings["agents"]["levels"])
    collected_food_types = np.unique(
        game_data["food_type"][game_data["food_type"] > 0]
    )
    n_individual_actions = np.logical_and(
        game_data["food"] > 0, game_data["cooperative_actions"] == 0
    ).sum()
    assert len(collected_food_types) == 2
    assert game_data["cooperative_actions"].sum() > 0
    assert n_individual_actions > 0

    _assert_goal_value_is_nan(game_data)


def test_cooperative():
    """Test behavior of COOPERATIVE agent on fixed environment."""
    settings = read_settings("test_settings.yml")
    Path(settings["outpath"]).mkdir(parents=True, exist_ok=True)
    settings["seed"] = 741
    settings["environment"]["coop"] = 1.0
    settings.agents["heuristic"] = "MultiHeuristicAgent"
    settings.agents["abilities"] = "COOPERATIVE"

    run_experiment(settings)
    game_data = pd.read_csv(
        Path(settings["outpath"]).joinpath(
            "Foraging-25x25-2p-10f-"
            + f"{settings['environment']['coop']}coop-{settings['environment']['distractors']}"
            + "dist-v2_game_data_trial_00.csv"
        )
    )
    # Assert cooperative actions were performed and the food type is logged correctly.
    expected_coop_food_type = np.sum(settings["agents"]["levels"])
    assert game_data["cooperative_actions"].sum() > 0
    assert (
        game_data["food_type"][game_data["cooperative_actions"] > 0]
        == expected_coop_food_type
    ).all()

    # Assert that for targeted steps, a goal value was computed.
    for col in ["goal_value_ego", "goal_value_other", "goal_value_together"]:
        assert not game_data[col].isna().all()


def test_adaptive():
    """Test behavior of ADAPTIVE agent on fixed environment."""
    settings = read_settings("test_settings.yml")
    Path(settings["outpath"]).mkdir(parents=True, exist_ok=True)
    settings["seed"] = 741
    settings["environment"]["coop"] = 1.0
    settings.agents["heuristic"] = "MultiHeuristicAgent"
    settings.agents["abilities"] = "ADAPTIVE"

    run_experiment(settings)
    game_data = pd.read_csv(
        Path(settings["outpath"]).joinpath(
            "Foraging-25x25-2p-10f-"
            + f"{settings['environment']['coop']}coop-{settings['environment']['distractors']}"
            + "dist-v2_game_data_trial_00.csv"
        )
    )
    # Assert cooperative actions were performed and the food type is logged correctly.
    expected_coop_food_type = np.sum(settings["agents"]["levels"])
    assert game_data["cooperative_actions"].sum() > 0
    assert (
        game_data["food_type"][game_data["cooperative_actions"] > 0]
        == expected_coop_food_type
    ).all()

    # Assert that for targeted steps, a goal value was computed.
    for col in ["goal_value_ego", "goal_value_other", "goal_value_together"]:
        assert not game_data[col].isna().all()

    # Test behavior in mixed/partially cooperative environment.
    settings["environment"]["coop"] = 0.3
    run_experiment(settings)
    game_data = pd.read_csv(
        Path(settings["outpath"]).joinpath(
            "Foraging-25x25-2p-10f-"
            + f"{settings['environment']['coop']}coop-{settings['environment']['distractors']}"
            + "dist-v2_game_data_trial_00.csv"
        )
    )
    # Assert cooperative and individual actions were performed and the food
    # type is logged correctly.
    expected_coop_food_type = np.sum(settings["agents"]["levels"])
    collected_food_types = np.unique(
        game_data["food_type"][game_data["food_type"] > 0]
    )
    n_individual_actions = np.logical_and(
        game_data["food"] > 0, game_data["cooperative_actions"] == 0
    ).sum()
    assert len(collected_food_types) == 2
    assert game_data["cooperative_actions"].sum() > 0
    assert n_individual_actions > 0


if __name__ == "__main__":
    test_baseline()
    test_egoistic()
    test_social1()
    test_cooperative()
    test_adaptive()
