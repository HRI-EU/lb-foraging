#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Test heuristic agents for LB-Foraging.
# Each agent has a parameter that specifies the radius of its sight
#     H1 always goes to the closest visible food.
#     H2 goes to the one visible food which is closest to the center of all visible players.
#     H3 always goes to the closest visible food with compatible level (i.e. it can load it).
#     H4 goes to the one visible food which is closest to all visible players such that the sum of their and H4's level is sufficient to load the food.
# H1-4 try to load the food once they are next to it. If they do not see a food, they go into a random direction.
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
import pytest
import numpy as np

import gym
from lbforaging.agents.heuristic_agent import H1, H2, H3, H4, H5, H6
from lbforaging.utils.experiments import _calculate_distance

import register_environments

HEURISTICS = [H1, H2, H3, H4, H5, H6]


def test_heuristics():
    """Test initialization of agent heuristics H1-H6"""
    id_env = "Foraging-8x8-2p-3f-v2"
    for Heuristic in HEURISTICS:
        env = gym.make(id_env)
        _, nobs = env.reset()
        agents = [
            Heuristic(player, i) for i, player in enumerate(nobs[0].players)
        ]

        # Run environment for three steps for each heuristic
        for _ in range(3):
            actions = []
            for i, agent in enumerate(agents):
                actions.append(agent.step(nobs[i]))
            _, nobs, _, _, _ = env.step(actions)


def test_heuristics_sight1():
    """Test reduced agent sight, s=1."""
    id_env = "Foraging-8x8-2p-3f-v2"
    for Heuristic in HEURISTICS:
        env = gym.make(id_env, sight=1)
        _, nobs = env.reset()
        agents = [
            Heuristic(player, i) for i, player in enumerate(nobs[0].players)
        ]

        # Run environment for three steps for each heuristic
        for _ in range(3):
            actions = []
            for i, agent in enumerate(agents):
                actions.append(agent.step(nobs[i]))
            _, nobs, _, _, _ = env.step(actions)


def test_heuristics_sight0():
    """Test no agent sight, s=0."""
    id_env = "Foraging-8x8-2p-3f-v2"
    for Heuristic in HEURISTICS:
        env = gym.make(id_env, sight=0)
        _, nobs = env.reset()
        agents = [
            Heuristic(player, i) for i, player in enumerate(nobs[0].players)
        ]

        # Run environment for three steps for each heuristic
        for _ in range(3):
            actions = []
            for i, agent in enumerate(agents):
                actions.append(agent.step(nobs[i]))
            _, nobs, _, _, _ = env.step(actions)


def test_distance_metrics():
    """Test calculation of distance between agent and food item."""
    id_env = "Foraging-8x8-2p-3f-coop-v2"
    env = gym.make(id_env)
    env.reset()
    dist_euclid = _calculate_distance(
        np.array([2, 3, -1]), np.array([4, 1, -2]), metric="euclidean"
    )
    assert dist_euclid == 3, "Euclidean distance not calculated correctly"
    dist_cityblock = _calculate_distance(
        np.array([0, 0]), np.array([5, 5]), metric="cityblock"
    )
    assert dist_cityblock == 10, "Cityblock distance not calculated correctly"

    with pytest.raises(RuntimeError):
        _calculate_distance(np.arange(5), np.arange(4))


if __name__ == "__main__":
    test_distance_metrics()
