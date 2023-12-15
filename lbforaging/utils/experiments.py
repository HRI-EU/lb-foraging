#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Utility functions for running experiments.
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
import logging
import copy

import numpy as np
from scipy.spatial import distance


def create_environment_id(settings, grid_obs=False):
    """Create environment ID based on environment settings"""
    sight = (
        ""
        if settings.environment["sight"] == settings.environment["size"]
        else f"-{settings.environment['sight']}s"
    )
    return "Foraging{6}{5}-{0}x{0}-{1}p-{2}f-{3}coop-{4}dist-v2".format(
        settings.environment["size"],
        settings.environment["nplayers"],
        settings.environment["nfood"],
        float(settings.environment["coop"]),
        float(settings.environment["distractors"]),
        sight,
        "-grid" if grid_obs else "",  # not sure what this option does
    )


def distance_to_closest_food(
    player, field, metric="euclidean", max_food_level=None
):
    """Calculate distance to closest food.

    Calculate distance to food closest to agent's position or start position.
    Identification may be limited to foods of a maximum level (e.g., one
    that is compatible with current agent).

    Note,  function calculates distance irrespective of the agent's sight,
    i.e., irrespective of the agent 'sees' the food.

    Parameters
    ----------
    agent : lbforaging.foraging.environment.Player instance
        Player instance in current environment
    field : numpy array
        State of the field in current environment
        lbforaging.foraging.environment.ForagingEnv
    norm : str, optional
        Distance metric to use, can be 'euclidean' or 'cityblock', by
        default 'euclidean'
    max_food_level : int, optional
        Maximum food level possible, if None, foods of all levels are
        considered, by default None

    Returns
    -------
    float
        distance to food
    """
    rows, cols = np.where(field > 0)
    dist = []
    for row, col in zip(rows, cols):
        dist.append(
            _calculate_distance(
                np.array(player.position), np.array([row, col]), metric
            )
        )
    return np.min(dist)


def distance_to_closest_agent(player_id, players, metric="euclidean"):
    """Calculate distance to closest player.

    Calculate distance to player closest to own position. Identification may be
    limited to agents of a minimum level (e.g., one that is required to pick up
    a certain food).

    Note,  function calculates distance irrespective of the agent's sight,
    i.e., irrespective of the agent 'sees' the closest agent.

    Parameters
    ----------
    player_id : int
        ID of reference player
    players : list of lbforaging.foraging.environment.Player instances
        Player instances in current environment
    metric : str, optional
        Distance metric to use, can be 'euclidean' or 'cityblock', by
        default 'euclidean'

    Returns
    -------
    float
        distance to agent
    """
    reference_player = copy.copy(players[player_id])
    other_players = copy.copy(players)
    del other_players[player_id]
    dist = []
    for player in other_players:
        dist.append(
            _calculate_distance(
                np.array(reference_player.position),
                np.array(player.position),
                metric,
            )
        )
    return np.min(dist)


def _calculate_distance(X, Y, metric="euclidean"):
    if X.shape != Y.shape:
        logging.info(  # pylint: disable=W1201,C0209
            "Shape X: %s, Shape Y: %s"  # pylint: disable=C0209
            % (X.shape, Y.shape)
        )
        raise RuntimeError("Position arrays X and Y must have same shape")
    if metric == "euclidean":
        return np.linalg.norm(X - Y)
    elif metric == "cityblock":
        if X.ndim < 2:
            X = np.expand_dims(X.copy(), axis=0)
            Y = np.expand_dims(Y.copy(), axis=0)
        return np.squeeze(distance.cdist(X, Y, metric="cityblock"))
    else:
        raise RuntimeError(  # pylint: disable=W1201
            "Unknown metric %s" % metric  # pylint: disable=C0209
        )
