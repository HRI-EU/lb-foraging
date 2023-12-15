#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Independent helper functions for the MultiHeuristicAgent class
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
import numpy as np
from scipy.spatial import distance

# ----------------------------------------------------------------------------------- #
# ability definition and checking


def build_abilities_dict():
    """
    Construct a dictionary through which the abilities of
    a MultiHeuristicAgent can be specified.
    """
    abilities = {}
    for lv1key in ["goal", "own", "other"]:
        abilities[lv1key] = {}
        if lv1key == "goal":
            lv2keys = [
                "location",
                "location_history",
                "distance",
                "level",
                "focus",
            ]
        elif lv1key == "own":
            lv2keys = [
                "location",
                "location_history",
                "level",
                "understands_action",
                "remembers_action",
                "next_action",
                "next_location",
                "competitive",
                "opportunistic",
                "selfless",
            ]
        elif lv1key == "other":
            lv2keys = [
                "location",
                "level",
                "goal",
                "fairness",
                "next_action",
                "next_location",
            ]
        for lv2key in lv2keys:
            if (lv1key == "other") and (lv2key == "goal"):
                lv3keys = ["location", "distance", "level"]
                abilities[lv1key][lv2key] = {}
                for lv3key in lv3keys:
                    abilities[lv1key][lv2key][lv3key] = False
            else:
                abilities[lv1key][lv2key] = False
    return abilities


def same_structure(d1, d2, level=0, order=0):
    """
    Checks if two dictionaries of arbitrary depth have the same keys and types
    of data

    Attributes
    ----------
    d1 : dict
        first dictionary of interest
    d2 : dict
        second dictionary which should have the same structure as d1 for
        returning True
    level: int
        current level that is being investigated in case of nested dicts
        (starts from 0).
        This is only used internally for recursive valls.
    order: bool
        order in which dictionaries should be compared.
        This is only used internally for recursive calls. Should be 0 for
        initial call.
    """
    for key in d1:
        # first check for keys that are in d1 but not in d2
        if key in d2:
            if type(d1[key]) != type(d2[key]):
                logging.info(  # pylint: disable=W1201
                    "Different types for key %s at level %d"  # pylint: disable=C0209
                    % (key, level)
                )
                return False
            elif isinstance(d1[key], dict) and isinstance(d2[key], dict):
                # recursive check of subdicts
                if not same_structure(d1[key], d2[key], level=level + 1):
                    print(
                        'different structures for key "{}" at level {}'.format(
                            key, level
                        )
                    )
                    return False
        else:
            return False
    # finally compare the inversion to get keys that are in d2 but not in d1
    if order == 0:
        if not same_structure(d2, d1, level=level, order=1):
            return False
    return True


def verify_abilities_dict(abilities):
    """
    Check if a given abilities dictionary has the correct structure
    Wrapper for same_structure().

    Attributes
    ----------
    abilities : dict
        dictionary that should have the structure specified by
        build_abilities_dict()
    """
    return same_structure(abilities, build_abilities_dict())


# --------------------------------------------------------------------------- #
# functions that were previously part of heuristic agent externalized because
# they are universally applicable and require no self references


def shuffle_actions(allowed):
    """Create list of shuffled actions for unbiased action selection"""
    actions = list(range(len(allowed)))
    np.random.shuffle(actions)
    return actions


def calculate_distance(X, Y, metric="euclidean"):
    """Taken from agent.py for potential adaptation"""
    if X.shape != Y.shape:
        logging.info(  # pylint: disable=W1201
            "Shape X: %s, Shape Y: %s"
            % (X.shape, Y.shape)  # pylint: disable=C0209
        )
        raise RuntimeError("Position arrays X and Y must have same shape")
    if metric == "euclidean":
        return np.linalg.norm(X - Y)
    if metric == "cityblock":
        if X.ndim < 2:
            X = np.expand_dims(X.copy(), axis=0)
            Y = np.expand_dims(Y.copy(), axis=0)
        return np.squeeze(distance.cdist(X, Y, metric="cityblock"))
    raise RuntimeError("Unknown metric")


def center_of_players(players):
    """
    Return location that lies between a list of players (centroid)
    Taken from HeuristicAgent class

    Optional ToDo: create a cooperative variant by expanding to a center
    of mass in which an agent's normalized inverse accumulated reward is taken
    as that agent's weight.
    """
    coords = np.array([player.position for player in players])
    return np.rint(coords.mean(axis=0))


def center_of_locations(locations):
    """given a list of locations, calculate the centroid"""
    return np.rint(np.array(locations).mean(axis=0))


def intheway(loc_ego, loc_other, loc_target):
    """True if two agents potentially block one another, otherwise False"""
    other_next_to_ego = (
        calculate_distance(
            X=np.array(loc_ego), Y=np.array(loc_other), metric="cityblock"
        )
        == 1
    )
    other_diagonal_to_target = (
        round(
            calculate_distance(
                X=np.array(loc_other),
                Y=np.array(loc_target),
                metric="euclidean",
            ),
            2,
        )
        == 1.41
    )
    return other_next_to_ego and not other_diagonal_to_target
