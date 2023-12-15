#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Collection of ability configurations for instances of the MultiHeuristicAgent
# class. Configurations are available both as individual dictionaries and as
# part of a # dictionary for convenience purposes.
#
# Usage:
# >>> from mh_agent_configurations import ability_sets
# >>> agent = MultiHeuristicAgent(player, player_id, abilities=ability_sets['Opportunistic1']
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
from copy import deepcopy

# own
from lbforaging.agents import mh_agent_helper as mhah

# --------------------------------------------------------------------------- #
# build base ability dictionary
abilities = mhah.build_abilities_dict()

# Ability definitions for different classes of agents
# all definitions are stored in a dictionary for easy string-based access
# (better compatibility with settings.yml usage)
ability_sets = {}

# baseline agent, no abilities except action (not explicitly defined)
# dictionary requires deepcopy for not affecting the states of its "ancestors"
BASELINE = deepcopy(abilities)
ability_sets["BASELINE"] = BASELINE

# ------------------------------------- #
# Exploration with memory
H0 = deepcopy(abilities)
H0["own"]["location_history"] = True
H0["own"]["location"] = True
H0["own"]["understands_action"] = True
H0["own"]["remembers_action"] = True
ability_sets["H0"] = H0

# ------------------------------------- #
# EGOISTIC (former H1) agent, always goes to closest visible food
# classification:
EGOISTIC = deepcopy(abilities)
EGOISTIC["goal"]["location"] = True
EGOISTIC["goal"]["focus"] = True
EGOISTIC["goal"]["level"] = True
EGOISTIC["own"]["level"] = True
EGOISTIC["own"]["location"] = True
EGOISTIC["own"]["location_history"] = True  # added 04-26
EGOISTIC["own"]["understands_action"] = True
EGOISTIC["own"]["remembers_action"] = True  # added 04-26
ability_sets["EGOISTIC"] = EGOISTIC

# ------------------------------------- #
# SOCIAL1 (former H2) agent, goes to the visible food that is closest to the
# center of all visible players
SOCIAL1 = deepcopy(
    EGOISTIC
)  # H1 can do everything H1 can do and is aware of others
SOCIAL1["other"]["location"] = True
SOCIAL1["goal"]["level"] = False
ability_sets["SOCIAL1"] = SOCIAL1

# ------------------------------------- #
# SOCIAL2 (former H5) agent, goes to the food that is closest to all visible
# players such that the sum of all agent's levels is sufficient to load the
# food
SOCIAL2 = deepcopy(SOCIAL1)  # SOCIAL2 can do everything SOCIAL1 can do
SOCIAL2["goal"]["level"] = True
ability_sets["SOCIAL2"] = SOCIAL2


# ------------------------------------- #
# H6 agent, egoistic agent that first targets goals it can load alone, only if
# no suitable goals are left it starts to act cooperatively (via SOCIAL2)
H6 = deepcopy(SOCIAL2)
H6["own"]["remembers_action"] = True
ability_sets["H6"] = H6

# ------------------------------------- #
# Goes to the one visible food which is targeted by any other player such
# that the sum of their levels is sufficient to load the food
# = coming for help
COOPERATIVE = deepcopy(SOCIAL2)
COOPERATIVE["other"]["goal"]["location"] = True
ability_sets["COOPERATIVE"] = COOPERATIVE

# ------------------------------------- #
# Goes to the one visible food which is targeted by any other player such
# that the sum of their levels is sufficient to load the food
# = coming for help
ADAPTIVE = deepcopy(SOCIAL2)
ability_sets["ADAPTIVE"] = ADAPTIVE

# ------------------------------------- #
# Goes to the closest visible food with compatible level
# that is approached by another agent („steals food“)
COMPETITIVE_EGOISTIC = deepcopy(SOCIAL2)
ability_sets["COMPETITIVE_EGOISTIC"] = COMPETITIVE_EGOISTIC
