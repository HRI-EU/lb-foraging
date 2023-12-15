#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Heuristic agent with different behavior options based on available abilities.
# Agents of this class can perform in different ways depending on the
# capabilities and traits they are initialized with.
# For more details on these traits see mh_agent_configurations.py and
# mh_agent_helper.py
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
import sys
from collections import deque

import numpy as np

from lbforaging.foraging.agent import Agent
from lbforaging.foraging.environment import Action
from lbforaging.agents import mh_agent_helper as mhah

# --------------------------------------------------------------------------- #


def goal_value(start_pos, goal_pos, goal_level):
    """Calculate goal value for a given target relative to own position.

    Parameters
    ----------
    start_pos : iterable
        Start position, x-y-coordinates
    goal_pos : iterable
        Target position, x-y-coordinates
    goal_level : int
        Target level

    Returns
    -------
    float
        goal value
    """
    goal_distance = mhah.calculate_distance(
        X=np.array(start_pos), Y=np.array(goal_pos), metric="cityblock"
    )
    return (1.0 / goal_distance) * goal_level


def joint_goal_value(start_pos_ego, start_pos_other, goal_pos, goal_level):
    """Calculate joint goal value for two agents' goals, assume shared reward

    Parameters
    ----------
    start_pos_ego : iterable
        Start position ego agent, x-y-coordinates
    start_pos_other : iterable
        Start position other agent, x-y-coordinates
    goal_pos : iterable
        Target position, x-y-coordinates
    goal_level : int
        Target level

    Returns
    -------
    float
        joint goal value
    """
    goal_distance_ego = mhah.calculate_distance(
        X=np.array(start_pos_ego), Y=np.array(goal_pos), metric="cityblock"
    )
    goal_value_ego = (1.0 / goal_distance_ego) * (goal_level / 2.0)
    goal_distance_other = mhah.calculate_distance(
        X=np.array(start_pos_other), Y=np.array(goal_pos), metric="cityblock"
    )
    goal_value_other = (1.0 / goal_distance_other) * (goal_level / 2.0)

    if goal_value_ego >= goal_value_other:
        return goal_value_other
    return goal_value_ego  # goal_value_ego < goal_value_other


class MultiHeuristicAgent(Agent):
    """Agent heuristic which can perform in different ways depending on
    the selected set of available capabilities/cooperation requirements.

    Attributes
    ----------
    player : int
        Player ID
    level : int
        Player level
    history : list of int
        Past actions

    Methods
    -------
    step(obs):
        Generate next step based on heuristic

    """

    name = "MultiHeuristicAgent"

    def __init__(
        self,
        player,
        player_id,
        location_memory_size=10,
        patience=6,
        strategy=None,
        abilities=None,
    ):
        # initialize agent instance
        super(MultiHeuristicAgent, self).__init__(player, player_id)
        # The MultiHeuristicAgent has additional attributes
        self.goal_location = (0, 0)
        self.observed_position = (0, 0)
        self.location_history = deque(maxlen=location_memory_size)
        self.action_history = deque(maxlen=location_memory_size)
        self.patience = patience
        self.forbidden_action = None
        self.attempted_load = 0
        self.fairness = 0.5  # capped at 0 and 1
        self.fairness_slope = 0.05
        self.focus = 4  # how long to (at least) focus on a goal
        self.focused = 0
        if strategy is not None:
            self.strategy = strategy.lower()
        else:
            self.strategy = strategy
        self.set_abilities(abilities=abilities)
        logging.info(
            "Setup Agent as MultiHeuristicAgent %s with level %s and strategy %s",
            self.player,
            self.level,
            self.strategy,
        )

    def set_abilities(self, abilities=None):
        """Determine abilities of the agent.

        Abilities can be specified as a dictionary with one boolean value for
        each dict entry. Dictionaries must follow the structure defined by
        mhah.build_abilities_dict().

        Attributes
        ----------
        abilities : dict or None
            dictionary of booleans through which the agent can be configured
            if abilities is not specified, the dictionary is initialized
            with default values
        """
        if abilities is not None:
            logging.info("Agent abilities: %s", abilities)
            if mhah.verify_abilities_dict(abilities):
                self.abilities = abilities
            else:
                raise RuntimeError("Invalid specification of abilities.")
        else:
            logging.warning(
                "No abilities specified, applying default abilities (None)"
            )
            self.abilities = mhah.build_abilities_dict()

    def _consumable(self, obs, coords, max_food_level=1):
        """Check if food item at location can be consumed

        Given food coordinates, check if food at that location is consumable.
        If an item is consumable is defined by its presence and level at "coords".

        Attributes
        ----------
        obs : Observation instance
            Observed state of environment lbforaging.foraging.ForagingEnv
        coords : tuple or list
            coordinates of location that should be checked
        max_food_level : int
            maximum level of food at coords to be considered consumable

        Returns
        -------
        bool
            True for consumable fields, False otherwise
        """
        if max_food_level is None:
            # assume maximum ability if unknown
            max_food_level = 1
        if coords is None:
            return False
        try:
            return obs.field[coords] <= max_food_level and obs.field[coords] > 0
        except IndexError:
            logging.info("coordinates out of bounds")
            return False

    def _all_goals(self, obs, max_food_level=None, ignore_locs=None):
        """Return all goals with given food level."""
        if max_food_level == 0:
            max_food_level = None
        field = np.copy(obs.field)
        # if the own level is supposed to be used as reference point
        # (implied by missing start value), check if a given level can be considered
        if not self.abilities["own"]["level"] and max_food_level is not None:
            max_food_level = None
        # if max_food_level:
        # if the abilities to see goal levels exist and a maximum level has
        # been specified consider that maximum level
        if max_food_level and self.abilities["goal"]["level"]:
            field[field > max_food_level] = 0
        # remove forbidden locations
        if ignore_locs is not None:
            # print('ignoring locations ', ignore_locs)
            for location in ignore_locs:
                field[location] = 0
        return np.nonzero(field)

    def _compute_cf_based_goals(
        self, obs, level_ego, level_other, start_ego, start_other, ignore_locs
    ):
        """Compute optimal goal for both, ego, and other based on goal value."""

        all_food_items = self._all_goals(
            obs, max_food_level=None, ignore_locs=ignore_locs
        )
        cf_best_for_ego = {
            "value": 0,
            "location": None,
            "level": None,
        }
        cf_best_for_other = {
            "value": 0,
            "location": None,
            "level": None,
        }
        cf_best_for_together = {
            "value": 0,
            "location": None,
            "level": None,
        }

        # Check for each food item on the field if it is a better choice for
        # ego, other, or both agents, based on goal value and food level
        # compatibility.
        goal_found = False
        for idx, i in enumerate(all_food_items[0]):
            j = all_food_items[1][idx]
            current_food_level = obs.field[i, j]
            current_food_position = [i, j]

            goal_value_ego = goal_value(
                start_ego, current_food_position, current_food_level
            )
            goal_value_other = goal_value(
                start_other, current_food_position, current_food_level
            )
            summed_goal_value = joint_goal_value(
                start_ego,
                start_other,
                current_food_position,
                current_food_level,
            )

            if (
                goal_value_ego > cf_best_for_ego["value"]
                and current_food_level <= level_ego
            ):
                cf_best_for_ego["value"] = goal_value_ego
                cf_best_for_ego["location"] = tuple(current_food_position)
                cf_best_for_ego["level"] = current_food_level
                goal_found = True

            if (
                goal_value_other > cf_best_for_other["value"]
                and current_food_level <= level_other
            ):
                cf_best_for_other["value"] = goal_value_other
                cf_best_for_other["location"] = tuple(current_food_position)
                cf_best_for_other["level"] = current_food_level
                goal_found = True

            if summed_goal_value > cf_best_for_together[
                "value"
            ] and current_food_level <= (level_other + level_ego):
                cf_best_for_together["value"] = summed_goal_value
                cf_best_for_together["location"] = tuple(current_food_position)
                cf_best_for_together["level"] = current_food_level
                goal_found = True

        if not goal_found:
            logging.info("No suitable goal found")
        return cf_best_for_ego, cf_best_for_other, cf_best_for_together

    def _return_comp_ego_strategic_goal(
        self,
        obs,
        start_ego,
        start_other,
        level_ego,
        level_other,
        goal_other,
        ignore_locs,
    ):
        if goal_other is None:
            goal_other = self._closest_food(
                obs,
                max_food_level=level_other,
                start=start_other,
                ignore_locs=ignore_locs,
            )
        goal_other_ego_distance = mhah.calculate_distance(
            X=np.array(goal_other), Y=np.array(start_ego), metric="cityblock"
        )
        goal_other_other_distance = mhah.calculate_distance(
            X=np.array(goal_other), Y=np.array(start_other), metric="cityblock"
        )
        consumable_by_ego_alone = self._consumable(
            obs, goal_other, max_food_level=level_ego
        )
        # Steal other's goal if feasible, otherwise look for the closest food that
        # can be consumed alone.
        if (
            consumable_by_ego_alone
            and goal_other_ego_distance <= goal_other_other_distance
        ):
            return goal_other
        return self._closest_food(
            obs,
            max_food_level=level_ego,
            start=start_ego,
            ignore_locs=ignore_locs,
        )

    def _return_adaptive_strategic_goal(self, cf_together, cf_ego):
        # Same as cooperative but goal loadable alone are prioritized.
        # Check first which ego items can be loaded alone...
        if cf_together["value"] > cf_ego["value"]:
            logging.info(  # pylint: disable=W1201
                "Agent %d adaptive cooperative goal selected: loc %s"  # pylint: disable=C0209
                % (self.player, cf_together["location"])
            )
            return cf_together["location"]
        logging.info(  # pylint: disable=W1201
            "Agent %d adaptive ego goal selected: loc %s"  # pylint: disable=C0209
            % (self.player, cf_ego["location"])
        )
        return cf_ego["location"]

    def _strategic_goal(
        self,
        obs,
        start_ego,
        start_other,
        level_ego,
        level_other,
        goal_other=None,
        ignore_locs=None,
    ):
        """Select next goal based on specified strategy.

        Given two locations start_ego and start_other, as well as levels of
        supposed agents at each of these locations. Determine a target location
        (from obs) from the perspective of start_ego based on a selected
        strategy.

        The ego and other agent are not automatically extracted from obs to
        allow for hypothetical starting locations and perspective switching,
        e.g., calling the method with the current locations of another agent
        as start_ego to estimate the target location of the other agent
        assuming a specific strategy.

        Available strategies: competitive_egoistic, social1, social2,
        cooperative, adaptive

        Attributes
        ----------
        obs : Observation instance
            Observed state of environment lbforaging.foraging.ForagingEnv
        start_ego: int tuple
            "ego" coordinates
        start_other: int tuple
            coordinates of another agent
        level_ego: int
            level of ego
        level_other: int
            level of other agent
        goal_other: int tuple
            if the goal of other is already known it can be specified
        ignore_locs: list of tuples
            optional list of locations that should be ignored as potential
            targets

        Return:
            next goal location under consideration of given strategy
        """
        cf_ego, cf_other, cf_together = self._compute_cf_based_goals(
            obs,
            level_ego,
            level_other,
            start_ego,
            start_other,
            ignore_locs=ignore_locs,
        )
        goal_values = {
            "ego": cf_ego["value"],
            "other": cf_other["value"],
            "together": cf_together["value"],
        }
        center = mhah.center_of_locations([start_ego, start_other])

        # print(goal_values)
        # print(goal_values_locations)

        if self.strategy == "competitive_egoistic":
            return (
                self._return_comp_ego_strategic_goal(
                    obs,
                    start_ego,
                    start_other,
                    level_ego,
                    level_other,
                    goal_other,
                    ignore_locs,
                ),
                goal_values,
            )

        if self.strategy == "social1":
            goal_center = self._closest_food(
                obs, max_food_level=None, start=center, ignore_locs=ignore_locs
            )
            return goal_center, goal_values

        if self.strategy == "social2":
            goal_center = self._closest_food(
                obs,
                max_food_level=level_ego + level_other,  # difference to social1
                start=center,
                ignore_locs=ignore_locs,
            )
            return goal_center, goal_values

        if self.strategy == "cooperative":
            logging.debug("Return location %s", cf_together["location"])
            return cf_together["location"], goal_values

        if self.strategy == "adaptive":
            return (
                self._return_adaptive_strategic_goal(cf_together, cf_ego),
                goal_values,
            )

        raise RuntimeError(
            "Unknown basic strategy %s" % self.strategy  # pylint: disable=C0209
        )

    def _closest_food(
        self, obs, max_food_level=None, start=None, ignore_locs=None
    ):
        """Find closest food (substitutes agent method with same name)

        Find food closest to own position or start position. Identification may
        be limited to foods of a maximum level (e.g., one that is compatible
        with current agent).

        Parameters
        ----------
        obs : Observation instance
            Observed state of environment lbforaging.foraging.ForagingEnv
        max_food_level : int, optional
            Maximum food level possible, if None, foods of all levels are
            considered, by default None
        start : iterable of two ints, optional
            Reference point for finding food (x-, y-coordinates), if None, the
            agent's location is used, by default None
        ignore_locs : list of tuples
            coordinates that should not be considered

        Returns
        -------
        (int, int)
            row and column index of identified food
            returns None in case no food is identified
        """
        if max_food_level == 0:
            max_food_level = None
        if start is None:
            x, y = self.observed_position
        else:
            x, y = start
        field = np.copy(obs.field)
        # if the own level is supposed to be used as reference point
        # (implied by missing start value), check if a given level can be considered
        if (
            not self.abilities["own"]["level"]
            and start is None
            and max_food_level is not None
        ):
            print("Overwriting max food level")
            max_food_level = None
        # if max_food_level:
        # if the abilities to see goal levels exist and a maximum level has been specified
        # consider that maximum level
        if max_food_level and self.abilities["goal"]["level"]:
            field[field > max_food_level] = 0
        # remove forbidden locations
        if ignore_locs is not None:
            for location in ignore_locs:
                field[location] = 0
        r, c = np.nonzero(field)
        try:
            min_idx = ((r - x) ** 2 + (c - y) ** 2).argmin()
        except ValueError:
            logging.debug("no food meets the requirements")
            print("No food found")
            print("Field w/o ignored locs", np.nonzero(obs.field))
            print("Field values:", obs.field[np.nonzero(obs.field)])
            print("own level:", max_food_level)
            print("Locations ignored:", ignore_locs)
            return None
        return (r[min_idx], c[min_idx])

    def ready_to_load(self, food_loc, maxattempts=None):
        """Checks whether conditions to consume a target have been met.

        The output can be affected by understanding and memory abilities
        of the agent.

        Attributes
        ----------
        food_loc : int tuple
            field location of food

        Returns
        -------
        bool
            True if ready, else False
        """
        if maxattempts is None:
            maxattempts = self.patience
        if self.abilities["own"]["location"]:
            if food_loc is None:
                return False
            y, x = self.observed_position
            food_in_neighborhood = (
                abs(food_loc[0] - y) + abs(food_loc[1] - x)
            ) == 1
            # if goal has been reached, take it
            # (unless the agent has just failed doing so)
            if (
                self.abilities["own"]["understands_action"]
                and food_in_neighborhood
            ):
                if (
                    self.abilities["own"]["remembers_action"]
                    and self.attempted_load >= maxattempts
                ):
                    logging.info(
                        "Agent %s: item can not be loaded. Patience exceeded",
                        self.player,
                    )
                    return False
                else:
                    return True
            else:
                # one who does not understand actions
                return food_in_neighborhood
        else:
            return False

    def _load(self, food_loc, maxattempts=None):
        """Carry out load action + logging of attempts if capable"""
        if maxattempts is None:
            maxattempts = self.patience
        if self.abilities["own"]["remembers_action"]:
            self.attempted_load += 1
            if self.attempted_load >= maxattempts:
                self.location_history.append(food_loc)
                logging.info(
                    "Agent %s patience exceeded. Adding goal location %s to ignore list",
                    self.player,
                    food_loc,
                )
        return Action.LOAD

    def _check_goal_status(self, obs):
        """reset goal location if it has been consumed"""
        if self.goal_location is not None and not obs.field[self.goal_location]:
            self.focused = 0
            self.goal_location = None
        elif self.goal_location is not None and obs.field[self.goal_location]:
            self.focused += 1

    def _move_towards(self, target, allowed):
        """HeuristicAgent _move_towards upgrade which does not
        create a north, south, east, west order bias due to
        command order by making that order random

        Attributes
        ----------
        target : int tuple
            target location
        allowed: list of Action objects
            included actions are considered valid action choices

        Returns
        -------
        int
            selected action
        """
        self.attempted_load = 0
        y, x = self.observed_position
        r, c = target
        # warning:
        # in case forbidden_action should be utilized elsewhere,
        # make sure to check for compatibility with load checking
        # e.g. if self.forbidden_action != Action.LOAD: ...
        if self.forbidden_action in allowed:
            logging.debug("removing action: %s", self.forbidden_action)
            allowed.remove(self.forbidden_action)
        actions = mhah.shuffle_actions(allowed)
        for i in actions:
            if allowed[i] == Action.NORTH and r < y:
                return Action.NORTH
            if r > y and allowed[i] == Action.SOUTH:
                return Action.SOUTH
            if c > x and allowed[i] == Action.EAST:
                return Action.EAST
            if c < x and allowed[i] == Action.WEST:
                return Action.WEST
        # if there is no good way towards, explore
        return self.exploration_step(actions)

    def _move_around(self, target, allowed=None):
        """Instead of moving towards a target, move around it.

        As a convention/preference/trait agents all agents circle in the same
        direction.

        Attributes
        ----------
        target : int tuple
            target location
        allowed: list of Action objects
            included actions are considered valid action choices

        Returns
        -------
        int
            selected action
        """
        if allowed is None:
            allowed = [0, 1, 2, 3, 4, 5]

        self.attempted_load = 0
        y, x = self.observed_position
        r, c = target

        if r < y and Action.EAST in allowed:
            return Action.EAST
        if r > y and Action.WEST in allowed:
            return Action.WEST
        if c > x and Action.SOUTH in allowed:
            return Action.SOUTH
        if c < x and Action.NORTH in allowed:
            return Action.NORTH
        else:
            raise ValueError("No simple path found")

    def random_step(self, allowed):
        """Select a random action

        Attributes
        ----------

        allowed: list of Action objects
            included actions are considered valid action choices
        """
        return np.random.choice(allowed)

    def exploration_step(self, allowed):
        """Move somewhere new if possible.

        Implements order-unbiased action selection (see _move_towards) but
        additionally avoids visited prior locations to promote exploratory
        behavior if self.abilities['own']['location_history'] is True

        Exploration optionally includes a loading action that may be selected
        depending on agent abilities

        Parameters
        ----------
        allowed: list of Action objects
            included actions are considered valid action choices

        Returns
        -------
        int
            selected action
        """
        y, x = self.observed_position
        actions = mhah.shuffle_actions(allowed)
        if (
            self.abilities["own"]["location_history"]
            and self.abilities["own"]["understands_action"]
        ):
            for action in actions:
                if (allowed[action] == Action.EAST) and (
                    (y, x + 1) not in self.location_history
                ):
                    self.attempted_load = 0
                    return Action.EAST
                if (allowed[action] == Action.WEST) and (
                    (y, x - 1) not in self.location_history
                ):
                    self.attempted_load = 0
                    return Action.WEST
                if (allowed[action] == Action.NORTH) and (
                    (y - 1, x) not in self.location_history
                ):
                    self.attempted_load = 0
                    return Action.NORTH
                if (allowed[action] == Action.SOUTH) and (
                    (y + 1, x) not in self.location_history
                ):
                    self.attempted_load = 0
                    return Action.SOUTH
                if (
                    (allowed[action] == Action.LOAD)
                    and self.abilities["own"]["remembers_action"]
                    and not self.attempted_load
                ):
                    # TODO: proper ability check and loading call
                    self.attempted_load += 1
                    return Action.LOAD
        # if all valid directions have been visited, pick a random one
        return self.random_step(allowed)

    def focus_step(self, obs, maxattempts=None):
        """Take a step towards focused goal.

        Step carried out if focus after goal selection is still below a limit.
        Only consists of pursuing a set goal. The execution of this step will
        circumvent all remaining 'intelligence' for the respective iteration.
        """
        if maxattempts is None:
            maxattempts = self.patience
        if self.ready_to_load(self.goal_location, maxattempts=maxattempts):
            return self._load(self.goal_location, maxattempts=maxattempts)
        try:
            return self._move_towards(self.goal_location, obs.actions)
        except ValueError:
            return np.random.choice(obs.actions)

    def egoistic_step(self, obs, maxattempts=None):
        """Step that does not consider other agents at all.

        Implements H1/Greedy1 if food_loc can be known and otherwise
        exploration.
        Exploration also becomes the fallback if no food is within visible
        reach.

        Parameters
        ----------
        obs : Observation instance
            Observed state of environment lbforaging.foraging.ForagingEnv

        Returns
        -------
        int
            selected action
        """
        # greedy selection based on proximity and optionally the goal level
        # Caution: this information may only be utilized by an agent later if
        # self.abilities['goal']['location'] is True.
        if maxattempts is None:
            maxattempts = self.patience
        if self.abilities["own"]["location_history"]:
            ignore_locs = list(self.location_history)
        else:
            ignore_locs = None
        if self.abilities["own"]["level"]:
            egolevel = self.level
        else:
            egolevel = None

        food_loc = self._closest_food(
            obs, max_food_level=egolevel, ignore_locs=ignore_locs
        )
        # explore if there is no food within reach
        if food_loc is None:
            logging.debug("No food found. Exploring.")
            return self.exploration_step(obs.actions)
        # if the food has already been reached try loading
        if self.ready_to_load(food_loc, maxattempts=maxattempts):
            return self._load(food_loc, maxattempts=maxattempts)
        # if the goal can be known, go for it
        if self.abilities["goal"]["location"]:
            try:
                logging.info(
                    "Agent %s: moving towards target with location %s and level %s",
                    self.player,
                    food_loc,
                    obs.field[food_loc],
                )
                self.goal_location = food_loc
                self.focused = 0
                return self._move_towards(food_loc, obs.actions)
            except ValueError:
                return np.random.choice(obs.actions)
        else:  # if location can not be known, explore
            return self.exploration_step(obs.actions)

    def considerate_step_with_strategy(self, obs, maxattempts=None):
        """Select goal and corresponding step using specified agent strategy.

        The action selection is carried out according to a given strategy
        identifier. For available strategies, see self._strategic_goal().

        Parameters
        ----------
        obs: numpy array
            gym observations (locations and levels of food, players, seen_players)
        maxattempts: int
            maximum number of subsequent attempts to load a particular food
            will be checked by self.ready_to_load()

        Returns
        -------
        int
            selected action
        dict
            goal values
        """
        if maxattempts is None:
            maxattempts = self.patience

        # get locations of other agents
        others_locations = np.array([player.position for player in obs.players])
        # get goal locations of other agents
        food_locations = []
        goal_values = []
        egolevel = None
        if self.abilities["own"]["level"]:
            egolevel = self.level

        if self.abilities["own"][
            "remembers_action"
        ]:  # and self.attempted_load > maxattempts:
            ignore_locs = list(self.location_history)
        else:
            ignore_locs = None

        # for each player other than oneself
        for player in obs.players:
            if not player.is_self:
                # select a target location based on a given strategy
                food_loc, values = self._strategic_goal(
                    obs,
                    start_ego=self.observed_position,
                    start_other=player.position,
                    level_ego=egolevel,
                    level_other=player.level,
                    goal_other=player.goal_position,
                    ignore_locs=ignore_locs,
                )
                food_locations.append(food_loc)
                goal_values.append(values)

        # First identified food location in strategic step is selected.
        # TODO adapt once there are envs with 3+ agents
        food_loc = food_locations[0]
        goal_values = goal_values[0]
        logging.debug("Food location %s", food_loc)
        # If no food is available at all with the selected strategy make a
        # random step.
        if food_loc is None:
            logging.info(
                "Agent %s no food found, ignorant step instead", self.player
            )
            return self.random_step(obs.actions), goal_values
        self.goal_location = food_loc
        self.focused = 0
        if self.ready_to_load(food_loc, maxattempts):
            if self._consumable(
                obs, coords=food_loc, max_food_level=self.level
            ):
                return self._load(food_loc, maxattempts), goal_values
            blocking_other_agent = any(
                [
                    mhah.intheway(self.observed_position, loc_other, food_loc)
                    for loc_other in others_locations
                ]
            )
            if not blocking_other_agent and self.attempted_load < maxattempts:
                return self._load(food_loc, maxattempts), goal_values
            if blocking_other_agent:  # or circle (make room)
                return self._move_around(food_loc, obs.actions), goal_values
            # Not in the way, but maxattempts reached.
            return self.random_step(obs.actions), goal_values
        return self._move_towards(food_loc, obs.actions), goal_values

    def _step(self, obs):
        """Choose next action according to agent strategy.

        Parameters
        ----------
        obs: numpy.ndarray
            gym observations (locations and levels of food, players,
            seen_players)

        Returns
        -------
        int
            selected action
        """
        # record the current location (required for exploration)
        if self.abilities["own"]["location_history"]:
            self.location_history.append(self.observed_position)
            if self.abilities["own"]["remembers_action"]:
                try:
                    if (
                        self.action_history[-2] == self.action_history[-1]
                        and self.location_history[-2]
                        == self.location_history[-2]
                    ):
                        self.forbidden_action = self.action_history[-1]
                    else:
                        self.forbidden_action = None
                except IndexError:
                    logging.info(
                        "no sufficient action history for forbidding actions"
                    )

        # Focus step: pursue goal if a goal was selected, otherwise reset goal
        # location (e.g., if goal has been consumed).
        if self.abilities["goal"]["location"]:
            self._check_goal_status(obs)

            if (
                self.goal_location is not None
                and self.abilities["goal"]["focus"]
                and self.focused < self.focus
            ):
                step = self.focus_step(obs)
                self.action_history.append(step)
                return step

        # BASELINE - random movement: if the agent does not know its own
        # position it cannot move in a purposeful manner (Baseline behavior).
        if self.strategy == "baseline":
            if self.abilities["own"]["location"]:
                raise RuntimeError(
                    "Ability own location is set, but baseline strategy is requested"
                )
            step = self.random_step(obs.actions)
            self.action_history.append(step)
            return step

        # EGOISTIC/H1 with exploration as fallback (when goal loc is not known)
        # behavior options when the other agents are not considered:
        if self.strategy == "egoistic":
            if (
                self.abilities["other"]["location"]
                or not self.abilities["goal"]["location"]
            ):
                raise RuntimeError(
                    "Selected strategy egoistic does not fit the specified abilities %s"
                    % self.abilities,
                )
            step = self.egoistic_step(obs)
            self.action_history.append(step)
            return step

        # H2, H5 - behavior options when the other agents are considered:
        # if the agent knows its own location, the goal locations,
        # and the location of the other agent, it can move towards the
        # closest goal that is not the closest goal for the other agent
        # a considerate step can also realize competitive strategies
        # depending on the 'competitive' ability
        # competitive objective: maximize relative success
        if self.strategy in ["social1", "social2"]:
            if (
                not self.abilities["other"]["location"]
                or not self.abilities["goal"]["location"]
            ):
                raise RuntimeError(
                    "Selected strategy egoistic does not fit the specified abilities %s"  # pylint: disable=C0209
                    % self.abilities,
                )
            # Don't return goal value for social strategies as these are calculated but
            # not used. The agent chooses the goal closest to the Euclidean center,
            # irrespective of the goal values.
            step, _ = self.considerate_step_with_strategy(obs)
            self.action_history.append(step)
            return step
        elif self.strategy in ["cooperative", "adaptive"]:
            if (
                not self.abilities["other"]["location"]
                or not self.abilities["goal"]["location"]
            ):
                raise RuntimeError(
                    "Selected strategy egoistic does not fit the specified abilities %s"  # pylint: disable=C0209
                    % self.abilities,
                )
            step, goal_values = self.considerate_step_with_strategy(obs)
            self.action_history.append(step)
            return step, goal_values

        # COMPETITIVE: Agent tries to steal other agent's food.
        elif self.strategy == "competitive_egoistic":
            if (
                not self.abilities["other"]["goal"]["location"]
                or not self.abilities["other"]["location"]
                or not self.abilities["goal"]["location"]
            ):
                raise RuntimeError(
                    "Selected strategy egoistic does not fit the specified abilities %s"  # pylint: disable=C0209
                    % self.abilities,
                )
            step, goal_values = self.considerate_step_with_strategy(obs)
            self.action_history.append(step)
            return step, goal_values

        else:  # agent did not find a strategy to select next step based on abilities
            raise RuntimeError(
                "Unknown strategy %s" % self.strategy  # pylint: disable=C0209
            )
