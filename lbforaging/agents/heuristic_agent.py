import logging
import numpy as np
from lbforaging.foraging.agent import Agent
from lbforaging.foraging.environment import Action


class HeuristicAgent(Agent):
    """Abstract class for heuristic agents.

    Parameters
    ----------
        player : PlayerObservation instance
            Player as used in the environment lbforaging.foraging.ForagingEnv
        player_id : int
            Player ID

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
    name = "Heuristic Agent"

    def _center_of_players(self, players):
        coords = np.array([player.position for player in players])
        return np.rint(coords.mean(axis=0))

    def _move_towards(self, target, allowed):

        y, x = self.observed_position
        r, c = target

        if r < y and Action.NORTH in allowed:
            return Action.NORTH
        elif r > y and Action.SOUTH in allowed:
            return Action.SOUTH
        elif c > x and Action.EAST in allowed:
            return Action.EAST
        elif c < x and Action.WEST in allowed:
            return Action.WEST
        else:
            raise ValueError("No simple path found")

    def _step(self, obs):
        raise NotImplementedError("Heuristic agent is implemented by H1-H4")


class H1(HeuristicAgent):
    """Agent heuristic H1

    H1 agent always goes to the closest food.

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

    name = "H1"

    def _step(self, obs):
        food_loc = self._closest_food(obs)
        if food_loc is None:
            logging.debug('No food found. Selecting random action.')
            return np.random.choice(obs.actions)
        y, x = self.observed_position

        if (abs(food_loc[0] - y) + abs(food_loc[1] - x)) == 1:
            return Action.LOAD

        try:
            return self._move_towards(food_loc, obs.actions)
        except ValueError:
            return np.random.choice(obs.actions)


class H2(HeuristicAgent):
    """Agent heuristic H2

    H2 Agent goes to the one visible food which is closest to the centre of
    visible players.

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

    name = "H2"

    def _step(self, obs):

        players_center = self._center_of_players(obs.players)

        food_loc = self._closest_food(obs, None, players_center)
        if food_loc is None:
            logging.debug('No food found. Selecting random action.')
            return np.random.choice(obs.actions)
        y, x = self.observed_position

        if (abs(food_loc[0] - y) + abs(food_loc[1] - x)) == 1:
            return Action.LOAD

        try:
            return self._move_towards(food_loc, obs.actions)
        except ValueError:
            return np.random.choice(obs.actions)


class H3(HeuristicAgent):
    """Agent heuristic H3

    H3 Agent always goes to the closest food with compatible level.

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

    name = "H3"

    def _step(self, obs):

        food_loc = self._closest_food(obs, self.level)
        if food_loc is None:
            logging.debug('No food found. Selecting random action.')
            return np.random.choice(obs.actions)
        y, x = self.observed_position

        if (abs(food_loc[0] - y) + abs(food_loc[1] - x)) == 1:
            return Action.LOAD

        try:
            return self._move_towards(food_loc, obs.actions)
        except ValueError:
            return np.random.choice(obs.actions)


class H4(HeuristicAgent):
    """Agent heuristic H4

    H4 Agent goes to the one visible food which is closest to all visible
    players such that the sum of their and H4's level is sufficient to load the
    food.

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
    name = "H4"

    def _step(self, obs):

        players_center = self._center_of_players(obs.players)
        players_sum_level = sum([a.level for a in obs.players])

        food_loc = self._closest_food(obs, players_sum_level, players_center)
        if food_loc is None:
            logging.debug('No food found. Selecting random action.')
            return np.random.choice(obs.actions)
        y, x = self.observed_position

        if (abs(food_loc[0] - y) + abs(food_loc[1] - x)) == 1:
            return Action.LOAD

        try:
            return self._move_towards(food_loc, obs.actions)
        except ValueError:
            return np.random.choice(obs.actions)


class H5(HeuristicAgent):
    """Agent heuristic H5

    H5 Agent goes to the one visible food which is closest to all visible
    players such that the sum of their and H5's level is sufficient to load the
    food.
    H4 agents have the issue of standing in each other's way.
    To avoid this, here they start circling the prey after arrival to make room
    for the partner.

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
    name = "H5"

    def _move_towards(self, target, allowed):
        """upgrade which does not create north, south,
        east, west order bias"""
        y, x = self.observed_position
        r, c = target
        actions = list(range(len(allowed)))
        np.random.shuffle(actions)
        for i in actions:
            if allowed[i] == Action.NORTH and r < y:
                return Action.NORTH
            elif r > y and allowed[i] == Action.SOUTH:
                return Action.SOUTH
            elif c > x and allowed[i] == Action.EAST:
                return Action.EAST
            elif c < x and allowed[i] == Action.WEST:
                return Action.WEST
        raise ValueError("No simple path found")

    def _move_around(self, target, allowed):
        y, x = self.observed_position
        r, c = target

        if r < y: # and Action.EAST in allowed:
            return Action.EAST
            #return np.random.choice([Action.WEST,Action.EAST])
        elif r > y:# and Action.WEST in allowed:
            return Action.WEST
            #return np.random.choice([Action.WEST,Action.EAST])
        elif c > x: # and Action.SOUTH in allowed:
            return Action.SOUTH
            #return np.random.choice([Action.NORTH,Action.SOUTH])
        elif c < x: # and Action.NORTH in allowed:
            return Action.NORTH
            #return np.random.choice([Action.NORTH,Action.SOUTH])
        else:
            raise ValueError("No simple path found")

    def _step(self, obs):

        players_center = self._center_of_players(obs.players)
        players_sum_level = sum([a.level for a in obs.players])

        food_loc = self._closest_food(obs, players_sum_level, players_center)
        if food_loc is None:
            logging.debug('No food found. Selecting random action.')
            return np.random.choice(obs.actions)
        y, x = self.observed_position

        if (abs(food_loc[0] - y) + abs(food_loc[1] - x)) == 1:
            # if the level is sufficient, take the resource
            if food_loc == self._closest_food(obs, self.level):
                return Action.LOAD
            # if it is not move around from time to time
            else:
                if (np.random.randn() < 0.75):
                    return Action.LOAD
                else:
                    # or circle (make room)
                    print(obs.actions)
                    return self._move_around(food_loc, obs.actions)
        else:
            try:
                return self._move_towards(food_loc, obs.actions)
            except ValueError:
                return np.random.choice(obs.actions)


class H6(H5):
    """Agent heuristic H6

    Extension of H5 which is competitive in the sense of prioritizing
    targets it can consume on its own (for its own reward).
    Only after such targets are no longer available does it become
    'cooperative'.

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
    name = "H6"

    def _step(self, obs):

        players_center = self._center_of_players(obs.players)
        players_sum_level = sum([a.level for a in obs.players])
        player_center = self.observed_position
        independent = False
        food_loc_alone = self._closest_food(obs, self.level, player_center)
        if food_loc_alone is None:
            logging.debug('cooperation')
            food_loc = self._closest_food(obs, players_sum_level, players_center)
            # if there is no joint goal either
            if food_loc is None:
                logging.debug('No food found. Selecting random action.')
                return np.random.choice(obs.actions)
        else:
            logging.debug('competition')
            food_loc = food_loc_alone
            independent = True
        y, x = self.observed_position

        # if arrived
        if (abs(food_loc[0] - y) + abs(food_loc[1] - x)) == 1:
            if independent:
                return Action.LOAD
            else:
                if (np.random.randn() < 0.75):
                    return Action.LOAD
                else:
                    # or circle (make room)
                    print(obs.actions)
                    return self._move_around(food_loc, obs.actions)
        else:
            try:
                return self._move_towards(food_loc, obs.actions)
            except ValueError:
                return np.random.choice(obs.actions)
