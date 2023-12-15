import logging

import numpy as np

_MAX_INT = 999999


class Agent:
    """Abstract class for agents.

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
    name = "Prototype Agent"

    def __repr__(self):
        return self.name

    def __init__(self, player, player_id):
        self.logger = logging.getLogger(__name__)
        self.player = player_id
        self.level = player.level
        self.history = []
        self.goal_value_ego = []
        self.goal_value_other = []
        self.goal_value_together = []
        self.reward = []
        self.goal_location = (0, 0)
        self.fairness = 0.5
        logging.info(f'Setting up agent {self.player} with level {self.level}')

    def __getattr__(self, item):
        return getattr(self.player, item)

    def step(self, obs):
        self.observed_position = next(
            (x for x in obs.players if x.is_self), None
        ).position

        # Identify action and save to history.
        action = self._step(obs)
        if type(action) is tuple:
            self.history.append(action[0])
            goal_values = action[1]
            self.goal_value_ego.append(goal_values['ego'])
            self.goal_value_other.append(goal_values['other'])
            self.goal_value_together.append(goal_values['together'])
            return action[0]
        else:
            self.history.append(action)
            self.goal_value_ego.append(np.nan)
            self.goal_value_other.append(np.nan)
            self.goal_value_together.append(np.nan)
            return action

    def _step(self, obs):
        raise NotImplementedError("You must implement an agent")

    def _closest_food(self, obs, max_food_level=None, start=None):
        """Find closest food

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

        Returns
        -------
        (int, int)
            row and column index of identified food
        """

        if start is None:
            x, y = self.observed_position
        else:
            x, y = start

        field = np.copy(obs.field)

        if max_food_level:
            field[field > max_food_level] = 0

        r, c = np.nonzero(field)
        try:
            min_idx = ((r - x) ** 2 + (c - y) ** 2).argmin()
        except ValueError:
            return None

        return (r[min_idx], c[min_idx])

    def _make_state(self, obs):

        state = str(obs.field)
        for c in ["]", "[", " ", "\n"]:
            state = state.replace(c, "")

        for a in obs.players:
            state = state + str(a.position[0]) + str(a.position[1]) + str(a.level)

        return int(state)

    def cleanup(self):
        pass
