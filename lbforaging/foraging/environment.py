import logging
from collections import namedtuple, defaultdict
from enum import Enum
from itertools import product
from gym import Env
import gym
from gym.utils import seeding
import numpy as np

from lbforaging.agents import mh_agent_helper as mhah


class Action(Enum):
    NONE = 0
    NORTH = 1
    SOUTH = 2
    WEST = 3
    EAST = 4
    LOAD = 5


class CellEntity(Enum):
    # entity encodings for grid observations
    OUT_OF_BOUNDS = 0
    EMPTY = 1
    FOOD = 2
    AGENT = 3


class Player:
    def __init__(self):
        self.controller = None
        self.position = None
        self.level = None
        self.field_size = None
        self.score = None
        self.reward = 0
        self.cooperative_reward = 0
        self.food = 0
        self.food_type = 0
        self.history = None
        self.cooperative_actions = None
        self.current_step = None
        self.goal_position = None  # added for optional goal transparency
        self.fairness = 0.5  # added for optional fairness transparency
        self.self_observed_position = (
            None  # added for optional perspective transparency
        )

    def setup(self, position, level, field_size):
        self.history = []
        self.collected_food = []
        self.collected_food_type = []
        self.cooperative_actions = []
        self.position = position
        self.level = level
        self.field_size = field_size
        self.score = 0

    def set_controller(self, controller):
        self.controller = controller

    def step(self, obs):
        return self.controller._step(obs)

    @property
    def name(self):
        if self.controller:
            return self.controller.name
        else:
            return "Player"


class ForagingEnv(Env):
    """
    A class that contains rules/actions for the game level-based foraging.
    """

    metadata = {"render.modes": ["human"]}

    action_set = [Action.NORTH, Action.SOUTH, Action.WEST, Action.EAST, Action.LOAD]
    Observation = namedtuple(
        "Observation",
        ["field", "actions", "players", "game_over", "sight", "current_step"],
    )
    PlayerObservation = namedtuple(
        "PlayerObservation",
        [
            "position",
            "level",
            "history",
            "reward",
            "is_self",
            "goal_position",
            "self_observed_position",
        ],
    )  # reward is available only if is_self

    def __init__(
        self,
        players,
        player_levels,
        field_size,
        max_food,
        sight,
        max_episode_steps,
        thresh_respawn_food,
        force_coop,
        respawn_min_dist_to_agents=0.0,
        force_distractors=0.0,
        normalize_reward=True,
        grid_observation=False,
    ):
        self.logger = logging.getLogger(__name__)
        self.seed()
        self.players = [Player() for _ in range(players)]

        self.field = np.zeros(field_size, np.int32)

        self.max_food = max_food
        self._food_spawned = 0.0
        self.max_player_level = np.max(player_levels)
        self.player_levels = player_levels
        self.sight = sight
        self.force_coop = force_coop
        self.force_distractors = force_distractors
        self._game_over = None

        self._rendering_initialized = False
        self._valid_actions = None
        self._max_episode_steps = max_episode_steps
        self._thresh_respawn_food = thresh_respawn_food
        self._respawn_min_dist_to_agents = respawn_min_dist_to_agents

        self._normalize_reward = normalize_reward
        self._grid_observation = grid_observation

        self.action_space = gym.spaces.Tuple(
            tuple([gym.spaces.Discrete(6)] * len(self.players))
        )
        self.observation_space = gym.spaces.Tuple(
            tuple([self._get_observation_space()] * len(self.players))
        )

        self.viewer = None

        self.n_agents = len(self.players)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_observation_space(self):
        """The Observation Space for each agent.
        - all of the board (board_size^2) with foods
        - player description (x, y, level)*player_count
        """
        if not self._grid_observation:
            field_x = self.field.shape[1]
            field_y = self.field.shape[0]
            # field_size = field_x * field_y

            max_food = self.max_food
            max_food_level = self.max_player_level * len(self.players)

            min_obs = [-1, -1, 0] * max_food + [0, 0, 1] * len(self.players)
            max_obs = [field_x, field_y, max_food_level] * max_food + [
                field_x,
                field_y,
                self.max_player_level,
            ] * len(self.players)
        else:
            # grid observation space
            grid_shape = (1 + 2 * self.sight, 1 + 2 * self.sight)

            # agents layer: agent levels
            agents_min = np.zeros(grid_shape, dtype=np.float32)
            agents_max = np.ones(grid_shape, dtype=np.float32) * self.max_player_level

            # foods layer: foods level
            max_food_level = self.max_player_level * len(self.players)
            foods_min = np.zeros(grid_shape, dtype=np.float32)
            foods_max = np.ones(grid_shape, dtype=np.float32) * max_food_level

            # access layer: i the cell available
            access_min = np.zeros(grid_shape, dtype=np.float32)
            access_max = np.ones(grid_shape, dtype=np.float32)

            # total layer
            min_obs = np.stack([agents_min, foods_min, access_min])
            max_obs = np.stack([agents_max, foods_max, access_max])

        return gym.spaces.Box(np.array(min_obs), np.array(max_obs), dtype=np.float32)

    @classmethod
    def from_obs(cls, obs):
        players = []
        for p in obs.players:
            player = Player()
            player.setup(p.position, p.level, obs.field.shape)
            player.score = p.score if p.score else 0
            players.append(player)

        env = cls(players, None, None, None, None, None, None)
        env.field = np.copy(obs.field)
        env.current_step = obs.current_step
        env.sight = obs.sight
        env._gen_valid_moves()

        return env

    @property
    def field_size(self):
        return self.field.shape

    @property
    def rows(self):
        return self.field_size[0]

    @property
    def cols(self):
        return self.field_size[1]

    @property
    def game_over(self):
        return self._game_over

    def _gen_valid_moves(self):
        self._valid_actions = {
            player: [
                action for action in Action if self._is_valid_action(player, action)
            ]
            for player in self.players
        }

    def neighborhood(self, row, col, distance=1, ignore_diag=False):
        if not ignore_diag:
            return self.field[
                max(row - distance, 0) : min(row + distance + 1, self.rows),
                max(col - distance, 0) : min(col + distance + 1, self.cols),
            ]

        return (
            self.field[
                max(row - distance, 0) : min(row + distance + 1, self.rows), col
            ].sum()
            + self.field[
                row, max(col - distance, 0) : min(col + distance + 1, self.cols)
            ].sum()
        )

    def adjacent_food(self, row, col):
        """Sum of levels of adjacent food."""
        return (
            self.field[max(row - 1, 0), col]
            + self.field[min(row + 1, self.rows - 1), col]
            + self.field[row, max(col - 1, 0)]
            + self.field[row, min(col + 1, self.cols - 1)]
        )

    def adjacent_food_location(self, row, col):
        if row > 1 and self.field[row - 1, col] > 0:
            return row - 1, col
        elif row < self.rows - 1 and self.field[row + 1, col] > 0:
            return row + 1, col
        elif col > 1 and self.field[row, col - 1] > 0:
            return row, col - 1
        elif col < self.cols - 1 and self.field[row, col + 1] > 0:
            return row, col + 1

    def adjacent_players(self, row, col):
        """Find all players adjacent to a given position."""
        return [
            player
            for player in self.players
            if abs(player.position[0] - row) == 1
            and player.position[1] == col
            or abs(player.position[1] - col) == 1
            and player.position[0] == row
        ]

    def _spawn_items(self, food_level, items_to_spawn, min_dist_to_players):
        min_dist_to_players = int(min(min_dist_to_players, 0.8 * self.rows))
        attempts = 0
        spawned = 0
        while spawned < items_to_spawn and attempts < 1000:
            attempts += 1
            row = self.np_random.randint(1, self.rows - 1)
            col = self.np_random.randint(1, self.cols - 1)

            # check if it has neighbors:
            if (
                self.neighborhood(row, col).sum() > 0
                or self.neighborhood(row, col, distance=2, ignore_diag=True) > 0
                or self._is_within_distance_from_players(row, col, min_dist_to_players)
                or not self._is_empty_location(row, col)
            ):
                self.logger.info("Spawning food not successful")
                continue
            self.logger.info(
                f"\tSpawning food with level {food_level} at location ({row}, {col}) (coop={self.force_coop})"
            )
            self.field[row, col] = food_level
            spawned += 1

    def spawn_food(self, max_food, player_levels, min_dist_to_players=0):
        """Create and place energy sources in accordance with settings.

        This function places a given number of energy sources in the
        environment. The location for each food source is determined randomly,
        with the only constraints that two energy sources are not placed next
        to or on top of each other (no spawning within distance 2).

        The level of an energy source is set according to specified
        probabilities for 'cooperation' and 'distractor' sources. A cooperation
        source has a level equal to the sum of the agents' individual levels
        while a distractor level has a level that is larger than the sum of
        agents' levels, thus making it impossible to consume.

        Individual targets have level equal to the agents' levels. If both
        agents have the same level, all items have that level and both agents
        can collect any food item. If one agent has a higher level, only that
        agent can collect all individual items while the other agent can only
        collect the individual items that are compatible with its level. The
        distribution of individual items' levels is 50:50.

        Parameters
        ----------
        max_food : int
            number of energy sources which should be placed in the environment.
        player_levels: list of int
            list of player levels
        spawn_distractors: bool
            True if distractors should be created, False otherwise. This
            affects the probability of spawning energy sources that require
            no cooperation (lower when True, higher when False)
        min_dist_to_players: int
            minimum distance of new energy sources to any players capped at
            0.8 * self.rows
        """
        if (self.force_coop + self.force_distractors) > 1:
            raise RuntimeError(
                'The sum of settings "coop" and "distractors" exceeds 1 but it should not! '
                "This may lead to an unintentionally high proportion of cooperative items."
            )

        # Check if food items with high and low level are applicable or if all
        # agents have the same level.
        if len(np.unique(player_levels)) == 1:
            one_indiv_level = True
        else:
            one_indiv_level = False

        indiv_high_food_level = np.max(
            player_levels
        )  # food items that can be collected by the 'stronger' agent only
        indiv_low_food_level = np.min(player_levels)
        coop_food_level = np.sum(player_levels)
        dist_food_level = np.sum(player_levels) + 1
        if one_indiv_level:
            assert indiv_high_food_level == indiv_low_food_level

        expected_coop_food_count = np.around(max_food * self.force_coop).astype(int)
        expected_dist_food_count = np.around(max_food * self.force_distractors).astype(
            int
        )
        if one_indiv_level:
            expected_indiv_low_food_count = (
                max_food - expected_coop_food_count - expected_dist_food_count
            )
            expected_indiv_high_food_count = 0
        else:
            expected_indiv_low_food_count = np.around(
                (max_food - expected_coop_food_count - expected_dist_food_count) / 2
            ).astype(int)
            expected_indiv_high_food_count = (
                max_food
                - expected_coop_food_count
                - expected_dist_food_count
                - expected_indiv_low_food_count
            )
        assert (
            expected_coop_food_count
            + expected_dist_food_count
            + expected_indiv_low_food_count
            + expected_indiv_high_food_count
        ) == max_food

        food_count = np.count_nonzero(
            self.field
        )  # if called during initial environment reset, this is zero
        actual_coop_food_count = (self.field == coop_food_level).sum()
        actual_dist_food_count = (self.field == dist_food_level).sum()
        actual_indiv_low_food_count = (self.field == indiv_low_food_level).sum()
        if one_indiv_level:
            actual_indiv_high_food_count = 0
        else:
            actual_indiv_high_food_count = (self.field == indiv_high_food_level).sum()
        assert (
            actual_coop_food_count
            + actual_dist_food_count
            + actual_indiv_low_food_count
            + actual_indiv_high_food_count
        ) == food_count

        coop_items_to_spawn = expected_coop_food_count - actual_coop_food_count
        dist_items_to_spawn = expected_dist_food_count - actual_dist_food_count
        indiv_low_items_to_spawn = (
            expected_indiv_low_food_count - actual_indiv_low_food_count
        )
        if one_indiv_level:
            indiv_high_items_to_spawn = 0
            logging.info(
                f"Spawning {coop_items_to_spawn} cooperative, {indiv_low_items_to_spawn} individual, and "
                f"{dist_items_to_spawn} distractor items"
            )
        else:
            indiv_high_items_to_spawn = (
                expected_indiv_high_food_count - actual_indiv_high_food_count
            )
            logging.info(
                f"Spawning {coop_items_to_spawn} cooperative, {indiv_low_items_to_spawn} individual level "
                f"{indiv_low_food_level}, {indiv_high_items_to_spawn} individual level {indiv_high_food_level}, and "
                f"{dist_items_to_spawn} distractor items"
            )

        self._spawn_items(
            food_level=coop_food_level,
            items_to_spawn=coop_items_to_spawn,
            min_dist_to_players=min_dist_to_players,
        )
        self._spawn_items(
            food_level=indiv_low_food_level,
            items_to_spawn=indiv_low_items_to_spawn,
            min_dist_to_players=min_dist_to_players,
        )
        self._spawn_items(
            food_level=indiv_high_food_level,
            items_to_spawn=indiv_high_items_to_spawn,
            min_dist_to_players=min_dist_to_players,
        )
        self._spawn_items(
            food_level=dist_food_level,
            items_to_spawn=dist_items_to_spawn,
            min_dist_to_players=min_dist_to_players,
        )
        self._food_spawned = self.field.sum()

    def _is_empty_location(self, row, col):
        if self.field[row, col] != 0:
            return False
        for a in self.players:
            if a.position and row == a.position[0] and col == a.position[1]:
                return False

        return True

    def _is_within_distance_from_players(self, row, col, distance):
        """Returns True if a given coordinate lies within a specified distance from any agent on the field"""
        for a in self.players:
            if (
                mhah.calculate_distance(
                    np.array(a.position), np.array([row, col]), metric="cityblock"
                )
                < distance
            ):
                self.logger.debug("Player found within distance %s", distance)
                return True
        return False

    def spawn_players(self, player_levels):
        for player, player_level in zip(self.players, player_levels):
            attempts = 0
            player.reward = 0

            while attempts < 1000:
                row = self.np_random.randint(0, self.rows)
                col = self.np_random.randint(0, self.cols)
                if self._is_empty_location(row, col):
                    player.setup(
                        (row, col),
                        player_level,
                        self.field_size,
                    )
                    break
                attempts += 1

    def _is_valid_action(self, player, action):
        if action == Action.NONE:
            return True
        elif action == Action.NORTH:
            return (
                player.position[0] > 0
                and self.field[player.position[0] - 1, player.position[1]] == 0
            )
        elif action == Action.SOUTH:
            return (
                player.position[0] < self.rows - 1
                and self.field[player.position[0] + 1, player.position[1]] == 0
            )
        elif action == Action.WEST:
            return (
                player.position[1] > 0
                and self.field[player.position[0], player.position[1] - 1] == 0
            )
        elif action == Action.EAST:
            return (
                player.position[1] < self.cols - 1
                and self.field[player.position[0], player.position[1] + 1] == 0
            )
        elif action == Action.LOAD:
            return self.adjacent_food(*player.position) > 0

        self.logger.error("Undefined action {} from {}".format(action, player.name))
        raise ValueError("Undefined action")

    def _transform_to_neighborhood(self, center, sight, position):
        if position is None:
            return None
        return (
            position[0] - center[0] + min(sight, center[0]),
            position[1] - center[1] + min(sight, center[1]),
        )

    def get_valid_actions(self) -> list:
        return list(product(*[self._valid_actions[player] for player in self.players]))

    def _make_obs(self, player):
        return self.Observation(
            actions=self._valid_actions[player],
            players=[
                self.PlayerObservation(
                    position=self._transform_to_neighborhood(
                        player.position, self.sight, a.position
                    ),
                    level=a.level,
                    is_self=a == player,
                    history=a.history,
                    reward=a.reward if a == player else None,
                    # goal_position = a.goal_position, #self._transform_to_neighborhood(
                    #    player.position, self.sight, a.position
                    # ),
                    goal_position=self._transform_to_neighborhood(
                        player.position, self.sight, a.goal_position
                    ),
                    self_observed_position=a.self_observed_position,
                )
                for a in self.players
                if (
                    min(
                        self._transform_to_neighborhood(
                            player.position, self.sight, a.position
                        )
                    )
                    >= 0
                )
                and max(
                    self._transform_to_neighborhood(
                        player.position, self.sight, a.position
                    )
                )
                <= 2 * self.sight
            ],
            # todo also check max?
            field=np.copy(self.neighborhood(*player.position, self.sight)),
            game_over=self.game_over,
            sight=self.sight,
            current_step=self.current_step,
        )

    def _make_gym_obs(self):
        def make_obs_array(observation):
            obs = np.zeros(self.observation_space[0].shape, dtype=np.float32)
            # obs[: observation.field.size] = observation.field.flatten()
            # self player is always first
            seen_players = [p for p in observation.players if p.is_self] + [
                p for p in observation.players if not p.is_self
            ]

            for i in range(self.max_food):
                obs[3 * i] = -1
                obs[3 * i + 1] = -1
                obs[3 * i + 2] = 0

            for i, (y, x) in enumerate(zip(*np.nonzero(observation.field))):
                obs[3 * i] = y
                obs[3 * i + 1] = x
                obs[3 * i + 2] = observation.field[y, x]

            for i in range(len(self.players)):
                obs[self.max_food * 3 + 3 * i] = -1
                obs[self.max_food * 3 + 3 * i + 1] = -1
                obs[self.max_food * 3 + 3 * i + 2] = 0

            for i, p in enumerate(seen_players):
                obs[self.max_food * 3 + 3 * i] = p.position[0]
                obs[self.max_food * 3 + 3 * i + 1] = p.position[1]
                obs[self.max_food * 3 + 3 * i + 2] = p.level

            return obs

        def make_global_grid_arrays():
            """
            Create global arrays for grid observation space
            """
            grid_shape_x, grid_shape_y = self.field_size
            grid_shape_x += 2 * self.sight
            grid_shape_y += 2 * self.sight
            grid_shape = (grid_shape_x, grid_shape_y)

            agents_layer = np.zeros(grid_shape, dtype=np.float32)
            for player in self.players:
                player_x, player_y = player.position
                agents_layer[
                    player_x + self.sight, player_y + self.sight
                ] = player.level

            foods_layer = np.zeros(grid_shape, dtype=np.float32)
            foods_layer[
                self.sight : -self.sight, self.sight : -self.sight
            ] = self.field.copy()

            access_layer = np.ones(grid_shape, dtype=np.float32)
            # out of bounds not accessible
            access_layer[: self.sight, :] = 0.0
            access_layer[-self.sight :, :] = 0.0
            access_layer[:, : self.sight] = 0.0
            access_layer[:, -self.sight :] = 0.0
            # agent locations are not accessible
            for player in self.players:
                player_x, player_y = player.position
                access_layer[player_x + self.sight, player_y + self.sight] = 0.0
            # food locations are not accessible
            foods_x, foods_y = self.field.nonzero()
            for x, y in zip(foods_x, foods_y):
                access_layer[x + self.sight, y + self.sight] = 0.0

            return np.stack([agents_layer, foods_layer, access_layer])

        def get_agent_grid_bounds(agent_x, agent_y):
            return (
                agent_x,
                agent_x + 2 * self.sight + 1,
                agent_y,
                agent_y + 2 * self.sight + 1,
            )

        def get_player_reward(observation):
            for p in observation.players:
                if p.is_self:
                    return p.reward

        nobs = [self._make_obs(player) for player in self.players]
        if self._grid_observation:
            layers = make_global_grid_arrays()
            agents_bounds = [
                get_agent_grid_bounds(*player.position) for player in self.players
            ]
            nobsarray = tuple(
                [
                    layers[:, start_x:end_x, start_y:end_y]
                    for start_x, end_x, start_y, end_y in agents_bounds
                ]
            )
        else:
            nobsarray = tuple([make_obs_array(obs) for obs in nobs])
        nreward = [get_player_reward(obs) for obs in nobs]
        ndone = [obs.game_over for obs in nobs]
        # ninfo = [{'observation': obs} for obs in nobs]
        ninfo = {}

        return nobsarray, nobs, nreward, ndone, ninfo

    def reset(self):
        self.field = np.zeros(self.field_size, np.int32)
        self.spawn_players(self.player_levels)
        player_levels = sorted([player.level for player in self.players])

        self.spawn_food(self.max_food, player_levels=player_levels[:3])
        self.current_step = 0
        self._game_over = False
        self._gen_valid_moves()

        nobsarray, nobs, _, _, _ = self._make_gym_obs()
        return nobsarray, nobs

    def step(self, actions):
        self.current_step += 1

        for p in self.players:
            p.reward = 0
            p.cooperative_reward = 0
            p.food = 0
            p.food_type = 0

        actions = [
            Action(a) if Action(a) in self._valid_actions[p] else Action.NONE
            for p, a in zip(self.players, actions)
        ]

        # check if actions are valid
        for i, (player, action) in enumerate(zip(self.players, actions)):
            if action not in self._valid_actions[player]:
                self.logger.info(
                    "{}{} attempted invalid action {}.".format(
                        player.name, player.position, action
                    )
                )
                actions[i] = Action.NONE
            else:
                self.logger.debug(
                    "{}{} performing action {}.".format(
                        player.name, player.position, action
                    )
                )

        loading_players = set()

        # move players
        # if two or more players try to move to the same location they all fail
        collisions = defaultdict(list)

        # so check for collisions
        for player, action in zip(self.players, actions):
            if action == Action.NONE:
                collisions[player.position].append(player)
            elif action == Action.NORTH:
                collisions[(player.position[0] - 1, player.position[1])].append(player)
            elif action == Action.SOUTH:
                collisions[(player.position[0] + 1, player.position[1])].append(player)
            elif action == Action.WEST:
                collisions[(player.position[0], player.position[1] - 1)].append(player)
            elif action == Action.EAST:
                collisions[(player.position[0], player.position[1] + 1)].append(player)
            elif action == Action.LOAD:
                collisions[player.position].append(player)
                loading_players.add(player)

        # and do movements for non colliding players

        for k, v in collisions.items():
            if len(v) > 1:  # make sure no more than an player will arrive at location
                continue
            v[0].position = k

        # finally process the loadings:
        while loading_players:
            # find adjacent food
            player = loading_players.pop()
            frow, fcol = self.adjacent_food_location(*player.position)
            food = self.field[frow, fcol]

            adj_players = self.adjacent_players(frow, fcol)
            adj_players = [
                p for p in adj_players if p in loading_players or p is player
            ]

            adj_player_level = sum([a.level for a in adj_players])

            loading_players = loading_players - set(adj_players)

            if adj_player_level < food:
                # failed to load
                self.logger.info(
                    "Player level ({}) not sufficient to load food ({}), player position: {}".format(
                        adj_player_level, food, [a.position for a in adj_players]
                    )
                )
                continue
            else:
                self.logger.debug(
                    "Player level ({}) sufficient to load food ({}), player position: {}".format(
                        adj_player_level, food, [a.position for a in adj_players]
                    )
                )

            if len(adj_players) > 1 and food >= adj_player_level:
                food_type = 2
            else:
                food_type = 1

            # else the food was loaded and each player scores points
            for a in adj_players:
                a.reward = float(a.level * food)
                a.food = food
                a.food_type = food_type
                if self._normalize_reward:
                    a.reward = a.reward / float(
                        adj_player_level * self._food_spawned
                    )  # normalize reward
                if len(adj_players) > 1:
                    a.cooperative_reward = 1
            # and the food is removed
            self.field[frow, fcol] = 0

        if np.count_nonzero(self.field) <= self._thresh_respawn_food:
            self.logger.info(
                f"Respawning food ({np.count_nonzero(self.field)} foods on field)"
            )
            player_levels = sorted([player.level for player in self.players])
            self.spawn_food(
                self.max_food,
                player_levels=player_levels[:3],
                min_dist_to_players=self._respawn_min_dist_to_agents,
            )
        self._game_over = (
            self.field.sum() == 0 or self.current_step >= self._max_episode_steps
        )
        self._gen_valid_moves()

        for p in self.players:
            p.score += p.reward
            if p.cooperative_reward:  # share food if it was collected in a joint action
                p.collected_food.append(p.food / 2)
            else:
                p.collected_food.append(p.food)
            p.collected_food_type.append(p.food_type)
            p.cooperative_actions.append(p.cooperative_reward)

        return self._make_gym_obs()

    def _init_render(self):
        from .rendering import Viewer

        self.viewer = Viewer((self.rows, self.cols))
        self._rendering_initialized = True

    def render(self, mode="human"):
        if not self._rendering_initialized:
            self._init_render()

        return self.viewer.render(self, return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
