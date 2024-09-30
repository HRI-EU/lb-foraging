#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Run LBF environment using heuristic agents. Comments from https://gym.openai.com/docs/.
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
import argparse
import logging
import time
import gym
import numpy as np
import pandas as pd
from pathlib import Path

from gym.envs.registration import register

# import lbforaging  # the import is required to register a range of environments
from lbforaging.utils.io import initialize_logger, read_settings
from lbforaging.utils.experiments import (
    create_environment_id,
    distance_to_closest_agent,
    distance_to_closest_food,
)
from lbforaging.agents.heuristic_agent import H1, H2, H3, H4, H5, H6
from lbforaging.agents.multi_heuristic_agents_value_function import (
    MultiHeuristicAgent,
)

from mh_agent_configurations import ability_sets


def print_food_dist(env):
    """Log distribution of food items on game field"""
    values, counts = np.unique(env.field, return_counts=True)
    logging.info(  # pylint: disable=W1201
        "Foods on field: level=%s count=%s (player levels: %s, c=%f, d=%f)"  # pylint: disable=C0209
        % (
            values[1:],
            counts[1:],
            [player.level for player in env.players],
            env.force_coop,
            env.force_distractors,
        )
    )


def _game_loop(settings, env, render):
    """Run Game Loop"""
    # The process gets started by calling reset(), which returns an initial observation.
    _, nobs = env.reset()
    logging.info("Initial observation:")
    logging.info(nobs[0])  # pylint: disable=W1201,C0209
    print_food_dist(env)

    done = False
    n_players = len(env.players)

    sleeptime = 1.0 / settings["fps"]
    if render:
        env.render()
        time.sleep(sleeptime)

    # Every environment comes with an action_space and an observation_space.
    # These attributes are of type Space, and they describe the format of valid
    # actions and observations:
    logging.info(env.action_space)
    logging.info(env.observation_space)

    # Initialize agents.
    logging.info(  # pylint: disable=W1201
        "Using agents with heuristic %s"  # pylint: disable=C0209
        % settings.agents["heuristic"]
    )
    if settings.agents["heuristic"] == "H1":
        agents = [H1(player, i) for i, player in enumerate(env.players)]
    elif settings.agents["heuristic"] == "H2":
        agents = [H2(player, i) for i, player in enumerate(env.players)]
    elif settings.agents["heuristic"] == "H3":
        agents = [H3(player, i) for i, player in enumerate(env.players)]
    elif settings.agents["heuristic"] == "H4":
        agents = [H4(player, i) for i, player in enumerate(env.players)]
    elif settings.agents["heuristic"] == "H5":
        agents = [H5(player, i) for i, player in enumerate(env.players)]
    elif settings.agents["heuristic"] == "H6":
        agents = [H6(player, i) for i, player in enumerate(env.players)]
    elif settings.agents["heuristic"] == "MultiHeuristicAgent":
        logging.info(  # pylint: disable=W1201
            "Applying multiheuristic labeled '%s'"  # pylint: disable=C0209
            % settings.agents["abilities"]
        )
        agents = [
            MultiHeuristicAgent(
                player,
                i,
                strategy=settings.agents["abilities"],
                abilities=ability_sets[settings.agents["abilities"]],
                location_memory_size=settings.agents["memory"],
                patience=settings.agents["patience"],
            )
            for i, player in enumerate(env.players)
        ]
    distances_food = {}
    distances_agents = {}
    coordinates_agents = {}
    for i, player in enumerate(env.players):
        distances_food[i] = []
        distances_agents[i] = []
        coordinates_agents[i] = {"x": [], "y": []}
    frames = []  # for gif
    while not done:
        loopstart = time.time()

        # Generate actions for next round. Has to be iterable of the same
        # length as number of agents in the game.
        # actions = env.action_space.sample()  # sample a random action from space
        actions = []
        for agent, obs, i in zip(agents, nobs, range(n_players)):
            actions.append(agent.step(obs))

        # Rerun loop to calculate distances to closest food and partner after
        # calling step() for each agent to set the observed position.
        logging.debug(  # pylint: disable=W1201
            "Agent distance: %s" % distances_agents[0]  # pylint: disable=C0209
        )
        logging.debug(  # pylint: disable=W1201
            "Player 0 position: (%d, %d)"  # pylint: disable=C0209
            % env.players[0].position
        )
        logging.debug(  # pylint: disable=W1201
            "Player 1 position: (%d, %d)"  # pylint: disable=C0209
            % env.players[1].position
        )
        logging.debug("Selected actions: %s" % actions)  # pylint: disable=W1201,C0209
        for agent, obs, i in zip(env.players, nobs, range(n_players)):
            distances_food[i].append(
                distance_to_closest_food(agent, env.field.copy(), metric="cityblock")
            )
            distances_agents[i].append(
                distance_to_closest_agent(
                    player_id=i, players=env.players.copy(), metric="cityblock"
                )
            )
            coordinates_agents[i]["x"].append(agent.position[0])
            coordinates_agents[i]["y"].append(agent.position[1])

        # step() returns five values. These are:
        #  - nobsarray (list of objects): a list of environment-specific
        #    objects representing each agents' observation of the environment.
        #    Representations are encoded into arrays and are a compressed
        #    representation of the current state.
        #  - observation (list of objects): a list of environment-specific
        #    objects representing each agents' observation of the environment.
        #  - reward (float): amount of reward achieved by the previous action.
        #    The scale varies between environments, but the goal is always to
        #    increase your total reward.
        #  - done (boolean): whether it’s time to reset the environment again.
        #    Most (but not all) tasks are divided up into well-defined
        #    episodes, and done being True indicates the episode has
        #    terminated.
        #  - info (dict): diagnostic information useful for debugging. It can
        #    sometimes be useful for learning (for example, it might contain
        #    the raw probabilities behind the environment’s last state change).
        #    However, official evaluations of your agent are not allowed to use
        #    this for learning.
        # This is just an implementation of the classic "agent-environment
        # loop". Each timestep, the agent chooses an action, and the
        # environment returns an observation and a reward.
        _, nobs, nreward, ndone, info = env.step(actions)
        if sum(nreward) > 0:
            logging.info(  # pylint: disable=W1201
                "Step %d: Player(s) %s loaded food with level(s) %s (type: %s, location: %s)"  # pylint: disable=C0209
                % (
                    env.current_step,
                    np.where(nreward)[0],
                    np.array([p.collected_food[-1] for p in env.players])[
                        np.where(nreward)[0]
                    ],
                    np.array([p.collected_food_type[-1] for p in env.players])[
                        np.where(nreward)[0]
                    ],
                    np.array([p.position for p in env.players])[np.where(nreward)[0]],
                )
            )
            print_food_dist(env)
            logging.debug("Reward: %s" % nreward)  # pylint: disable=W1201,C0209
            # logging.info("Nobs: %s" % nobs) # pylint: disable=W1201,C0209
            logging.debug("Done: %s" % ndone)  # pylint: disable=W1201,C0209
            logging.debug("Info: %s" % info)  # pylint: disable=W1201,C0209

        for reward, agent in zip(nreward, agents):
            agent.reward.append(reward)

        # synchronize agent and player object properties
        for player, agent in zip(env.players, agents):
            player.goal_position = agent.goal_location
            player.self_observed_position = agent.observed_position
        #    # player.history = agent.history

        if render:
            env.render()
            time.sleep(max(sleeptime - (time.time() - loopstart), 0))
            # env.render()

        done = np.all(ndone)
    logging.info(  # pylint: disable=W1201
        "-------- SUCCESS - Step %d - N collected food per player: %s (total value: %s), N cooperative actions: %s"  # pylint: disable=C0209
        % (
            env.current_step,
            [np.sum(np.array(p.collected_food) > 0) for p in env.players],
            [np.sum(p.collected_food) for p in env.players],
            [np.sum(p.cooperative_actions) for p in env.players],
        )
    )

    if len(agents) == 2:
        assert distances_agents[0] == distances_agents[1]
    game_data = pd.DataFrame(
        {
            "agent_id": np.array(
                [a.player * np.ones(env.current_step, dtype=int) for a in agents]
            ).flatten(),
            "step": np.array([np.arange(env.current_step) for a in agents]).flatten(),
            "coord_x": np.array(
                [coordinates_agents[a.player]["x"] for a in agents]
            ).flatten(),
            "coord_y": np.array(
                [coordinates_agents[a.player]["y"] for a in agents]
            ).flatten(),
            "reward": np.array([a.reward for a in agents]).flatten(),
            "reward_sum": np.array([np.cumsum(a.reward) for a in agents]).flatten(),
            "cooperative_actions": np.array(
                [p.cooperative_actions for p in env.players]
            ).flatten(),
            "food": np.array([p.collected_food for p in env.players]).flatten(),
            "food_type": np.array(
                [p.collected_food_type for p in env.players]
            ).flatten(),
            "food_sum": np.array(
                [np.cumsum(p.collected_food) for p in env.players]
            ).flatten(),
            "action": np.array([a.history for a in agents]).flatten(),
            "goal_value_ego": np.array([a.goal_value_ego for a in agents]).flatten(),
            "goal_value_other": np.array(
                [a.goal_value_other for a in agents]
            ).flatten(),
            "goal_value_together": np.array(
                [a.goal_value_together for a in agents]
            ).flatten(),
            "dist_closest_food": np.array(
                [distances_food[a.player] for a in agents]
            ).flatten(),
            "dist_closest_agent": np.array(
                [distances_agents[a.player] for a in agents]
            ).flatten(),
        }
    )
    return game_data


def main(settings, game_count=1, render=False):
    """Initialize environment and run game loop"""
    env_id = create_environment_id(settings)
    initialize_logger(debug=settings.debug, logname=f"{env_id}")
    register(
        id=env_id,
        entry_point="lbforaging.foraging:ForagingEnv",
        kwargs={
            "players": settings.environment["nplayers"],
            "player_levels": settings.agents["levels"],
            "field_size": (
                settings.environment["size"],
                settings.environment["size"],
            ),
            "max_food": settings.environment["nfood"],
            "sight": (
                ""
                if settings.environment["sight"] == settings.environment["size"]
                else f"-{settings.environment['sight']}s"
            ),
            "max_episode_steps": settings.environment["max_episode_steps"],
            "thresh_respawn_food": settings.environment["thresh_respawn_food"],
            "force_coop": settings.environment["coop"],
            "force_distractors": settings.environment["distractors"],
            "grid_observation": False,
        },
    )
    logging.info("Making environment %s" % env_id)  # pylint: disable=W1201,C0209

    env = gym.make(env_id, sight=settings.environment["sight"])
    # to record a video uncomment + fix the following
    # env = gym.wrappers.Monitor(env, "./vid", force=True)
    # recording not functional atm due to added step return value
    # (only expects observation, reward, done, info)
    if settings.seed is not None:
        logging.info(  # pylint: disable=W1201
            "Setting environment main seed to %d"  # pylint: disable=C0209
            % settings.seed
        )
        env.seed(settings.seed)
        env.action_space.seed(settings.seed)
        np.random.seed(settings.seed)
    env.reset()
    for episode in range(game_count):
        logging.info(  # pylint: disable=W1201
            "Starting game number %d of %d"  # pylint: disable=C0209
            % (episode + 1, game_count)
        )
        df = _game_loop(settings, env, render)
        df.to_csv(Path(settings["outpath"]).joinpath(f"{env_id}_game_{episode}.csv"))
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play the level foraging game.")

    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--times", type=int, default=1, help="How many times to run the game"
    )
    args = parser.parse_args()
    settings = read_settings(Path("../config").joinpath("settings.yml"))
    main(settings, args.times, args.render)
