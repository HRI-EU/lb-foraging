#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Run experiments in LBF environment using simple heuristic agent classes.
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
import time
import argparse
import logging
import copy
import yaml
import gym
import numpy as np
from pathlib import Path

from gym.envs.registration import register

from lbforaging.utils.io import initialize_logger, read_settings
from lbforaging.utils.experiments import create_environment_id
from lbf_heuristic_agents import _game_loop


def run_experiment(settings, render=False):
    """Register environment and requested number of trials for experiment.

    Parameters
    ----------
    settings : dict
        Experiment settings
    render : bool, optional
        Whether to render the simulation, by default False
    """
    id_env = create_environment_id(settings)
    initialize_logger(
        debug=settings.debug, logdir=settings["outpath"], logname=f"{id_env}"
    )
    if settings.agents["heuristic"] == "MultiHeuristicAgent":
        logging.info(  # pylint: disable=W1201
            "Running experiments using the MultiHeuristicAgent class with ability %s"  # pylint: disable=C0209
            % (settings.agents["abilities"])
        )
    else:
        logging.info("Running experiments using simple HeuristicAgents classes")

    # Avoid re-registering environment over multiple experimental runs.
    if id_env not in gym.envs.registration.registry.env_specs:
        register(
            id=id_env,
            entry_point="lbforaging.foraging:ForagingEnv",
            kwargs={
                "players": settings.environment["nplayers"],
                "player_levels": settings.agents["levels"],
                "field_size": (
                    settings.environment["size"],
                    settings.environment["size"],
                ),
                "max_food": settings.environment["nfood"],
                "sight": ""
                if settings.environment["sight"] == settings.environment["size"]
                else f"-{settings.environment['sight']}s",
                "max_episode_steps": settings.environment["max_episode_steps"],
                "thresh_respawn_food": settings.environment[
                    "thresh_respawn_food"
                ],
                "force_coop": settings.environment["coop"],
                "force_distractors": settings.environment["distractors"],
                "grid_observation": False,
            },
        )
        logging.info(  # pylint: disable=W1201
            "Registering environment %s" % id_env  # pylint: disable=C0209
        )
    else:
        logging.info(  # pylint: disable=W1201
            "Environment %s already registered"
            % id_env  # pylint: disable=C0209
        )

    for trial in range(settings.experiment["ntrials"]):
        env = gym.make(id_env, sight=settings.environment["sight"])
        # to record a video uncomment + fix the following
        # env = gym.wrappers.Monitor(env, "./vid", force=True)
        # recording not functional atm due to added step return value
        # (only expects observation, reward, done, info)
        if settings.seed is not None:
            seed = settings.seed + trial  # use a different seed for each trial
            logging.info(  # pylint: disable=W1201
                "Setting environment main seed to %d"
                % seed  # pylint: disable=C0209
            )
            env.seed(seed)
            env.action_space.seed(seed)
            np.random.seed(seed)
        else:
            raise RuntimeError("No random seed was specified in the settings.")
        env.reset()

        logging.info(  # pylint: disable=W1201
            "Starting trial number %d" % trial  # pylint: disable=C0209
        )
        game_data = _game_loop(settings, env, render)
        game_data.to_csv(
            Path(settings["outpath"]).joinpath(
                f"{id_env}_game_data_trial_{trial:02d}.csv"
            )
        )
        logging.info(  # pylint: disable=W1201,C0209
            "Saving results to %s"
            % (
                Path(settings["outpath"]).joinpath(
                    f"{id_env}_game_data_trial_{trial:02d}.csv"
                )
            )
        )
        env.close()


def main(settings, outpath, render):
    """Read range of settings and run experiments."""
    sight = settings["environment"]["size"]  # set perfect sight for agents
    t_start = time.time()
    experiment_run = 0
    settings["experiment"]["coop_all"] = np.hstack(
        (
            np.arange(
                settings["experiment"]["coop_min"],
                settings["experiment"]["coop_max"],
                settings["experiment"]["coop_step"],
            ),
            settings["experiment"]["coop_max"],
        )
    )
    settings["experiment"]["coop_all"] = [
        float(c) for c in settings["experiment"]["coop_all"]
    ]  # required by yaml
    for enforce_cooperation in settings["experiment"]["coop_all"]:
        coop = float(np.around(float(enforce_cooperation), decimals=2))
        settings.environment["coop"] = coop
        for agent_heuristic in settings.experiment["heuristics"]:
            experiment_run += 1
            settings.agents["heuristic"] = "MultiHeuristicAgent"
            settings.agents["abilities"] = agent_heuristic
            print(agent_heuristic)
            # Generate folder and parents, also if it already exists.
            # Write modified settings file to output path.
            settings["outpath"] = str(  # cast to str, to write to yaml
                outpath.joinpath(
                    f"{experiment_run:02d}_s{sight}_coop{coop}_MHTrue_{agent_heuristic}"
                )
            )
            print(settings["outpath"])

            Path(settings["outpath"]).mkdir(parents=True, exist_ok=True)
            with open(
                Path(settings["outpath"]).joinpath("experiment_settings.yml"),
                "w",
            ) as outfile:
                yaml.safe_dump(
                    dict(copy.copy(settings)), outfile, default_flow_style=False
                )

            print(
                f"Running experiment {experiment_run:02d}: {agent_heuristic}, {coop}, {sight}"
            )
            print("Experiment settings:", settings)
            print(
                f"Running experiments for the following cooperation values: {settings['experiment']['coop_all']}"
            )
            print(
                f"Fraction of distractors: {settings['environment']['distractors']}"
            )
            run_experiment(settings, render)

    t_stop = time.time()
    logging.info(  # pylint: disable=W1201
        "Ran %d experiments in %6.2f min."  # pylint: disable=C0209
        % (experiment_run, (t_stop - t_start) / 60)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run experiment on LBF environment."
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--settings",
        type=str,
        default="../config/settings.yml",
    )
    parser.add_argument(
        "--outpath",
        "-o",
        type=str,
        default="../../lbf_experiments/",
    )
    args = parser.parse_args()

    settings = read_settings(args.settings)

    main(settings, Path(args.outpath), args.render)
