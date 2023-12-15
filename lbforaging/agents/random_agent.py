import numpy as np

from lbforaging.foraging.agent import Agent


class RandomAgent(Agent):
    name = "Random Agent"

    def _step(self, obs):
        return np.random.choice(obs.actions)
