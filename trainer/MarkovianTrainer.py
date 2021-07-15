
from tensorforce.agents import Agent
from tensorforce.environments import Environment
import numpy as np



class MarkovianTrainer(object):
    def __init__(self,agent, environment):

        """
        Desc: class that implements the markovian agent training (single color for gym sapentino).

        Args:
            agent:
            environment:
        """


        self.agent = agent
        self.environment = environment


