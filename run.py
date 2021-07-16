
from tensorforce.environments import Environment
from tensorforce.agents import Agent
import os
import matplotlib.pyplot as plt

import numpy as np

#Sapientino package
from gym.wrappers import TimeLimit
import os

#Custom observation wrapper for the gymsapientino environment
from gym_sapientino_case.env import SapientinoCase

import argparse
from utils.one_hot import *

import argparse


SINK_ID = 2


"""
Running script built for evaluation purposes
"""







if __name__ == '__main__':
    map_file = os.path.join('.', 'maps/map4_easy.txt')

    #Log directory for the automaton states.
    log_dir = os.path.join('.','log_dir')

    #Istantiate the gym sapientino environment.
    environment = SapientinoCase(

        colors = ['blue','yellow','green','red'],

        params = dict(
            reward_per_step=-1.0,
            reward_outside_grid=0.0,
            reward_duplicate_beep=0.0,
            acceleration=0.4,
            angular_acceleration=15.0,
            max_velocity=0.6,
            min_velocity=0.4,
            max_angular_vel=40,
            initial_position=[4, 2],
            tg_reward=1000.0,
        ),

        map_file = map_file,
        logdir =log_dir

    )

    #Handle command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_colors', type = int, default = 1, help="Number of distinct colors in the map.")
    parser.add_argument('--max_timesteps', type = int, default = 300, help= "Maximum number of timesteps each episode.")
    parser.add_argument('--episodes', type = int, default = 1000, help = "Number of training episodes.")


    args = parser.parse_args()



    #Load the correct agent according to the colors.
    num_colors = args.num_colors


    MAX_EPISODE_TIMESTEPS = args.max_timesteps



    #Set this value here to the maximum timestep value.
    MAX_EPISODE_TIMESTEPS = 350

    #Choose whether or not to visualize the environment
    VISUALIZE = True

    # Limit the length of the episode of gym sapientino.
    environment = TimeLimit(environment, MAX_EPISODE_TIMESTEPS)
    environment = Environment.create(environment =environment,max_episode_timesteps=MAX_EPISODE_TIMESTEPS,visualize =VISUALIZE)

    NUM_STATES_AUTOMATON = 4

    HIDDEN_STATE_SIZE = 128

    AUTOMATON_STATE_ENCODING_SIZE = HIDDEN_STATE_SIZE*NUM_STATES_AUTOMATON

    model_directory = os.path.join('saved_models','map2_easy','model')
    agent = Agent.load( path =model_directory, environment = environment)



    EVALUATION_EPISODES = 200

    sum_rewards = 0.0

    prevAutState = 0
    cum_reward = 0.0

    evaluation_rewards= list()

    for episode in range(EVALUATION_EPISODES):
        states = environment.reset()
        ep_reward =0.0


        internals = agent.initial_internals()
        terminal = False
        while not terminal:

           automaton_state = states['gymtpl1'][0]
           encoded_automaton_state = one_hot_encode(automaton_state,AUTOMATON_STATE_ENCODING_SIZE,NUM_STATES_AUTOMATON)

           states = dict(gymtpl0 = states['gymtpl0'],gymtpl1 = encoded_automaton_state)


           actions, internals = agent.act(
               states=states, internals=internals,
               independent=True, deterministic=True
           )
           states, terminal, reward = environment.execute(actions=actions)
           automaton_state = states['gymtpl1'][0]


           if automaton_state == SINK_ID:
               reward = -500.0
               ep_reward += reward
               cum_reward += reward
               terminal = True


           elif automaton_state == 1 and prevAutState!=1:
               reward = 500.0

           elif automaton_state == 3:
               reward = 500.0
               print("Visited the goal in episode: ", episode)

           prevAutState = automaton_state




           sum_rewards += reward

        evaluation_rewards.append(sum_rewards)
