import argparse

from tensorforce.environments import Environment

import matplotlib.pyplot as plt

import numpy as np

#Sapientino package
from gym.wrappers import TimeLimit
import os

#Custom observation wrapper for the gymsapientino environment
from gym_sapientino_case.env import SapientinoCase

from agent_config import  build_agent

from trainer.NonMarkovianTrainer import NonMarkovianTrainer

from argparse import ArgumentParser



SINK_ID = 2

DEBUG = True








if __name__ == '__main__':



    map_file = os.path.join('.', 'maps/map4_easy.txt')

    #Log directory for the automaton states.
    log_dir = os.path.join('.','log_dir')

    #Istantiate the gym sapientino environment.
    environment = SapientinoCase(

        colors = ['blue','red','yellow','green'],

        params = dict(
            reward_per_step=-1.0,
            reward_outside_grid=0.0,
            reward_duplicate_beep=0.0,
            acceleration=0.4,
            angular_acceleration=15.0,
            max_velocity=0.6,
            min_velocity=0.4,
            max_angular_vel=40,
            initial_position=[3, 3],
            tg_reward=1000.0,
        ),

        map_file = map_file,
        logdir =log_dir

    )


    #Handle command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type = int, default = 64,help= 'Experience batch size.')
    parser.add_argument('--memory', type = int, default = None,help= 'Memory buffer size. Used by agents that train with replay buffer.')
    parser.add_argument('--multi_step',type = int, default = 10, help="Agent update optimization steps.")
    parser.add_argument('--update_frequency', type = int, default = None, help="Frequency of the policy updates. Default equals to batch_size.")
    parser.add_argument('--num_colors', type = int, default = 1, help="Number of distinct colors in the map.")
    parser.add_argument('--learning_rate', type = float, default = 0.001, help="Learning rate for the optimization algorithm")
    parser.add_argument('--exploration', type = float, default = 0.0, help = "Exploration for the epsilon greedy algorithm.")
    parser.add_argument('--entropy_bonus', type = float, default = 0.0, help ="Entropy bonus for the 'extended' loss of PPO. It discourages the policy distribution from being “too certain” (default: no entropy regularization" )
    parser.add_argument('--hidden_size', type = int, default = 128, help="Number of neurons of the hidden layers of the network.")
    parser.add_argument('--max_timesteps', type = int, default = 300, help= "Maximum number of timesteps each episode.")
    parser.add_argument('--episodes', type = int, default = 1000, help = "Number of training episodes.")


    args = parser.parse_args()

    batch_size = args.batch_size
    memory = args.memory
    update_frequency = args.update_frequency
    multi_step = args.multi_step
    num_colors = args.num_colors
    learning_rate = args.learning_rate
    entropy_bonus = args.entropy_bonus
    exploration = args.exploration



    #Default tensorforce update frequency is batch size.
    if not update_frequency:
        update_frequency = batch_size

    #Default ppo memory.
    if not memory:
        memory = 'minimum'

    num_hidden = args.hidden_size



    #There are both the initial and the sink additional states.
    NUM_STATE_AUTOMATON = num_colors+2







    #Set this value here to the maximum timestep value.
    MAX_EPISODE_TIMESTEPS = 400

    #Choose whether or not to visualize the environment
    VISUALIZE = False

    # Limit the length of the episode of gym sapientino.
    environment = TimeLimit(environment, MAX_EPISODE_TIMESTEPS)
    environment = Environment.create(environment =environment,max_episode_timesteps=MAX_EPISODE_TIMESTEPS,visualize =VISUALIZE)

    NUM_STATES_AUTOMATON = 6

    HIDDEN_STATE_SIZE = 256

    AUTOMATON_STATE_ENCODING_SIZE = HIDDEN_STATE_SIZE*NUM_STATES_AUTOMATON

    agent = build_agent(agent = 'ppo', batch_size = 128,

                        environment = environment,
                        num_states_automaton =NUM_STATES_AUTOMATON,
                        automaton_state_encoding_size=AUTOMATON_STATE_ENCODING_SIZE,

                        hidden_layer_size=HIDDEN_STATE_SIZE,

                        exploration =0.0,
                        update_frequency =60



                        )

    trainer = NonMarkovianTrainer(agent,environment,NUM_STATES_AUTOMATON,AUTOMATON_STATE_ENCODING_SIZE,
                                  SINK_ID
                                  )




    EPISODES = 2000

    #Train the agent

    training_results = trainer.train(episodes=EPISODES,evaluate=False)

    print("Training of the agent complete: results are: ")
    print(training_results)

    #Plot the total reward on the various evaluation epochs.
    plt.figure(1)
    plt.title("Total reward on evaluation phases")
    plt.xlabel("Evaluation epoch")
    plt.ylabel("Total reward")
    plt.plot(np.arange(1,len(training_results['evaluation_rewards'])+1),training_results['evaluation_rewards'])
    plt.show()
