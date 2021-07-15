from tensorforce.environments import Environment

import matplotlib.pyplot as plt


#Sapientino package
from gym.wrappers import TimeLimit
import os

#Custom observation wrapper for the gymsapientino environment
from environment.env import *
from agent_config import  build_agent

from trainer.NonMarkovianTrainer import NonMarkovianTrainer

SINK_ID = 2

DEBUG = True








if __name__ == '__main__':



    map_file = os.path.join('.', 'maps/map4_easy.txt')

    #Log directory for the automaton states.
    log_dir = os.path.join('.','log_dir')

    #Istantiate the gym sapientino environment.
    environment = SapientinoCase(

        colors = ['blue','red','green','yellow'],

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

    #Set this value here to the maximum timestep value.
    MAX_EPISODE_TIMESTEPS = 350

    #Choose whether or not to visualize the environment
    VISUALIZE = False

    # Limit the length of the episode of gym sapientino.
    environment = TimeLimit(environment, MAX_EPISODE_TIMESTEPS)
    environment = Environment.create(environment =environment,max_episode_timesteps=MAX_EPISODE_TIMESTEPS,visualize =VISUALIZE)

    NUM_STATES_AUTOMATON = 6

    HIDDEN_STATE_SIZE = 256

    AUTOMATON_STATE_ENCODING_SIZE = HIDDEN_STATE_SIZE*NUM_STATES_AUTOMATON

    agent = build_agent(agent = 'ppo', batch_size = 256,

                        environment = environment,
                        num_states_automaton =NUM_STATES_AUTOMATON,
                        automaton_state_encoding_size=AUTOMATON_STATE_ENCODING_SIZE,

                        hidden_layer_size=HIDDEN_STATE_SIZE,

                        exploration =0.0,



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
