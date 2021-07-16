from tensorforce.agents import Agent
from tensorforce.environments import Environment


from utils.one_hot import one_hot_encode
from tqdm.auto import tqdm


DEBUG = True




class NonMarkovianTrainer(object):
    def __init__(self,agent,environment,num_state_automaton,
                 automaton_encoding_size,sink_id, automaton_states_params = None
                 ):

        """
        Desc: class that implements the non markovian training (multiple colors for gym sapientino).
        Keep in mind that the class instantiates the agent according to the parameter dictionary stored inside
        "agent_params" variable. The agent, and in particular the neural network, should be already non markovian.

        Args:
            agent_params:
            environment_params:
            num_state_automaton:
            automaton_encoding_size:
        """



        self.num_state_automaton = num_state_automaton
        self.automaton_encoding_size = automaton_encoding_size

        self.sink_id = sink_id


        #Create both the agent and the environment that will be used a training time.
        self.agent = agent
        self.environment = environment


        #Store the automaton states params if needed.
        self.automaton_states_params = automaton_states_params

        if DEBUG:
            architecture = self.agent.get_architecture()
            print(architecture)





    def train(self,episodes = 1000, evaluate = True):

        evaluation_rewards = list()


        cum_reward = 0.0



        def pack_states(states):

            obs = states['gymtpl0']
            automaton_state = states['gymtpl1'][0]

            one_hot_encoding = one_hot_encode(automaton_state,
                                              self.automaton_encoding_size,self.num_state_automaton)

            return dict(gymtpl0 =obs,
                        gymtpl1 = one_hot_encoding)

        agent = self.agent
        environment = self.environment

        try:
            for episode in tqdm(range(episodes),desc='training',leave = True):
                terminal = False

                states = environment.reset()

                automaton_state = states['gymtpl1'][0]
                states = pack_states(states)


                prevAutState = 0
                #Save the reward that you reach in the episode inside a linked list. This will be used for nice plots in the report.
                ep_reward = 0.0

                while not terminal:

                    actions = agent.act(states=states)


                    states, terminal, reward = environment.execute(actions=actions)

                    #Extract gym sapientino state and the state of the automaton.
                    automaton_state = states['gymtpl1'][0]
                    states = pack_states(states)

                    if automaton_state == self.sink_id:
                        reward = -500.0

                        terminal = True


                    elif automaton_state == 1 and prevAutState==0:
                        reward = 500.0

                    elif automaton_state == 3 and prevAutState==1:
                        reward = 500.0

                    elif automaton_state ==4 and prevAutState == 3:
                        reward = 500.0

                    elif automaton_state == 5:
                        reward = 500.0
                        print("Visited goal on episode: ", episode)

                    prevAutState = automaton_state


                    """
                    Check if the automaton state is sink. In this case, stop the episode.            
                    """




                    #Update the cumulative reward during the training.
                    cum_reward+=reward

                    #Update the episode reward during the training
                    ep_reward += reward



                    agent.observe(terminal=terminal, reward=reward)
                    if terminal == True:
                        states = environment.reset()

                EVALUATION_EPISODES = 100

                if evaluate == True:

                    if (episode+1) %100 == 0:
                        # Evaluate for 100 episodes
                        sum_rewards = 0.0

                        prevAutState = 0
                        for _ in range(EVALUATION_EPISODES):
                            states = environment.reset()


                            internals = agent.initial_internals()
                            terminal = False
                            while not terminal:
                                states = pack_states(states)



                                actions, internals = agent.act(
                                    states=states, internals=internals,
                                    independent=True, deterministic=True
                                )
                                states, terminal, reward = environment.execute(actions=actions)
                                automaton_state = states['gymtpl1'][0]


                                if automaton_state == self.sink_id:
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

                        print('Evaluation on episode:',episode,' total reward on', EVALUATION_EPISODES, " episodes: ", sum_rewards)
                        evaluation_rewards.append(sum_rewards)


            #Close both the agent and the environment.
            self.agent.close()
            self.environment.close()


            return dict(cumulative_reward_nodiscount = cum_reward,
                        average_reward_nodiscount = cum_reward/episodes,
                        evaluation_rewards = evaluation_rewards)
        finally:

           #Let the user interrupt
           pass




