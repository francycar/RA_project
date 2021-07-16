
import numpy as np
from tensorforce.agents import Agent




def build_agent(agent, batch_size,environment,num_states_automaton,
                      hidden_layer_size,automaton_state_encoding_size,
                      memory= 'minimum',
                     update_frequency = 20,multi_step = 10,exploration = 0.0, learning_rate = 0.001,
                non_markovian = True, entropy_regularization = 0.0):


    """
    Desc: simple function that creates the agent parameters dictionary and manages
    the code to define relevant hyperparameters. It defines also the structure
    of the policy (and the baseline) networks.

    Args:
        agent:
        memory:
        batch_size:
        environment:
        num_states_automaton:
        hidden_layer_size:
        non_markovian:

    Returns:

    """



    AUTOMATON_STATE_ENCODING_SIZE = automaton_state_encoding_size

    if non_markovian:
        agent = Agent.create(

            #Dictionary containing the agent configuration parameters
            agent = agent,
            memory = memory,
            batch_size = batch_size,
            environment = environment,
            update_frequency = update_frequency,
            multi_step = multi_step,
            states = dict(
                gymtpl0 = dict(type = 'float',shape= (7,),min_value = -np.inf,max_value = np.inf),
                gymtpl1 = dict(type ='float',shape=(AUTOMATON_STATE_ENCODING_SIZE,),min_value = 0.0, max_value = 1.0)
            ),

            #The actor network which computes the policy.

            network=dict(type = 'custom',
                         layers= [
                             dict(type = 'retrieve',tensors= ['gymtpl0']),
                             dict(type = 'linear_normalization'),
                             dict(type='dense', bias = True,activation = 'tanh',size=AUTOMATON_STATE_ENCODING_SIZE),
                             dict(type= 'register',tensor = 'gymtpl0-dense1'),

                             #Perform the product between the one hot encoding of the automaton and the output of the dense layer.
                             dict(type = 'retrieve',tensors=['gymtpl0-dense1','gymtpl1'], aggregation = 'product'),
                             dict(type='dense', bias = True,activation = 'tanh',size=AUTOMATON_STATE_ENCODING_SIZE),
                             dict(type= 'register',tensor = 'gymtpl0-dense2'),
                             dict(type = 'retrieve',tensors=['gymtpl0-dense2','gymtpl1'], aggregation = 'product'),
                             dict(type='register',tensor = 'gymtpl0-embeddings'),

                         ],

                         ),

            #learning_rate = dict(type = 'linear', initial_value = 0.001, unit = 'episodes',
            #                     num_steps = 500, final_value =0.0008),
            learning_rate = learning_rate,
            exploration = exploration,

            saver=dict(directory='model'),
            summarizer=dict(directory='summaries',summaries=['reward','graph']),
            entropy_regularization = entropy_regularization

        )

    else:
        agent = Agent.create(

        #Dictionary containing the agent configuration parameters
        agent = agent,
        memory = memory,
        batch_size = batch_size,
        environment= environment,
        states = dict(
                    gymtpl0 = dict(type = 'float',shape= (7,),min_value = -np.inf,max_value = np.inf),
                    gymtpl1 = dict(type ='int',shape=(1,))
                    ),

                             #The actor network which computes the policy.

                             network=dict(type = 'custom',
                                          layers= [
                                              dict(type = 'retrieve',tensors= ['gymtpl0']),
                                              dict(type = 'linear_normalization'),
                                              dict(type='dense', bias = True,activation = 'tanh',size=hidden_layer_size),

                                              #Perform the product between the one hot encoding of the automaton and the output of the dense layer.
                                              dict(type='dense', bias = True,activation = 'tanh',size=hidden_layer_size),
                                              dict(type='register',tensor = 'gymtpl0-embeddings'),

                                              ],

                            ),

                            #learning_rate = dict(type = 'linear', initial_value = 0.001, unit = 'episodes',
                            #                     num_steps = 500, final_value =0.0008),
                            learning_rate = learning_rate,
                            exploration = exploration,


                            saver=dict(directory='model'),
                            summarizer=dict(directory='summaries',summaries=['reward','graph']),




    )

    return agent

