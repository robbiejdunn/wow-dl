from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import sequential
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from typing import Tuple
import tensorflow as tf


class SimpleDQN:
    def __init__(
        self, 
        fc_layer_params: Tuple[int, int], 
        num_actions: int, 
        learning_rate: float, 
        train_env,
        replay_buffer_max_len: int,
    ):
        print(f"Building DQN with fc_layer_params {fc_layer_params}")
        dense_layers = [self.create_dense_layer(num_units) for num_units in fc_layer_params]
        q_values_layer = tf.keras.layers.Dense(
            num_actions,
            activation=None,
            kernel_initializer=tf.keras.initializers.RandomUniform(
                minval=-0.03, 
                maxval=0.03,
            ),
            bias_initializer=tf.keras.initializers.Constant(-0.2)
        )
        q_net = sequential.Sequential(dense_layers + [q_values_layer])
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        train_step_counter = tf.Variable(0)

        # define epsilon greedy decay (https://github.com/tensorflow/agents/issues/339 https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/exponential_decay)
        start_epsilon = 0.9
        n_of_steps = 20000
        end_epsilon = 0.1
        epsilon = tf.compat.v1.train.polynomial_decay(
            start_epsilon,
            train_step_counter,
            n_of_steps,
            end_learning_rate=end_epsilon
        )
        

        self.agent = dqn_agent.DqnAgent(
            train_env.time_step_spec(),
            train_env.action_spec(),
            q_network=q_net,
            optimizer=optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=train_step_counter,
            epsilon_greedy=epsilon,
        )
        self.agent.initialize()
        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            batch_size=train_env.batch_size,
            max_length=replay_buffer_max_len,
        )

    # Define a helper function to create Dense layers configured with the right
    # activation and kernel initializer.
    @staticmethod
    def create_dense_layer(num_units):
        return tf.keras.layers.Dense(
            num_units,
            activation=tf.keras.activations.relu,
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=2.0, 
                mode='fan_in', 
                distribution='truncated_normal',
            )
        )
