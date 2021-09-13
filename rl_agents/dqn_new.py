from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import sequential
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
        num_iterations: int,
    ):
        print(f"Building DQN with fc_layer_params {fc_layer_params}")
        dense_layers = [self.dense_layer(num_units) for num_units in fc_layer_params]
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
        agent = dqn_agent.DqnAgent(
            train_env.time_step_spec(),
            train_env.action_spec(),
            q_network=q_net,
            optimizer=optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=train_step_counter
        )
        agent.initialize()
        self.num_iterations = num_iterations

    def train(self):
        for _ in range(self.num_iterations):
            # Collect a few steps using collect_policy and save to the replay buffer.
            collect_data(train_env, agent.collect_policy, replay_buffer, collect_steps_per_iteration)

            # Sample a batch of data from the buffer and update the agent's network.
            experience, unused_info = next(iterator)
            train_loss = agent.train(experience).loss

            step = agent.train_step_counter.numpy()

            if step % log_interval == 0:
                print('step = {0}: loss = {1}'.format(step, train_loss))

            if step % eval_interval == 0:
                avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
                print('step = {0}: Average Return = {1}'.format(step, avg_return))
                returns.append(avg_return)

    # Define a helper function to create Dense layers configured with the right
    # activation and kernel initializer.
    @staticmethod
    def dense_layer(num_units):
        return tf.keras.layers.Dense(
            num_units,
            activation=tf.keras.activations.relu,
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=2.0, 
                mode='fan_in', 
                distribution='truncated_normal',
            )
        )
