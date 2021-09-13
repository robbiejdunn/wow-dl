import pyvirtualdisplay
from tf_agents.environments import suite_gym, tf_py_environment
from tf_agents.policies import random_tf_policy
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
import matplotlib.pyplot as plt

from rl_agents.models.dqn.dqn import SimpleDQN


class TrainingManager:
    def __init__(
        self, 
        env_name: str, 
        channels: int,
        num_iterations: int,
        collect_steps_per_iteration: int,
        learning_rate: float,
        replay_buffer_max_len: int,
        batch_size: int,
        num_eval_episodes: int,
        initial_collect_steps: int,
        log_interval: int,
        eval_interval: int,
    ):
        """
        Creates a `TrainingManager` object, used for RL training in a given
        gym environment.

        :param gym_env: name of the gym environment to use
        :param channels: number of channels to use in images (1=grayscale, 3=rgb)
        """
        train_py_env = suite_gym.load(env_name)
        print(f"Observation spec: {train_py_env.time_step_spec().observation}")
        print(f"Reward Spec: {train_py_env.time_step_spec().reward}")
        print(f"Action space: {train_py_env.action_spec()}")
        eval_py_env = suite_gym.load(env_name)

        # is this doing anything
        display = pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()

        self.train_env = tf_py_environment.TFPyEnvironment(train_py_env)
        self.eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

        self.num_iterations = num_iterations
        self.collect_steps_per_iteration = collect_steps_per_iteration
        self.num_eval_episodes = num_eval_episodes
        self.log_interval = log_interval
        self.eval_interval = eval_interval

        action_tensor_spec = tensor_spec.from_spec(train_py_env.action_spec())
        num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1
        print(f"Num actions = {num_actions}")
        self.dqn = SimpleDQN((100, 50), num_actions, learning_rate, self.train_env, replay_buffer_max_len)
        random_policy = random_tf_policy.RandomTFPolicy(self.train_env.time_step_spec(), self.train_env.action_spec())
        self.collect_data(initial_collect_steps)
        # Dataset generates trajectories with shape [Bx2x...]
        dataset = self.dqn.replay_buffer.as_dataset(
            num_parallel_calls=3, 
            sample_batch_size=batch_size, 
            num_steps=2
        ).prefetch(3)

        self.iterator = iter(dataset)
        print(f"Baseline: {self.compute_avg_return(random_policy)}")
    
    def collect_step(self):
        time_step = self.train_env.current_time_step()
        action_step = self.dqn.agent.policy.action(time_step)
        next_time_step = self.train_env.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)
         # Add trajectory to the replay buffer
        self.dqn.replay_buffer.add_batch(traj)

    def collect_data(self, steps: int):
        for _ in range(self.collect_steps_per_iteration):
            self.collect_step()
    
    def compute_avg_return(self, policy):
        total_return = 0.0
        for _ in range(self.num_eval_episodes):
            time_step = self.eval_env.reset()
            episode_return = 0.0

            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = self.eval_env.step(action_step.action)
                episode_return += time_step.reward
            total_return += episode_return

        avg_return = total_return / self.num_eval_episodes
        return avg_return.numpy()[0]


    def train(self):
        # (Optional) Optimize by wrapping some of the code in a graph using TF function.
        self.dqn.agent.train = common.function(self.dqn.agent.train)
        # Reset the train step
        self.dqn.agent.train_step_counter.assign(0)
        # Evaluate the agent's policy once before training.
        avg_return = self.compute_avg_return(self.dqn.agent.policy)
        returns = [avg_return]

        for _ in range(self.num_iterations):
            # Collect a few steps using collect_policy and save to the replay buffer.
            self.collect_data(self.collect_steps_per_iteration)

            # Sample a batch of data from the buffer and update the agent's network.
            experience, unused_info = next(self.iterator)
            train_loss = self.dqn.agent.train(experience).loss

            step = self.dqn.agent.train_step_counter.numpy()

            if step % self.log_interval == 0:
                print('step = {0}: loss = {1}'.format(step, train_loss))

            if step % self.eval_interval == 0:
                avg_return = self.compute_avg_return(self.dqn.agent.policy)
                print('step = {0}: Average Return = {1}'.format(step, avg_return))
                returns.append(avg_return)
        iterations = range(0, self.num_iterations + 1, self.eval_interval)
        plt.plot(iterations, returns)
        plt.ylabel('Average Return')
        plt.xlabel('Iterations')
        plt.ylim(top=250)
        plt.savefig("resultsnewish.png")
