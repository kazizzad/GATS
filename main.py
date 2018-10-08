import gym
import torch.optim as optim

from dqn_model import DQN
from dqn_learn import OptimizerSpec, dqn_learing
from utils.gym import get_env, get_wrapper_by_name
from utils.schedule import LinearSchedule
import sys
import pickle
import numpy as np
from collections import namedtuple
from itertools import count
import random
import gym.spaces
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from utils.replay_buffer import ReplayBuffer
from utils.gym import get_wrapper_by_name



BATCH_SIZE = 32
GAMMA = 0.99
REPLAY_BUFFER_SIZE = 1000000
LEARNING_STARTS = 50000
LEARNING_FREQ = 4
FRAME_HISTORY_LEN = 4
TARGER_UPDATE_FREQ = 10000
LEARNING_RATE = 0.00025
ALPHA = 0.95
EPS = 0.01


benchmark = gym.benchmark_spec('Atari40M')
task = benchmark.tasks[3]
seed = 0 # Use a seed of zero (you may want to randomize the seed!)
env = get_env(task, seed)
num_timesteps = task.max_timesteps
def stopping_criterion(env):
    # notice that here t is the number of steps of the wrapped env,
    # which is different from the number of steps in the underlying env
    return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps

optimizer_spec = OptimizerSpec(
    constructor=optim.RMSprop,
    kwargs=dict(lr=LEARNING_RATE, alpha=ALPHA, eps=EPS),
)

exploration_schedule = LinearSchedule(1000000, 0.1)

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])

Statistic = {"mean_episode_rewards": [],"best_mean_episode_rewards": []}


env=env
q_func=DQN
optimizer_spec=optimizer_spec
exploration=exploration_schedule
stopping_criterion=stopping_criterion
replay_buffer_size=REPLAY_BUFFER_SIZE
batch_size=BATCH_SIZE
gamma=GAMMA
learning_starts=LEARNING_STARTS
learning_freq=LEARNING_FREQ
frame_history_len=FRAME_HISTORY_LEN
target_update_freq=TARGER_UPDATE_FREQ


assert type(env.observation_space) == gym.spaces.Box
assert type(env.action_space)      == gym.spaces.Discrete

###############
# BUILD MODEL #
###############

if len(env.observation_space.shape) == 1:
    # This means we are running on low-dimensional observations (e.g. RAM)
    input_arg = env.observation_space.shape[0]
else:
    img_h, img_w, img_c = env.observation_space.shape
    input_arg = frame_history_len * img_c
num_actions = env.action_space.n

# Construct an epilson greedy policy with given exploration schedule
def select_epilson_greedy_action(model, obs, t):
    sample = random.random()
    eps_threshold = exploration.value(t)
    if sample > eps_threshold:
        obs = torch.from_numpy(obs).type(dtype).unsqueeze(0) / 255.0
        # Use volatile = True if variable is only used in inference mode, i.e. don’t save the history
        return model(Variable(obs, volatile=True)).data.max(1)[1].cpu()
    else:
        return torch.IntTensor([[random.randrange(num_actions)]])

# Initialize target q function and q function
Q = q_func(input_arg, num_actions).type(dtype)
target_Q = q_func(input_arg, num_actions).type(dtype)

# Construct Q network optimizer function
optimizer = optimizer_spec.constructor(Q.parameters(), **optimizer_spec.kwargs)

# Construct the replay buffer
replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

###############
# RUN ENV     #
###############
num_param_updates = 0
mean_episode_reward = -float('nan')
best_mean_episode_reward = -float('inf')
last_obs = env.reset()
LOG_EVERY_N_STEPS = 10000
t = 0
while t < 100000:
    t = t + 1 
    print(t)
    ### Check stopping criterion
    if stopping_criterion is not None and stopping_criterion(env):
        break

    ### Step the env and store the transition
    # Store lastest observation in replay memory and last_idx can be used to store action, reward, done
    last_idx = replay_buffer.store_frame(last_obs)
    # encode_recent_observation will take the latest observation
    # that you pushed into the buffer and compute the corresponding
    # input that should be given to a Q network by appending some
    # previous frames.
    recent_observations = replay_buffer.encode_recent_observation()

    # Choose random action if not yet start learning
    if t > learning_starts:
        action = select_epilson_greedy_action(Q, recent_observations, t)[0, 0]
    else:
        action = random.randrange(num_actions)
    # Advance one step
    obs, reward, done, _ = env.step(action)
    # clip rewards between -1 and 1
    reward = max(-1.0, min(reward, 1.0))
    # Store other info in replay memory
    replay_buffer.store_effect(last_idx, action, reward, done)
    # Resets the environment when reaching an episode boundary.
    if done:
        obs = env.reset()
    last_obs = obs

    ### Perform experience replay and train the network.
    # Note that this is only done if the replay buffer contains enough samples
    # for us to learn something useful -- until then, the model will not be
    # initialized and random actions should be taken
    if (t > learning_starts and
            t % learning_freq == 0 and
            replay_buffer.can_sample(batch_size)):
        # Use the replay buffer to sample a batch of transitions
        # Note: done_mask[i] is 1 if the next state corresponds to the end of an episode,
        # in which case there is no Q-value at the next state; at the end of an
        # episode, only the current state reward contributes to the target
        obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = replay_buffer.sample(batch_size)
        # Convert numpy nd_array to torch variables for calculation
        obs_batch = Variable(torch.from_numpy(obs_batch).type(dtype) / 255.0)
        act_batch = Variable(torch.from_numpy(act_batch).long())
        rew_batch = Variable(torch.from_numpy(rew_batch))
        next_obs_batch = Variable(torch.from_numpy(next_obs_batch).type(dtype) / 255.0)
        not_done_mask = Variable(torch.from_numpy(1 - done_mask)).type(dtype)

        if USE_CUDA:
            act_batch = act_batch.cuda()
            rew_batch = rew_batch.cuda()

        # Compute current Q value, q_func takes only state and output value for every state-action pair
        # We choose Q based on action taken.
        current_Q_values = Q(obs_batch).gather(1, act_batch.unsqueeze(1))
        # Compute next Q value based on which action gives max Q values
        # Detach variable from the current graph since we don't want gradients for next Q to propagated
        next_max_q = target_Q(next_obs_batch).detach().max(1)[0]
        next_Q_values = not_done_mask * next_max_q
        # Compute the target of the current Q values
        target_Q_values = rew_batch + (gamma * next_Q_values)
        # Compute Bellman error
        bellman_error = target_Q_values - current_Q_values
        # clip the bellman error between [-1 , 1]
        clipped_bellman_error = bellman_error.clamp(-1, 1)
        # Note: clipped_bellman_delta * -1 will be right gradient
        d_error = clipped_bellman_error * -1.0
        # Clear previous gradients before backward pass
        optimizer.zero_grad()
        # run backward pass
        current_Q_values.backward(d_error.data.unsqueeze(1))

        # Perfom the update
        optimizer.step()
        num_param_updates += 1

        # Periodically update the target network by Q network to target Q network
        if num_param_updates % target_update_freq == 0:
            target_Q.load_state_dict(Q.state_dict())

    ### 4. Log progress and keep track of statistics
    episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
    if len(episode_rewards) > 0:
        mean_episode_reward = np.mean(episode_rewards[-100:])
    if len(episode_rewards) > 100:
        best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)

    Statistic["mean_episode_rewards"].append(mean_episode_reward)
    Statistic["best_mean_episode_rewards"].append(best_mean_episode_reward)

    if t % LOG_EVERY_N_STEPS == 0 and t > learning_starts:
        print("Timestep %d" % (t,))
        print("mean reward (100 episodes) %f" % mean_episode_reward)
        print("best mean reward %f" % best_mean_episode_reward)
        print("episodes %d" % len(episode_rewards))
        print("exploration %f" % exploration.value(t))
        sys.stdout.flush()

        # Dump statistics to pickle
        with open('statistics.pkl', 'wb') as f:
            pickle.dump(Statistic, f)
            print("Saved to %s" % 'statistics.pkl')





