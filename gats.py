import gym
import torch.optim as optim

from dqn_model import DQN , _RP
# from dqn_learn import OptimizerSpec, dqn_learing
from utils.gym import get_env, get_env_by_id, get_wrapper_by_name
from utils.schedule import LinearSchedule
from utils.evaluation import evaluation
# from utils.Gradient_penalty import calc_gradient_penalty
    
import sys
import pickle
import logging, logging.handlers
import numpy as np
from collections import namedtuple
from itertools import count
import random
import gym.spaces

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable

import matplotlib.pyplot as plt


import torchvision.utils as vutils

from utils.replay_buffer import ReplayBuffer, GANReplayBuffer, onehot_action
from utils.gym import get_wrapper_by_name
from models.models import _netG, _netD, _netPatchD

lookahead = 3
TARGER_UPDATE_FREQ = 10000
skip_f = 4

env_name = 'Pong'
env_name1 = '%sNoFrameskip-v4' %(env_name)  # Set the desired environment
env_name = '%sDDQN-GAN_LA_%d_TargetFQ_%d_skip_%d_RNN' %(env_name1,lookahead,TARGER_UPDATE_FREQ,skip_f)
TARGER_UPDATE_FREQ = 10000
logger = logging.getLogger()
file_name = './data/results_%s.log' %(env_name)
fh = logging.handlers.RotatingFileHandler(file_name)
fh.setLevel(logging.DEBUG)#no matter what level I set here
formatter = logging.Formatter('%(asctime)s:%(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

BATCH_SIZE = 32
GAMMA = 0.99
REPLAY_BUFFER_SIZE = 1000000
GAN_REPLAY_BUFFER_SIZE = 10000
LEARNING_STARTS = 50000
LEARNING_STARTS_Q_GAN = 200000
REWARD_LEARNING_STARTS = 10000
LEARNING_FREQ = 4
Reward_learning_freq = 4
FRAME_HISTORY_LEN = 4
GAN_LEARNING_STARTS = 10000
GAN_learning_freq = 4
LEARNING_RATE = 0.00025
ALPHA = 0.95
EPS = 0.01
RP_beta1 = 0.5 
RP_beta2 = 0.999
NUM_ACTIONS = 5
NUM_REWARDS = 3
NORM_FRAME_VAL = 130.

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

# Run training
seed = 0 # Use a seed of zero (you may want to randomize the seed!)
env = get_env_by_id(env_name1, seed, skip_f)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv' or 'SNConv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])

optimizer_spec = OptimizerSpec(
    constructor=optim.RMSprop,
    kwargs=dict(lr=LEARNING_RATE, alpha=ALPHA, eps=EPS),
)

exploration_schedule = LinearSchedule(1000000, 0.1)
exploration_G = LinearSchedule(50000, 0.01)
env=env
q_func=DQN
optimizer_spec=optimizer_spec
exploration=exploration_schedule
replay_buffer_size=REPLAY_BUFFER_SIZE
gan_replay_buffer_size=GAN_REPLAY_BUFFER_SIZE
gan_learning_freq = GAN_learning_freq
batch_size=BATCH_SIZE
gamma=GAMMA
gan_learning_starts=GAN_LEARNING_STARTS
learning_starts=LEARNING_STARTS
learning_starts_Q_GAN = LEARNING_STARTS_Q_GAN
reward_learning_starts=REWARD_LEARNING_STARTS
learning_freq=LEARNING_FREQ
reward_learning_freq = Reward_learning_freq
frame_history_len=FRAME_HISTORY_LEN
target_update_freq=TARGER_UPDATE_FREQ
reward_bs = 128 
bs = 128
one = torch.FloatTensor([1]).cuda()
mone = one * -1
num_actions = NUM_ACTIONS
norm_frame_val = NORM_FRAME_VAL
num_rewards = NUM_REWARDS

def dispimage(num_samples = 1, rollout_len = 3):
    obs_batch1, act_batch1, next_obs_batch1 = replay_buffer.GAN_sample(num_samples,rollout_len)
    act_batch = onehot_action(act_batch1,num_actions,rollout_len,num_samples)
    obs_batch = Variable(norm_frame(torch.from_numpy(obs_batch1).type(dtype)))
    next_obs_batch = Variable(norm_frame(torch.from_numpy(next_obs_batch1).type(dtype)))
    act_batch = Variable(torch.from_numpy(act_batch).type(dtype))
    fig=plt.figure(figsize=(16, 6))
    
    trajectories = obs_batch
    for i in range(rollout_len):
        act = Variable(torch.from_numpy(onehot_action(np.expand_dims(act_batch1[:,i],1),num_actions,1,num_samples)).type(dtype))
        trajectories = torch.cat((trajectories,norm_frame_Q_GAN(G(trajectories[:,-1*frame_history_len:,:,:],act))), dim = 1)
    
    for j in range(num_samples):
        next_frame = unnorm_frame(next_obs_batch.data.cpu().numpy()[j]).astype('uint8')
        for i in range(rollout_len):
            x = np.expand_dims(next_frame[i], axis=0)
            img = np.transpose(np.repeat(x, [3], axis=0), (1, 2, 0))
            fig.add_subplot(2, num_samples * rollout_len, j* rollout_len + i+1)
            plt.imshow(img)

        next_frame = unnorm_frame(trajectories[:,-1*rollout_len:,:,:].data.cpu().numpy()[j]).astype('uint8')  
        for i in range(rollout_len):
            x = np.expand_dims(next_frame[i], axis=0)
            img = np.transpose(np.repeat(x, [3], axis=0), (1, 2, 0))
            fig.add_subplot(2, num_samples * rollout_len, (num_samples + j) * rollout_len + i + 1)
            plt.imshow(img)
    
    plt.show()

def norm_frame(obs):
    x = (obs - 127.5)/norm_frame_val
    return x

def norm_frame_Q(obs):
    x = obs/255.
    return x

def unnorm_frame(obs):
    return np.clip(obs * norm_frame_val + 127.5,0., 255.)

def norm_frame_Q_GAN(obs):
    return torch.clamp(obs,-1*127.5/norm_frame_val, 127.5/norm_frame_val)

assert type(env.observation_space) == gym.spaces.Box
assert type(env.action_space)      == gym.spaces.Discrete

img_h, img_w, img_c = env.observation_space.shape
input_arg = frame_history_len * img_c

# Set the base tree
leaves_size = num_actions**lookahead
def base_generator ():
    tree_base = np.zeros((leaves_size,lookahead)).astype('uint8')
    for i in range(leaves_size):
        n = i
        j = 0
        while n:
            n, r = divmod(n, num_actions)
            tree_base[i,lookahead-1-j] = r
            j = j + 1
    tree_base_onehot = torch.from_numpy(onehot_action(tree_base,num_actions,lookahead,leaves_size)).type(dtype)
    return tree_base, tree_base_onehot

tree_base, tree_base_onehot = base_generator()

# MCTS planner
# base_3 = torch.from_numpy(onehot_action(np.arange(3).reshape((3,1)),num_actions,1,3)).type(dtype)

# def state_generator(G,state,depth):
#     if depth == lookahead-1:
#         G(state,base_3)
#     else:
#         state_generator(state,depth+1):
        
    
def MCTS_planning(G, Q, RP, state, t):
    var_tree_base_onehot = Variable(tree_base_onehot)
    state = Variable(state.repeat(leaves_size,1,1,1))
    trajectories = state
    for i in range(lookahead):
        act = Variable(torch.from_numpy(onehot_action(np.expand_dims(tree_base[:,i],1),num_actions,1,leaves_size)).type(dtype))
        trajectories = torch.cat((trajectories,norm_frame_Q_GAN(G(trajectories[:,-1*frame_history_len:,:,:],act))), dim = 1)
#     trajectories = torch.cat((state,norm_frame_Q_GAN(G(state,var_tree_base_onehot))), dim = 1)
    leaves_Q = Q(trajectories[:,-4:,:,:])
    leaves_Q_max = gamma **(lookahead) * leaves_Q.data.max(1)[0].cpu()
    predicted_cum_rew = RP(norm_frame_Q_GAN(trajectories[:,:-1,:,:]),var_tree_base_onehot)
    predicted_cum_return = torch.zeros(leaves_size)
    for i in range(lookahead):
        predicted_cum_return = gamma * predicted_cum_return + \
            (predicted_cum_rew.data[:,((lookahead-i-1)*num_rewards):((lookahead-i)*num_rewards)].max(1)[1].cpu()-1).type(torch.FloatTensor)
    GATS_action = leaves_Q_max + predicted_cum_return
    return_action = int(tree_base[GATS_action.max(0)[1],0])
    # update gan replay buffer
    if t > learning_starts_Q_GAN:
        for i in range(lookahead):
            idx = lookahead - i - 1
            selected_idx = np.random.randint(0, leaves_size)
            # save memory by 1/3 by saving all frames, acts, rewards together
            frames = trajectories[selected_idx, idx:idx + frame_history_len + 1, :, :].data.cpu().numpy()
            act_batch = tree_base[selected_idx,idx]
            rew_batch = (predicted_cum_rew[selected_idx,((idx) * num_rewards):((idx + 1)*num_rewards)].max(0)[1] - 1).data.cpu().numpy()
            gan_replay_buffer.add_batch(frames, act_batch, rew_batch)
    return return_action


# Construct an epilson greedy policy with given exploration schedule
def select_epilson_greedy_action(Q,G,RP, state, t):
    sample = random.random()
    eps_threshold = exploration.value(t)
    if sample > eps_threshold:
        state = norm_frame(torch.from_numpy(state).type(dtype).unsqueeze(0))
        if t < learning_starts_Q_GAN:
            return Q(Variable(state, volatile=True)).detach().data.max(1)[1].cpu()
        else:
#             Use volatile = True if variable is only used in inference mode, i.e. don’t save the history
            return MCTS_planning(G, Q, RP, state, t)
    else:
        return torch.IntTensor([[random.randrange(num_actions)]])

# Initialize target q function and q function
Q = q_func(input_arg, num_actions).type(dtype)
# Q_GAN = q_func(input_arg, num_actions).type(dtype)
# Q_GAN.load_state_dict(Q.state_dict())
target_Q = q_func(input_arg, num_actions).type(dtype)

# Construct Q network optimizer function
optimizer = optimizer_spec.constructor(Q.parameters(), **optimizer_spec.kwargs)
# optimizer_Q_GAN = torch.optim.Adam(Q_GAN.parameters(),lr = 1e-4,weight_decay = 0., betas = (RP_beta1, RP_beta2)) 



# losses
softmax_cross_entropy_loss = nn.CrossEntropyLoss().cuda()
L1_loss = nn.L1Loss()
L2_loss = nn.MSELoss()
lossQ_GAN = torch.nn.MSELoss().cuda()
# Reward predictor

RP = _RP(num_actions,num_rewards,lookahead).type(dtype)
RP.apply(weights_init)
trainerR = torch.optim.Adam(RP.parameters(),lr = 2e-4,weight_decay = 0.0001, betas = (RP_beta1, RP_beta2)) 


# Construct the replay buffer
replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)
gan_replay_buffer = GANReplayBuffer(gan_replay_buffer_size, frame_history_len)


# GDM
critic_iters = 1
actor_iters = 1
G_lr = 1e-4
D_lr = 1e-5
lambda_l1_gan = 50.
lambda_l2_gan = 50. #5, 0.0002
gan_bs = 128
gan_warmup = 10000

G = _netG(in_channels=frame_history_len, num_actions = num_actions, lookahead = 1, ngf=24).cuda()
SND = _netD(in_channels=frame_history_len, lookahead = lookahead , num_actions=num_actions, ngf=24).cuda()
G.apply(weights_init)
SND.apply(weights_init)

trainerG = optim.Adam(G.parameters(), weight_decay = 0.001,lr=G_lr, betas=(0.5, 0.999))
trainerSND = optim.SGD(SND.parameters(), weight_decay = 0.1, lr = D_lr, momentum=0.9)

num_param_updates = 0
LOG_EVERY_N_STEPS = 10000

tot_clipped_reward = []
tot_reward = []
frame_count_record = []
tot_clipped_reward_epi = []
tot_reward_epi = []
frame_count_record_epi = []

moving_average_clipped = 0.
moving_average = 0.
cum_clipped_reward = 0
cum_reward = 0
cum_clipped_reward_epi = 0
cum_reward_epi = 0
epi = 0
t = 0
last_obs = env.reset()
tot_D = []
tot_G = []
tot_l2 = []
tot_l1 = []
tot_Q_GAN = []
tot_rew_err = []
tot_rew_err_nonzero = []


gen_steps = 0
tot_steps = 0

while t < 5000001:
    t = t + 1 
    if t == 10000:
        reward_learning_freq = 8
        gan_learning_freq = 8
    if t == 50000:
        reward_learning_freq = 16
        gan_learning_freq = 16
    if t == 100000:
        reward_learning_freq = 24
        gan_learning_freq = 24
    last_idx = replay_buffer.store_frame(last_obs)
    recent_observations = replay_buffer.encode_recent_observation()
    if t> learning_starts:
        action = select_epilson_greedy_action(Q, G , RP , recent_observations, t)
    else:
        action = random.randrange(num_actions)
    apply_action = action
    if int(action != 0):
        apply_action = action + 1
        
    obs, reward, done, done_epi, _ = env.step(apply_action)
    # clip rewards between -1 and 1
    cum_reward += reward
    cum_reward_epi += reward
    reward = max(-1.0, min(reward, 1.0))
    cum_clipped_reward += reward
    cum_clipped_reward_epi += reward    
    # Store other info in replay memory
    replay_buffer.store_effect(last_idx, action, reward, done)
    # Resets the environment when reaching an episode boundary.
    
    if t % 50000. == 0. :
        logging.error('env:%s,epis[%d],durat[%d],fnum=%d, cum_cl_rew = %d, cum_rew = %d,tot_cl = %d , tot = %d'\
          %(env_name, epi,frame_count_record[-1]-frame_count_record[-2],t,tot_clipped_reward[-1],tot_reward[-1],moving_average_clipped,moving_average))
    if done:
        obs = env.reset()
        epi = epi + 1
        tot_clipped_reward = np.append(tot_clipped_reward, cum_clipped_reward)
        tot_reward = np.append(tot_reward, cum_reward)
        if done_epi:
            tot_clipped_reward_epi = np.append(tot_clipped_reward_epi, cum_clipped_reward_epi)
            tot_reward_epi = np.append(tot_reward_epi, cum_reward_epi)
            frame_count_record_epi = np.append(frame_count_record_epi,t)
            cum_clipped_reward_epi = 0
            cum_reward_epi = 0
        cum_clipped_reward = 0
        cum_reward = 0
        frame_count_record = np.append(frame_count_record,t)
        moving_average_clipped = np.mean(tot_clipped_reward[-1*min(100,epi):])
        moving_average = np.mean(tot_reward[-1*min(100,epi):])
    if t % 100000 == 0:
        fnam = './data/clippted_rew_%s' %(env_name)
        np.save(fnam,tot_clipped_reward)
        fnam = './data/tot_rew_%s' %(env_name)
        np.save(fnam,tot_reward)
        fnam = './data/frame_count_%s' %(env_name)
        np.save(fnam,frame_count_record)
        fnam = './data/rew_err_%s' %(env_name)
        fnam = './data/clippted_rew_epi_%s' %(env_name)
        np.save(fnam,tot_clipped_reward_epi)
        fnam = './data/tot_rew_epi_%s' %(env_name)
        np.save(fnam,tot_reward_epi)
        fnam = './data/frame_count_epi_%s' %(env_name)
        np.save(fnam,frame_count_record_epi)
        fnam = './data/rew_err_%s' %(env_name)
        np.save(fnam,tot_rew_err)
        fnam = './data/rew_err_nonzero_%s' %(env_name)
        np.save(fnam,tot_rew_err_nonzero)
        fnam = './data/tot_D_%s' %(env_name)
        np.save(fnam,tot_D)
        fnam = './data/tot_G_%s' %(env_name)
        np.save(fnam,tot_G) 
        fnam = './data/tot_l2_%s' %(env_name)
        np.save(fnam,tot_l2)
        fnam = './data/tot_l1_%s' %(env_name)
        np.save(fnam,tot_l1)
#         fnam = './data/tot_Q_GAN_%s' %(env_name)
#         np.save(fnam,tot_Q_GAN)

    if t % 100000 == 0:
        fdqn = './data/target_%s_%d' % (env_name,int(epi / 10))
        torch.save(Q.state_dict(), fdqn)
            
    last_obs = obs
    
    # Learning the RP:
    if (t > reward_learning_starts and
            t % reward_learning_freq == 0 and
            replay_buffer.can_sample(bs)):
        obs, act, rew = replay_buffer.reward_sample(bs,lookahead)
        reward_obs, reward_act, reward_rew = replay_buffer.nonzero_reward_sample(reward_bs,lookahead)
        obs_batch = Variable(norm_frame(torch.from_numpy(np.concatenate((obs,reward_obs),axis=0)).type(dtype)))
        act_ = np.concatenate((act,reward_act),axis=0)
        act_batch = onehot_action(act_,num_actions,lookahead,bs+reward_bs)
        act_batch = Variable(torch.from_numpy(act_batch).type(dtype))
        rew_batch = Variable(torch.from_numpy(np.concatenate((rew,reward_rew),axis=0)).long().cuda())
        reward_labels = rew_batch + 1
        trainerR.zero_grad()
        loss = 0

        trajectories = obs_batch
        for ijk in range(lookahead-1):
            act = Variable(torch.from_numpy(onehot_action(np.expand_dims(act_[:,ijk],1),num_actions,1,reward_bs+bs)).type(dtype))
            trajectories = torch.cat((trajectories,norm_frame_Q_GAN(G(trajectories[:,-1*frame_history_len:,:,:],act))), dim = 1)

        predicted_cum_rew = RP(trajectories.detach(),act_batch)
        for ind in range(lookahead):
            outputs = predicted_cum_rew[:,((lookahead-ind-1)*3):((lookahead-ind)*3)]
            loss = loss + softmax_cross_entropy_loss(outputs , reward_labels[:,lookahead-ind-1,0])

        loss.backward()
        trainerR.step()
        if t % 100000 == 0:
            reg_rew, non_ze = evaluation(RP,G,replay_buffer,norm_frame,norm_frame_Q_GAN,lookahead,num_actions,frame_history_len)
            logging.error('Accuracy of RP model on rewards:%d and on non_zero %d' %(reg_rew, non_ze))
            tot_rew_err = np.append(tot_rew_err, reg_rew)
            tot_rew_err_nonzero = np.append(tot_rew_err_nonzero, non_ze)


    # Learning the GDM:
    if (t > gan_learning_starts and
            t % gan_learning_freq == 0 and
            replay_buffer.can_sample(gan_bs)):
        for ii in range(critic_iters):
            obs_batch, act_batch_, next_obs_batch = replay_buffer.GAN_sample(gan_bs,lookahead)
            act_batch = onehot_action(act_batch_,num_actions,lookahead,gan_bs)
            obs_batch = Variable(norm_frame(torch.from_numpy(obs_batch).type(dtype)))
            next_obs_batch = Variable(norm_frame(torch.from_numpy(next_obs_batch).type(dtype)))
            act_batch = Variable(torch.from_numpy(act_batch).type(dtype))

            SND.zero_grad()
            cat_real = torch.cat((obs_batch,next_obs_batch), dim = 1)

            cat_fake = obs_batch
            trajectories = obs_batch
            for ijk in range(lookahead):
                act = Variable(torch.from_numpy(onehot_action(np.expand_dims(act_batch_[:,ijk],1),num_actions,1,gan_bs)).type(dtype))
                if gen_steps > gan_warmup:
                    sample = random.random()
                    eps_threshold = exploration_G.value(gen_steps-gan_warmup)
                else:
                    sample = 1
                    eps_threshold = 0
                fake = norm_frame_Q_GAN(G(trajectories[:,-1*frame_history_len:,:,:].detach(),act).detach())
                cat_fake = torch.cat((cat_fake,fake), dim = 1)
                if sample > eps_threshold:
                    trajectories = torch.cat((trajectories, fake), dim = 1)
                else:
                    trajectories = torch.cat((trajectories,next_obs_batch[:,ijk,:,:].unsqueeze(1)), dim = 1) 

            D_real = SND(cat_real,act_batch).mean()
            D_real.backward(mone)

            D_fake = SND(cat_fake,act_batch).mean()
            D_fake.backward(one)

            D_cost = D_fake - D_real 
            Wasserstein_D = D_real - D_fake
            trainerSND.step()
            tot_steps += 1

        tot_D = np.append(tot_D, Wasserstein_D.data.cpu().numpy())
        
        for ii in range(actor_iters):
            obs_batch, act_batch_, next_obs_batch = replay_buffer.GAN_sample(gan_bs,lookahead)
            act_batch = onehot_action(act_batch_,num_actions,lookahead,gan_bs)
            obs_batch = Variable(norm_frame(torch.from_numpy(obs_batch).type(dtype)))
            next_obs_batch = Variable(norm_frame(torch.from_numpy(next_obs_batch).type(dtype)))
            act_batch = Variable(torch.from_numpy(act_batch).type(dtype))

            G.zero_grad()

            cat_fake = obs_batch
            trajectories = obs_batch
            for ijk in range(lookahead):
                act = Variable(torch.from_numpy(onehot_action(np.expand_dims(act_batch_[:,ijk],1),num_actions,1,gan_bs)).type(dtype))
                if gen_steps > gan_warmup:
                    sample = random.random()
                    eps_threshold = exploration_G.value(gen_steps-gan_warmup)
                else:
                    sample = 1
                    eps_threshold = 0
                fake = norm_frame_Q_GAN(G(trajectories[:,-1*frame_history_len:,:,:].detach(),act))
                cat_fake = torch.cat((cat_fake, fake), dim = 1)
                if sample > eps_threshold:
                    trajectories = torch.cat((trajectories, fake), dim = 1)
                else:
                    trajectories = torch.cat((trajectories, next_obs_batch[:,ijk,:,:].unsqueeze(1)), dim = 1) 

            cat_real = torch.cat((obs_batch,next_obs_batch), dim = 1)
            G_fake = SND(cat_fake,act_batch).mean()
            G_cost = G_fake
            l1_loss = L1_loss(cat_fake[:,-1*lookahead:,:,:],next_obs_batch).mean()
            l2_loss = L2_loss(cat_fake[:,-1*lookahead:,:,:], next_obs_batch).mean()
            G_fake = l2_loss * lambda_l2_gan  + l1_loss * lambda_l1_gan - G_fake
            G_fake.backward()
            trainerG.step()
            gen_steps += 1
            tot_steps += 1
       
        tot_G = np.append(tot_G, G_cost.data.cpu().numpy())
        tot_l1 = np.append(tot_l1, l1_loss.data.cpu().numpy())
        tot_l2 = np.append(tot_l2, l2_loss.data.cpu().numpy())

        if gen_steps % 500 == 0:
            print('[%d] D: %.4f G: %.4f L2 = %.4f Gen_steps = %d Tot_steps = %d'
                  % (t,Wasserstein_D.data.cpu().numpy(), G_cost.data.cpu().numpy(), l2_loss.data.cpu().numpy(), gen_steps, tot_steps))
#             G.eval()
#             dispimage(num_samples = 1, rollout_len = lookahead)
#             G.train()
            
#             fonts = 10
#             tim = np.arange(len(tot_G))
#             belplt = plt.plot(tim,tot_G,"b", label = "tot_G")
#             belplt = plt.plot(tim,tot_D,"c", label = "tot_D")
#             belplt = plt.plot(tim,tot_l2,"r", label = "tot_l2")
#             belplt = plt.plot(tim,tot_l1,"g", label = "tot_l1")
#             plt.legend(fontsize=fonts)
#             plt.ylabel("loss",fontsize=fonts, family = 'serif')
#             plt.title("%s - D_lr: %f, G_lr: %f, l_l2: %f, bs %d" %(env_name, D_lr, G_lr, lambda_l2_gan, gan_bs),fontsize=fonts, family = 'serif')
#             plt.show()

    # Learning Q
    if (t > learning_starts and t % learning_freq == 0 and replay_buffer.can_sample(batch_size)):
        obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = replay_buffer.sample(batch_size,lookahead)
        obs_batch = norm_frame(obs_batch)
        next_obs_batch = norm_frame(next_obs_batch)
        total_batch_size = batch_size
        if t > learning_starts_Q_GAN and gan_replay_buffer.can_sample(batch_size):
            gan_obs_batch, gan_act_batch, gan_rew_batch, gan_next_obs_batch = gan_replay_buffer.sample_batch(batch_size)
            obs_batch = np.concatenate((obs_batch, gan_obs_batch), axis=0)
            act_batch = np.concatenate((act_batch, gan_act_batch), axis=0)
            rew_batch = np.concatenate((rew_batch, gan_rew_batch), axis=0)
            next_obs_batch = np.concatenate((next_obs_batch, gan_next_obs_batch), axis=0)
            gan_done_mask = np.zeros(done_mask.shape)
            done_mask = np.concatenate((done_mask, gan_done_mask), axis=0)
            total_batch_size = batch_size * 2
        obs_batch = Variable(norm_frame_Q_GAN(torch.from_numpy(obs_batch).type(dtype)))
        act_batch = Variable(torch.from_numpy(act_batch).long())
        rew_batch = Variable(torch.from_numpy(rew_batch)).type(dtype)
        next_obs_batch = Variable(norm_frame_Q_GAN(torch.from_numpy(next_obs_batch).type(dtype)))
        not_done_mask = Variable(torch.from_numpy(1 - done_mask)).type(dtype)
        if USE_CUDA:
            act_batch = act_batch.cuda()
        current_Q_values = Q(obs_batch).gather(1, act_batch.unsqueeze(1)).squeeze(1)
        ddqn_action = Q(next_obs_batch).max(1)[1]
        next_max_q = target_Q(next_obs_batch).detach()[torch.arange(total_batch_size).type(torch.cuda.LongTensor).unsqueeze(1),\
                          Q(next_obs_batch).max(1)[1].data.unsqueeze(1)].squeeze(1)
        next_Q_values = not_done_mask * next_max_q
        target_Q_values = rew_batch + (gamma * next_Q_values)
        bellman_error = target_Q_values - current_Q_values
        clipped_bellman_error = bellman_error.clamp(-1, 1)
        d_error = clipped_bellman_error * -1.0
        optimizer.zero_grad()
        current_Q_values.backward(d_error.data)

        optimizer.step()
        num_param_updates += 1

        if num_param_updates % target_update_freq == 0:
            target_Q.load_state_dict(Q.state_dict())
            
            
    if t % 500000 == 0:
        fdqn = './data/GAN_%s_%d' % (env_name,t)
        torch.save(G.state_dict(), fdqn)
        fdqn = './data/SND_%s_%d' % (env_name,t)
        torch.save(SND.state_dict(), fdqn)
        fdqn = './data/Reward_%s_%d' % (env_name,t)
        torch.save(RP.state_dict(), fdqn)
#         fdqn = './data/Q_GAN_%s_%d' % (env_name,t)
#         torch.save(Q_GAN.state_dict(), fdqn)
        fdqn = './data/Q_%s_%d' % (env_name,t)
        torch.save(Q.state_dict(), fdqn)
