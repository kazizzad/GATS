import numpy as np
import random
import torch
from torch.autograd import Variable
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
from torch.autograd import Variable
from utils.replay_buffer import onehot_action

def evaluation(RP,G,replay_buffer,norm_frame,norm_frame_Q_GAN,lookahead,num_actions,frame_history_len,num_rewards):
    bs = 128
    iterat = 10
    correct = 0
    total = bs * iterat
    for jj in range(iterat):
        obs, act_, rew = replay_buffer.reward_sample(bs,lookahead)
        obs_batch = Variable(norm_frame(torch.from_numpy(obs).type(dtype)))
        act_batch = onehot_action(act_,num_actions,lookahead + 1,bs)
        act_batch = Variable(torch.from_numpy(act_batch).type(dtype))
        rew_batch = torch.from_numpy(rew).long().cuda()
        reward_labels = rew_batch + 1
        trajectories = obs_batch
        for i in range(lookahead):
            act = Variable(torch.from_numpy(onehot_action(np.expand_dims(act_[:,i],1),num_actions,1,bs)).type(dtype))
            trajectories = torch.cat((trajectories,norm_frame_Q_GAN(G(trajectories[:,-1*frame_history_len:,:,:],act))), dim = 1)
        predicted_cum_rew = RP(trajectories,act_batch)
        if lookahead == 1:
            for ind in range(lookahead + 1):
                outputs = predicted_cum_rew[:,num_rewards * ind: num_rewards * (ind + 1)]
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == reward_labels[:,ind,0]).sum()
        else:
            for ind in range(lookahead + 1):
                outputs = predicted_cum_rew[:,num_rewards * ind: num_rewards * (ind + 1)]
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == reward_labels.squeeze(1)[:,ind,0]).sum()
            

    reg_rew = 100. * (float(correct) / total / (lookahead + 1))

    correct = 0
    for jj in range(iterat):
        obs, act_, rew = replay_buffer.nonzero_reward_sample(bs,lookahead)
        obs_batch = Variable(norm_frame(torch.from_numpy(obs).type(dtype)))
        act_batch = onehot_action(act_,num_actions,lookahead + 1,bs)
        act_batch = Variable(torch.from_numpy(act_batch).type(dtype))
        rew_batch = torch.from_numpy(rew).long().cuda()
        reward_labels = rew_batch + 1
        
        trajectories = obs_batch
        for i in range(lookahead):
            act = Variable(torch.from_numpy(onehot_action(np.expand_dims(act_[:,i],1),num_actions,1,bs)).type(dtype))
            trajectories = torch.cat((trajectories,norm_frame_Q_GAN(G(trajectories[:,-1*frame_history_len:,:,:],act))), dim = 1)

        predicted_cum_rew = RP(trajectories,act_batch)
        if lookahead == 1:
            for ind in range(lookahead + 1):
                outputs = predicted_cum_rew[:,num_rewards * ind: num_rewards * (ind + 1)]
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == reward_labels[:,ind,0]).sum()
        else:
            for ind in range(lookahead + 1):
                outputs = predicted_cum_rew[:,num_rewards * ind: num_rewards * (ind + 1)]
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == reward_labels.squeeze(1)[:,ind,0]).sum()

    catas_rew = 100. * (float(correct) / total / (lookahead + 1))
    return reg_rew, catas_rew