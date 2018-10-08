import numpy as np
import random
import scipy.misc
import torch

img_h,img_w = 84,84

def sample_n_unique(sampling_f, n):
    res = []
    while len(res) < n:
        candidate = sampling_f()
#         print(candidate)
        if candidate not in res:
            res.append(candidate)
    return res

class GANReplayBuffer(object):
    def __init__(self, size, frame_history_len):
        self.size = size
        self.frame_history_len = frame_history_len
        self.num_in_buffer = 0
        self.next_idx = 0

        self.obs = None
        self.action = None
        self.reward = None
    
    def add_batch(self, frames, act, rew):
        if self.obs is None:
            self.obs = np.empty([self.size] + list(frames.shape), dtype=np.float32)
            self.action = np.empty([self.size], dtype=np.uint8)
            self.reward = np.empty([self.size], dtype=np.float32)
                               
        self.obs[self.next_idx] = frames
        self.action[self.next_idx] = act
        self.reward[self.next_idx] = rew

        self.next_idx = (self.next_idx + 1) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)
    
    def can_sample(self, batch_size):
        """Returns true if `batch_size` different transitions can be sampled from the buffer."""
        return batch_size + 1 <= self.num_in_buffer
    
    def sample_batch(self, batch_size):
        idxes = sample_n_unique(lambda: random.randint(0, self.num_in_buffer - 1), batch_size)
        obs_batch = np.stack([self.obs[idx] for idx in idxes], 0)
        act_batch = np.stack([self.action[idx] for idx in idxes], 0)
        rew_batch = np.stack([self.reward[idx] for idx in idxes], 0)
        return obs_batch, act_batch, rew_batch
        
class ReplayBuffer(object):
    def __init__(self, size, frame_history_len):
        self.size = size
        self.frame_history_len = frame_history_len

        self.next_idx      = 0
        self.num_in_buffer = 0
        self.nonzero_rewards = []
        self.overwrite_idx = None


        self.obs      = None
        self.action   = None
        self.reward   = None
        self.done     = None

    def can_sample(self, batch_size):
        """Returns true if `batch_size` different transitions can be sampled from the buffer."""
        return batch_size + 1 <= self.num_in_buffer

    def _encode_sample(self, idxes):
        obs_batch      = np.concatenate([self._encode_observation(idx)[np.newaxis, :] for idx in idxes], 0)
        act_batch      = self.action[idxes]
        rew_batch      = self.reward[idxes]
        next_obs_batch = np.concatenate([self._encode_observation(idx + 1)[np.newaxis, :] for idx in idxes], 0)
        done_mask      = np.array([1.0 if self.done[idx] else 0.0 for idx in idxes], dtype=np.float32)

        return obs_batch, act_batch, rew_batch, next_obs_batch, done_mask
    
    
    def GAN_encode_sample(self, idxes, lookahead):
        obs_batch      = np.concatenate([self._encode_observation(idx)[np.newaxis, :] for idx in idxes], 0)
        gan_seq = [self.GAN_encode_observation_action(idx + 1, lookahead) for idx in idxes]
        act_batch = np.concatenate([gan_seq[i][1][np.newaxis, :,0] for i in range(len(idxes))], 0)
        reward_batch = np.concatenate([gan_seq[i][2][np.newaxis, :,0] for i in range(len(idxes))], 0)
        next_obs_batch = np.concatenate([gan_seq[i][0][np.newaxis, :] for i in range(len(idxes))], 0)

        return obs_batch, act_batch, next_obs_batch
    
    def reward_encode_sample(self, idxes, lookahead):
        obs_batch      = np.concatenate([self._encode_observation(idx)[np.newaxis, :] for idx in idxes], 0)
        seq = [self._encode_reward_action(idx + 1, lookahead) for idx in idxes]
        act_batch = np.concatenate([seq[i][0][np.newaxis, :, 0] for i in range(len(idxes))], 0)
        rew_batch = np.concatenate([seq[i][1][np.newaxis, :] for i in range(len(idxes))], 0)
        return obs_batch, act_batch, rew_batch


    def sample(self, batch_size, lookahead):
        assert self.can_sample(batch_size)
        idxes = sample_n_unique(lambda: random.randint(lookahead + self.frame_history_len, self.num_in_buffer - 2 - lookahead), batch_size)

        return self._encode_sample(idxes)
    
    
    def GAN_sample(self, batch_size, lookahead):
        assert self.can_sample(batch_size)
        #idxes = sample_n_unique(lambda: random.randint(0, self.num_in_buffer-2-lookahead), batch_size)
        idxes = sample_n_unique(lambda: (self.next_idx-random.randint(lookahead+self.frame_history_len,60000))%(self.num_in_buffer-2*lookahead-2*self.frame_history_len-1)+lookahead+self.frame_history_len, batch_size)
        self.next_idx
        return self.GAN_encode_sample(idxes,lookahead)
    
    def reward_sample(self, batch_size, lookahead):
        assert self.can_sample(lookahead)
        #idxes = sample_n_unique(lambda: random.randint(lookahead, self.num_in_buffer - 2 - lookahead), batch_size)
        idxes = sample_n_unique(lambda: (self.next_idx-random.randint(lookahead+self.frame_history_len,60000))%(self.num_in_buffer-lookahead-self.frame_history_len), batch_size)
        return self.reward_encode_sample(idxes,lookahead)

    def get_rand_nonzero_idx(self, lookahead):
        nonzero_idx = np.random.choice(self.nonzero_rewards, size=1)[0] - random.randint(0, lookahead)
        while nonzero_idx % (self.num_in_buffer-lookahead-2) != nonzero_idx:
            nonzero_idx = np.random.choice(self.nonzero_rewards, size=1)[0] - random.randint(0, lookahead)
        start_idx = nonzero_idx
        # for idx in range(start_idx, nonzero_idx):
        #     if self.done[idx % self.size]:
        #         start_idx = idx + 1
        return start_idx
    
    def nonzero_reward_sample(self, batch_size, lookahead):
        #assert self.can_sample_nonzero_rewards(lookahead)
        nonzero_idxes = np.random.choice(self.nonzero_rewards, size=batch_size)
        idxes = [self.get_rand_nonzero_idx(lookahead) for i in range(batch_size)]
        return self.reward_encode_sample(idxes, lookahead)
    
    
    def encode_recent_observation(self):
        assert self.num_in_buffer > 0
        return self._encode_observation((self.next_idx - 1) % self.num_in_buffer)

    def _encode_observation(self, idx):
        end_idx   = idx + 1 # make noninclusive
        start_idx = end_idx - self.frame_history_len
        # this checks if we are using low-dimensional observations, such as RAM
        # state, in which case we just directly return the latest RAM.
        if len(self.obs.shape) == 2:
            return self.obs[end_idx-1]
        # if there weren't enough frames ever in the buffer for context
        if start_idx < 0 and self.num_in_buffer != self.size:
            start_idx = 0
        for idx in range(start_idx, end_idx - 1):
            if self.done[idx % self.num_in_buffer]:
                start_idx = idx + 1
        missing_context = self.frame_history_len - (end_idx - start_idx)
        # if zero padding is needed for missing context
        # or we are on the boundry of the buffer
        repeat_frame = self.obs[start_idx % self.num_in_buffer]
        if start_idx < 0 or missing_context > 0:
            frames = [repeat_frame for _ in range(missing_context)]
            for idx in range(start_idx, end_idx):
                frames.append(self.obs[idx % self.num_in_buffer])
            return np.concatenate(frames, 0)
        else:
            # this optimization has potential to saves about 30% compute time \o/
            img_h, img_w = self.obs.shape[2], self.obs.shape[3]
            return self.obs[start_idx:end_idx].reshape(-1, img_h, img_w)

    def GAN_encode_observation_action(self, idx, lookahead):
        end_idx   = idx + lookahead # make noninclusive
        start_idx = idx
        # this checks if we are using low-dimensional observations, such as RAM
        # state, in which case we just directly return the latest RAM.
        if len(self.obs.shape) == 2:
            return self.obs[end_idx-1]
        # if there weren't enough frames ever in the buffer for context
        if start_idx < 0 and self.num_in_buffer != self.size:
            start_idx = 0
        for idx in range(start_idx, end_idx - 1):
            if self.done[idx % self.num_in_buffer]:
                start_idx = idx + 1
        missing_context = lookahead - (end_idx - start_idx)
        # if zero padding is needed for missing context
        # or we are on the boundry of the buffer
        repeat_frame = self.obs[start_idx % self.num_in_buffer]
        if start_idx < 0 or missing_context > 0:
            frames = [repeat_frame for _ in range(missing_context)]
            action = [0 for _ in range(missing_context)]
            reward = [0 for _ in range(missing_context)]
            for idx in range(start_idx, end_idx):
                frames.append(self.obs[idx % self.num_in_buffer])
                action.append(self.action[idx-1 % self.num_in_buffer])
                reward.append(self.reward[idx-1 % self.num_in_buffer])
            return np.concatenate(frames, 0), np.asarray(action).reshape(-1, 1),np.asarray(reward).reshape(-1, 1)
        else:
            # this optimization has potential to saves about 30% compute time \o/
            img_h, img_w = self.obs.shape[2], self.obs.shape[3]
            return self.obs[start_idx:end_idx].reshape(-1, img_h, img_w), self.action[start_idx - 1:end_idx - 1].reshape(-1, 1)\
                                                        , self.reward[start_idx - 1:end_idx - 1].reshape(-1, 1)
        
    def _encode_reward_action(self, idx, lookahead):
        end_idx   = idx + lookahead + 1  # make noninclusive
        start_idx = idx 
        if start_idx < 0 and self.num_in_buffer != self.size:
            start_idx = 0
        for idx in range(start_idx, end_idx - 1):
            if self.done[idx % self.num_in_buffer]:
                start_idx = idx + 1
        missing_context = (lookahead + 1) - (end_idx - start_idx)
        # if zero padding is needed for missing context
        # or we are on the boundry of the buffer
        if start_idx < 0 or missing_context > 0:
            action = [0 for _ in range(missing_context)]
            reward = [0 for _ in range(missing_context)]
            for idx in range(start_idx, end_idx):
                action.append(self.action[(idx-1) % self.num_in_buffer])
                reward.append(self.reward[(idx-1) % self.num_in_buffer])
            return np.asarray(action).reshape(-1, 1),np.asarray(reward).reshape(-1, 1)
        else:
            # this optimization has potential to saves about 30% compute time \o/
            return self.action[start_idx-1:end_idx-1].reshape(-1, 1), self.reward[start_idx-1:end_idx-1].reshape(-1, 1)
        
    def store_frame(self, frame):
        # make sure we are not using low-dimensional observations, such as RAM
        if len(frame.shape) > 1:
            # transpose image frame into (img_c, img_h, img_w)
            frame = scipy.misc.imresize(frame.mean(2), (img_h,img_w)).reshape([1,img_h,img_w])
#             frame = frame.transpose(2, 0, 1)

        if self.obs is None:
            self.obs      = np.empty([self.size] + list(frame.shape), dtype=np.uint8)
            self.action   = np.empty([self.size],                     dtype=np.int32)
            self.reward   = np.empty([self.size],                     dtype=np.float32)
            self.done     = np.empty([self.size],                     dtype=np.bool)

        self.obs[self.next_idx] = frame

        ret = self.next_idx
        
        if(self.overwrite_idx != None and self.next_idx == self.nonzero_rewards[self.overwrite_idx]):
            self.nonzero_rewards.pop(self.overwrite_idx)
            if self.overwrite_idx >= len(self.nonzero_rewards):
                self.overwrite_idx = None
        
        if (self.next_idx + 1) >= self.size and len(self.nonzero_rewards):
            self.overwrite_idx = 0
            
        self.next_idx = (self.next_idx + 1) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)

        return ret

    def store_effect(self, idx, action, reward, done):
        self.action[idx] = action
        self.reward[idx] = reward
        self.done[idx]   = done
        
        if (reward != 0):
            if self.overwrite_idx == None:
                self.nonzero_rewards.append(idx)
            else:
                self.nonzero_rewards.insert(self.overwrite_idx, idx)
                self.overwrite_idx += 1

        
def onehot_action(act_batch,num_actions,lookahead,batch_size):
    bias = np.arange(lookahead)*num_actions
    act_batch_tran = np.transpose(np.clip(act_batch,0,num_actions-1)+bias,(1,0))
    act_batch = np.zeros((batch_size, num_actions*lookahead))
    act_batch[np.arange(batch_size), act_batch_tran ] = 1
    return act_batch