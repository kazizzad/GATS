import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, in_channels=4, num_actions=18):
        """
        Initialize a deep Q-learning network as described in
        https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
        Arguments:
            in_channels: number of channel of input.
                i.e The number of most recent frames stacked together as describe in the paper
            num_actions: number of action-value to output, one-to-one correspondence to action in game.
        """
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.fc5(x)
    
class _RP(nn.Module):
    def __init__(self, num_actions, num_rewards, lookahead, frame_history_len):
        super(_RP, self).__init__()
        self.en_conv1 = nn.Conv2d(in_channels=frame_history_len+lookahead, out_channels = 32, kernel_size=8, stride=4, padding=0)
        self.en_conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0)
        self.en_conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0)
        self.dense4   = nn.Linear(in_features = 6272, out_features = 512)
        self.dense5   = nn.Linear(in_features = 512 + num_actions*(lookahead + 1), out_features = num_rewards * (lookahead + 1))
        
        
    def forward(self, x , a ):
        x = F.relu(self.en_conv1(x))
        x = F.relu(self.en_conv2(x))
        x = F.relu(self.en_conv3(x))
        x = F.relu(self.dense4(x.view(x.size(0), -1)))
        x = torch.cat((x,a), dim=1)
        return self.dense5(x)    
    
    