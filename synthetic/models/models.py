import torch.nn as nn
import torch
from torch.nn.modules import conv, Linear
import torch.nn.functional as F
from src.snlayers.snconv2d import SNConv2d

# class _netG(nn.Module):
#     def __init__(self, in_channels=4, num_actions = 18 , lookahead = 1, ngf = 32 ):
#         super(_netG, self).__init__()
        
#         self.en_conv1 = nn.Conv2d(in_channels = in_channels, out_channels = ngf, kernel_size=4, stride=2, padding=1) #42
#         self.en_norm1 = nn.BatchNorm2d(num_features = ngf)
#         self.en_relu1 = nn.LeakyReLU(negative_slope=0.2)
        
#         self.en_conv2 = nn.Conv2d(in_channels = ngf, out_channels= ngf, kernel_size=4, stride=2, padding=2) #22
#         self.en_norm2 = nn.BatchNorm2d(num_features = ngf)
#         self.en_relu2 = nn.LeakyReLU(negative_slope=0.2)

#         self.en_conv3 = nn.Conv2d(in_channels = ngf , out_channels = 2 * ngf, kernel_size=4, stride=2, padding=2) #12
#         self.en_norm3 = nn.BatchNorm2d(num_features = 2*ngf)
#         self.en_relu3 = nn.LeakyReLU(negative_slope=0.2)
        
#         self.en_conv4 = nn.Conv2d(in_channels=ngf*2, out_channels=ngf*4, kernel_size=4, stride=2, padding=1) # 6
#         self.en_norm4 = nn.BatchNorm2d(num_features = 4*ngf)
#         self.en_relu4 = nn.LeakyReLU(negative_slope=0.2)

#         self.en_conv5 = nn.Conv2d(in_channels=ngf*4, out_channels=ngf*8, kernel_size=2, stride=1, padding=1) # 6
#         self.en_norm5 = nn.BatchNorm2d(num_features = 8*ngf)
#         self.en_relu5 = nn.LeakyReLU(negative_slope=0.2)
        
#         self.en_conv6 = nn.Conv2d(in_channels=ngf*8, out_channels=ngf*16, kernel_size=2, stride=1, padding=1) # 6
#         self.en_norm6 = nn.BatchNorm2d(num_features = 16*ngf)
#         self.en_relu6 = nn.LeakyReLU(negative_slope=0.2)
        
#         self.en_conv7 = nn.Conv2d(in_channels=ngf*16, out_channels=ngf*16, kernel_size=2, stride=1, padding=1) # 6
#         self.en_norm7 = nn.BatchNorm2d(num_features = 16*ngf)
#         self.en_relu7 = nn.LeakyReLU(negative_slope=0.2)
        
#         ### DeConv
#         self.de_conv7 = nn.ConvTranspose2d(in_channels=ngf*16+num_actions*lookahead, out_channels=ngf*16, kernel_size=2, stride=1, padding=1) #6
#         self.de_norm7 = nn.BatchNorm2d(num_features = 16*ngf)
#         self.de_relu7 = nn.ReLU()
        
        
        
#         self.de_conv6 = nn.ConvTranspose2d(in_channels=ngf*16*2+num_actions*lookahead, out_channels=ngf*8, kernel_size=2, stride=1, padding=1) #6
#         self.de_norm6 = nn.BatchNorm2d(num_features = 8*ngf)
#         self.de_relu6 = nn.ReLU()
        
#         self.de_conv5 = nn.ConvTranspose2d(in_channels=ngf*8* 2+num_actions*lookahead, out_channels=ngf*4, kernel_size=2, stride=1, padding=1) #6
#         self.de_norm5 = nn.BatchNorm2d(num_features = 4*ngf)
#         self.de_relu5 = nn.ReLU()
        
#         self.de_conv4 = nn.ConvTranspose2d(in_channels=ngf * 4 * 2+num_actions*lookahead, out_channels=ngf * 2, kernel_size=4, stride=2, padding=1) #12
#         self.de_norm4 = nn.BatchNorm2d(num_features = 2*ngf)
#         self.de_relu4 = nn.ReLU()
        
#         self.de_conv3 = nn.ConvTranspose2d(in_channels=ngf * 2 * 2+num_actions*lookahead, out_channels=ngf , kernel_size=4, stride=2, padding=2) #22
#         self.de_norm3 = nn.BatchNorm2d(num_features = ngf)
#         self.de_relu3 = nn.ReLU()

#         self.de_conv2 = nn.ConvTranspose2d(in_channels=ngf * 2 + num_actions*lookahead, out_channels=ngf, kernel_size=4, stride=2, padding=2) #42
#         self.de_norm2 = nn.BatchNorm2d(num_features = ngf)
#         self.de_relu2 = nn.ReLU()

#         self.de_conv1 = nn.ConvTranspose2d(in_channels=ngf + num_actions*lookahead, out_channels=lookahead, kernel_size=4, stride=2, padding=1) #84
#         # self.de_act1  = nn.Sigmoid() 
#         self.de_act1  = nn.Tanh()    
        
#     def forward(self, x, a):
#         x = self.en_conv1(x)
#         x = self.en_norm1(x)
#         x = self.en_relu1(x)#42
        
#         x = self.en_conv2(x)
#         x = self.en_norm2(x)
#         x2 = self.en_relu2(x)#22
        
#         x = self.en_conv3(x2)
#         x = self.en_norm3(x)
#         x3 = self.en_relu3(x)#12
        
#         x = self.en_conv4(x3)
#         x = self.en_norm4(x)
#         x4 = self.en_relu4(x)#6
        
#         x = self.en_conv5(x4)
#         x = self.en_norm5(x)
#         x5 = self.en_relu5(x)#6
        
#         x = self.en_conv6(x5)
#         x = self.en_norm6(x)
#         x6 = self.en_relu6(x)#6
        
#         x = self.en_conv7(x6)
#         x = self.en_norm7(x)
#         x7 = self.en_relu7(x)#6
        
#         #deconv
        
#         a_tile7 = a.repeat(x7.shape[-2],x7.shape[-1],1,1).permute(2, 3, 0,1)
#         x       = self.de_conv7(torch.cat((x7,a_tile7),dim=1))
#         x       = self.de_norm7(x)
#         dx6     = self.de_relu7(x)
        
#         a_tile6 = a.repeat(x6.shape[-2],x6.shape[-1],1,1).permute(2, 3, 0,1)
#         x       = self.de_conv6(torch.cat((dx6,x6,a_tile6),dim=1))
#         x       = self.de_norm6(x)
#         dx5     = self.de_relu6(x)
        
#         a_tile5 = a.repeat(dx5.shape[-2],dx5.shape[-1],1,1).permute(2, 3, 0,1)
#         x       = self.de_conv5(torch.cat((dx5,x5,a_tile5),dim=1))
#         x       = self.de_norm5(x)
#         dx4     = self.de_relu5(x)
        
#         a_tile4 = a.repeat(dx4.shape[-2],dx4.shape[-1],1,1).permute(2, 3, 0,1)
#         x       = self.de_conv4(torch.cat((dx4,x4,a_tile4),dim=1))
#         x       = self.de_norm4(x)
#         dx3     = self.de_relu4(x)

#         a_tile3 = a.repeat(dx3.shape[-2],dx3.shape[-1],1,1).permute(2, 3, 0,1)
#         x       = self.de_conv3(torch.cat((dx3,x3,a_tile3),dim=1))
#         x       = self.de_norm3(x)
#         dx2     = self.de_relu3(x)
        
#         a_tile2 = a.repeat(dx2.shape[-2],dx2.shape[-1],1,1).permute(2, 3, 0,1)
#         x       = self.de_conv2(torch.cat((dx2,x2,a_tile2),dim=1))
#         x       = self.de_norm2(x)
#         dx1     = self.de_relu2(x)
        
#         a_tile1 = a.repeat(dx1.shape[-2],dx1.shape[-1],1,1).permute(2, 3, 0,1)
#         x       = self.de_conv1(torch.cat((dx1, a_tile1), dim=1))
#         x       = self.de_act1(x)
#         return x

class _netG(nn.Module):
    def __init__(self, in_channels=4, num_actions = 18 , lookahead = 1, ngf = 32 ):
        super(_netG, self).__init__()
        
        self.en_conv1 = nn.Conv2d(in_channels = in_channels, out_channels = ngf, kernel_size=4, stride=2, padding=1) #42
        self.en_norm1 = nn.BatchNorm2d(num_features = ngf)
        self.en_relu1 = nn.LeakyReLU(negative_slope=0.2)
        
        self.en_conv2 = nn.Conv2d(in_channels = ngf, out_channels= ngf * 2, kernel_size=4, stride=2) #20
        self.en_norm2 = nn.BatchNorm2d(num_features = 2*ngf)
        self.en_relu2 = nn.LeakyReLU(negative_slope=0.2)

        self.en_conv3 = nn.Conv2d(in_channels = ngf * 2, out_channels = 4 * ngf, kernel_size=4, stride=2, padding=1) #10
        self.en_norm3 = nn.BatchNorm2d(num_features = 4*ngf)
        self.en_relu3 = nn.LeakyReLU(negative_slope=0.2)
        
        self.en_conv4 = nn.Conv2d(in_channels=ngf*4, out_channels=ngf*8, kernel_size=4, stride=2, padding=1) # 5
        self.en_norm4 = nn.BatchNorm2d(num_features = 8*ngf)
        self.en_relu4 = nn.LeakyReLU(negative_slope=0.2)

        self.en_conv5 = nn.Conv2d(in_channels=ngf*8, out_channels=ngf*8, kernel_size=3, stride=1, padding=1) # 5
        self.en_norm5 = nn.BatchNorm2d(num_features = 8*ngf)
        self.en_relu5 = nn.LeakyReLU(negative_slope=0.2)
        
        self.en_conv6 = nn.Conv2d(in_channels=ngf*8, out_channels=ngf*8, kernel_size=3, stride=1, padding=1) # 5
        self.en_norm6 = nn.BatchNorm2d(num_features = 8*ngf)
        self.en_relu6 = nn.LeakyReLU(negative_slope=0.2)
        
        ### DeConv
        self.de_conv6 = nn.ConvTranspose2d(in_channels=ngf*8+num_actions*lookahead, out_channels=ngf*8, kernel_size=3, stride=1, padding=1) #6
        self.de_norm6 = nn.BatchNorm2d(num_features = 8*ngf)
        self.de_relu6 = nn.ReLU()
        
        self.de_conv5 = nn.ConvTranspose2d(in_channels=ngf*8* 2+num_actions*lookahead, out_channels=ngf*8, kernel_size=3, stride=1, padding=1) #6
        self.de_norm5 = nn.BatchNorm2d(num_features = 8*ngf)
        self.de_relu5 = nn.ReLU()
        
        self.de_conv4 = nn.ConvTranspose2d(in_channels=ngf * 8 * 2+num_actions*lookahead, out_channels=ngf * 4, kernel_size=4, stride=2, padding=1) #12
        self.de_norm4 = nn.BatchNorm2d(num_features = 4*ngf)
        self.de_relu4 = nn.ReLU()
        
        self.de_conv3 = nn.ConvTranspose2d(in_channels=ngf * 4 * 2+num_actions*lookahead, out_channels=ngf * 2, kernel_size=4, stride=2, padding=1) #22
        self.de_norm3 = nn.BatchNorm2d(num_features = 2*ngf)
        self.de_relu3 = nn.ReLU()

        self.de_conv2 = nn.ConvTranspose2d(in_channels=ngf * 2 * 2 + num_actions*lookahead, out_channels=ngf, kernel_size=4, stride=2) #42
        self.de_norm2 = nn.BatchNorm2d(num_features = ngf)
        self.de_relu2 = nn.ReLU()

        self.de_conv1 = nn.ConvTranspose2d(in_channels=ngf + num_actions*lookahead, out_channels=lookahead, kernel_size=4, stride=2, padding=1) #84
        # self.de_act1  = nn.Sigmoid() 
        self.de_act1  = nn.Tanh()    
        
    def forward(self, x, a):
        x = self.en_conv1(x)
        x = self.en_norm1(x)
        x = self.en_relu1(x)#42
        
        x = self.en_conv2(x)
        x = self.en_norm2(x)
        x2 = self.en_relu2(x)#22
        
        x = self.en_conv3(x2)
        x = self.en_norm3(x)
        x3 = self.en_relu3(x)#12
        
        x = self.en_conv4(x3)
        x = self.en_norm4(x)
        x4 = self.en_relu4(x)#6
        
        x = self.en_conv5(x4)
        x = self.en_norm5(x)
        x5 = self.en_relu5(x)#6
        
        x = self.en_conv6(x5)
        x = self.en_norm6(x)
        x6 = self.en_relu6(x)#6
        
        #deconv
        
        a_tile6 = a.repeat(x6.shape[-2],x6.shape[-1],1,1).permute(2, 3, 0,1)
        x       = self.de_conv6(torch.cat((x6,a_tile6),dim=1))
        x       = self.de_norm6(x)
        dx5     = self.de_relu6(x)
        
        a_tile5 = a.repeat(dx5.shape[-2],dx5.shape[-1],1,1).permute(2, 3, 0,1)
        x       = self.de_conv5(torch.cat((dx5,x5,a_tile5),dim=1))
        x       = self.de_norm5(x)
        dx4     = self.de_relu5(x)
        
        a_tile4 = a.repeat(dx4.shape[-2],dx4.shape[-1],1,1).permute(2, 3, 0,1)
        x       = self.de_conv4(torch.cat((dx4,x4,a_tile4),dim=1))
        x       = self.de_norm4(x)
        dx3     = self.de_relu4(x)

        a_tile3 = a.repeat(dx3.shape[-2],dx3.shape[-1],1,1).permute(2, 3, 0,1)
        x       = self.de_conv3(torch.cat((dx3,x3,a_tile3),dim=1))
        x       = self.de_norm3(x)
        dx2     = self.de_relu3(x)
        
        a_tile2 = a.repeat(dx2.shape[-2],dx2.shape[-1],1,1).permute(2, 3, 0,1)
        x       = self.de_conv2(torch.cat((dx2,x2,a_tile2),dim=1))
        x       = self.de_norm2(x)
        dx1     = self.de_relu2(x)
        
        a_tile1 = a.repeat(dx1.shape[-2],dx1.shape[-1],1,1).permute(2, 3, 0,1)
        x       = self.de_conv1(torch.cat((dx1, a_tile1), dim=1))
        x       = self.de_act1(x)
        return x
    
class _netD(nn.Module):
    def __init__(self, in_channels=4, lookahead = 1, num_actions = 18, ngf = 64):
        super(_netD, self).__init__()
        self.en_conv1 = SNConv2d(in_channels=in_channels+lookahead, out_channels = ngf, kernel_size=8, stride=4, padding=1)
        self.en_norm1 = nn.BatchNorm2d(num_features = ngf)

        self.en_relu2 = nn.LeakyReLU(negative_slope = 0.2, inplace=True)
        self.en_conv2 = SNConv2d(in_channels=ngf, out_channels=ngf*2, kernel_size=4, stride=2, padding=1)
        self.en_norm2 = nn.BatchNorm2d(num_features = ngf*2)

        self.en_relu3 = nn.LeakyReLU(negative_slope = 0.2, inplace=True)
        self.en_conv3 = SNConv2d(in_channels=ngf*2, out_channels=ngf*4, kernel_size=4, stride=2, padding=1)
        self.en_norm3 = nn.BatchNorm2d(num_features = ngf*4)
                
        self.en_relu4 = nn.LeakyReLU(negative_slope = 0.2, inplace=True)
        self.en_conv4 = SNConv2d(in_channels=ngf*4, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.en_norm4 = nn.BatchNorm2d(num_features = 16)

        self.dense6   = nn.Linear(in_features = 16 * 25+ num_actions*lookahead , out_features = 18)
        self.dense7   = nn.Linear(in_features = 18 + num_actions*lookahead , out_features = 1)
        
    def forward(self, x , a ):
        x = self.en_conv1(x)
        x = self.en_norm1(x)
        
        x = self.en_relu2(x)
        x = self.en_conv2(x)
        x = self.en_norm2(x)
        
        x = self.en_relu3(x)
        x = self.en_conv3(x)
        x = self.en_norm3(x)
        
        x = self.en_relu4(x)
        x = self.en_conv4(x)
        x = self.en_norm4(x)
        
        x = torch.cat((x.view(x.size(0), -1), a),dim=1)
        x = F.leaky_relu(self.dense6(x),negative_slope=0.2)
        x = self.dense7(torch.cat((x,a), dim = 1))
        return x

# class _netD(nn.Module):
#     def __init__(self, in_channels=4, lookahead = 1, num_actions = 18, ngf = 64):
#         super(_netD, self).__init__()
#         self.en_conv1 = SNConv2d(in_channels=in_channels+lookahead, out_channels = ngf, kernel_size=8, stride=4, padding=1)
#         self.en_norm1 = nn.BatchNorm2d(num_features = ngf)

#         self.en_relu2 = nn.LeakyReLU(negative_slope = 0.2, inplace=True)
#         self.en_conv2 = SNConv2d(in_channels=ngf, out_channels=ngf*2, kernel_size=4, stride=2, padding=1)
#         self.en_norm2 = nn.BatchNorm2d(num_features = ngf*2)

#         self.en_relu3 = nn.LeakyReLU(negative_slope = 0.2, inplace=True)
#         self.en_conv3 = SNConv2d(in_channels=ngf*2, out_channels=ngf*4, kernel_size=4, stride=2, padding=1)
#         self.en_norm3 = nn.BatchNorm2d(num_features = ngf*4)
        
#         self.en_relu4 = nn.LeakyReLU(negative_slope = 0.2, inplace=True)
#         self.en_conv4 = SNConv2d(in_channels=ngf*4, out_channels=16, kernel_size=3, stride=1, padding=1)
#         self.en_norm4 = nn.BatchNorm2d(num_features = 16)
        
#         # self.en_relu4 = nn.LeakyReLU(negative_slope = 0.2, inplace=True)
#         # self.en_conv4 = SNConv2d(in_channels=ngf*4, out_channels=ngf*8, kernel_size=4, stride=2, padding=1)
#         # self.en_norm4 = nn.BatchNorm2d(num_features = ngf*8)
        
#         # self.en_relu5 = nn.LeakyReLU(negative_slope = 0.2, inplace=True)
#         # self.en_conv5 = SNConv2d(in_channels=ngf*8, out_channels=8, kernel_size=4, stride=1, padding=1,bias=False)
#         # self.en_norm5 = nn.BatchNorm2d(num_features = 8)

#         self.dense6   = nn.Linear(in_features = 16 * 25+ num_actions*lookahead , out_features = 18)
#         self.dense7   = nn.Linear(in_features = 18 + num_actions*lookahead , out_features = 1)
        
#     def forward(self, x , a ):
#         x = self.en_conv1(x)
#         x = self.en_norm1(x)
        
#         x = self.en_relu2(x)
#         x = self.en_conv2(x)
#         x = self.en_norm2(x)
        
#         x = self.en_relu3(x)
#         x = self.en_conv3(x)
#         x = self.en_norm3(x)
        
#         x = self.en_relu4(x)
#         x = self.en_conv4(x)
#         x = self.en_norm4(x)
        
#         # x = self.en_relu5(x)
#         # x = self.en_conv5(x)
#         # x = self.en_norm5(x)
#         x = torch.cat((x.view(x.size(0), -1), a),dim=1)
#         x = F.leaky_relu(self.dense6(x),negative_slope=0.2)
#         x = self.dense7(torch.cat((x,a), dim = 1))
#         # x = F.sigmoid(x)
        
#         return x
   
class _netACVP(nn.Module):
    def __init__(self, in_channels=4, num_actions=3):
        super(_netACVP, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, 8, 2, (1,1))
        self.conv2 = nn.Conv2d(64, 128, 6, 2, (0, 0))
        self.conv3 = nn.Conv2d(128, 128, 6, 2, (0, 0))
        self.conv4 = nn.Conv2d(128, 128, 3, 2, (0, 0))

        self.hidden_units = 128 * 3 * 3

        self.fc5 = nn.Linear(self.hidden_units, 2048)
        self.fc_encode = nn.Linear(2048, 2048)
        self.fc_action = nn.Linear(num_actions, 2048)
        self.fc_decode = nn.Linear(2048, 2048)
        self.fc8 = nn.Linear(2048, self.hidden_units)

        self.deconv9 = nn.ConvTranspose2d(128, 128, 3, 2)
        self.deconv10 = nn.ConvTranspose2d(128, 128, 6, 2)
        self.deconv11 = nn.ConvTranspose2d(128, 128, 6, 2)
        self.deconv12 = nn.ConvTranspose2d(128, 1, 8, 2, (1,1))

        self.de_act1  = nn.Tanh()    

#         self.init_weights()
#         self.criterion = nn.MSELoss()
#         self.opt = torch.optim.Adam(self.parameters(), 1e-4)


    def init_weights(self):
        for layer in self.children():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
                nn.init.xavier_uniform(layer.weight.data)
            if not isinstance(layer, nn.Tanh) and not isinstance(layer, nn.LeakyReLU):
                nn.init.constant(layer.bias.data, 0)
        nn.init.uniform(self.fc_encode.weight.data, -1, 1)
        nn.init.uniform(self.fc_decode.weight.data, -1, 1)
        nn.init.uniform(self.fc_action.weight.data, -0.1, 0.1)

    def forward(self, obs, action):
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view((-1, self.hidden_units))
        x = F.relu(self.fc5(x))
        x = self.fc_encode(x)
        action = self.fc_action(action)
        x = torch.mul(x, action)
        x = self.fc_decode(x)
        x = F.relu(self.fc8(x))
        x = x.view((-1, 128, 3, 3))
        x = F.relu(self.deconv9(x))
        x = F.relu(self.deconv10(x))
        x = F.relu(self.deconv11(x))
        x = self.deconv12(x)
        x = self.de_act1(x)
        return x
    
# Defines the PatchGAN discriminator with the specified arguments.
# Copied from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
# Receptive field size: n=1 -> 16, n=2 -> 34, n=3 -> 70
class _netPatchD(nn.Module):
    def __init__(self, in_channels=4, lookahead = 1, num_actions = 3, ndf=64, n_layers=1):
        super(_netPatchD, self).__init__()

        input_nc = in_channels + lookahead
        norm_layer=nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=False),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        self.noaction_convs = nn.Sequential(*sequence)
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        self.action_conv1 = nn.Conv2d(ndf * nf_mult_prev + num_actions*lookahead, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=False)
        self.action_norm1 = norm_layer(ndf * nf_mult)
        self.action_relu1 = nn.LeakyReLU(0.2, True)
        self.action_conv2 = nn.Conv2d(ndf * nf_mult + num_actions*lookahead, 1, kernel_size=kw, stride=1, padding=padw)

    def forward(self, input, a):
        x = self.noaction_convs(input)
        a_tile1 = a.repeat(x.shape[-2], x.shape[-1],1,1).permute(2, 3, 0, 1)
        x = torch.cat((x, a_tile1), dim=1)
        x = self.action_conv1(x)
        x = self.action_norm1(x)
        x = self.action_relu1(x)
        a_tile2 = a.repeat(x.shape[-2], x.shape[-1],1,1).permute(2, 3, 0, 1)
        x = torch.cat((x, a_tile2), dim=1)
        x = self.action_conv2(x)
        return x