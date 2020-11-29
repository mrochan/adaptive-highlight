import torch
import torch.nn as nn
from torch.nn import functional as F


class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = None
        self.bias = None
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and \
               self.bias is not None, "Please assign AdaIN weight first"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])
        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)
        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class Self_Attn(nn.Module):
    """ Self attention Network"""
    def __init__(self, in_dim, activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)

        self.softmax  = nn.Softmax(dim=-1) # across columns for each row

    def forward(self, x):
        """
            inputs :
            -------
                x : input feature maps (B X C X W X H)
            
            returns :
            --------
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N
        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)

        return out, attention


class FCSN_encoder(nn.Module):
    def __init__(self):
        super(FCSN_encoder, self).__init__()
        # conv1 (input shape (batch_size X Channel X H X W))
        self.conv1_1 = nn.Conv2d(8192, 64, (1, 3), padding=(0, 100))
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, (1, 3), padding=(0, 1))
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d((1, 2), stride=(1, 2), ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, (1, 3), padding=(0, 1))
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, (1, 3), padding=(0, 1))
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d((1, 2), stride=(1, 2), ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, (1, 3), padding=(0, 1))
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, (1, 3), padding=(0, 1))
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, (1, 3), padding=(0, 1))
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d((1, 2), stride=(1, 2), ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, (1, 3), padding=(0, 1))
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, (1, 3), padding=(0, 1))
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, (1, 3), padding=(0, 1))
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d((1, 2), stride=(1, 2), ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, (1, 3), padding=(0, 1))
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, (1, 3), padding=(0, 1))
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, (1, 3), padding=(0, 1))
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d((1, 2), stride=(1, 2), ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, (1, 7))
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, (1, 1))
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

    def forward(self, x):
        """
        input:
        ------
                Video, x:  batch_size X 8192 X 1 X NFrames) [note: NFrames indicate strided (e.g. 8 or 16) C3D features from video V]
        returns:
        --------
                drop7: torch.Size([1, 4096, 1, k])
                pool4: torch.Size([1, 512, 1, m])
        """
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)
        pool4 = h  # 1/16

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)
        drop7 = h

        return drop7, pool4

class FCSN_decoder(nn.Module):
    def __init__(self, n_class=2, is_history=False, is_dec_affine=False, is_insnorm_layer=True):
        super(FCSN_decoder, self).__init__()

        self.is_history = is_history
        self.is_dec_affine = is_dec_affine
        self.is_insnorm_layer = is_insnorm_layer

        self.score_fr = nn.Conv2d(4096, n_class, (1, 1))
        self.score_pool4 = nn.Conv2d(512, n_class, (1, 1))

        self.upscore2 = nn.ConvTranspose2d(
            n_class, n_class, (1, 4), stride=(1, 2), bias=False)
        self.upscore16 = nn.ConvTranspose2d(
            n_class, n_class, (1, 32), stride=(1, 16), bias=False)

        # adaptive instance norm layers in the decoder
        if self.is_history == True and self.is_insnorm_layer==True:
            self.instance_norm_enc_out_drop7 = AdaptiveInstanceNorm2d(4096)
            self.instance_norm_enc_out_pool4_skip = AdaptiveInstanceNorm2d(512)
        # instance norm layers in the decoder
        if self.is_history==False and self.is_insnorm_layer==True:
            self.instance_norm_enc_out_drop7 = nn.InstanceNorm2d(4096, affine=self.is_dec_affine)
            self.instance_norm_enc_out_pool4_skip = nn.InstanceNorm2d(512, affine=self.is_dec_affine)

        if self.is_history==False and self.is_insnorm_layer==False:
            # no instance norm layer in the decoder
            pass

    def forward(self, enc_drop7, enc_pool4_skip, inp):
        """
        input:
        ------
                Video, x:  (batch_size X 8192 X 1 X NFrames) [note: NFrames indicate strided (e.g. 8 or 16) C3D features from video V]
                enc_drop7: torch.Size([1, 4096, 1, k]) from FCSN_encoder
                enc_pool4_skip: torch.Size([1, 512, 1, m]) from FCSN_encoder
        returns:
        --------
                h: (batch_size X 2 X 1 X NFrames)
        """
        if enc_drop7.shape[3] > 1 and self.is_insnorm_layer==True:
            enc_drop7 = self.instance_norm_enc_out_drop7(enc_drop7)
            # print(self.instance_norm_enc_out_drop7.weight.sum())
            # print(self.instance_norm_enc_out_drop7.bias.sum())
            # print('\n')
        else:
            pass
        h = self.score_fr(enc_drop7)
        h = self.upscore2(h)
        upscore2 = h  # 1/16
        
        if enc_pool4_skip.shape[3] > 1 and self.is_insnorm_layer==True:
            enc_pool4_skip = self.instance_norm_enc_out_pool4_skip(enc_pool4_skip)
        else:
            pass
        h = self.score_pool4(enc_pool4_skip)
        h = h[:, :, :, 5:5 + upscore2.size()[3]]
        score_pool4c = h  # 1/16

        h = upscore2 + score_pool4c
        h = self.upscore16(h)
        h = h[:, :, :, 27:27 + inp.size()[3]].contiguous()

        return h

