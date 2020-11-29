import torch
from torch import nn
import torch.nn.functional as F

from models.highlight_fcsn import FCSN_encoder, Self_Attn

class HistoryNet(nn.Module):
    def __init__(self, net_type=None):
        super(HistoryNet, self).__init__()
        self.net_type=net_type
        
        if self.net_type == 'conv':
            self.enc_net_history = FCSN_encoder()
            self.linear_residual = nn.Linear(8192, 4096)
            self.MLP = nn.Linear(4096, 2*(4096+512)) #[enc_drop7 + enc_pool4_skip]
 
        if self.net_type == 'attn':
            self.enc_net_history = Self_Attn(8192, 'relu') 
            self.MLP = nn.Linear(8192, 2*(4096+512)) #[enc_drop7 + enc_pool4_skip]

    
    def forward(self, usr_hist_list):
        # merge all in one tensor
        usr_hist_tensors = torch.cat(usr_hist_list, dim=0).transpose(0,1).unsqueeze(0).unsqueeze(2) # reshape it to forward to enc_net_history 
 
        if self.net_type == 'conv':
            enc_usr_histories, _ = self.enc_net_history(usr_hist_tensors) # enc_usr_histories: torch.Size([1, 4096, 1, k])
        
            # Since the 'enc_usr_histories' size may vary, apply avg. pooling to get the final feature of all the user histories
            nw = enc_usr_histories.size()[3]
            combined_feat_hists = F.avg_pool2d(enc_usr_histories, kernel_size=(1,nw)) # combined_feat_hists: torch.Size([1, 4096, 1, 1])
            feat_dim = combined_feat_hists.size()[1] # should be 4096
            # user history latent code
       	    Zy = combined_feat_hists.view(1, feat_dim)  # Zy: (1X4096)
            #print("Zy before:{}".format(Zy.sum()))
            # skip connection
            residual = F.avg_pool2d(usr_hist_tensors, kernel_size=(1, usr_hist_tensors.size()[3])) # torch.Size([1, 8192, 1, 1]) 
            residual = residual.squeeze(3).squeeze(2)
            lin_residual = self.linear_residual(residual)
            Zy += lin_residual
            # single layer MLP
            adaIN_para = self.MLP(Zy) # adaIN_para: (1X8 or 9216 (enc_drop7 (4096*2)+ pool4_skip (512*2))) # 4096 -> 8 or 9216 (adaIN_para directly)
        
        
        if self.net_type == 'attn':       
            # pass the video-level feats of a each user query to the history encoder
            self_attended_feats, attn_matrix = self.enc_net_history(usr_hist_tensors) # self_attended_feats: torch.Size([1,8192, 1, #num_usr_hist])      
            nw = self_attended_feats.size()[3]
            full_history_feat = F.avg_pool2d(self_attended_feats, kernel_size=(1,nw)) # full_sentence_feat: torch.Size([1, 8192, 1, 1])
            feat_dim = full_history_feat.size()[1] # should be 8192
            # latent code for user query
            Zy = full_history_feat.view(1, feat_dim)  # Zy: (1, 8192)
		    # single layer MLP
            adaIN_para = self.MLP(Zy) # adaIN_para: (9216 (enc_drop7 (4096*2)+ pool4_skip (512*2))) # 300 -> 9216 (adaIN_para directly)        
        
        return adaIN_para

