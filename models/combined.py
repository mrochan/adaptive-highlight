from torch import nn
import torch.nn.functional as F

from models.highlight_fcsn import FCSN_encoder, FCSN_decoder
from models.history import HistoryNet

# Ref: https://github.com/nvlabs/FUNIT/
def assign_adain_params(adain_params, model):
    # assign the adain_params to the AdaIN layers in model (i.e. FCSN_decoder())
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
            mean = adain_params[:, :m.num_features]
            std = adain_params[:, m.num_features:2*m.num_features]
            m.bias = mean.contiguous().view(-1)
            m.weight = std.contiguous().view(-1)
            if adain_params.size(1) > 2*m.num_features:
                adain_params = adain_params[:, 2*m.num_features:]

class CombinedNet(nn.Module):
    def __init__(self, is_history=False, is_dec_affine=False, is_insnorm_layer=True, hist_net_type=None):
        super(CombinedNet, self).__init__()
        self.is_history = is_history
        self.is_dec_affine = is_dec_affine
        self.is_insnorm_layer = is_insnorm_layer
        self.hist_net_type = hist_net_type
        self.enc_net_highlight = FCSN_encoder()
        
        # if self.is_history = True ==> AdaptiveInstanceNorm2d added to the FCSN_decoder()
        # if self.is_history = False ==> nn.InstanceNorm2d added to to the FCSN_decoder()
        self.dec_net_highlight = FCSN_decoder(n_class=2, is_history=self.is_history, is_dec_affine=self.is_dec_affine, is_insnorm_layer=self.is_insnorm_layer)

        if self.is_history:
            self.net_history = HistoryNet(net_type=self.hist_net_type)
        else:
            pass

    def forward(self, vid, usr_hist_list=list()):
        # pass video to FCSN_encoder()
        enc_out, enc_pool4_skip = self.enc_net_highlight(vid)
        
        # predict & assign the params for AdaptiveInstanceNorm2d in FCSN_decoder() if is_history=True
        if self.is_history:
            assert len(usr_hist_list)>0,"usr_hist_list is empty!"
            adaIN_params = self.net_history(usr_hist_list)
            assign_adain_params(adaIN_params, self.dec_net_highlight)            

        # FCSN_encoder outputs to the FCSN_decoder()
        pred = self.dec_net_highlight(enc_out, enc_pool4_skip, vid)
        return pred

