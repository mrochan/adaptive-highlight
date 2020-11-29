'''
Usage: python test.py --hist -m [[./checkpoints/path/to/checkpoint.pt/]] --hist_net attn
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import argparse

from config.config import Config
from dataloader.make_dataloader_final_dumps import get_loader
from models.combined import CombinedNet as CombinedNet16s

from tqdm import tqdm
from utils.utils import cross_entropy2d, load_checkpoint
from evaluation.metrics import *


def test(model, test_loader):
    print("===> Testing initiated...")
    # test_losses = []
    model.eval()
    
    all_ap=np.zeros(len(test_loader))
    all_msd=np.zeros(len(test_loader))

    # count the number of users skipped due to missing history (in the case of is_history=True)
    count_user_skipped = 0

    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(test_loader), total=len(test_loader)):
            ## handle the case of history vs. non-history training
            if len(data) == 4: # is_history = False
                vid_feat_tensor, gt_strided_binary, user_path, nframes = data
                # convert data to cuda
                vid_feat_tensor, gt_strided_binary = vid_feat_tensor.unsqueeze(dim=2).transpose(1, 3).cuda(), gt_strided_binary.view(1,1,-1).cuda()
                # forward to model
                output = model(vid_feat_tensor)
            else: # is_history = True i.e. len(data) = 5
                vid_feat_tensor, gt_strided_binary, usr_hist_list, user_path, nframes = data
                if len(usr_hist_list) == 0:
                    count_user_skipped = count_user_skipped + 1
                    continue
                else:
                    pass
                # convert data to cuda
                vid_feat_tensor, gt_strided_binary, usr_hist_list = vid_feat_tensor.unsqueeze(dim=2).transpose(1, 3).cuda(), gt_strided_binary.view(1,1,-1).cuda(), [hist.float().cuda() for hist in usr_hist_list]
                # forward to the model with history
                output = model(vid_feat_tensor, usr_hist_list)
                
            # softmax the logits predicted from the model 
            output = F.softmax(output, dim=1)

            # select the index 1 (which belongs to highlight prob.) in output as pred  
            output = output[:,1,:,:]

            # postprocess the output (strided) -> output (nframes) for evaluation
            processed_output = postprocess_prediction(output, nframes.item())

            # process the ground-truth of every user as per https://github.com/gyglim/video2gif_code
            processed_gt = get_user_ground_truth(user_path[0])

            # mean average precision
            mean_ap = average_precision(np.array(processed_gt).max(axis=0), processed_output)
            # append to all_ap, all_msd
            all_ap[batch_idx] = mean_ap

        avg_ap = 100*(np.sum(all_ap)/(len(test_loader)-count_user_skipped))
        print('AP=%.2f%%' % (avg_ap))

def main(args):

    # create configuration
    config = Config(is_history=args.hist, hist_net=args.hist_net)

    # set seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # prepare dataloaders
    print("===> Creating the test dataloader...")
        
    # test loader
    _, _, test_loader = get_loader(config)

    # create model
    print("===> Creating the model (is_history={})".format(str(config.is_history)))
    model = CombinedNet16s(is_history=config.is_history, is_dec_affine=config.is_dec_affine, is_insnorm_layer=config.is_insnorm_layer, hist_net_type=config.hist_net).cuda()
    print(model)

    # load checkpoint
    print('===> Loading the checkpoint:{}'.format(args.model))
    load_checkpoint(model, args.model)

    # call test()
    test(model, test_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m","--model",required=True,help="enter checkpoint path")
    parser.add_argument("--hist",help="model with user history or no history",action='store_true')
    parser.add_argument("--hist_net",type=str, required=True, help="enter history net type: attn or conv")
    args = parser.parse_args()
    main(args)
