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
from tensorboardX import SummaryWriter
from utils.utils import cross_entropy2d, load_checkpoint
from utils.earlystopping import EarlyStopping
from evaluation.metrics import *


def train(config, model, train_loader, val_loader, optimizer):
    if not os.path.exists('./runs'):
  	    os.mkdir('./runs')
    if not os.path.exists('./checkpoints'):
        os.mkdir('./checkpoints')
    writer = SummaryWriter('./runs/{}'.format(config.exp_name)) 
    early_stopping = EarlyStopping(save_dir=config.save_dir, model_type=config.exp_name, patience=config.patience, verbose=True)

    for epoch in tqdm(range(1, config.n_epochs+1)):
        highlight_loss = []
        # Training
        model.train()
        for batch_idx, data in enumerate(train_loader):         
            # zero the grads
            optimizer.zero_grad()
            # handle the case of history vs. non-history training
            if len(data) == 4: # is_history = False
                vid_feat_tensor, gt_strided_binary, user_path, nframes = data
                # convert data to cuda
                vid_feat_tensor, gt_strided_binary = vid_feat_tensor.unsqueeze(dim=2).transpose(1, 3).cuda(), gt_strided_binary.view(1,1,-1).cuda()
                # forward to model
                output = model(vid_feat_tensor)
            else: # is_history = True i.e. len(data) = 5
                vid_feat_tensor, gt_strided_binary, usr_hist_list, usr_path, nframes = data
                # check if usr_hist_list has some data
                if len(usr_hist_list) == 0:
                    continue
                else:
                    pass
                # convert data to cuda
                vid_feat_tensor, gt_strided_binary, usr_hist_list = vid_feat_tensor.unsqueeze(dim=2).transpose(1, 3).cuda(), gt_strided_binary.view(1,1,-1).cuda(), [hist.float().cuda() for hist in usr_hist_list]
                # forward to the model with history
                output = model(vid_feat_tensor, usr_hist_list)
            
            # compute loss
            loss = cross_entropy2d(output, gt_strided_binary)

            # backward and update the model
            loss.backward()
            optimizer.step()

            highlight_loss.append(loss.item())

            if batch_idx % config.print_interval == 0:
                print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                    epoch, batch_idx+1, len(train_loader), loss.item()))

        mean_highlight_loss = np.average(highlight_loss)
        writer.add_scalar('Train/loss', mean_highlight_loss, epoch)

        # Validation
        if config.is_validate and epoch % config.validate_interval == 0:
            avg_map, avg_val_loss = validate(config, model, val_loader)

            # val avg_map for early stopping 
            early_stopping(avg_map, model, epoch)
            
            if early_stopping.early_stop:
                print("Early stopping")
                break 

            writer.add_scalar('Val/mAP', avg_map, epoch)
            writer.add_scalar('Val/Loss', avg_val_loss, epoch)

    # close summary writer
    writer.close()
    return


def validate(config, model, val_loader):
    print("Validation initiated...")
    valid_losses = []
    model.eval()
    
    all_ap=np.zeros(len(val_loader))
    all_msd=np.zeros(len(val_loader))

    # count the number of users skipped due to missing history (in the case of is_history=True)
    count_user_skipped = 0

    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(val_loader), total=len(val_loader)):
            ## handle the case of history vs. non-history training
            if len(data) == 4: # is_history = False
                vid_feat_tensor, gt_strided_binary, user_path, nframes = data
                # convert data to cuda
                vid_feat_tensor, gt_strided_binary = vid_feat_tensor.unsqueeze(dim=2).transpose(1, 3).cuda(), gt_strided_binary.view(1,1,-1).cuda()
                # forward to model
                output = model(vid_feat_tensor)
            else: # is_history = True i.e. len(data) = 5
                vid_feat_tensor, gt_strided_binary, usr_hist_list, user_path, nframes = data
                # check if usr_hist_list has some data (empty history.json created for few users, see train() for comments)
                if len(usr_hist_list) == 0:
                    # print('Skipping the (val) user as history is not present, user_info: {}'.format(user_path))
                    count_user_skipped = count_user_skipped + 1
                    continue
                else:
                    pass
                # convert data to cuda
                vid_feat_tensor, gt_strided_binary, usr_hist_list = vid_feat_tensor.unsqueeze(dim=2).transpose(1, 3).cuda(), gt_strided_binary.view(1,1,-1).cuda(), [hist.float().cuda() for hist in usr_hist_list]
                # forward to the model with history
                output = model(vid_feat_tensor, usr_hist_list)
                
            # compute losses
            loss = cross_entropy2d(output, gt_strided_binary)

            valid_losses.append(loss.item())

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

        ## Since some of the users are skipped due to missing history => remove their count while aggregating mAP
        avg_ap = 100*(np.sum(all_ap)/(len(val_loader)-count_user_skipped))

        avg_val_loss = np.average(valid_losses)

        print('AP=%.2f%%' % (avg_ap))

    return avg_ap, avg_val_loss 


def test_net_with_random_data():
    # create the model
    model = CombinedNet16s(is_history=True, is_dec_affine=False, is_insnorm_layer=True, hist_net_type='attn').cuda()
    print(model)
    # create random data
    vid = torch.randn(1, 8192, 1, 848).cuda()
    hist_vids = [torch.randn(1, 8192), torch.randn(1, 8192), torch.randn(1,8192)]
    hist_vids = [hist.cuda() for hist in hist_vids]

    # forward
    out = model(vid, hist_vids)

    # dummy backward to check .grad populate
    loss = torch.sum(out, dim=1)
    loss = loss.sum()
    loss.backward()

    print("\nHighlight encoder parameter info:")
    for name, param in model.enc_net_highlight.named_parameters():
        if param.requires_grad:
            print(name, param.grad.sum()) # check the gradient sum of the parameter

    print("\nHighlight decoder parameter info:")
    for name, param in model.dec_net_highlight.named_parameters():
        if param.requires_grad:
            print(name, param.grad.sum()) # check the gradient sum of the parameter

    print("\nHighlight history net parameter info:")
    for name, param in model.net_history.named_parameters():
        if param.requires_grad:
            print(name, param.grad.sum()) # check the gradient sum of the parameter
   
    # should print (1x2x1XNframes) i.e.(highlight or non-highlight)
    print('\nCombined (encoder+decoder+history) highlight net output shape: {}'.format(out.shape)) 

def main(args):

    # create configuration
    config = Config(hist_net=args.hist_net)

    # set seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # prepare dataloaders
    print("===> Creating the dataloaders...")
    # train/val/test loader not having restriction on number of user history (when config.is_history=True)
    # also invoked when training baseline models (i.e. config.is_history=False)
    train_loader, val_loader, test_loader = get_loader(config)

    # create model
    print("===> Creating the model (is_history={})".format(str(config.is_history)))
    model = CombinedNet16s(is_history=config.is_history, is_dec_affine=config.is_dec_affine, is_insnorm_layer=config.is_insnorm_layer, hist_net_type=config.hist_net).cuda()
    print(model)

    # check if resuming training
    if args.resume:
        try:
            print('\n===> Loading checkpoint to resume:{}'.format(args.checkpoint))
            load_checkpoint(model, args.checkpoint)
        except ValueError:
            print("args.checkpoint is empty or not valid path!")
    else:
        print("\n===> No re-training...training from scratch!")

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    print("===> Training initiated..")
    train(config, model, train_loader, val_loader, optimizer)

if __name__ == '__main__':
    #test_net_with_random_data()
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume",help="resume training",action='store_true')
    parser.add_argument("-c","--checkpoint",type=str, default=None, help="enter checkpoint path")
    parser.add_argument("--hist_net",type=str, required=True, help="enter history net type: attn or conv")
    args = parser.parse_args()
    main(args)

