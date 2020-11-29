import numpy as np
import torch
import os

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.
    Refer: https://github.com/Bjarten/early-stopping-pytorch
    """
    def __init__(self, save_dir, model_type, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_dir = save_dir
        self.model_type = model_type

    def __call__(self, val_loss, model, epoch):

        score = val_loss  # if using mAP, since mAP should be high
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch):
        '''Saves model when performance on validation increases.'''
        if self.verbose:
            print(f'Validation performance increased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        model_dir = os.path.join(self.save_dir,self.model_type)
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
            print("checkpoint directory " , model_dir ,  " created!")
        else:
            print("checkpoint directory " , model_dir ,  " already exists!")

        # torch.save(model.state_dict(), '{}/{}/epoch-{}-checkpoint.pt'.format(self.save_dir, self.model_type, str(epoch)))
        torch.save(model.state_dict(), '{}/{}/checkpoint.pt'.format(self.save_dir, self.model_type))
        self.val_loss_min = val_loss
