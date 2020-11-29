import torch
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli
import json

# Reference: https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/loss/loss.py
def cross_entropy2d(input, target, weight=None, reduction='mean'):
    # # class weights for weighted cross-entropy
    # weight = torch.Tensor(2).zero_().cuda()
    # weight[0] = 1
    # weight[1] = 10

    # change to make code compatible with pytorch v0.4
    if '0.4' in torch.__version__:
        reduction = 'elementwise_mean'

    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h != ht and w != wt:  # upsample labels
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(
        input, target, weight=weight, ignore_index=-100, reduction=reduction
    )
    return loss


def read_json(json_file_path):
    """ Parsing a .json file
    input:
    ------
        file_path: file_path of .json file as str
    
    returns:
    --------
        data: contains the data as dict
    """
    with open(json_file_path, 'r') as fp:
        data = json.loads(fp.read())
    return data


def write_json(json_save_path, file_name, data):
    """ Dumping to a .json file
    input:
    ------
        json_save_path: directory to save
        file_name: fith name with .json extension to be saved
        data: data (e.g. dict) to be dumped in .json
    
    returns:
    --------
        None
    """
    assert file_name.endswith('.json'), "JSON writing: file_name should end with .json extension!"
    json_save_path = os.path.join(json_save_path, file_name)
    with open(json_save_path, 'w') as f:
        json.dump(data, f)

def load_checkpoint(model, chkpt_path):
    """ Loading a pre-trained model
    input:
    ------
            model: The model
            chkpt_path: checkpoint path
    returns:
    --------
            model
    """
    model.load_state_dict(torch.load(chkpt_path))
    return model

