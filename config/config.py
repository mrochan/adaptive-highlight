'''
Configuration file
'''
import pprint


class Config():
    """Config class"""

    def __init__(self, **kwargs):

        # Paths
        self.save_dir = './checkpoints'
        self.preds_dir = './predictions'
        self.train_data_path = './data/final_train.json'
        self.val_data_path = './data/final_val.json'
        self.test_data_path = './data/final_test_dict_v2.json'
        self.history_gt_dir = '/path/to/highlight/phd-gif/ground_truth_final'

        # Main params
        self.exp_name = 'adain-attn'
        self.is_history = True # True: to use history data (even for baselines), False: no history data
        self.is_l2_normalize = True  # l2 normalize features
        self.is_dec_affine = False # set 'True' for when fcsn decoder InstanceNorm2d has learnable affine. This flag applicable only when self.is_history = False
        self.lr = 1e-4
        self.is_insnorm_layer = True # True: to use InstanceNorm2d or AdaptiveInstanceNorm2d layer in decoder, False: no form ins/adain layer use in decoder 
        
        self.n_epochs = 10
        self.patience = 5 # used for earlystopping

        # other parameters
        self.batch_size = 1
        self.lr_factor = 0.1
        self.seed = 1
        self.validate_interval = 1
        self.print_interval = 1

        # other flags
        self.is_validate = True
        self.scale_lrate = False

        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        config_str = 'Configurations\n' + pprint.pformat(self.__dict__)
        return config_str


if __name__ == '__main__':
    config = Config(hist_net='attn')
    print(config)
