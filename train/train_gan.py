import os
os.sys.path.append('..')

from train import utils_wf,utils_shs
from attacks.gan import gan
import sys



def main(opts):

    mode = opts['mode']

    "load data"
    if mode == 'wf':
        "load data"
        train_data = utils_wf.load_data_main(opts['train_data_path'], opts['batch_size'])

    elif mode == 'shs':
        "load data"
        train_data = utils_shs.load_data_main(opts['train_data_path'], opts['batch_size'])

    else:
        print('mode not in ["wf","shs"], system will exit.')
        sys.exit()

    adv_gan = gan.advGan(opts,opts['x_box_min'],opts['x_box_max'],opts['pert_box'])
    adv_gan.train(train_data)



def get_opts_wf(mode):

    return {
        'epochs': 50,
        'batch_size': 64,
        'pert_box': 0.3,
        'x_box_min': -1,
        'x_box_max': 1,
        'alpha':1,
        'num_class': 95,
        'input_size': 512,
        'mode': mode,
        'train_data_path': '../data/wf/train_NoDef_burst.csv',
    }


def get_opts_shs(mode):
    "only in SHS, adv_x need to clamp to [-1,0] to maintain incoming traffic"
    return {
        'epochs': 50,
        'batch_size': 64,
        'pert_box': 0.3,
        'x_box_min': -1,
        'x_box_max': 0,
        'alpha':1,
        'num_class': 101,
        'input_size': 256,
        'mode': mode,
        'train_data_path': '../data/shs/traffic_train.csv',
    }



if __name__ == '__main__':

    mode = 'wf'

    if mode == 'wf':
        opts = get_opts_wf(mode)
    elif mode == 'shs':
        opts = get_opts_shs(mode)
    else:
        print("mode shoud in ['wf','shs'], system will exit.")

    main(opts)
