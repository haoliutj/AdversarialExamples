import os
os.sys.path.append('..')

import torch
from website_fingerprinting import utils_wf



# def params(num_class,input_size):
#     return {
#         'conv1_input_channel': 1,
#         'conv2_input_channel': 128,
#         'conv3_input_channel': 128,
#         'conv4_input_channel': 64,
#         'conv1_output_channel': 128,
#         'conv2_output_channel': 128,
#         'conv3_output_channel': 64,
#         'conv4_output_channel': 256,
#         'kernel_size1': 7,
#         'kernel_size2': 19,
#         'kernel_size3': 13,
#         'kernel_size4': 23,
#         'stride1': 1,
#         'stride2': 1,
#         'stride3': 1,
#         'stride4': 1,
#         'padding1': 3,
#         'padding2': 9,
#         'padding3': 6,
#         'padding4': 11,
#         'drop_rate1': 0.1,
#         'drop_rate2': 0.3,
#         'drop_rate3': 0.1,
#         'drop_rate4': 0.0,
#         'pool1': 2,
#         'pool2': 2,
#         'pool3': 2,
#         'pool4': 2,
#         'num_classes': num_class,
#         'input_size':input_size
#     }


def get_advX_gan(x,pert,mode,pert_box=0.3,x_box_min=-1,x_box_max=1):
    """
    given different mode, compute the adv_x
    x: torch.Tensor
    pert: torch.Tensor
    mode: ['wf','shs']
    return: torch.Tensor. adversarial example,
    """
    if mode == 'wf':
        adv_x = utils_wf.get_advX_wf_main(x,pert,pert_box)
    else:
        pert = torch.clamp(pert, -pert_box, pert_box)
        adv_x = pert + x
        adv_x = torch.clamp(adv_x, x_box_min, x_box_max)

    return adv_x
