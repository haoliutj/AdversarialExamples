import torch
import torch.optim as optim
import numpy as np
from optparse import OptionParser
import math





def get_args():

    parser = OptionParser()
    parser.add_option('--epochs', dest='epochs',type='int',
                      default=200, help='number of epochs')
    parser.add_option('--iterations', dest='ITERS', default=1000, type='int',
                      help='number of critic iterations per generator iteration')
    parser.add_option('--channels', dest='channels', default=1, type='int',
                      help='number of channels')
    parser.add_option('--width', dest='width', default=256, type='int',
                      help='input width')
    parser.add_option('--height', dest='height', default=1, type='int',
                      help='input height')
    parser.add_option('--batch_size', dest='batch_size', default=64, type='int',
                      help='batch size')
    parser.add_option('--test_batch_size', dest='test_batch_size', default=1, type='int',
                      help='test batch size')
    parser.add_option('--learning_rate', dest='lr', default=0.0002, type='float',
                      help='learning rate')
    parser.add_option('--get_z_learning_rate', dest='get_z_lr', default=10.0, type='float',
                      help='learning rate of get z sets')
    parser.add_option('--latent_dim)', dest='latent_dim', default=100, type='int',
                      help='dimensionality of the latent space')
    parser.add_option('--display_steps)', dest='display_steps', default=10, type='int',
                      help='number of steps interval to display')
    parser.add_option('--lambda_gp)', dest='lambda_gp', default=10, type='int',
                      help='Loss weight for gradient penalty')
    parser.add_option('--inital_epoch)', dest='inital_epoch', default=0, type='int',
                      help='inital epoch')
    parser.add_option('--n_critic)', dest='n_critic', default=5, type='int',
                      help= 'number of training steps for generator per iter')
    parser.add_option('--rec_iters)', dest='rec_iters', default=[200], type='int',
                      help='the number of L of Gradient Descent steps')
    parser.add_option('--rec_rrs)', dest='rec_rrs', default=[10], type='int',
                      help='the number of different random initialization of z')
    parser.add_option('--global_step)', dest='global_step', default=3.0, type='float',
                      help='global step for adjusting learing rate')

    (options, args) = parser.parse_args()
    return options


"partial borrow from https://github.com/sky4689524/DefenseGAN-Pytorch"

def adjust_lr(optimizer, cur_lr, decay_rate=0.1, global_step=1, rec_iter=200):
    lr = cur_lr * decay_rate ** (global_step / int(math.ceil(rec_iter * 0.8)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


"""

To get R random different initializations of z from L steps of Gradient Descent.
rec_iter : the number of L of Gradient Descent steps 
rec_rr : the number of different random initialization of z

"""
def get_z_sets(model, data, lr, loss, device, rec_iter=200, rec_rr=10, input_latent=64, global_step=1):

    # the output of R random different initializations of z from L steps of GD
    z_hats_recs = torch.Tensor(rec_rr, data.size(0), input_latent)

    # the R random differernt initializations of z before L steps of GD
    z_hats_orig = torch.Tensor(rec_rr, data.size(0), input_latent)

    for idx in range(len(z_hats_recs)):

        "gaussial distribution initiation "
        # z_hat = np.random.normal(0, 1, (data.shape[0], input_latent)) # gussian distribution
        # z_hat = torch.Tensor(np.expand_dims(z_hat,axis=1)).to(device)
        "uniform distribution initiation"
        z_hat = torch.rand(data.size(0), input_latent).to(device)  #  rand:unifom distribution, randn: gaussian distribution
        z_hat = z_hat.detach().requires_grad_()

        cur_lr = lr

        optimizer = optim.SGD([z_hat], lr=cur_lr, momentum=0.7)

        z_hats_orig[idx] = z_hat.cpu().detach().clone()

        for iteration in range(rec_iter):
            optimizer.zero_grad()

            fake_x = model(z_hat)

            fake_x = fake_x.view(-1, data.size(1), data.size(2))

            reconstruct_loss = loss(fake_x, data)

            reconstruct_loss.backward()

            optimizer.step()

            cur_lr = adjust_lr(optimizer, cur_lr, global_step=global_step, rec_iter=rec_iter)

        z_hats_recs[idx] = z_hat.cpu().detach().clone()

    return z_hats_orig, z_hats_recs


"""

To get z* so as to minimize reconstruction error between generator G and an original x

"""
def get_z_star(model, data, z_hats_recs, loss, device):
    reconstructions = torch.Tensor(len(z_hats_recs))

    for i in range(len(z_hats_recs)):
        z = model(z_hats_recs[i].to(device))

        z = z.view(-1, data.size(1), data.size(2))

        reconstructions[i] = loss(z, data).cpu().item()

    min_idx = torch.argmin(reconstructions)

    return z_hats_recs[min_idx]



