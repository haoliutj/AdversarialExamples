import os
os.sys.path.append('..')

import torch
import torch.nn as nn
import torch.optim as optim
import utils_DenGAN
from utils_DenGAN import adjust_lr, get_z_star
import utils
from gan_model import Generator
from advGAN import models



"partial borrow from https://github.com/sky4689524/DefenseGAN-Pytorch"
class Defense_GAN:
    def __init__(self,test_data,netG,target_model,args):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.test_data = test_data
        self.netG = netG.to(self.device)
        # self.netG = Generator().to(self.device)
        self.target_model = target_model.to(self.device)
        self.loss = nn.MSELoss()
        self.opt = args


    """
    To get R random different initializations of z from L steps of Gradient Descent
    rec_iter : the number of L of Gradient Descent steps 
    rec_rr : the number of different random initialization of z
    """
    def get_z_sets(self,data,rec_iter=200,rec_rr=10):

        # the output of R random different initializations of z from L steps of GD
        z_hats_recs = torch.Tensor(rec_rr, data.size(0), self.opt.latent_dim)

        # the R random differernt initializations of z before L steps of GD
        z_hats_orig = torch.Tensor(rec_rr, data.size(0), self.opt.latent_dim)

        for idx in range(len(z_hats_recs)):

            "gaussial distribution initiation "
            # z_hat = np.random.normal(0, 1, (data.shape[0], input_latent)) # gussian distribution
            # z_hat = torch.Tensor(np.expand_dims(z_hat,axis=1)).to(device)
            "uniform distribution initiation"
            z_hat = torch.rand(data.size(0), self.opt.latent_dim).to(self.device)  # rand:unifom distribution, randn: gaussian distribution
            z_hat = z_hat.detach().requires_grad_()

            cur_lr = self.opt.get_z_lr

            optimizer = optim.SGD([z_hat], lr=cur_lr, momentum=0.7)

            z_hats_orig[idx] = z_hat.cpu().detach().clone()

            for iteration in range(rec_iter):
                optimizer.zero_grad()

                fake_x = self.netG(z_hat)

                fake_x = fake_x.view(-1, data.size(1), data.size(2))

                reconstruct_loss = self.loss(fake_x, data)

                reconstruct_loss.backward()

                optimizer.step()

                cur_lr = adjust_lr(optimizer, cur_lr, global_step=self.opt.global_step, rec_iter=rec_iter)

            z_hats_recs[idx] = z_hat.cpu().detach().clone()

        return z_hats_orig, z_hats_recs


    def defense_gan(self):

        corrects = 0
        data_epoch_size = 0


        for rec_iter in self.opt.rec_iters:
            for rec_rr in self.opt.rec_rrs:

                for batch_idx, (inputs, labels) in enumerate(self.test_data):
                    data = inputs.to(self.device)

                    "find z*"
                    _,z_sets = self.get_z_sets(data,rec_iter=rec_iter,rec_rr=rec_rr)
                    z_star = get_z_star(self.netG,data,z_sets,self.loss,self.device)

                    "generate data"
                    data_star = self.netG(z_star.to(self.device)).detach()

                    "evaluate with target model"
                    data_star = data_star.to(self.device)
                    labels = labels.to(self.device)

                    "visualize data"
                    # utils.single_traffic_plot(batch_idx,data.cpu().numpy().squeeze(),data_star.cpu().numpy().squeeze())

                    outputs = self.target_model(data_star)
                    _,preds = torch.max(outputs,1)

                    corrects += torch.sum(preds == labels.data)
                    data_epoch_size += inputs.size(0)

                    "display result every display_step"
                    if batch_idx % self.opt.display_steps == 0:
                        print('{:>3}/{:>3} average acc {:.4f}\r'.format(batch_idx + 1, len(self.test_data),
                                                                        corrects.double() / data_epoch_size))

                    del labels,outputs,preds,data,data_star,z_star

                test_acc = corrects.double() / data_epoch_size

                print('rec_iter : {}, rec_rr : {}, Test Acc: {:.4f}'.format(rec_iter, rec_rr, test_acc))




def main(adversary):

    "parameters init"
    opt = utils_DenGAN.get_args()
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu')

    "load data"
    if adversary == 'FGSM':
        print('adversary is FGSM...')
        # train_path = '../data/traffic_train_advFGSM.csv'
        test_path =  '../data/traffic_test_advFGSM.csv'
    elif adversary == 'DeepFool':
        print('adversary is DeepFool...')
        # train_path = '../data/traffic_train_advDeepFool.csv'
        test_path = '../data/traffic_test_advDeepFool.csv'
    elif adversary == 'PGD':
        print('adversary is PGD...')
        # train_path = '../data/traffic_train_advPGD.csv'
        test_path = '../data/traffic_test_advPGD.csv'
    elif adversary == 'Test':
        print('no adversary ...')
        # train_path = '../data/traffic_train.csv'
        test_path = '../data/traffic_test.csv'

    print('loading data...')
    # train_data = utils.load_data_main(train_path,opt.batch_size)  # input_shape (batch_size,1,wide 256)
    test_data = utils.load_data_main(test_path, opt.batch_size)
    # print('train_loader size: {} * {} = {}'.format(len(train_data), opt.batch_size, len(train_data) * opt.batch_size))
    print('test_loader size: {} * {} = {}'.format(len(test_data), opt.batch_size,len(test_data) * opt.test_batch_size))

    "load generator"
    print('loading generator...')
    netG = Generator()
    pretrained_generator_path = '../model/DefenseGan_G.pth'
    netG.load_state_dict(torch.load(pretrained_generator_path, map_location=device))
    netG.eval()

    "load target model"
    print('loading target model...')
    pretrained_model = "../model/target_model.pth"
    params = utils.params()
    target_model = models.target_model_1(params)
    target_model.load_state_dict(torch.load(pretrained_model, map_location=device))
    target_model.eval()


    dGAN_test = Defense_GAN(test_data,netG,target_model,opt)
    dGAN_test.defense_gan()


if __name__ == '__main__':
    adversary = ['Test','FGSM','PGD','DeepFool']
    # adversary = ['Test']
    for adv in adversary:
        main(adv)









