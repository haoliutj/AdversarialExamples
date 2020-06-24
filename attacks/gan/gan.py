import os
os.sys.path.append('..')

import torch
from train import models
from train import utils_wf,utils_shs,utils_gan
import torch.nn.functional as F
import os,sys




class advGan:
    def __init__(self,opts,x_box_min=-1,x_box_max=1,pert_box=0.3):

        self.opts = opts
        self.mode = opts['mode']
        self.model_path = '../model/' + self.mode
        self.pert_box = pert_box
        self.x_box_min = x_box_min
        self.x_box_max = x_box_max
        self.input_nc, self.gen_input_nc = 1, 1
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("CUDA Available: ", torch.cuda.is_available())

        self.generator = models.Generator(self.gen_input_nc, self.input_nc).to(self.device)
        self.discriminator = models.Discriminator(self.input_nc).to(self.device)
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=0.001)
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=0.001)

        if self.mode == 'wf':
            print('Website Fingerprinting...')
        elif self.mode == 'shs':
            print('Smart Home Speaker Fingerprinting...')

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)


    def train_batch(self,x,labels,target_model):

        "train D"
        for i in range(1):
            pert = self.generator(x)

            "cal adv_x given different mode"
            adv_x = utils_gan.get_advX_gan(x,pert,self.mode,self.pert_box,self.x_box_min,self.x_box_max,self.opts['alpha'])
            adv_x = adv_x.to(self.device)

            self.optimizer_D.zero_grad()
            loss_D = torch.mean(self.discriminator(adv_x)) - torch.mean(self.discriminator(x))
            loss_D.backward(retain_graph=True)
            self.optimizer_D.step()

        "train G"
        for i in range(1):

            self.optimizer_G.zero_grad()

            "cal G's loss in GAN"
            pred_fake = self.discriminator(adv_x)
            loss_G_fake = F.mse_loss(pred_fake, torch.ones_like(pred_fake, device=self.device))
            loss_G_fake.backward(retain_graph=True)

            "calculate perturbation norm"
            C = 0.1
            loss_perturb = torch.mean(torch.norm(pert.view(pert.shape[0], -1), 2, dim=1))
            # loss_perturb = torch.max(loss_perturb - C, torch.zeros(1, device=self.device))

            "cal adv loss"
            logits_model = target_model(adv_x)
            probs_model = F.softmax(logits_model, dim=1)
            onehot_labels = torch.eye(self.opts['num_class'], device=self.device)[labels]

            # C&W loss function
            real = torch.sum(onehot_labels * probs_model, dim=1)
            other, _ = torch.max((1 - onehot_labels) * probs_model - onehot_labels * 10000, dim=1)
            zeros = torch.zeros_like(other)
            loss_adv = torch.max(real - other, zeros)
            loss_adv = torch.sum(loss_adv)

            # maximize cross_entropy loss
            # loss_adv = -F.mse_loss(logits_model, onehot_labels)
            # loss_adv = - F.cross_entropy(logits_model, labels)

            adv_lambda = 10
            pert_lambda = 1
            loss_G = adv_lambda * loss_adv + pert_lambda * loss_perturb
            loss_G.backward()
            self.optimizer_G.step()

            return loss_D.item(), loss_G_fake.item(),loss_perturb.item(), loss_adv.item()



    def train(self,train_dataloader):

        "load target model"
        if self.mode == 'wf':
            "load target model structure"
            params = utils_wf.params(self.opts['num_class'], self.opts['input_size'])
            target_model = models.target_model_wf(params).to(self.device)

        elif self.mode == 'shs':
            "load target model structure"
            params = utils_shs.params(self.opts['num_class'], self.opts['input_size'])
            target_model = models.target_model_shs(params).to(self.device)

        else:
            print('mode not in ["wf","shs"], system will exit.')
            sys.exit()

        model_name = self.model_path + '/target_model.pth'
        target_model.load_state_dict(torch.load(model_name, map_location=self.device))
        target_model.eval()

        for epoch in range(1, self.opts['epochs'] + 1):

            if epoch == 50:
                self.optimizer_G = torch.optim.Adam(self.generator.parameters(),
                                                    lr=0.0001)
                self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(),
                                                    lr=0.0001)
            if epoch == 80:
                self.optimizer_G = torch.optim.Adam(self.generator.parameters(),
                                                    lr=0.00001)
                self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(),
                                                    lr=0.00001)
            loss_D_sum = 0
            loss_G_fake_sum = 0
            loss_perturb_sum = 0
            loss_adv_sum = 0
            for i, data in enumerate(train_dataloader, start=0):
                train_x, train_y = data
                train_x, train_y = train_x.to(self.device), train_y.to(self.device)

                loss_D_batch, loss_G_fake_batch, loss_perturb_batch, loss_adv_batch = \
                    self.train_batch(train_x, train_y,target_model)
                loss_D_sum += loss_D_batch
                loss_G_fake_sum += loss_G_fake_batch
                loss_perturb_sum += loss_perturb_batch
                loss_adv_sum += loss_adv_batch

            "print statistics"
            num_batch = len(train_dataloader)
            print("epoch %d:\nloss_D: %.3f, loss_G_fake: %.3f,\
             \nloss_perturb: %.3f, loss_adv: %.3f, \n" %
                  (epoch, loss_D_sum / num_batch, loss_G_fake_sum / num_batch,
                   loss_perturb_sum / num_batch, loss_adv_sum / num_batch))

            "save generator"
            if epoch % 10 == 0:
                torch.save(self.generator.state_dict(), self.model_path + '/adv_generator.pth')
                torch.save(self.discriminator.state_dict(), self.model_path + '/adv_discriminator.pth')









