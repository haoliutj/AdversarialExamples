# import os
# os.sys.path.append('..')
# import torch
# import torch.nn.functional as F
# from models import  target_model, target_model_1
# import utils
# import models


# if __name__ == "__main__":
#
#     class opt:
#         batch_size = 64
#         test_batch_size = 1
#         learning_rate = 0.00005
#         epochs = 60
#         input_height = 1
#         input_wide = 256
#         clip_value = 0.01  # lower and upper clip value for disc. weights
#         n_discriminator = 5  # number of training steps for discriminator per iteration
#         sample_interval = 400  # interval between input data samples
#         latent_dim = 100  # the dimensionlity of the generator's first input channel. default 100, can change
#         delay = 0.6 # adversarial training
#         x_box_min,x_box_max = -1,0
#         pert_box_min,pert_box_max = -0.3,0.3
#
#
#     use_cuda = True
#     image_nc = 1
#     gen_input_nc = 1
#
#
#     # Define what device we are using
#     print("CUDA Available: ", torch.cuda.is_available())
#     device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
#
#     # load data
#     train_dataloader = utils.load_data_main('../data/traffic_train.csv',opt.batch_size) # input_shape (batch_size,1,wide 256)
#     test_dataloader = utils.load_data_main('../data/traffic_test.csv',opt.test_batch_size)
#     print('train_loader size: {}'.format(len(train_dataloader)))
#     print('test_loader size: {}'.format(len(test_dataloader)))
#
#     # load the generator of adversarial examples
#     pretrained_generator_path = '../model/adv_generator.pth'
#     pretrained_G = models.Generator(gen_input_nc, image_nc).to(device)
#     pretrained_G.load_state_dict(torch.load(pretrained_generator_path, map_location=device))
#     pretrained_G.eval()
#
#     # training the target model
#     params = utils.params()
#     target_model = target_model_1(params).to(device)
#     print('model structure:', target_model.parameters)
#     target_model.train()
#     optimizer = torch.optim.Adam(target_model.parameters(), lr=0.001)
#
#     for epoch in range(1,opt.epochs+1):
#         loss_epoch = 0
#         for i, data in enumerate(train_dataloader, 0):
#
#             # normal training
#             train_imgs, train_labels = data
#             train_imgs, train_labels = train_imgs.to(device), train_labels.to(device)
#             optimizer.zero_grad()
#             logits_model = target_model(train_imgs)
#             loss_model = F.cross_entropy(logits_model, train_labels)
#             loss_epoch += loss_model
#
#             # adversarial training
#             if epoch > int(1-opt.delay)*opt.epochs:
#                 if epoch == int(opt.delay*opt.epochs) + 1:
#                     print("with {} epochs adversarial training...".format(int((1-opt.delay)*opt.epochs)))
#
#                 # # use predicted label to prevent label leaking
#                 # _,y_pred = torch.max(target_model(train_imgs),1)
#
#                 # produce adversarial examples
#                 perturbation = pretrained_G(train_imgs)
#                 perturbation = torch.clamp(perturbation, opt.pert_box_min, opt.pert_box_max)
#                 adv_img = perturbation + train_imgs
#                 adv_img = torch.clamp(adv_img, opt.x_box_min, opt.x_box_max)
#
#                 loss_adv = F.cross_entropy(target_model(adv_img),train_labels)
#                 loss_model = (loss_model + loss_adv) / 2
#
#             if i % 100 == 0:
#                 _, predicted = torch.max(logits_model,1)
#                 correct = int(sum(predicted == train_labels))
#                 accuracy = correct / opt.batch_size
#                 msg = 'Epoch {:5}, Step {:5}, Loss: {:6.2f}, Accuracy:{:8.2%}.'
#                 print(msg.format(epoch,i,loss_model,accuracy))
#
#             loss_model.backward()
#             optimizer.step()
#
#
#         # print('loss in epoch %d: %f' % (epoch, loss_epoch.item()))
#
#     # save model
#     targeted_model_file_name = '../model/adv_target_model_GAN.pth'
#     torch.save(target_model.state_dict(), targeted_model_file_name)
#     target_model.eval()
#
#
#     # test model
#     num_correct = 0
#     for i, data in enumerate(test_dataloader, 0):
#         test_img, test_label = data
#         test_img, test_label = test_img.to(device), test_label.to(device)
#         pred_lab = torch.argmax(target_model(test_img), 1)
#         num_correct += torch.sum(pred_lab==test_label,0)
#
#     print('accuracy in testing set: %f\n'%(num_correct.item()/(len(test_dataloader)*opt.test_batch_size)))







import os
os.sys.path.append('..')
import torch
import torch.nn.functional as F
from advGAN import models as models_gan
from advGAN import utils_advGAN
from website_fingerprinting import models as models_wf
from website_fingerprinting import utils_wf
import utils as utils_gb



class adv_train_gan:
    def __init__(self,
                 opts,
                 x_box_min=-1,
                 x_box_max=0,
                 pert_box=0.3):
        self.opts = opts
        self.mode = opts['mode']
        self.model_path = '../model/' + self.mode
        self.x_box_min = x_box_min
        self.x_box_max = x_box_max
        self.pert_box = pert_box
        self.input_nc =1
        self.gen_input_nc = 1
        self.device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("CUDA Available: ", torch.cuda.is_available())

        if self.mode == 'wf':
            print('Website Fingerprinting...')
        elif self.mode == 'shs':
            print('Smart Home Speaker Fingerprinting...')

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)


    def adv_train(self):

        "load data"
        if self.mode == 'wf':
            train_dataloader = utils_wf.load_data_main(self.opts['train_data_path'],self.opts['batch_size'])
            test_dataloader = utils_wf.load_data_main(self.opts['test_data_path'], self.opts['test_batch_size'])
        elif self.mode == 'shs':
            train_dataloader = utils_gb.load_data_main(self.opts['train_data_path'], self.opts['batch_size'])
            test_dataloader = utils_gb.load_data_main(self.opts['test_data_path'], self.opts['test_batch_size'])


        "load G"
        generator = models_gan.Generator(self.gen_input_nc,self.input_nc).to(self.device)
        generator.load_state_dict(torch.load(self.opts['adv_generator_path'],map_location=self.device))
        generator.eval()


        "load target model structure"
        if self.mode == 'wf':   # website fingerprinting
            params = utils_wf.params(self.opts['num_class'],self.opts['input_size'])
            target_model = models_wf.target_model(params).to(self.device)
        elif self.mode == 'shs': #'smart home speaker
            params = utils_advGAN.params(self.opts['num_class'],self.opts['input_size'])
            target_model = models_gan.target_model_1(params).to(self.device)


        # ****************************************************
        "training process"
        target_model.train()
        optimizer = torch.optim.Adam(target_model.parameters(), lr=0.001)

        for epoch in range(1,self.opts['epochs']+1):
            loss_epoch = 0
            for i, data in enumerate(train_dataloader, 0):

                # normal training
                X, y = data
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                logits_model = target_model(X)
                loss_model = F.cross_entropy(logits_model, y)
                loss_epoch += loss_model

                "adversarial training"
                if epoch > int(1-self.opts['delay'])*self.opts['epochs']:
                    if epoch == int(self.opts['delay']*self.opts['epochs']) + 1:
                        print("with {} epochs adversarial training...".format(int((1-self.opts['delay'])*self.opts['epochs'])))

                    # # use predicted label to prevent label leaking
                    # _,y_pred = torch.max(target_model(X),1)

                    "produce adversarial examples"
                    pert = generator(X)

                    if self.mode == 'wf':

                        "normalize data to [-1,1]"
                        X = utils_wf.data_normalize(X)

                        "clamp perturbation"
                        pert = torch.clamp(pert, self.pert_box_min, self.pert_box_max)

                        "add pert to x given the restrictions"
                        adv_x = utils_wf.add_perturbation(X, pert)

                        "inverse the normalized data"
                        adv_x = utils_wf.inverse_normalize(adv_x)

                        "add round function to inversed data"
                        adv_x = utils_wf.round_data(adv_x)

                        "convert ndarray to torch.Tensor"
                        adv_x = torch.Tensor(adv_x).to(self.device)

                    elif self.mode == 'shs':
                        adv_x = pert + X
                        adv_x = torch.clamp(adv_x, self.x_box_min, self.x_box_max)

                    loss_adv = F.cross_entropy(target_model(adv_x),y)
                    loss_model = (loss_model + loss_adv) / 2

                if i % 100 == 0:
                    _, predicted = torch.max(logits_model,1)
                    correct = int(sum(predicted == y))
                    accuracy = correct / len(y)
                    msg = 'Epoch {:5}, Step {:5}, Loss: {:6.2f}, Accuracy:{:8.2%}.'
                    print(msg.format(epoch,i,loss_model,accuracy))

                loss_model.backward()
                optimizer.step()

            "save model"
            if epoch % 10 == 0:
                targeted_model_path = self.model_path + '/adv_target_model_GAN.pth'
                torch.save(target_model.state_dict(), targeted_model_path)


        #****************************************************
        "test the adversarial training model"
        target_model.eval()
        num_correct = 0
        total_case = 0
        for i, data in enumerate(test_dataloader, 0):
            test_x, test_label = data
            test_x, test_label = test_x.to(self.device), test_label.to(self.device)
            pred_lab = torch.argmax(target_model(test_x), 1)
            num_correct += torch.sum(pred_lab==test_label,0)
            total_case += len(test_label)

        print('accuracy in testing set: %f\n'%(num_correct.item()/total_case))


def main(opts):

    advTrainGAN = adv_train_gan(opts)
    advTrainGAN.adv_train()



def get_params():

    return{
    'epochs':50,
    'batch_size':64,
    'test_batch_size':64,
    'pert_box':0.3,
    'delay': 0.5,

    #-----------------------------------
    # params for website fingerprinting (wf)
    #-----------------------------------

    'mode': 'wf',
    'x_box_min': -1,
    'x_box_max':0,
    'num_class':95,
    'input_size':512,
    'train_data_path':'../data/NoDef/train_NoDef_burst.csv',
    'test_data_path':'../data/NoDef/test_NoDef_burst.csv',
    'adv_generator_path': '../model/wf/adv_generator.pth',



    # -----------------------------------
    # params for smart home speaker (shs)
    # -----------------------------------

    # 'x_box_min': -1,
    # 'x_box_max': 0,
    # 'num_class': 101,
    # 'input_size': 256,
    # 'mode': 'shs',   #shs and wf
    # 'input_data_path': '../data/traffic_train.csv',

    }


if __name__ =='__main__':
    opts = get_params()
    main(opts)