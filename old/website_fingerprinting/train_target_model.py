# import os
# os.sys.path.append('..')
# import torch
# import torch.nn.functional as F
# from website_fingerprinting.models import target_model
# import utils
# from website_fingerprinting import utils_wf
#
#
# if __name__ == "__main__":
#
#     class opt:
#         batch_size = 64
#         test_batch_size = 32
#         epochs = 50
#         input_size = 512
#         num_class = 95
#
#
#     use_cuda = True
#
#     # Define what device we are using
#     print("CUDA Available: ", torch.cuda.is_available())
#     device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
#
#     # load data
#     train_dataloader = utils_wf.load_data_main('../data/NoDef/train_NoDef_burst.csv',opt.batch_size) # input_shape (batch_size,1,wide 256)
#     test_dataloader = utils_wf.load_data_main('../data/NoDef/test_NoDef_burst.csv',opt.test_batch_size)
#
#     print('train_loader batch: {}'.format(len(train_dataloader)))
#     print('test_loader batch: {}'.format(len(test_dataloader)))
#
#     # training the target model
#     params = utils_wf.params(opt.num_class,opt.input_size)
#     target_model = target_model(params).to(device)
#     print('model structure:', target_model.parameters)
#     target_model.train()
#     optimizer = torch.optim.Adam(target_model.parameters(), lr=0.001)
#
#     for epoch in range(opt.epochs):
#         loss_epoch = 0
#         for i, data in enumerate(train_dataloader, 0):
#             train_x, train_labels = data
#             train_x, train_labels = train_x.to(device), train_labels.to(device)
#             optimizer.zero_grad()
#             logits_model = target_model(train_x)
#             loss_model = F.cross_entropy(logits_model, train_labels)
#             loss_epoch += loss_model
#
#             loss_model.backward()
#             optimizer.step()
#
#
#             if i % 100 == 0:
#                 _, predicted = torch.max(logits_model,1)
#                 correct = int(sum(predicted == train_labels))
#                 accuracy = correct / len(train_labels)
#                 msg = 'Epoch {:5}, Step {:5}, Loss: {:6.2f}, Accuracy:{:8.2%}.'
#                 print(msg.format(epoch,i,loss_model,accuracy))
#
#         # save model
#         if epoch % 20 == 0:
#             targeted_model_file_name = '../model/wf_model/target_model.pth'
#             torch.save(target_model.state_dict(), targeted_model_file_name)
#
#     target_model.eval()
#
#
#     # test model
#     num_correct = 0
#     total_instances = 0
#     for i, data in enumerate(test_dataloader, 0):
#         test_x, test_label = data
#         test_x, test_label = test_x.to(device), test_label.to(device)
#         pred_lab = torch.argmax(target_model(test_x), 1)
#         num_correct += torch.sum(pred_lab==test_label,0)
#         total_instances += len(test_label)
#
#     print('accuracy in testing set: %f\n'%(num_correct.item()/total_instances))
#
#
#
#
#
#
#
#
#


import os
os.sys.path.append('..')

import torch
import torch.nn.functional as F
from website_fingerprinting import models as models_wf
from website_fingerprinting import utils_wf
import os



class train_target_model:
    def __init__(self,opts):

        self.opts = opts
        self.mode = opts['mode']
        self.model_path = '../model/' + self.mode
        self.device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('CUDA AVAILABEL: ', torch.cuda.is_available())

        if self.mode == 'wf':
            print('Website Fingerprinting...')
        elif self.mode == 'shs':
            print('Smart Home Speaker Fingerprinting...')

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)


    def train_model(self):

        "load data"
        train_data = utils_wf.load_data_main(self.opts['train_data_path'],self.opts['batch_size'])
        test_data = utils_wf.load_data_main(self.opts['test_data_path'],self.opts['batch_size'])

        "load target model structure"
        params = utils_wf.params(self.opts['num_class'],self.opts['input_size'])
        target_model = models_wf.target_model(params).to(self.device)
        target_model.train()

        "train process"
        optimizer = torch.optim.Adam(target_model.parameters(),lr=0.001)
        for epoch in range(self.opts['epochs']):
            loss_epoch = 0
            for i, data in enumerate(train_data, 0):
                train_x, train_y = data
                train_x, train_y = train_x.to(self.device), train_y.to(self.device)
                optimizer.zero_grad()
                logits_model = target_model(train_x)
                loss_model = F.cross_entropy(logits_model, train_y)
                loss_epoch += loss_model

                loss_model.backward()
                optimizer.step()

                if i % 100 == 0:
                    _, predicted = torch.max(logits_model, 1)
                    correct = int(sum(predicted == train_y))
                    accuracy = correct / len(train_y)
                    msg = 'Epoch {:5}, Step {:5}, Loss: {:6.2f}, Accuracy:{:8.2%}.'
                    print(msg.format(epoch, i, loss_model, accuracy))

            "save model every 10 epochs"
            if epoch % 10 == 0:
                targeted_model_path = self.model_path + '/target_model.pth'
                torch.save(target_model.state_dict(), targeted_model_path)


        "test target model"
        target_model.eval()

        num_correct = 0
        total_instances = 0
        for i, data in enumerate(test_data, 0):
            test_x, test_y = data
            test_x, test_y = test_x.to(self.device), test_y.to(self.device)
            pred_lab = torch.argmax(target_model(test_x), 1)
            num_correct += torch.sum(pred_lab == test_y, 0)
            total_instances += len(test_y)

        print('accuracy of target model against test dataset: %f\n' % (num_correct.item() / total_instances))



def main(opts):
    trainTargetModel = train_target_model(opts)
    trainTargetModel.train_model()


def get_opts():
    return{
        'mode':'wf',
        'num_class': 95,
        'input_size': 512,
        'batch_size': 64,
        'epochs':50,
        'train_data_path': '../data/NoDef/train_NoDef_burst.csv',
        'test_data_path': '../data/NoDef/test_NoDef_burst.csv',
    }


if __name__ == '__main__':
    opts = get_opts()
    main(opts)
