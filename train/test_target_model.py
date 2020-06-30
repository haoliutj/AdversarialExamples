"test the performance of target model"

import os
os.sys.path.append('..')

from train import models,utils_wf,utils_shs
import torch
import sys


class performance_target_model:
    def __init__(self,opts):
        self.opts = opts
        self.mode = opts['mode']
        self.classifier_type = opts['classifier_type']
        self.model_path = '../model/'  + self.mode + '/' + opts['classifier_type']
        self.device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('CUDA AVAILABEL:', torch.cuda.is_available())
        print('testing model path:  ', self.model_path)


    def test_peformance(self):

        "load data and target model"
        if self.mode == 'wf':
            "load data"
            test_data = utils_wf.load_data_main(self.opts['test_data_path'],self.opts['batch_size'])

            "load target model structure"
            if self.classifier_type == 'cnn':
                params = utils_wf.params_cnn(self.opts['num_class'],self.opts['input_size'])
                target_model = models.cnn_norm(params).to(self.device)

            elif self.classifier_type == 'lstm':
                params = utils_wf.params_lstm(self.opts['num_class'], self.opts['input_size'], self.opts['batch_size'])
                target_model = models.lstm(params).to(self.device)

        elif self.mode == 'shs':
            "load data"
            test_data = utils_shs.load_data_main(self.opts['test_data_path'],self.opts['batch_size'])

            "load target model structure"
            params = utils_shs.params(self.opts['num_class'],self.opts['input_size'])
            target_model = models.cnn_noNorm(params).to(self.device)

        else:
            print('mode not in ["wf","shs"], system will exit.')
            sys.exit()

        model_name = self.model_path + '/target_model.pth'
        target_model.load_state_dict(torch.load(model_name, map_location=self.device))
        target_model.eval()


        "testing process..."

        num_correct = 0
        total_case = 0
        for i, data in enumerate(test_data, 0):
            test_x, test_y = data
            test_x, test_y = test_x.to(self.device), test_y.to(self.device)
            pred_lab = torch.argmax(target_model(test_x), 1)
            num_correct += torch.sum(pred_lab == test_y, 0)
            total_case += len(test_y)

        print('accuracy in testing set: %f\n' % (num_correct.item() / total_case))


def main(opts):

    test_target_model = performance_target_model(opts)
    test_target_model.test_peformance()


def get_opts_wf(mode, classifier_type):

    return{
        'mode':mode,
        'classifier_type':classifier_type,
        'batch_size':64,
        'num_class':95,
        'input_size':512,
        'test_data_path': '../data/wf/test_NoDef_burst.csv'
    }


def get_opts_shs(mode, classifier_type):
    return {
        'mode': mode,
        'classifier_type': classifier_type,
        'batch_size': 64,
        'num_class': 101,
        'input_size': 256,
        'test_data_path': '../data/shs/traffic_test.csv'
    }


if __name__ == '__main__':
    mode = 'wf'
    classifier_type = 'lstm'

    if mode == 'wf':
        opts = get_opts_wf(mode, classifier_type)
    elif mode == 'shs':
        opts = get_opts_shs(mode, classifier_type)
    else:
        print('mode should in ["wf","shs"]. system will exit.')
        sys.exit()

    main(opts)