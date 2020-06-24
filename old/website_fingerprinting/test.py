
import os
os.sys.path.append('..')

import torch
import torch.nn.functional as F
from website_fingerprinting import models as models_wf
from website_fingerprinting import utils_wf
import os


test_data = utils_wf.load_data_main('../data/wf/test_NoDef_burst.csv',8)

device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
"load target model structure"
params = utils_wf.params(95,512)
target_model = models_wf.target_model(params).to(device)
target_model.load_state_dict(torch.load('../model/wf/target_model.pth', map_location = device))
target_model.eval()


num_correct = 0
total_instances = 0
for i, data in enumerate(test_data, 0):
    test_x, test_y = data
    test_x, test_y = test_x.to(device), test_y.to(device)
    pred_lab = torch.argmax(target_model(test_x), 1)
    num_correct += torch.sum(pred_lab == test_y, 0)
    total_instances += len(test_y)

print('accuracy of target model against test dataset: %f\n' % (num_correct.item() / total_instances))