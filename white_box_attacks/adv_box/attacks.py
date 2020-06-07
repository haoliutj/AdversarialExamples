import numpy as np
import torch
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
import torch.nn as nn
import copy



class FGSM(object):

    def __init__(self, model=None, epsilon=0.1):
        self.model = model
        self.epsilon = epsilon
        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.box_min,self.box_max = -1,0

    def perturbation(self, X_i, y, epsilons=None):
        """
        given examples (X,y), returns corresponding adversarial example
        y should be a integer
        """
        if epsilons is not None:
            self.epsilon = epsilons

        X = np.copy(X_i.cpu())  #the input X_i is a tensor, so before copy, need to copy it in cpu; array
        X = torch.from_numpy(X)

        X_var = Variable(X.to(self.device), requires_grad=True)
        y_var = Variable(y.to(self.device))

        output = self.model(X_var)
        loss = self.criterion(output, y_var)
        loss.backward()
        grad_sign = X_var.grad.data.cpu().sign().numpy()  # convert cuda/cpu data to numpy (array)

        X += torch.from_numpy(self.epsilon * grad_sign)
        X = torch.clamp(X,self.box_min,self.box_max)                            # make variable in the range [0,1]
        output_pert = self.model(X.to(self.device))
        _,y_pert = torch.max(output_pert.data,1)

        return y_pert, X    # X in numpy format




class DeepFool_batch_train(object):
    "applied in mini batch training, but attack performance is not good as single train"

    def __init__(self,model=None, num_classes=10):
        """
        num_classes: limits the number of classes to test against
        overshoot: used as a termination criterion to prevent vanishing updates
        max_iter: maximum number of iterations for deepfool
        return: perturbed output (adversarial examples)
        """
        self.model = model
        self.num_classes = num_classes
        self.overshoot = 0.02
        self.max_iter = 50
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.box_min, self.box_max = -1, 0


    def perturbation(self, X_input, y=None):
        "y: label is not needed here"

        X_var = Variable(X_input.to(self.device),requires_grad=True)
        preds = self.model(X_var)       # [0,num_labels], the first dimenson is empty

        # preds_label_sort = (np.array(preds.data.cpu().numpy())).argsort()[::-1]    # return index of sorted value, one dimension vector V[::-1], means sort in reverse order from large to small
        # preds_label_sort = preds_label_sort[:,0:self.num_classes]
        # pred_label = preds_label_sort[:,0]      #original predict label

        preds_label_sort = torch.argsort(preds,descending=True)     # return index of sorted value in descending direction, compatable to mini batch
        preds_label_sort = preds_label_sort[:,0:self.num_classes]
        pred_label = preds_label_sort[:,0]


        input_shape = X_input.cpu().numpy().shape
        w = np.zeros(input_shape)
        noise_total = np.zeros(input_shape)
        pert_x = copy.deepcopy(X_input)
        loop_i = 0                  # number of iteratioins applied
        pert_label_i = pred_label   # in the end, pert_label_i refer to the perturbed label corresponding to perturbed output

        # only all the labels are different with original ones, then stop, which is not efficient,
        # one of the label is different at the first step, but need iterate since others need to be perturbed
        flag = ((pert_label_i.data.cpu().numpy().all() == pred_label.data.cpu().numpy()).all())


        while flag and loop_i < self.max_iter:

            pert = np.inf     # noise at each step, format: infinity float

            #obtain all rows of max prediction, shape len(rows) vector
            var_backgrad = []
            for i in range(len(X_input)):
                eee = preds[i,preds_label_sort[i,0]]
                eee1 = torch.unsqueeze(eee,dim=-1)
                var_backgrad.append(eee1)
            var_backgrad_tensor = torch.cat(var_backgrad)   #convert to tensor

            # Non scalar Y for backward, need add "grad_tensors = torch.one_like(Y)" to backward to obtain gradient
            torch.autograd.backward(var_backgrad_tensor,grad_tensors=torch.ones_like(var_backgrad_tensor),retain_graph=True)
            grad_orig = X_var.grad.data.cpu().numpy().copy()


            # var_backgrad_ttt1 = var_backgrad_tensor.backward(retain_graph=True)


            # yyy = preds[:,preds_label_sort[:,0]].sum().backward(retain_graph=True)      # preds is 2-dimension [0,num_labels], 1st dimension is empty
            # grad_orig = X_var.grad.data.cpu().numpy().copy()

            for k in range(1,self.num_classes):

                #back propagation
                zero_gradients(X_var)

                #obtain all rows of prediction in position k
                var_backgrad_k = []
                for i in range(len(X_input)):
                    temp = preds[i,preds_label_sort[i,k]]
                    temp = torch.unsqueeze(temp,dim=-1)
                    var_backgrad_k.append(temp)
                var_backgrad_k_tensor = torch.cat(var_backgrad_k)
                torch.autograd.backward(var_backgrad_k_tensor,grad_tensors=torch.ones_like(var_backgrad_k_tensor),retain_graph=True)
                cur_grad = X_var.grad.data.cpu().numpy().copy()

                #set new w_k and new preds_k
                w_k = cur_grad - grad_orig
                preds_k = (var_backgrad_k_tensor - var_backgrad_tensor).data.cpu().numpy()

                # Normalization: ||w_k||**2, 默认情况求元素平方和开根号
                w_k_norm = []
                for i in range(len(X_input)):
                    temp = np.linalg.norm(w_k[i,:].flatten())
                    w_k_norm.append(temp)

                # w_k_norm = np.linalg.norm(w_k.flatten())

                pert_k = abs(preds_k) / w_k_norm

                # determin which w_k to use, select the smalles one. Orginally, pert_k just one scalar value
                # Different to original paper, we add mini-bach training,
                # so find the min value of all mean(pert_k)
                if np.mean(pert_k) < pert:
                    pert = np.mean(pert_k)
                    w = w_k

            # calculate noise_i and noise_total
            # add 1e-4 for numerical stability
            noise_i = (pert + 1e-4) * w / np.linalg.norm(w)
            noise_total = np.float32(noise_total + noise_i)

            # obain perturbated x and corresponding perturbated prediction
            pert_x = X_input.to(self.device) + ((1 + self.overshoot) * torch.from_numpy(noise_total)).to(self.device)
            pert_x = torch.clamp(pert_x,self.box_min,self.box_max)        # range[0,1] incoming packet should [-1500,0]
            X_var = Variable(pert_x.to(self.device), requires_grad = True)

            preds = self.model(X_var)

            # the orginal code from the paper, may have errors in terms of preds_label_sort
            # they did not iterate parameter "preds_label_sort", instead they only computed it
            # once at the very first begin based on the original input X_input
            # we correct it by add "preds_label_sort" to the iteration step based on X_input first
            # and then iterately based on perturbed x

            # preds_label_sort = (np.array(preds.data.cpu().numpy())).flatten().argsort()[::-1]
            # preds_label_sort = preds_label_sort[0:self.num_classes]
            # pert_label_i = preds_label_sort[0] # index of max value, label of perturbed input

            preds_label_sort = torch.argsort(preds, descending=True)  # return index of sorted value in descending direction in orginal shape, compatable to mini batch
            preds_label_sort = preds_label_sort[:, 0:self.num_classes]
            pert_label_i = preds_label_sort[:,0]

            flag = ((pert_label_i.data.cpu().numpy().all() == pred_label.data.cpu().numpy()).all())

            loop_i += 1

        noise_total = (1 + self.overshoot) * noise_total    # minimal pertubation that fools the classifier

        return pert_label_i, pert_x       #return pert label and pert x
        # return pert_x



class DeepFool(object):

    def __init__(self,model=None, num_classes=5):
        """
        num_classes: limits the number of classes to test against
        overshoot: used as a termination criterion to prevent vanishing updates
        max_iter: maximum number of iterations for deepfool
        return: perturbed output
        """
        self.model = model
        self.num_classes = num_classes
        self.overshoot = 0.02
        self.max_iter = 25  #default 50
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.box_min, self.box_max = -1, 0


    def perturbation(self, X_input, y=None):
        "y: label is not needed here"

        X_var = Variable(X_input.to(self.device),requires_grad=True)
        preds = self.model(X_var)       # [0,num_labels], the first dimenson is empty

        # preds_label_sort = (np.array(preds.data.cpu().numpy())).argsort()[::-1]    # return index of sorted value, one dimension vector V[::-1], means sort in reverse order from large to small
        # preds_label_sort = preds_label_sort[:,0:self.num_classes]
        # pred_label = preds_label_sort[:,0]      #original predict label

        preds_label_sort = torch.argsort(preds,descending=True)     # return index of sorted value in descending direction, compatable to mini batch
        preds_label_sort = preds_label_sort[:,0:self.num_classes]
        pred_label = preds_label_sort[:,0]


        input_shape = X_input.cpu().numpy().shape
        w = np.zeros(input_shape)
        noise_total = np.zeros(input_shape)
        pert_x = copy.deepcopy(X_input)
        loop_i = 0                  # number of iteratioins applied
        pert_label_i = pred_label   # in the end, pert_label_i refer to the perturbed label corresponding to perturbed output


        while pert_label_i == pred_label and loop_i < self.max_iter:

            pert = np.inf     # noise at each step, format: infinity float

            preds[:,preds_label_sort[:,0]].backward(retain_graph=True)
            grad_orig = X_var.grad.data.cpu().numpy().copy()


            for k in range(1,self.num_classes):

                #back propagation
                zero_gradients(X_var)

                preds[:,preds_label_sort[:,k]].backward(retain_graph=True)
                cur_grad = X_var.grad.data.cpu().numpy().copy()

                #set new w_k and new preds_k
                w_k = cur_grad - grad_orig
                preds_k = (preds[:,preds_label_sort[:,k]] - preds[:,preds_label_sort[:,0]]).data.cpu().numpy()

                # Normalization: ||w_k||**2, 默认情况求元素平方和开根号
                w_k_norm = np.linalg.norm(w_k.flatten())
                # smoothing avoid to divide by zero
                if w_k_norm == 0:
                    overshoot_noise = 1e-4   # the smaller, the higher of 'pert_k', will be dropped during the iterations
                    pert_k = abs(preds_k) / (overshoot_noise + w_k_norm)
                else:
                    pert_k = abs(preds_k) / w_k_norm

                # determin which w_k to use, select the smalles one.
                if pert_k < pert:
                    pert = pert_k
                    w = w_k

            # calculate noise_i and noise_total
            # add 1e-4 for numerical stability
            w_norm = np.linalg.norm(w)
            if w_norm == 0:
                noise_i = (pert + 1e-4) * w / (w_norm + 1)  # add 1, avoid divide by zero
            else:
                noise_i = (pert + 1e-4) * w / w_norm

            noise_total = np.float32(noise_total + noise_i)

            # obain perturbated x and corresponding perturbated prediction
            noise_total_overshoot = (1 + self.overshoot) * torch.from_numpy(noise_total)
            # must push the noise_total_overshoot to the device (if its cuda), otherwise rise error
            pert_x = X_input + noise_total_overshoot.to(self.device)
            pert_x = torch.clamp(pert_x, self.box_min,self.box_max)
            # pert_x = X_input + (1 + self.overshoot) * torch.from_numpy(noise_total)
            X_var = Variable(pert_x.to(self.device), requires_grad = True)

            preds = self.model(X_var)

            # the orginal code from the paper, may have errors in terms of preds_label_sort
            # they did not iterate parameter "preds_label_sort", instead they only computed it
            # once at the very first begin based on the original input X_input
            # we correct it by add "preds_label_sort" to the iteration step based on X_input first
            # and then iterately based on perturbed x


            preds_label_sort = torch.argsort(preds, descending=True)  # return index of sorted value in descending direction in orginal shape, compatable to mini batch
            preds_label_sort = preds_label_sort[:, 0:self.num_classes]
            pert_label_i = preds_label_sort[:,0]

            loop_i += 1

        noise_total = (1 + self.overshoot) * noise_total    # minimal pertubation that fools the classifier

        return pert_label_i, pert_x       #return pert label and pert x
        # return pert_x       #return pert label and pert x



class LinfPGDAttack(object):
    def __init__(self, model=None, epsilon=0.3, k=40, a=0.01,
        random_start=True):
        """
        Attack parameter initialization. The attack performs k steps of
        size a, while always staying within epsilon from the initial
        point.
        https://github.com/MadryLab/mnist_challenge/blob/master/pgd_attack.py
        """
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.a = a
        self.rand = random_start
        self.loss_fn = nn.CrossEntropyLoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.box_min, self.box_max = -1, 0

    def perturbation(self, X_nat, y):
        """
        Given examples (X_nat, y), returns adversarial
        examples within epsilon of X_nat in l_infinity norm.
        """
        if self.rand:
            X = X_nat + torch.from_numpy(np.random.uniform(-self.epsilon, self.epsilon,
                X_nat.shape).astype('float32'))
        else:
            X = copy.deepcopy(X_nat)


        for i in range(self.k):
            X_var = Variable(X.to(self.device), requires_grad=True)
            # y_var = Variable(torch.LongTensor(y).to(self.device))
            y_var = Variable(y.to(self.device))

            scores = self.model(X_var)
            loss = self.loss_fn(scores, y_var)
            loss.backward()
            grad_sign = X_var.grad.data.cpu().sign().numpy()

            X += torch.from_numpy(self.a * grad_sign).to(self.device)

        X = torch.clamp(X, self.box_min,self.box_max)
        y_adv = self.model(X.to(self.device))
        _,y_adv = torch.max(y_adv,1)

        return y_adv, X






