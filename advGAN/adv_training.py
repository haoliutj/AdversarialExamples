import os
os.sys.path.append('..')
import torch
import torch.nn.functional as F
from models import  target_model, target_model_1
import utils
import models


if __name__ == "__main__":

    class opt:
        batch_size = 64
        test_batch_size = 1
        learning_rate = 0.00005
        epochs = 60
        input_height = 1
        input_wide = 256
        clip_value = 0.01  # lower and upper clip value for disc. weights
        n_discriminator = 5  # number of training steps for discriminator per iteration
        sample_interval = 400  # interval between input data samples
        latent_dim = 100  # the dimensionlity of the generator's first input channel. default 100, can change
        delay = 0.6 # adversarial training
        x_box_min,x_box_max = -1,0
        pert_box_min,pert_box_max = -0.3,0.3


    use_cuda = True
    image_nc = 1
    gen_input_nc = 1


    # Define what device we are using
    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    # load data
    train_dataloader = utils.load_data_main('../data/traffic_train.csv',opt.batch_size) # input_shape (batch_size,1,wide 256)
    test_dataloader = utils.load_data_main('../data/traffic_test.csv',opt.test_batch_size)
    print('train_loader size: {}'.format(len(train_dataloader)))
    print('test_loader size: {}'.format(len(test_dataloader)))

    # load the generator of adversarial examples
    pretrained_generator_path = '../model/adv_generator.pth'
    pretrained_G = models.Generator(gen_input_nc, image_nc).to(device)
    pretrained_G.load_state_dict(torch.load(pretrained_generator_path, map_location=device))
    pretrained_G.eval()

    # training the target model
    params = utils.params()
    target_model = target_model_1(params).to(device)
    print('model structure:', target_model.parameters)
    target_model.train()
    optimizer = torch.optim.Adam(target_model.parameters(), lr=0.001)

    for epoch in range(1,opt.epochs+1):
        loss_epoch = 0
        for i, data in enumerate(train_dataloader, 0):

            # normal training
            train_imgs, train_labels = data
            train_imgs, train_labels = train_imgs.to(device), train_labels.to(device)
            optimizer.zero_grad()
            logits_model = target_model(train_imgs)
            loss_model = F.cross_entropy(logits_model, train_labels)
            loss_epoch += loss_model

            # adversarial training
            if epoch > int(1-opt.delay)*opt.epochs:
                if epoch == int(opt.delay*opt.epochs) + 1:
                    print("with {} epochs adversarial training...".format(int((1-opt.delay)*opt.epochs)))

                # # use predicted label to prevent label leaking
                # _,y_pred = torch.max(target_model(train_imgs),1)

                # produce adversarial examples
                perturbation = pretrained_G(train_imgs)
                perturbation = torch.clamp(perturbation, opt.pert_box_min, opt.pert_box_max)
                adv_img = perturbation + train_imgs
                adv_img = torch.clamp(adv_img, opt.x_box_min, opt.x_box_max)

                loss_adv = F.cross_entropy(target_model(adv_img),train_labels)
                loss_model = (loss_model + loss_adv) / 2

            if i % 100 == 0:
                _, predicted = torch.max(logits_model,1)
                correct = int(sum(predicted == train_labels))
                accuracy = correct / opt.batch_size
                msg = 'Epoch {:5}, Step {:5}, Loss: {:6.2f}, Accuracy:{:8.2%}.'
                print(msg.format(epoch,i,loss_model,accuracy))

            loss_model.backward()
            optimizer.step()


        # print('loss in epoch %d: %f' % (epoch, loss_epoch.item()))

    # save model
    targeted_model_file_name = '../model/adv_target_model_GAN.pth'
    torch.save(target_model.state_dict(), targeted_model_file_name)
    target_model.eval()


    # test model
    num_correct = 0
    for i, data in enumerate(test_dataloader, 0):
        test_img, test_label = data
        test_img, test_label = test_img.to(device), test_label.to(device)
        pred_lab = torch.argmax(target_model(test_img), 1)
        num_correct += torch.sum(pred_lab==test_label,0)

    print('accuracy in testing set: %f\n'%(num_correct.item()/(len(test_dataloader)*opt.test_batch_size)))
