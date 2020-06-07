import os
os.sys.path.append('..')
import torch
import torch.nn.functional as F
from models import target_model
import utils, utils_wf


if __name__ == "__main__":

    class opt:
        batch_size = 64
        test_batch_size = 1
        learning_rate = 0.00005
        epochs = 50
        input_height = 1
        input_wide = 512
        clip_value = 0.01       # lower and upper clip value for disc. weights
        sample_interval = 100   # interval between input data samples



    use_cuda = True

    # Define what device we are using
    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    # load data
    train_dataloader = utils_wf.load_data_main('../data/NoDef/train_NoDef.csv',opt.batch_size) # input_shape (batch_size,1,wide 256)
    test_dataloader = utils_wf.load_data_main('../data/NoDef/test_NoDef.csv',opt.test_batch_size)

    print('train_loader size: {}'.format(len(train_dataloader)*opt.batch_size))
    print('test_loader size: {}'.format(len(test_dataloader)*opt.test_batch_size))

    # training the target model
    params = utils_wf.params()
    target_model = target_model(params).to(device)
    print('model structure:', target_model.parameters)
    target_model.train()
    optimizer = torch.optim.Adam(target_model.parameters(), lr=0.001)

    for epoch in range(opt.epochs):
        loss_epoch = 0
        for i, data in enumerate(train_dataloader, 0):
            train_x, train_labels = data
            train_x, train_labels = train_x.to(device), train_labels.to(device)
            optimizer.zero_grad()
            logits_model = target_model(train_x)
            loss_model = F.cross_entropy(logits_model, train_labels)
            loss_epoch += loss_model

            loss_model.backward()
            optimizer.step()


            if i % 100 == 0:
                _, predicted = torch.max(logits_model,1)
                correct = int(sum(predicted == train_labels))
                accuracy = correct / opt.batch_size
                msg = 'Epoch {:5}, Step {:5}, Loss: {:6.2f}, Accuracy:{:8.2%}.'
                print(msg.format(epoch,i,loss_model,accuracy))


        # print('loss in epoch %d: %f' % (epoch, loss_epoch.item()))

    # save model
    targeted_model_file_name = '../model/wf_model/target_model.pth'
    torch.save(target_model.state_dict(), targeted_model_file_name)
    target_model.eval()


    # test model
    num_correct = 0
    for i, data in enumerate(test_dataloader, 0):
        test_x, test_label = data
        test_x, test_label = test_x.to(device), test_label.to(device)
        pred_lab = torch.argmax(target_model(test_x), 1)
        num_correct += torch.sum(pred_lab==test_label,0)

    print('accuracy in testing set: %f\n'%(num_correct.item()/(len(test_dataloader)*opt.test_batch_size)))
