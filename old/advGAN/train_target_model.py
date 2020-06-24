import torch
import torch.nn.functional as F
from models import target_model_1
import data_loader
import read_data


if __name__ == "__main__":

    class opt:
        batch_size = 64
        test_batch_size = 1
        learning_rate = 0.00005
        epochs = 50
        input_height = 1
        input_wide = 256
        clip_value = 0.01  # lower and upper clip value for disc. weights
        n_discriminator = 5  # number of training steps for discriminator per iteration
        sample_interval = 400  # interval between input data samples
        latent_dim = 100  # the dimensionlity of the generator's first input channel. default 100, can change

    def params():
        return{
            'conv1_input_channel':1,
            'conv2_input_channel':128,
            'conv3_input_channel':128,
            'conv4_input_channel':64,
            'conv1_output_channel':128,
            'conv2_output_channel':128,
            'conv3_output_channel':64,
            'conv4_output_channel':256,
            'kernel_size1':7,
            'kernel_size2':19,
            'kernel_size3':13,
            'kernel_size4':23,
            'stride1':1,
            'stride2':1,
            'stride3':1,
            'stride4':1,
            'padding1':3,
            'padding2':9,
            'padding3':6,
            'padding4':11,
            'drop_rate1':0.1,
            'drop_rate2':0.3,
            'drop_rate3':0.1,
            'drop_rate4':0.0,
            'pool1':2,
            'pool2':2,
            'pool3':2,
            'pool4':2,
            'num_classes':101,
            'dim':256
        }


    use_cuda = True
    image_nc = 1
    # batch_size = 256

    # Define what device we are using
    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    # load data
    train_dataloader = read_data.main('../traffic_train.csv',opt.batch_size) # input_shape (batch_size,1,wide 256)
    test_dataloader = read_data.main('../traffic_test.csv',opt.test_batch_size)

    print('train_loader size: {}'.format(len(train_dataloader)*opt.batch_size))
    print('test_loader size: {}'.format(len(test_dataloader)*opt.test_batch_size))

    # training the target model
    params = params()
    target_model = target_model_1(params).to(device)
    print('model structure:', target_model.parameters)
    target_model.train()
    optimizer = torch.optim.Adam(target_model.parameters(), lr=0.001)
    for epoch in range(opt.epochs):
        loss_epoch = 0
        for i, data in enumerate(train_dataloader, 0):
            train_imgs, train_labels = data
            train_imgs, train_labels = train_imgs.to(device), train_labels.to(device)
            optimizer.zero_grad()
            logits_model = target_model(train_imgs)
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
    targeted_model_file_name = './target_model.pth'
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
