import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.functional as F
import torchvision
from torchvision import models, datasets
import torchvision.transforms as transforms
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
from CreateDataset import FlowerDataset



def train(args):

    #split the dataset
    flowers_dataset = FlowerDataset(args.data_dir)
    flowers_dataset.dataset_is_existed()

    #import data with torchvision
    data_dir = 'dataset'
    sets = ['train', 'test']
    #use transforms
    data_transforms = {
        'train'      : transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(args.image_size),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])]),

        'test'      : transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(args.image_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])}

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                        for x in sets}

    dataloaders = {x: DataLoader(dataset=image_datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=4)
                        for x in sets}

    print('Training data loaded successfully. \n Training image number: %d \n Training image size %d %d \n Batch size: %d \n labels: %s \n' %(len(dataloaders['train'].dataset), args.image_size, args.image_size, args.batch_size, image_datasets['train'].classes))
    print('Test data loaded successfully. \n Test image number: %d \n Test image size %d %d \n Batch size: %d \n labels: %s \n' %(len(dataloaders['test'].dataset), args.image_size, args.image_size, args.batch_size, image_datasets['test'].classes))

    #load pre-trained model
    model = getattr(models, args.network)(pretrained = args.pre_train)

    #freeze all the parameters in the former layers
    if args.pre_train and args.freeze:
        flag = True
        for param in model.parameters():
            param.requires_grad = False
    else:
        flag = False

    #design new classifiers of each network
    #5 species of flowers
    out_features = 5

    #resnet
    if args.network == 'resnet18':
        in_features = model.fc.in_features
        if args.new_fc:
            model.fc = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(in_features, 128, bias=True)),
                            ('relu1', nn.ReLU(inplace=True)),
                            ('dropout1', nn.Dropout(0.5, inplace=False)),
                            ('fc2', nn.Linear(in_features=128, out_features=128, bias=True)),
                            ('relu2', nn.ReLU(inplace=True)),
                            ('dropout2', nn.Dropout(0.5, inplace=False)),
                            ('fc3', nn.Linear(in_features=128, out_features=out_features, bias=True))
                            ]))
        else:
            model.fc = nn.Linear(in_features, out_features)
        print('Network loaded successfully \n name: %s \n freezing parameters: %s \n' %(args.network, str(flag)))
        print(model)

    #densnet
    if args.network == 'densenet161':
        in_features = model.classifier.in_features
        if args.new_fc:
            model.classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(in_features, 552, bias=True)),
                            ('relu1', nn.ReLU(inplace=True)),
                            ('dropout1', nn.Dropout(0.5, inplace=False)),
                            ('fc2', nn.Linear(in_features=552, out_features=552, bias=True)),
                            ('relu2', nn.ReLU(inplace=True)),
                            ('dropout2', nn.Dropout(0.5, inplace=False)),
                            ('fc3', nn.Linear(in_features=552, out_features=out_features, bias=True))
                            ]))
        else:
            model.classifier = nn.Linear(in_features, out_features, bias = True)
        print('Network: %s loaded successfully \n freezing parameters: %s \n' %(args.network, str(flag)))
        print(model)

    #vgg16
    if args.network == 'vgg16':
        in_features = model.classifier[0].in_features
        if args.new_fc:
            model.classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(in_features, 4096, bias=True)),
                            ('relu1', nn.ReLU(inplace=True)),
                            ('dropout1', nn.Dropout(0.5, inplace=False)),
                            ('fc2', nn.Linear(in_features=4096, out_features=4096, bias=True)),
                            ('relu2', nn.ReLU(inplace=True)),
                            ('dropout2', nn.Dropout(0.5, inplace=False)),
                            ('fc3', nn.Linear(in_features=4096, out_features=out_features, bias=True))
                            ]))
        else:
            model.classifier = nn.Linear(in_features, out_features, bias = True)
        print('Network: %s loaded successfully \n freezing parameters: %s \n' %(args.network, str(flag)))
        print(model)


    #MobileNetV2
    if args.network == 'mobilenet_v2':
        in_features = model.classifier[1].in_features
        if args.new_fc:
            model.classifier = nn.Sequential(OrderedDict([
                            ('dropout', nn.Dropout(0.2, inplace = False)),
                            ('fc1', nn.Linear(in_features=in_features, out_features=320, bias=True)),
                            ('relu1', nn.ReLU(inplace=True)),
                            ('dropout1', nn.Dropout(0.5, inplace= False)),
                            ('fc2', nn.Linear(in_features=320, out_features=320, bias=True)),
                            ('relu2', nn.ReLU(inplace=True)),
                            ('dropout2', nn.Dropout(0.5, inplace=False)),
                            ('fc3', nn.Linear(in_features=320, out_features=out_features, bias=True))
                            ]))
        else:
            model.classifier = nn.Sequential(OrderedDict([
                            ('dropout', nn.Dropout(0.2, inplace = False)),
                            ('fc1', nn.Linear(in_features=in_features, out_features=out_features, bias=True))
                            ]))
        print('Network loaded successfully \n name: %s \n freezing parameters: %s \n' %(args.network, str(flag)))
        print(model)

    #load model to device, default device: GPU
    device = torch.device('cuda' if args.gpu else 'cpu')
    model.to(device)
    print('\nUse GPU: %s \n devive: %s \n' %(torch.cuda.is_available(), torch.cuda.get_device_name()))

    #criterion
    criterion = nn.CrossEntropyLoss()
    #optimizer
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)

    #scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma=0.9, last_epoch=-1)

    #start training
    #tensorboard
    tensorboard_path = os.path.join(args.logdir, args.network)
    tensorboard_path = os.path.join(tensorboard_path, 'new_fc' + '_' + str(args.new_fc))
    print('\nNew FC layer: %s' %(str(args.new_fc)))

    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)
    writer = SummaryWriter(tensorboard_path)

    epochs = args.epochs
    train_iteration = len(dataloaders['train'])
    test_iteration = len(dataloaders['test'])

    plt_train_precision = []
    plt_train_recall = []
    plt_train_F1 = []
    plt_train_losses= []
    plt_test_precision = []
    plt_test_recall = []
    plt_test_F1 = []
    plt_test_losses = []

    for epoch in range(epochs):

        train_precision = []
        train_recall = []
        train_F1 = []
        train_losses= []
        test_precision = []
        test_recall = []
        test_F1 = []
        test_losses = []

        #train
        model.train()
        print('Start training ... \n\n')
        for i, (inputs, labels) in enumerate(dataloaders['train']):
            #load data to gpu
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model.forward(inputs)
            #loss
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            #calculation
            _, pre_labels = torch.max(output.data, 1)
            precision, recall, f1 = evaluation(labels, pre_labels)

            train_precision.append(precision)
            train_recall.append(recall)
            train_F1.append(f1)

            training_loss = loss.item()
            train_losses.append(training_loss)
            #print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{train_iteration}], Loss: {training_loss:.6f}, precision: {precision:.6f}, recall: {recall:.6f}, F1: {f1:.6f}')
            print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{train_iteration}], Loss: {training_loss:.6f}')

        train_precision = np.asarray(train_precision)
        train_recall = np.asarray(train_recall)
        train_F1 = np.asarray(train_F1)

        train_epoch_precision = sum(train_precision) / train_iteration
        train_epoch_recall = sum(train_recall) / train_iteration
        train_epoch_F1 = sum(train_F1) / train_iteration
        train_epoch_loss = sum(train_losses) / train_iteration

        #write precision
        writer.add_scalar('Training/total_precision', float(sum(train_epoch_precision) / 5), epoch)
        writer.add_scalars('Training/precision', {'daisy':train_epoch_precision[0], 'dandelion':train_epoch_precision[1], 'rose':train_epoch_precision[2], 'sunflower':train_epoch_precision[3], 'tulip':train_epoch_precision[4]}, epoch)
        #write recall
        writer.add_scalar('Training/total_recall', float(sum(train_epoch_recall) / 5), epoch)
        writer.add_scalars('Training/recall', {'daisy':train_epoch_recall[0], 'dandelion':train_epoch_recall[1], 'rose':train_epoch_recall[2], 'sunflower':train_epoch_recall[3], 'tulip':train_epoch_recall[4]}, epoch)
        #write f1
        writer.add_scalar('Training/total_F1', float(sum(train_epoch_F1) / 5), epoch)
        writer.add_scalars('Training/F1', {'daisy':train_epoch_F1[0], 'dandelion':train_epoch_F1[1], 'rose':train_epoch_F1[2], 'sunflower':train_epoch_F1[3], 'tulip':train_epoch_F1[4]}, epoch)
        #write training loss
        writer.add_scalar('Training/loss', train_epoch_loss, epoch)

        plt_train_precision.append(train_epoch_precision)
        plt_train_recall.append(train_epoch_recall)
        plt_train_F1.append(train_epoch_F1)
        plt_train_losses.append(train_epoch_loss)


        #test
        model.eval()
        print('\n\nStart testing ... \n\n')
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloaders['test']):
                #load data to gpu
                inputs, labels = inputs.to(device), labels.to(device)
                output = model.forward(inputs)
                loss = criterion(output, labels)
                test_loss = loss.item()

                #calculation
                _, pre_labels = torch.max(output.data, 1)
                precision, recall, f1 = evaluation(labels, pre_labels)

                test_precision.append(precision)
                test_recall.append(recall)
                test_F1.append(f1)
                test_losses.append(test_loss)
                #print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{test_iteration}], Loss: {test_loss:.6f}, precision: {precision:.6f}, recall: {recall:.6f}, F1: {f1:.6f}')
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{test_iteration}], Loss: {test_loss:.6f}')

            test_precision = np.asarray(test_precision)
            test_recall = np.asarray(test_recall)
            test_F1 = np.asarray(test_F1)

            test_epoch_precision = sum(test_precision) / test_iteration
            test_epoch_recall = sum(test_recall) / test_iteration
            test_epoch_F1 = sum(test_F1) / test_iteration
            test_epoch_loss = sum(test_losses) / test_iteration
            #write precision
            writer.add_scalar('Test/total_precision', float(sum(test_epoch_precision) / 5), epoch)
            writer.add_scalars('Test/precision', {'daisy':test_epoch_precision[0], 'dandelion':test_epoch_precision[1], 'rose':test_epoch_precision[2], 'sunflower':test_epoch_precision[3], 'tulip':test_epoch_precision[4]}, epoch)
            #write recall
            writer.add_scalar('Test/total_recall', float(sum(test_epoch_recall) / 5), epoch)
            writer.add_scalars('Test/recall', {'daisy':test_epoch_recall[0], 'dandelion':test_epoch_recall[1], 'rose':test_epoch_recall[2], 'sunflower':test_epoch_recall[3], 'tulip':test_epoch_recall[4]}, epoch)
            #write f1
            writer.add_scalar('Test/total_F1', float(sum(test_epoch_F1) / 5), epoch)
            writer.add_scalars('Test/F1', {'daisy':test_epoch_F1[0], 'dandelion':test_epoch_F1[1], 'rose':test_epoch_F1[2], 'sunflower':test_epoch_F1[3], 'tulip':test_epoch_F1[4]}, epoch)
            #write training loss
            writer.add_scalar('Test/loss', test_epoch_loss, epoch)

            plt_test_precision.append(test_epoch_precision)
            plt_test_recall.append(test_epoch_recall)
            plt_test_F1.append(test_epoch_F1)
            plt_test_losses.append(test_epoch_loss)


        #scheduler
        scheduler.step()

    evaluations = [plt_train_precision, plt_train_recall, plt_train_F1, plt_train_losses, plt_test_precision, plt_test_recall, plt_test_F1, plt_test_losses]
    draw_curces(evaluations, args.epochs, tensorboard_path)

    #save model
    model_path = os.path.join(os.getcwd(), args.save_dir)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_name = args.network + '_' + 'freeze_' + str(args.freeze) + '.pth'
    model_path = os.path.join(model_path, model_name)
 

def evaluation(gt, pre):

    gt = F.one_hot(gt, 5).cpu()
    pre = F.one_hot(pre, 5).cpu()
    epsilon = 1e-7

    pp = pre.t().numpy()
    gg = gt.numpy()

    res = np.zeros(shape = (5, 5))
    for i in range(np.size(pp, 1)):
        res += np.dot(pp[:, i].reshape(5, 1), gg[i].reshape(1,5))

    precision_classes = []
    recall_classes = []
    f1_classes = []
    for i in range(5):
        TN = (int(np.sum(np.delete(np.delete(res, i, axis = 0), i, axis = 1))))
        FP = (int(np.sum(res[:, i]) - res[i, i]))
        FN = (int(np.sum(res[i]) - res[i, i]))
        TP = (int(res[i, i]))

        precision = TP / (TP + FP + epsilon)
        recall = TP / (TP + FN + epsilon)
        f1 = 2 * (precision * recall) / (precision + recall + epsilon)

        precision_classes.append(precision)
        recall_classes.append(recall)
        f1_classes.append(f1)

    return precision_classes, recall_classes, f1_classes


def draw_curces(matrix, epoch, path):
    x = range(epoch)
    #train precision
    plt.plot(x, np.array(matrix[0])[:, 0], 'r', label = 'daisy')
    plt.plot(x, np.array(matrix[0])[:, 1], 'g', label = 'dandelion')
    plt.plot(x, np.array(matrix[0])[:, 2], 'b', label = 'sunflower')
    plt.plot(x, np.array(matrix[0])[:, 3], 'y', label = 'rose')
    plt.plot(x, np.array(matrix[0])[:, 4], 'c', label = 'tulip')
    plt.plot(x, np.sum(np.array(matrix[0]), axis = 1) / 5, 'k', label = 'average')

    plt.xlabel('epochs')
    plt.ylabel('Value')
    plt.title('Training precision')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(path, './training_precision.png'))
    plt.cla()

    #train recall
    plt.plot(x, np.array(matrix[1])[:, 0], 'r', label = 'daisy')
    plt.plot(x, np.array(matrix[1])[:, 1], 'g', label = 'dandelion')
    plt.plot(x, np.array(matrix[1])[:, 2], 'b', label = 'sunflower')
    plt.plot(x, np.array(matrix[1])[:, 3], 'y', label = 'rose')
    plt.plot(x, np.array(matrix[1])[:, 4], 'c', label = 'tulip')
    plt.plot(x, np.sum(np.array(matrix[1]), axis = 1) / 5, 'k', label = 'average')

    plt.xlabel('epochs')
    plt.ylabel('Value')
    plt.title('Training recall')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(path, './training_recall.png'))
    plt.cla()

    #train F1
    plt.plot(x, np.array(matrix[2])[:, 0], 'r', label = 'daisy')
    plt.plot(x, np.array(matrix[2])[:, 1], 'g', label = 'dandelion')
    plt.plot(x, np.array(matrix[2])[:, 2], 'b', label = 'sunflower')
    plt.plot(x, np.array(matrix[2])[:, 3], 'y', label = 'rose')
    plt.plot(x, np.array(matrix[2])[:, 4], 'c', label = 'tulip')
    plt.plot(x, np.sum(np.array(matrix[2]), axis = 1) / 5, 'k', label = 'average')

    plt.xlabel('epochs')
    plt.ylabel('Value')
    plt.title('Training F1')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(path, './training_F1.png'))
    plt.cla()

    #train loss
    plt.plot(x, np.array(matrix[3]), 'r', label = 'training loss')

    plt.xlabel('epochs')
    plt.ylabel('Value')
    plt.title('Training loss')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(path, './training_loss.png'))
    plt.cla()


    #test precision
    plt.plot(x, np.array(matrix[4])[:, 0], 'r', label = 'daisy')
    plt.plot(x, np.array(matrix[4])[:, 1], 'g', label = 'dandelion')
    plt.plot(x, np.array(matrix[4])[:, 2], 'b', label = 'sunflower')
    plt.plot(x, np.array(matrix[4])[:, 3], 'y', label = 'rose')
    plt.plot(x, np.array(matrix[4])[:, 4], 'c', label = 'tulip')
    plt.plot(x, np.sum(np.array(matrix[4]), axis = 1) / 5, 'k', label = 'average')

    plt.xlabel('epochs')
    plt.ylabel('Value')
    plt.title('Test precision')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(path, './test_precision.png'))
    plt.cla()

    #test recall
    plt.plot(x, np.array(matrix[5])[:, 0], 'r', label = 'daisy')
    plt.plot(x, np.array(matrix[5])[:, 1], 'g', label = 'dandelion')
    plt.plot(x, np.array(matrix[5])[:, 2], 'b', label = 'sunflower')
    plt.plot(x, np.array(matrix[5])[:, 3], 'y', label = 'rose')
    plt.plot(x, np.array(matrix[5])[:, 4], 'c', label = 'tulip')
    plt.plot(x, np.sum(np.array(matrix[5]), axis = 1) / 5, 'k', label = 'average')

    plt.xlabel('epochs')
    plt.ylabel('Value')
    plt.title('Test recall')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(path, './test_recall.png'))
    plt.cla()

    #test F1
    plt.plot(x, np.array(matrix[6])[:, 0], 'r', label = 'daisy')
    plt.plot(x, np.array(matrix[6])[:, 1], 'g', label = 'dandelion')
    plt.plot(x, np.array(matrix[6])[:, 2], 'b', label = 'sunflower')
    plt.plot(x, np.array(matrix[6])[:, 3], 'y', label = 'rose')
    plt.plot(x, np.array(matrix[6])[:, 4], 'c', label = 'tulip')
    plt.plot(x, np.sum(np.array(matrix[6]), axis = 1) / 5, 'k', label = 'average')

    plt.xlabel('epochs')
    plt.ylabel('Value')
    plt.title('Test F1')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(path,'./test_F1.png'))
    plt.cla()

    #test loss
    plt.plot(x, np.array(matrix[7]), 'r', label = 'Test loss')

    plt.xlabel('epochs')
    plt.ylabel('Value')
    plt.title('Test loss')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(path, './test_loss.png'))
    plt.cla()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
#
    parser.add_argument('--data_dir', type=str, default='flowers', help='Directory of the original data')
    parser.add_argument('--save_dir', type=str, default='weights', help='Directory for saving the checkpoint')
    parser.add_argument('--batch_size', type=int, default='64', help='Batch size')
    parser.add_argument('--image_size', type=int, default='224', help='Resize all images to a specific size')
    parser.add_argument('--network', type=str, default = 'vgg16', help='The CNN model architecture for training, option: vgg16, densenet161, resnet18, mobilenet_v2')
    parser.add_argument('--learning_rate', type=float, default = 0.001, help='Learning rate')
    parser.add_argument('--gpu', type=bool, default=True, help='Use GPU if its available')
    parser.add_argument('--epochs', type=int, default = 100 , help='Epochs')
    parser.add_argument('--logdir', type=str, default='log/', help='Logging directory for tensorboard')
    parser.add_argument('--freeze', type=bool, default=True, help='Freezing all the parameters from former layers')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer')
    parser.add_argument('--pre_train', type=bool, default=True, help='pre_train or not')
    parser.add_argument('--new_fc', type=bool, default=False, help='type of fc you want to fine-tune on, 0 is for the original fc layer in the network, 1 is for new fc layer')

    args = parser.parse_args()

    train(args)
