from scipy.io import loadmat
import numpy as np 

import argparse
import time 

from torch.utils.data.dataset import Dataset
import torch 
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch import nn  
from torchsummary import summary
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from utils import evaluate_accuracy, aa_and_each_accuracy, record_output
import torch.optim as optim
import torch.nn.functional as F
import math 


parser = argparse.ArgumentParser(description='Training for HSI')

parser.add_argument('--dataset', type=str, default='DUP', choices=["DUP"])
parser.add_argument('--model', type=str, default='ViT')

parser.add_argument('--patch_size', type=int, default=11, help="Patch size.") 
parser.add_argument('--iter', type=int, default=5, help="No of iter.")   
parser.add_argument('--learning_rate', type=float, default=0.001, help="No of iter.")  
parser.add_argument('--epochs', type=int, default=500, help="No of epochs.")   
parser.add_argument('--train_percent', default=0.9, type=float, help='Percentage of validation split.')
args = parser.parse_args()




# Data Loading
def loaddata(name, patch):
    if patch == 11:
        if name == 'DUT':
            x_train = loadmat('./dataset/Trento11x11/HSI_Tr.mat')['Data']
            y_train = loadmat('./dataset/Trento11x11/TrLabel.mat')['Data']
            y_train = np.squeeze(y_train) - 1

            x_test = loadmat('./dataset/Trento11x11/HSI_Te.mat')['Data']
            y_test = loadmat('./dataset/Trento11x11/TeLabel.mat')['Data']
            y_test = np.squeeze(y_test) - 1

            bands = x_train.shape[3]
            classes = len(np.unique(y_train))
        elif name == 'DUH':
            x_train = loadmat('./dataset/Houston11x11/HSI_Tr.mat')['Data']
            y_train = loadmat('./dataset/Houston11x11/TrLabel.mat')['Data']
            y_train = np.squeeze(y_train) - 1

            x_test = loadmat('./dataset/Houston11x11/HSI_Te.mat')['Data']
            y_test = loadmat('./dataset/Houston11x11/TeLabel.mat')['Data']
            y_test = np.squeeze(y_test) - 1

            bands = x_train.shape[3]
            classes = len(np.unique(y_train)) 
        elif name == 'DIP':
            x_train = loadmat('./dataset/IP11x11/HSI_Tr.mat')['Data']
            y_train = loadmat('./dataset/IP11x11/TrLabel.mat')['Data']
            y_train = np.squeeze(y_train) - 1

            x_test = loadmat('./dataset/IP11x11/HSI_Te.mat')['Data']
            y_test = loadmat('./dataset/IP11x11/TeLabel.mat')['Data']
            y_test = np.squeeze(y_test) - 1

            bands = x_train.shape[3]
            classes = len(np.unique(y_train))
        elif name == 'DUP':
            x_train = loadmat('./dataset/UP11x11/HSI_Tr_minmaxnormalize.mat')['Data']
            y_train = loadmat('./dataset/UP11x11/TrLabel_minmaxnormalize.mat')['Data']
            # y_train = np.squeeze(y_train) - 1
            y_train = np.squeeze(y_train) # during normalization stage, we have -1 already, 

            x_test = loadmat('./dataset/UP11x11/HSI_Te_minmaxnormalize.mat')['Data']
            y_test = loadmat('./dataset/UP11x11/TeLabel_minmaxnormalize.mat')['Data']
            # y_test = np.squeeze(y_test) - 1
            y_test = np.squeeze(y_test)

            bands = x_train.shape[3]
            classes = len(np.unique(y_train))
        else:
            print('NO DATASET')
            exit()
    else:
        path = './dataset/DUP/'+ str(patch) + 'x' + str(patch) +'/'
        x_train = loadmat(path + 'HSI_Tr.mat')['Data']
        y_train = loadmat(path + 'TrLabel.mat')['Data']
        y_train = np.squeeze(y_train) 

        x_test = loadmat(path + 'HSI_Te.mat')['Data']
        y_test = loadmat(path + 'TeLabel.mat')['Data']
        y_test = np.squeeze(y_test)

        bands = x_train.shape[3]
        classes = len(np.unique(y_train))
    x_train = np.transpose(x_train, (0,3,1,2))
    x_test = np.transpose(x_test, (0,3,1,2))
    # x_train, x_test = Normalization(x_train), Normalization(x_test)
    
    return x_train, y_train, x_test, y_test, bands, classes 


class HyperDataset(Dataset):
    def __init__(self, dataset):
        self.data = dataset[0].astype(np.float32)
        self.labels = []
        for n in dataset[1]: self.labels += [int(n)]

    def __getitem__(self, index):
        img = torch.from_numpy(np.asarray(self.data[index, :, :, :]))
        label = self.labels[index]
        return img, label

    def __len__(self):
        return len(self.labels)

    def __labels__(self):
        return self.labels

from sklearn.decomposition import PCA

# def applyPCA(X, numComponents):
#     newX = np.reshape(X, (-1, X.shape[1]))
#     pca = PCA(n_components=numComponents, whiten=True)
#     newX = pca.fit_transform(newX)
#     newX = np.reshape(newX, (X.shape[0], numComponents, X.shape[2], X.shape[3]))
#     return newX




X_DATA, Y_DATA, x_test, y_test, bands, classes = loaddata(name = args.dataset,patch=args.patch_size)


from nat_token import MS_STN_NAT2

def get_model(name, bands, classes, patch):
    
    if name == 'SSRN':
        net = SSRN_network(bands,classes,patch)

    elif name == 'MS_STN_NAT':
        net = MS_STN_NAT2(bands=bands, num_classes=classes)
 

    optimizer = optim.Adam(
            net.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0)    
    loss = torch.nn.CrossEntropyLoss()
    return net, optimizer, loss 


from pytorchtools import EarlyStopping

def train(net,
          train_iter,
          valida_iter,
          loss,
          optimizer,
          device,
          epochs,
          early_stopping=True,
          early_num=20):
    loss_list = [100]
    early_epoch = 0

    net = net.to(device)
    print("training on ", device)
    start = time.time()
    train_loss_list = []
    valida_loss_list = []
    train_acc_list = []
    valida_acc_list = []
    early_stopping = EarlyStopping(patience=50, verbose=True, path='checkpoint_dis_DUP.pt')
    for epoch in range(epochs):
        train_acc_sum, n = 0.0, 0
        time_epoch = time.time()
        lr_adjust = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 15, eta_min=0.0, last_epoch=-1)
        for X, y in train_iter:

            batch_count, train_l_sum = 0, 0
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            #print(y_hat.shape, y.shape)
            #exit()
            l = loss(y_hat, y.long())

            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        lr_adjust.step()
        valida_acc, valida_loss = evaluate_accuracy(
            valida_iter, net, loss, device)
        loss_list.append(valida_loss)

        train_loss_list.append(train_l_sum)  # / batch_count)
        train_acc_list.append(train_acc_sum / n)
        valida_loss_list.append(valida_loss)
        valida_acc_list.append(valida_acc)

        print(
            'epoch %d, train loss %.6f, train acc %.3f, valida loss %.6f, valida acc %.3f, time %.1f sec'
            % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n,
               valida_loss, valida_acc, time.time() - time_epoch))

        early_stopping(valida_loss, net)

        if early_stopping.early_stop:
            print("Early stopping")
            break

        # PATH = "./net_DBA_a2s2k.pt"

        # if early_stopping and loss_list[-2] < loss_list[-1]:
        #     if early_epoch == 0:
        #         torch.save(net.state_dict(), PATH)
        #     early_epoch += 1
        #     loss_list[-1] = loss_list[-2]
        #     if early_epoch == early_num:
        #         net.load_state_dict(torch.load(PATH))
        #         break
        # else:
        #     early_epoch = 0

    net.load_state_dict(torch.load('checkpoint_dis_DUP.pt'))

    print('epoch %d, loss %.4f, train acc %.3f, time %.1f sec'
          % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n,
             time.time() - start))




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seeds = [1331, 1332, 1333, 1334, 1335, 1336, 1337, 1338, 1339, 1340, 1341]

KAPPA = []
OA = []
AA = []
TRAINING_TIME = []
TESTING_TIME = []
ELEMENT_ACC = np.zeros((args.iter, classes))

for index_iter in range(args.iter):

    x_train, x_val, y_train, y_val = train_test_split(X_DATA,Y_DATA , train_size=args.train_percent, stratify=Y_DATA, random_state=seeds[index_iter])

    print('x_train shape:',x_train.shape)
    print('y_train shape:',y_train.shape)
    print('x_val shape:',x_val.shape)
    print('y_val shape:',y_val.shape)
    print('x_test shape:',x_test.shape)
    print('y_test shape:',y_test.shape)
    print('bands:', bands)
    print('classes:',classes)

    train_dataset = HyperDataset((x_train.astype('float32'),y_train))
    val_dataset = HyperDataset((x_val.astype('float32'),y_val))
    test_dataset = HyperDataset((x_test.astype('float32'), y_test))


    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    
    net, optimizer, loss = get_model(args.model, bands=bands, classes=classes, patch=args.patch_size)
    tic1 = time.time()
    train(
        net, 
        train_loader,
        val_loader, 
        loss,
        optimizer, 
        device, 
        epochs=args.epochs)
    toc1 = time.time()

    pred_test = []
    tic2 = time.time()
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            net.eval()
            y_hat = net(X)
            pred_test.extend(np.array(y_hat.cpu().argmax(axis=1)))
    toc2 = time.time()

    gt_test = y_test 

    overall_acc = metrics.accuracy_score(pred_test, gt_test)
    confusion_matrix = metrics.confusion_matrix(gt_test, pred_test)
    each_acc, average_acc = aa_and_each_accuracy(confusion_matrix)
    kappa = metrics.cohen_kappa_score(pred_test, gt_test)

    KAPPA.append(kappa)
    OA.append(overall_acc)
    AA.append(average_acc)
    TRAINING_TIME.append(toc1 - tic1)
    TESTING_TIME.append(toc2 - tic2)
    ELEMENT_ACC[index_iter, :] = each_acc

    if args.train_percent == 0.9:
        if args.patch_size == 11:
            torch.save(net.state_dict(), './checkpoints/'+str(args.dataset) + '_' + 
            str(args.model) + '_' + str(round(overall_acc, 5)) + '.pt')
        else: 
            torch.save(net.state_dict(), './checkpoints/checkpoints_with_different_patch_size'+str(args.dataset) + '_' + 
            str(args.model) + '_' + str(args.patch_size) +'x' + str(args.patch_size) + '_' + str(round(overall_acc, 5)) + '.pt')
    else:
        torch.save(net.state_dict(), './checkpoints/checkpoints_with_different_train_percent/'+str(args.dataset) + '_' + 
        str(args.model) + '_train_percent_' + str(args.train_percent) + '_' + str(round(overall_acc, 5)) + '.pt')


print(OA, AA, KAPPA, ELEMENT_ACC, TRAINING_TIME, TESTING_TIME)

if args.train_percent == 0.9:
    if args.patch_size == 11:
        record_output(
            OA, AA, KAPPA, ELEMENT_ACC, TRAINING_TIME, TESTING_TIME,
            './'+ str(args.dataset) + '/' + str(args.dataset) + '_' +
            str(args.model) + '_lr_' + str(args.learning_rate) +'_' + str(round(overall_acc, 5)) + '.txt')
    else: 
        record_output(
            OA, AA, KAPPA, ELEMENT_ACC, TRAINING_TIME, TESTING_TIME,
            './'+ str(args.dataset) + '/' + str(args.dataset) + '_' +
            str(args.model) + '_' + str(args.patch_size) +'x' + str(args.patch_size) + '_lr_' + str(args.learning_rate) +'_' + str(round(overall_acc, 5)) + '.txt')
else:
    record_output(
        OA, AA, KAPPA, ELEMENT_ACC, TRAINING_TIME, TESTING_TIME,
        './'+ str(args.dataset) + '/' + str(args.dataset) + '_' +
        str(args.model) + '_train_percent_' + str(args.train_percent) + '_' + str(round(overall_acc, 5)) + '.txt')


