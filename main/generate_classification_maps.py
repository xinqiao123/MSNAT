# generate classification maps


import numpy as np 
import torch 
from scipy.io import loadmat

import argparse
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

from torch import nn 
import torch.nn.functional as F
import math


from sklearn.preprocessing import MinMaxScaler



parser = argparse.ArgumentParser(description='Training for HSI')

parser.add_argument('--dataset', type=str, default='IN', choices=["IN", "DUP", "DUH","DUT"])
parser.add_argument('--model', type=str, default='SSRN')
parser.add_argument('--checkpoint_path', type=str, default='IN_0.1_SSRN_0.976.pt')

args = parser.parse_args()





def loaddata(name):
    if name == 'IN':
        data = loadmat('./dataset/Indian_pines_corrected')['indian_pines_corrected']
        labels = loadmat('./dataset/Indian_pines_gt')['indian_pines_gt']
    elif name == 'DUP':
        data = loadmat('./dataset/PaviaU.mat')['paviaU']
        labels = loadmat('./dataset/PaviaU_gt.mat')['paviaU_gt']
    elif name == 'SA':
        data = loadmat('./dataset/Salinas_corrected.mat')['salinas_corrected']
        labels = loadmat('./dataset/Salinas_gt.mat')['salinas_gt']
    elif name == 'KSC':
        data = loadmat('./dataset/KSC.mat')['KSC']
        labels = loadmat('./dataset/KSC_gt.mat')['KSC_gt']
    elif name == 'DUH':
        data = loadmat('./disjoint_dataset/Houston/houston.mat')['houston']
        tr_labels = loadmat('./disjoint_dataset/Houston/TRLabel/TRLabel.mat')['TRLabel']
        te_labels = loadmat('./disjoint_dataset/Houston/TSLabel.mat')['TSLabel']
        labels = tr_labels + te_labels
    elif name == 'DUT':
        data = loadmat('./disjoint_dataset/Trento/HSI.mat')['HSI']
        tr_labels = loadmat('./disjoint_dataset/Trento/TRLabel.mat')['TRLabel']
        te_labels = loadmat('./disjoint_dataset/Trento/TSLabel.mat')['TSLabel']
        labels = tr_labels + te_labels
    else:
        print('NO DATASET')
        exit()

    # data = Normalization(data)
    hh,ww,cc = data.shape[0], data.shape[1], data.shape[2]
    data = data.reshape(-1, data.shape[-1])
    data = MinMaxScaler().fit_transform(data)
    data = data.reshape(hh, ww, cc)

    bands = data.shape[2]
    classes = len(np.unique(labels)) - 1

    
    return data, labels, bands, classes


def sampling_fixed_percent(groundTruth):
    labels_loc = {}
    # train = {}
    # val = {}
    # test = {}
    m = np.max(groundTruth)
    for i in range(m):
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1]
        # np.random.shuffle(indices)
        labels_loc[i] = indices
        # nb_train =int(np.ceil(trainproportion * len(indices)))
        # nb_val = int(np.ceil(valproportion * len(indices)))
        # nb_test = len(indices) - nb_train - nb_val 
        # train[i] = indices[:nb_train]
        # val[i] = indices[nb_train:nb_train+nb_val]
        # test[i] = indices[nb_train+nb_val:]
    whole_indices = []
    # train_indices = []
    # val_indices = []
    # test_indices = []
    for i in range(m):
        whole_indices += labels_loc[i]
        # train_indices += train[i]
        # val_indices += val[i]
        # test_indices += test[i]
        # np.random.shuffle(whole_indices)
        # np.random.shuffle(train_indices) 
        # np.random.shuffle(val_indices)  
        # np.random.shuffle(test_indices)
    return whole_indices

def zeroPadding_3D(old_matrix, pad_length, pad_depth = 0):
    new_matrix = np.lib.pad(old_matrix, ((pad_depth, pad_depth), (pad_length, pad_length), (pad_length, pad_length)), 'constant', constant_values=0)
    return new_matrix


def indexToAssignment(index_, pad_length, Row, Col):
    new_assign = {}
    for counter, value in enumerate(index_):
        assign_0 = value // Col + pad_length
        assign_1 = value % Col + pad_length
        new_assign[counter] = [assign_0, assign_1]
    return new_assign

def selectNeighboringPatch(matrix, ex_len, pos_row, pos_col):
    # print(matrix.shape)
    selected_rows = matrix[:,range(pos_row-ex_len,pos_row+ex_len+1), :]
    selected_patch = selected_rows[:, :, range(pos_col-ex_len, pos_col+ex_len+1)]
    return selected_patch
    



def predVisIN(indices, pred, size1, size2):
    
    if pred.ndim > 1:
        pred = np.ravel(pred)
    
    x = np.zeros(size1*size2)
    x[indices] = pred
    
    y = np.ones((x.shape[0], 3))

    for index, item in enumerate(x):
        if item == 0:
            y[index] = np.array([0, 0, 0]) / 255.  # np.array([255, 255, 255]) / 255.
        if item == 1:
            y[index] = np.array([255, 0, 0]) / 255.
        if item == 2:
            y[index] = np.array([0, 255, 0]) / 255.
        if item == 3:
            y[index] = np.array([0, 0, 255]) / 255.
        if item == 4:
            y[index] = np.array([255, 255, 0]) / 255.
        if item == 5:
            y[index] = np.array([0, 255, 255]) / 255.
        if item == 6:
            y[index] = np.array([255, 0, 255]) / 255.
        if item == 7:
            y[index] = np.array([192, 192, 192]) / 255.
        if item == 8:
            y[index] = np.array([128, 128, 128]) / 255.
        if item == 9:
            y[index] = np.array([128, 0, 0]) / 255.
        if item == 10:  
            y[index] = np.array([128, 128, 0]) / 255.
        if item == 11:
            y[index] = np.array([0, 128, 0]) / 255.
        if item == 12:
            y[index] = np.array([128, 0, 128]) / 255.
        if item == 13:
            y[index] = np.array([0, 128, 128]) / 255.
        if item == 14:
            y[index] = np.array([0, 0, 128]) / 255.
        if item == 15:
            y[index] = np.array([255, 165, 0]) / 255.
        if item == 16:
            y[index] = np.array([255, 215, 0]) / 255.
    
    y_rgb = np.reshape(y, (size1, size2, 3))
    
    return y_rgb




data, labels, bands, classes = loaddata(args.dataset)
gt = labels.reshape(np.prod(labels.shape[:2]))
whole_indices = sampling_fixed_percent(gt)
data = np.transpose(data, (2,0,1))
h = data.shape[1]
w = data.shape[2]

PATCH_LENGTH = 5
INPUT_DIMENSION_CONV = bands 

padded_data = zeroPadding_3D(data, PATCH_LENGTH)
all_data = np.zeros((len(whole_indices), INPUT_DIMENSION_CONV, 2*PATCH_LENGTH + 1, 2*PATCH_LENGTH + 1))
y_all = gt[whole_indices] - 1


whole_data = data 


all_assign = indexToAssignment(whole_indices, PATCH_LENGTH, whole_data.shape[1], whole_data.shape[2])
for i in range(len(all_assign)):
    all_data[i] = selectNeighboringPatch(padded_data, PATCH_LENGTH, all_assign[i][0], all_assign[i][1])





class HSIDataset(Dataset):
    def __init__(self, list_IDs, samples, labels):
        
        self.list_IDs = list_IDs
        self.samples = samples.astype(np.float32)
        self.labels = labels

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = self.samples[ID]
        y = self.labels[ID]

        return X, y

all_set = HSIDataset(range(len(whole_indices)), all_data, y_all)
allloader = DataLoader(all_set, batch_size=32, shuffle=False)




from nat_token import MS_STN_NAT2



def get_model(name, bands, classes):
    
    if name == 'SSRN':
        net = SSRN_network(bands,classes)
  
    elif name == 'MS_STN_NAT':
        net = MS_STN_NAT2(bands=bands, num_classes=classes)
 

    return net

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


net= get_model(args.model, bands=bands, classes=classes)

net.to(device)
path = args.checkpoint_path
net.load_state_dict(torch.load(path))
net.eval()



all_pred = []
gt = []

with torch.no_grad():
    for data in allloader:
        images, y = data
        images, y = images.to(device), y.to(device)
        outputs = net(images.float())
        _, predicted = torch.max(outputs.data, 1)
        all_pred.append(predicted)
        # gt.append(y)

all_pred = torch.cat(all_pred)
all_pred = all_pred.cpu().numpy() + 1

y_pred = predVisIN(whole_indices, all_pred, h, w)


import matplotlib.pyplot as plt 
# plt.plot(x, y)
plt.imshow(y_pred)
plt.axis('off')
fig_path = './Maps-TSNE/maps-' + str(args.dataset) + '_' + str(args.model) + '.png'
plt.savefig(fig_path, bbox_inches='tight')
#plt.savefig(fig_path, bbox_inches='tight')



# gt = torch.cat(gt)
# gt = gt.cpu().numpy() + 1
# gt = predVisIN(whole_indices, gt, 145, 145)
# plt.imshow(gt)
# plt.axis('off')
# fig_path = './figs/gt_' + str(args.model) + '.png'
# plt.savefig(fig_path, bbox_inches=0)
# #plt.savefig(fig_path, bbox_inches='tight')