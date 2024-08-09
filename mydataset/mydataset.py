import numpy as np
import glob
import os
from os.path import join
import random
import h5py
from torchvision.io import read_image
from torch.utils.data import Dataset
import torch
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

class recurnot(Dataset):
    '''
    dataset for lung carcinoma pilot study data. predict recur or not in 5 yrs
    '''
    def __init__(self, train='train', transform=None, args=None, split=42):

        self.img_dir = './data/{}embedding/pilotstudyl0p{}s{}'.format(args.encoder, args.psize, args.psize)

        self.split = pd.read_csv(join('./splits/recurnot', 'splits_{}.csv'.format(split-42)), header=0)
        self.labels = pd.read_excel('./dataset_csv/filedfclean.xlsx', header=0, index_col='AI ID')
        self.nreckeys = np.load('./data/keys/recurnot/{}-p{}-nrec-{}-f{}.npy'.format(args.encoder, args.psize, args.t, split-42))
        self.reckeys = np.load('./data/keys/recurnot/{}-p{}-rec-{}-f{}.npy'.format(args.encoder, args.psize, args.t, split-42))
        self.train = train
        
        if train == 'train':
            self.img_names = self.split.loc[:, ['train']].dropna()
        elif train == 'test':
            self.img_names = self.split.loc[:, ['test']].dropna()
        elif train == 'val':
            self.img_names = self.split.loc[:, ['val']].dropna()
            
        self.transform = transform

    def __len__(self):
        return len(self.img_names)
    
    def get_weights(self):
        # get weights for weight random sampler (training only)
        if self.train != 'train':
            raise TypeError('WEIGHT SAMPLING FOR TRAINING SET ONLY')
        N = len(self.img_names)
        
        labels = self.labels.loc[self.img_names['train']]
        labels['Months until recurrence (60 if not recur)'] = (labels['Months until recurrence (60 if not recur)'] < 60) * 1

        w_per_cls = {1: N/labels['Months until recurrence (60 if not recur)'].sum(), 0: N/(N-labels['Months until recurrence (60 if not recur)'].sum())}
        weights = [w_per_cls[labels.loc[name, 'Months until recurrence (60 if not recur)']] for name in self.img_names['train']]

        return torch.DoubleTensor(weights)
    
    def get_testnames(self):
        return self.split['test'].dropna().tolist()

    def get_keysetdims(self):
        return [self.nreckeys.shape[0], self.reckeys.shape[0]]

    def __getitem__(self, idx):
        # print('*****************', idx)
        iid = self.img_names.iloc[idx][self.train]
        img_path = os.path.basename(self.labels.loc[iid, 'filename'])[:-5]
        img_path = join(self.img_dir, img_path+'.npy')
        image = np.load(img_path)

        label = int(self.labels.loc[iid, 'Months until recurrence (60 if not recur)'] < 60)
            
        return torch.Tensor(image), torch.Tensor(self.reckeys), torch.Tensor(self.nreckeys), label
