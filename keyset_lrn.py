import argparse
import os
from os.path import join
import glob
import numpy as np
import pandas as pd
import h5py

parser = argparse.ArgumentParser(description='lung')
parser.add_argument('--datadf', default='../filedfclean_zs.xlsx', 
                    type=str, help='dataset excel')
parser.add_argument('--featdir', default='./data/ctpembedding/pilotstudyl0p448s448', type=str,
                    help='path to the feats of normal WSIs in trainng set')
parser.add_argument('--task', default='recurnot', type=str,
                    help='task')
parser.add_argument('--encoder', default='ctp', type=str,
                    help='encoder name')
parser.add_argument('--fold', default=0, type=int,
                    help='fold index')                    
parser.add_argument('--savedir', default='./data/keys', type=str,
                    help='path to save learned key set')
parser.add_argument('--curdir', default='./data/cur', type=str,
                    help='path to save learned key set')
parser.add_argument('-t', default=100, type=int,
                    help='maximum number of keys from each normal WSI')       
parser.add_argument('--dim', default=768, type=int,
                    help='feat dim')  
parser.add_argument('--psize', default=448, type=int,
                    help='patch size')
parser.add_argument('--ns', default=35, type=int,
                    help='number of samples')
parser.add_argument('--cur', action='store_true', default=False, 
                    help='run cur and save col ranking')  
parser.add_argument('--extract', action='store_true', default=False, 
                    help='extract keys based on cur results')         



def extractTopKColumns(matrix):
    '''
    Learn representative negative instances from each normal WSI
    '''
    score  = {}
    rank = np.linalg.matrix_rank(matrix)
    u, s, vh = np.linalg.svd(matrix, full_matrices=True)
    
    for j in range(0, matrix.shape[1]):
        cscore = sum(np.square(vh[0:rank,j]))
        cscore /= rank
        score[j] = min(1, rank*cscore)
        
    prominentColumns = sorted(score, key=score.get, reverse=True)[:rank]
    #Removal of extra dimension\n",
    C = np.squeeze(matrix[:, [prominentColumns]])
    
    return ({"columns": prominentColumns, "matrix": C, "scores": sorted(score.values(), reverse = True)[:rank]})

def extract(choices, traindf, featdir, curdir, t, keyset):
    for index in choices:
        print(index)
        basename = os.path.splitext(os.path.basename(traindf.loc[index, 'filename']))[0]
        filename = join(featdir, basename+'.npy')
        curname = join(curdir, basename+'.npy')

        feats = np.load(filename).T
        cols = np.load(curname)
        keys = np.transpose(np.squeeze(feats[:, cols])) #back to n x dim
        print(keys.shape)

        length = keys.shape[0]

        if length <= t:
            keyset = np.vstack([keyset, keys])
        else:
            keyset = np.vstack([keyset, keys[:t]])

    return keyset

def run(args):
    sizecode = args.featdir.split('/')[-1] # indicate the size, stride, level of our patch
    if args.cur:
        # save all CUR results for all slides
        if not os.path.exists(join(args.curdir, sizecode)):
            os.mkdir(join(args.curdir, sizecode))

        filedf = pd.read_excel(args.datadf, index_col='AI ID')
        for index in filedf.index:
            basename = os.path.splitext(os.path.basename(filedf.loc[index, 'filename']))[0]
            filename = join(args.featdir, basename+'.npy')
            feats = np.load(filename).T
            res = extractTopKColumns(feats)
            cols = res["columns"]

            np.save(join(args.curdir, sizecode, f"{basename}.npy"), cols)

    if args.extract:
        N = len(glob.glob('./splits/{}/splits_*.csv'.format(args.task)))
        for i in range(N):
            args.fold = i

            # prepare keyset file
            if not os.path.exists(join(args.savedir, args.task)):
                os.mkdir(join(args.savedir, args.task))

            split = pd.read_csv('./splits/{}/splits_{}.csv'.format(args.task, args.fold), header=0)
            filedf = pd.read_excel(args.datadf, index_col='AI ID')

            trainindex = split['train'].dropna()
            trainlabels = split['train_label'].dropna()

            negindices = trainindex[~trainlabels]
            posindices = trainindex[trainlabels]

            negchoices = negindices.sample(n=args.ns, random_state=42)
            poschoices = posindices.sample(n=args.ns, random_state=42)

            negkeyset = np.empty((0, args.dim))
            poskeyset = np.empty((0, args.dim))

            negkeyset = extract(negchoices, filedf, args.featdir, join(args.curdir, sizecode), args.t, negkeyset)
            poskeyset = extract(poschoices, filedf, args.featdir, join(args.curdir, sizecode), args.t, poskeyset)
            
            np.save(join(args.savedir, args.task, '{}-p{}-nrec-{}-f{}'.format(args.encoder, args.psize, args.t, args.fold)), negkeyset)
            np.save(join(args.savedir, args.task, '{}-p{}-rec-{}-f{}'.format(args.encoder, args.psize, args.t, args.fold)), poskeyset)

if __name__ == '__main__':
    args = parser.parse_args()

    run(args)
