import os
import glob
import torch
import torchvision
import numpy as np
import scanpy as sc
import pandas as pd 
import scprep as scp
import anndata as ad
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import scipy
import torchvision.transforms as transforms
from PIL import ImageFile, Image
import cv2
from utils import read_tiff, get_data
from graph_construction import calcADJ
from collections import defaultdict as dfd
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

class ViT_HER2ST(torch.utils.data.Dataset):
    """Some Information about HER2ST"""
    def __init__(self,train=True,fold=0,r=4,flatten=True,ori=False,adj=False,prune='Grid',neighs=4):
        super(ViT_HER2ST, self).__init__()
        
        self.cnt_dir = 'data/her2st/data/ST-cnts'
        self.img_dir = 'data/her2st/data/ST-imgs'
        self.pos_dir = 'data/her2st/data/ST-spotfiles'
        self.lbl_dir = 'data/her2st/data/ST-pat/lbl'
        self.r = 224//r

        # gene_list = list(np.load('data/her_hvg.npy',allow_pickle=True))
        gene_list = list(np.load('data/her_hvg_cut_1000.npy',allow_pickle=True))
        self.gene_list = gene_list
        names = os.listdir(self.cnt_dir)
        names.sort()
        names = [i[:2] for i in names]
        self.train = train
        self.ori = ori
        self.adj = adj
        # samples = ['A1','B1','C1','D1','E1','F1','G2','H1']
        samples = names[1:33]

        te_names = [samples[fold]]
        print(te_names)
        tr_names = list(set(samples)-set(te_names))

        if train:
            self.names = tr_names
        else:
            self.names = te_names

        print('Loading imgs...')
        self.img_dict = {i:torch.Tensor(np.array(self.get_img(i))) for i in self.names}
        print('Loading metadata...')
        self.meta_dict = {i:self.get_meta(i) for i in self.names}
        self.label={i:None for i in self.names}
        self.lbl2id={
            'invasive cancer':0, 'breast glands':1, 'immune infiltrate':2, 
            'cancer in situ':3, 'connective tissue':4, 'adipose tissue':5, 'undetermined':-1
        }
        if not train and self.names[0] in ['A1','B1','C1','D1','E1','F1','G2','H1','J1']:
            self.lbl_dict={i:self.get_lbl(i) for i in self.names}
            # self.label={i:m['label'].values for i,m in self.lbl_dict.items()}
            idx=self.meta_dict[self.names[0]].index
            lbl=self.lbl_dict[self.names[0]]
            lbl=lbl.loc[idx,:]['label'].values
            # lbl=torch.Tensor(list(map(lambda i:self.lbl2id[i],lbl)))
            self.label[self.names[0]]=lbl
        elif train:
            for i in self.names:
                idx=self.meta_dict[i].index
                if i in ['A1','B1','C1','D1','E1','F1','G2','H1','J1']:
                    lbl=self.get_lbl(i)
                    lbl=lbl.loc[idx,:]['label'].values
                    lbl=torch.Tensor(list(map(lambda i:self.lbl2id[i],lbl)))
                    self.label[i]=lbl
                else:
                    self.label[i]=torch.full((len(idx),),-1)
        self.gene_set = list(gene_list)
        self.exp_dict = {
            i:scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values)) 
            for i,m in self.meta_dict.items()
        }
        if self.ori:
            self.ori_dict = {i:m[self.gene_set].values for i,m in self.meta_dict.items()}
            self.counts_dict={}
            for i,m in self.ori_dict.items():
                n_counts=m.sum(1)
                sf = n_counts / np.median(n_counts)
                self.counts_dict[i]=sf
        self.center_dict = {
            i:np.floor(m[['pixel_x','pixel_y']].values).astype(int) 
            for i,m in self.meta_dict.items()
        }
        self.loc_dict = {i:m[['x','y']].values for i,m in self.meta_dict.items()}
        self.adj_dict = {
            i:calcADJ(m,neighs,pruneTag=prune)
            for i,m in self.loc_dict.items()
        }
        self.patch_dict=dfd(lambda :None)
        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(self.names))
        self.flatten=flatten
    def __getitem__(self, index):
        ID=self.id2name[index]
        im = self.img_dict[ID]
        im = im.permute(1,0,2)
        # im = torch.Tensor(np.array(self.im))
        exps = self.exp_dict[ID]
        if self.ori:
            oris = self.ori_dict[ID]
            sfs = self.counts_dict[ID]
        centers = self.center_dict[ID]
        loc = self.loc_dict[ID]
        adj = self.adj_dict[ID]
        patches = self.patch_dict[ID]
        positions = torch.LongTensor(loc)
        patch_dim = 3 * self.r * self.r * 4
        label=self.label[ID]
        exps = torch.Tensor(exps)
        if patches is None:
            n_patches = len(centers)
            if self.flatten:
                patches = torch.zeros((n_patches,patch_dim))
            else:
                patches = torch.zeros((n_patches,3,2*self.r,2*self.r))
            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im[(x-self.r):(x+self.r),(y-self.r):(y+self.r),:]
                if self.flatten:
                    patches[i] = patch.flatten()
                else:
                    patches[i]=patch.permute(2,0,1)
            self.patch_dict[ID]=patches
        data=[patches, positions, exps]
        if self.adj:
            data.append(adj)
        if self.ori:
            data+=[torch.Tensor(oris),torch.Tensor(sfs)]
        data.append(torch.Tensor(centers))
        return data
        
    def __len__(self):
        return len(self.exp_dict)

    def get_img(self,name):
        pre = self.img_dir+'/'+name[0]+'/'+name
        fig_name = os.listdir(pre)[0]
        path = pre+'/'+fig_name
        im = Image.open(path)
        return im

    def get_cnt(self,name):
        path = self.cnt_dir+'/'+name+'.tsv'
        df = pd.read_csv(path,sep='\t',index_col=0)

        return df

    def get_pos(self,name):
        path = self.pos_dir+'/'+name+'_selection.tsv'
        # path = self.pos_dir+'/'+name+'_labeled_coordinates.tsv'
        df = pd.read_csv(path,sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i])) 
        df['id'] = id

        return df

    def get_meta(self,name,gene_list=None):
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = cnt.join((pos.set_index('id')))

        return meta

    def get_lbl(self,name):
        # path = self.pos_dir+'/'+name+'_selection.tsv'
        path = self.lbl_dir+'/'+name+'_labeled_coordinates.tsv'
        df = pd.read_csv(path,sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i])) 
        df['id'] = id
        df.drop('pixel_x', inplace=True, axis=1)
        df.drop('pixel_y', inplace=True, axis=1)
        df.drop('x', inplace=True, axis=1)
        df.drop('y', inplace=True, axis=1)
        df.set_index('id',inplace=True)
        return df

class ViT_SKIN(torch.utils.data.Dataset):
    """Some Information about ViT_SKIN"""
    def __init__(self,train=True,r=4,norm=False,fold=0,flatten=True,ori=False,adj=False,prune='NA',neighs=4):
        super(ViT_SKIN, self).__init__()

        self.dir = './data/GSE144240_RAW/'
        self.r = 224//r

        patients = ['P2', 'P5', 'P9', 'P10']
        reps = ['rep1', 'rep2', 'rep3']
        names = []
        for i in patients:
            for j in reps:
                names.append(i+'_ST_'+j)
        gene_list = list(np.load('data/skin_hvg_cut_1000.npy',allow_pickle=True))

        self.ori = ori
        self.adj = adj
        self.norm = norm
        self.train = train
        self.flatten = flatten
        self.gene_list = gene_list
        samples = names
        te_names = [samples[fold]]
        tr_names = list(set(samples)-set(te_names))

        if train:
            self.names = tr_names
        else:
            self.names = te_names

        print(te_names)
        print('Loading imgs...')
        self.img_dict = {i:torch.Tensor(np.array(self.get_img(i))) for i in self.names}
        print('Loading metadata...')
        self.meta_dict = {i:self.get_meta(i) for i in self.names}

        self.gene_set = list(gene_list)
        if self.norm:
            self.exp_dict = {
                i:sc.pp.scale(scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values)))
                for i,m in self.meta_dict.items()
            }
        else:
            self.exp_dict = {
                i:scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values)) 
                for i,m in self.meta_dict.items()
            }
        if self.ori:
            self.ori_dict = {i:m[self.gene_set].values for i,m in self.meta_dict.items()}
            self.counts_dict={}
            for i,m in self.ori_dict.items():
                n_counts=m.sum(1)
                sf = n_counts / np.median(n_counts)
                self.counts_dict[i]=sf
        self.center_dict = {
            i:np.floor(m[['pixel_x','pixel_y']].values).astype(int)
            for i,m in self.meta_dict.items()
        }
        self.loc_dict = {i:m[['x','y']].values for i,m in self.meta_dict.items()}
        self.adj_dict = {
            i:calcADJ(m,neighs,pruneTag=prune)
            for i,m in self.loc_dict.items()
        }
        self.patch_dict=dfd(lambda :None)
        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(self.names))


    def filter_helper(self):
        a = np.zeros(len(self.gene_list))
        n = 0
        for i,exp in self.exp_dict.items():
            n += exp.shape[0]
            exp[exp>0] = 1
            for j in range((len(self.gene_list))):
                a[j] += np.sum(exp[:,j])


    def __getitem__(self, index):
        ID=self.id2name[index]
        im = self.img_dict[ID].permute(1,0,2)

        exps = self.exp_dict[ID]
        if self.ori:
            oris = self.ori_dict[ID]
            sfs = self.counts_dict[ID]
        adj=self.adj_dict[ID]
        centers = self.center_dict[ID]
        loc = self.loc_dict[ID]
        patches = self.patch_dict[ID]
        positions = torch.LongTensor(loc)
        patch_dim = 3 * self.r * self.r * 4
        exps = torch.Tensor(exps)
        if patches is None:
            n_patches = len(centers)
            if self.flatten:
                patches = torch.zeros((n_patches,patch_dim))
            else:
                patches = torch.zeros((n_patches,3,2*self.r,2*self.r))

            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im[(x-self.r):(x+self.r),(y-self.r):(y+self.r),:]
                if self.flatten:
                    patches[i] = patch.flatten()
                else:
                    patches[i]=patch.permute(2,0,1)
            self.patch_dict[ID]=patches
        data=[patches, positions, exps]
        if self.adj:
            data.append(adj)
        if self.ori:
            data+=[torch.Tensor(oris),torch.Tensor(sfs)]
        data.append(torch.Tensor(centers))
        return data
        
    def __len__(self):
        return len(self.exp_dict)

    def get_img(self,name):
        path = glob.glob(self.dir+'*'+name+'.jpg')[0]
        im = Image.open(path)
        return im

    def get_cnt(self,name):
        path = glob.glob(self.dir+'*'+name+'_stdata.tsv')[0]
        df = pd.read_csv(path,sep='\t',index_col=0)
        return df

    def get_pos(self,name):
        path = glob.glob(self.dir+'*spot*'+name+'.tsv')[0]
        df = pd.read_csv(path,sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i])) 
        df['id'] = id

        return df

    def get_meta(self,name,gene_list=None):
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = cnt.join(pos.set_index('id'),how='inner')

        return meta

    def get_overlap(self,meta_dict,gene_list):
        gene_set = set(gene_list)
        for i in meta_dict.values():
            gene_set = gene_set&set(i.columns)
        return list(gene_set)

class ViT_DLPFC(torch.utils.data.Dataset):
    def __init__(self, train=True, gene_list=None, ds=None, sr=False, aug=False, norm=False, fold=0):
        super(ViT_DLPFC, self).__init__()
        self.dir = './data/st_data/DLPFC12/'
        self.r = 224 // 4

        sample_names = ['151507', '151508', '151509', 
                        '151510', '151669', '151670',
                        '151671', '151672', '151673',
                        '151674', '151675', '151676']

        gene_list = list(np.load('./data/dlpfc_hvg_cut_1000.npy', allow_pickle=True))

        self.gene_list = gene_list

        self.train = train
        self.sr = sr
        self.aug = aug
        self.transforms = transforms.Compose([transforms.ColorJitter(0.5, 0.5, 0.5), transforms.ToTensor()])
        self.norm = norm

        samples = sample_names
        te_names = [samples[fold]]
        tr_names = list(set(samples) - set(te_names))

        if train:
            names = tr_names
        else:
            names = te_names

        print('Loading imgs...')
        if self.aug:
            self.img_dict = {i: self.get_img(i) for i in names}
        else:
            self.img_dict = {i: torch.Tensor(np.array(self.get_img(i))) for i in names}
        print('Loading metadata...')
        self.meta_dict = {i: self.get_meta(i) for i in names}

        self.gene_set = list(gene_list)

        if self.norm:
            self.exp_dict = {
                i: sc.pp.scale(scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values))) for
                i, m in self.meta_dict.items()}
        else:
            self.exp_dict = {i: scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values)) for
                             i, m in self.meta_dict.items()}
        self.center_dict = {i: np.floor(m[['pixel_x', 'pixel_y']].values).astype(int) for i, m in
                            self.meta_dict.items()}
        self.loc_dict = {i: m[['x', 'y']].values for i, m in self.meta_dict.items()}

        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(names))

        self.patch_dict = {}

        self.adj_dict = {i: calcADJ(coord=m, k=4, pruneTag='NA') for i, m in self.loc_dict.items()}

    def filter_helper(self):
        a = np.zeros(len(self.gene_list))
        n = 0
        for i, exp in self.exp_dict.items():
            n += exp.shape[0]
            exp[exp > 0] = 1
            for j in range((len(self.gene_list))):
                a[j] += np.sum(exp[:, j])

    def __getitem__(self, index):
        i = index
        name = self.id2name[i]
        im = self.img_dict[self.id2name[i]]
        if self.aug:
            im = self.transforms(im)
            im = im.permute(2, 1, 0)
        else:
            im = im.permute(1, 0, 2)  # im = im

        exps = self.exp_dict[self.id2name[i]]
        centers = self.center_dict[self.id2name[i]]
        loc = self.loc_dict[self.id2name[i]]
        positions = torch.LongTensor(loc)
        patch_dim = 3 * self.r * self.r * 4

        if self.id2name[i] in self.patch_dict:
            patches = self.patch_dict[self.id2name[i]]
        else:
            patches = None
        adj = self.adj_dict[name]

        if self.sr:
            centers = torch.LongTensor(centers)
            max_x = centers[:, 0].max().item()
            max_y = centers[:, 1].max().item()
            min_x = centers[:, 0].min().item()
            min_y = centers[:, 1].min().item()
            r_x = (max_x - min_x) // 30
            r_y = (max_y - min_y) // 30

            centers = torch.LongTensor([min_x, min_y]).view(1, -1)
            positions = torch.LongTensor([0, 0]).view(1, -1)
            x = min_x
            y = min_y

            while y < max_y:
                x = min_x
                while x < max_x:
                    centers = torch.cat((centers, torch.LongTensor([x, y]).view(1, -1)), dim=0)
                    positions = torch.cat((positions, torch.LongTensor([x // r_x, y // r_y]).view(1, -1)), dim=0)
                    x += 56
                y += 56

            centers = centers[1:, :]
            positions = positions[1:, :]

            n_patches = len(centers)
            patches = torch.zeros((n_patches, patch_dim))
            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im[(x - self.r):(x + self.r), (y - self.r):(y + self.r), :]
                patches[i] = patch.flatten()

            return patches, positions, centers

        else:
            n_patches = len(centers)
            exps = torch.Tensor(exps)
            if patches is None:
                patches = torch.zeros((n_patches, 3, 2 * self.r, 2 * self.r))

                for i in range(n_patches):
                    center = centers[i]
                    x, y = center
                    patch = im[(x - self.r):(x + self.r), (y - self.r):(y + self.r),
                            :]
                    patches[i] = patch.permute(2, 0, 1)
                self.patch_dict[name] = patches
            if self.train:
                return patches, positions, exps, adj
            else:
                return patches, positions, exps, torch.Tensor(centers), adj

    def __len__(self):
        return len(self.exp_dict)

    def get_img(self, name):
        # path = f'{self.dir}/{name}/{name}_full_image.tif'
        path = os.path.join(self.dir, name, 'spatial', 'tissue_hires_image.png')
        im = Image.open(path)
        return im

    def get_meta(self, name, gene_list=None):
        meta = pd.read_csv(f'{self.dir}{name}/metadata.tsv', index_col=0, sep='\t')
        return meta
    
    def get_adata(self, name):
        h5_path = os.path.join(self.dir, name)
        adata = sc.read_visium(h5_path, count_file=name+'_filtered_feature_bc_matrix.h5', load_images=True)
        adata.var_names_make_unique()
        
        gt_df = self.get_labels(name)
        adata.obs['gt_clusters'] = gt_df.loc[:, 6] # Last column
        
        # Drop na
        adata = adata[adata.obs.dropna().index].copy()
        adata['gt_clusters'] = adata.obs['gt_clusters'].astype(int).astype(str)
        return adata
    
    def get_spatial_positions(self, name):
        pos_path = os.path.join(self.dir, name, 'spatial', 'tissue_positions_list.csv')
        positions = pd.read_csv(pos_path, header=None, index_col=0)
        positions.columns = ['barcode', 'in_tissue', 'array_row', 'array_col', 'pxl_col_in_fullres', 'pxl_row_in_fullres']
        return positions

    def get_overlap(self, meta_dict, gene_list):
        gene_set = set(gene_list)
        for i in meta_dict.values():
            gene_set = gene_set & set(i.columns)
        return list(gene_set)
    
    def get_labels(self, name):
        gt_dir = os.path.join(self.dir, name, 'gt')
        gt_df = pd.read_csv(os.path.join(gt_dir, 'tissue_positions_list_GTs.txt'),
                            header=None, sep=',', index_col=0)
        return gt_df

        
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, section_id, train=False, aug=False, norm=False, gene_list=None):
        super(MyDataset, self).__init__()

        self.r = 224 // 4
        self.train = train
        self.aug = aug
        self.norm = norm
        self.gene_list = gene_list
        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.5, 0.5, 0.5), 
            transforms.ToTensor()
        ])        

        if dataset.lower() == 'dlpfc':
            self.dir = './data/st_data/DLPFC12/'
            
            sample_names = ['151507', '151508', '151509', '151510', '151669', '151670', '151671', '151672', '151673',
                        '151674', '151675', '151676']
            
            
            te_names = [section_id]
            tr_names = list(set(sample_names) - set(te_names))
            names = tr_names if train else te_names

            # Load data
            print('Loading data...')
            self.adata_dict = {i: self.get_adata(i) for i in names}
            
            print('Loading image...')
            # full_img = np.array(self.get_img(section_id))
            if self.aug:
                self.img_dict = {i: self.get_img(i) for i in names}
            else:
                self.img_dict = {i: torch.Tensor(np.array(self.get_img(i))) for i in names}

            # self.spatial = self.adata.obsm['spatial']
            self.spatial = {i: self.adata_dict[i].obsm['spatial'] for i in names}

            print('Extracting patches...')
            self.patch_dict = {}

            # self.patches = np.array([
            #     self.full_img[y - self.r:y + self.r, x - self.r:x + self.r, :]
            #     for x, y in self.spatial
            # ])
            self.labels = {i: self.adata_dict[i].obs['gt_clusters'] for i in names}
                
            if gene_list is None:
                self.gene_set = list(np.load('./data/dlpfc_hvg_cut_1000.npy', allow_pickle=True))
                        
            # expression_matrix = {i: self.adata[i][:, self.gene_list].X for i in names}
            # # Convert sparse matrix to dense if needed
            # if scipy.sparse.issparse(expression_matrix):
            #     expression_matrix = expression_matrix.toarray()
            
            # # Normalize and process
            # if self.norm:
            #     self.exp_data = sc.pp.scale(
            #         scp.transform.log(
            #             scp.normalize.library_size_normalize(
            #                 expression_matrix
            #             )
            #         )
            #     )
            # else:
            #     self.exp_data = scp.transform.log(
            #         scp.normalize.library_size_normalize(
            #             expression_matrix
            #         )
            #     )
            
            # Process expression data
            if self.norm:
                self.exp_dict = {
                    i: sc.pp.scale(
                        scp.transform.log(
                            scp.normalize.library_size_normalize(
                                adata[:, self.gene_set].X.toarray() if scipy.sparse.issparse(adata[:, self.gene_set].X) 
                                else adata[:, self.gene_set].X
                            )
                        )
                    ) for i, adata in self.adata_dict.items()
                }
            else:
                self.exp_dict = {
                    i: scp.transform.log(
                        scp.normalize.library_size_normalize(
                            adata[:, self.gene_set].X.toarray() if scipy.sparse.issparse(adata[:, self.gene_set].X) 
                            else adata[:, self.gene_set].X
                        )
                    ) for i, adata in self.adata_dict.items()
                }
                
            # Ensure exp_data is dense numpy array
            self.exp_dict = {i: np.array(exp_data) for i, exp_data in self.exp_dict.items()}

            self.center_dict = {i: np.floor(adata.obsm['spatial']).astype(int) for i, adata in self.adata_dict.items()}
            # self.loc = self.adata.obs[['array_row', 'array_col']].values
            self.loc_dict = {i: adata.obs[['array_row', 'array_col']].values for i, adata in self.adata_dict.items()}

            self.lengths = [len(adata) for adata in self.adata_dict.values()]
            self.cumlen = np.cumsum(self.lengths)
            self.n_clusters = len(np.unique(self.labels))
            # self.n_pos = self.spatial.max() + 1
            self.id2name = dict(enumerate(names))

            # Calculate adjacency matrix
            # self.adj = calcADJ(coord=self.loc, k=4, pruneTag='NA')
            self.adj_dict = {i: calcADJ(coord=loc, k=4, pruneTag='NA') for i, loc in self.loc_dict.items()}
            
            # print(self.patch_dict)
            # print(self.loc_dict)
            # print(self.adj_dict)
            # print(self.center_dict)
            # print(self.adata_dict)
            
            print('Done!')
                
                
        elif dataset == 'BRCA':
            print('Not implemented yet!')
        else:
            raise ValueError('Dataset not supported!')

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        name = self.id2name[index]
        im = self.img_dict[name]
        # patch = self.patches
        # position = torch.LongTensor(self.loc)        
        # exp = torch.Tensor(self.exp_data)

        # Apply augmentation if needed
        if self.aug:
            im = self.transforms(im)
            im = im.permute(2, 1, 0)
        else:
            im = im.permute(1, 0, 2)
            
        exps = self.exp_dict[name]   
        centers = self.center_dict[name]
        loc = self.loc_dict[name]
        positions = torch.LongTensor(loc)
        
        n_patches = len(centers)
         
        if name in self.patch_dict:
            patches = self.patch_dict[name]
        else:
            patches = torch.zeros((n_patches, 3, 2 * self.r, 2 * self.r))
            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im[(y - self.r):(y + self.r), (x - self.r):(x + self.r), :]
                patches[i] = patch.permute(2, 0, 1)
            self.patch_dict[name] = patches
            
        adj = self.adj_dict[name]
        if self.train:
            return patches, positions, exps, adj
        else:
            return patches, positions, exps, torch.Tensor(centers), adj
        
    def get_img(self, section_id):
        # path = f'{self.dir}/{section_id}/{section_id}_full_image.tif'
        # path = os.path.join(self.dir, section_id, 'spatial', 'tissue_hires_image.png')
        
        # Use new path because the current one may have damaged tif images
        path = os.path.join('data/st_data/DLPFC_new/', section_id, 'spatial', section_id+'_full_image.tif')
        # print(path)
        # im = Image.open(path)
        im = cv2.imread(path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return im

    def get_meta(self, section_id, gene_list=None):
        meta = pd.read_csv(f'{self.dir}{section_id}/metadata.tsv', index_col=0, sep='\t')
        return meta
    
    def get_adata(self, section_id):
        print(f'Loading {section_id}...')
        h5_path = os.path.join(self.dir, section_id)
        adata = sc.read_visium(h5_path, count_file=section_id+'_filtered_feature_bc_matrix.h5', load_images=True)
        adata.var_names_make_unique()
        
        gt_df = self.get_labels(section_id)
        adata.obs['gt_clusters'] = gt_df.loc[:, 6] # Last column
        
        # Drop na
        adata = adata[adata.obs.dropna().index].copy()
        adata.obs['gt_clusters'] = adata.obs['gt_clusters'].astype(int).astype(str)
        
        return adata
    
    def get_spatial_positions(self, section_id):
        pos_path = os.path.join(self.dir, section_id, 'spatial', 'tissue_positions_list.csv')
        positions = pd.read_csv(pos_path, header=None, index_col=0)
        positions.columns = ['barcode', 'in_tissue', 'array_row', 'array_col', 'pxl_col_in_fullres', 'pxl_row_in_fullres']
        return positions

    def get_overlap(self, meta_dict, gene_list):
        gene_set = set(gene_list)
        for i in meta_dict.values():
            gene_set = gene_set & set(i.columns)
        return list(gene_set)
    
    def get_labels(self, section_id):
        gt_dir = os.path.join(self.dir, section_id, 'gt')
        gt_df = pd.read_csv(os.path.join(gt_dir, 'tissue_positions_list_GTs.txt'),
                            header=None, sep=',', index_col=0)
        return gt_df

         