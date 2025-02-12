import scanpy as sc
import anndata
import pandas as pd
import os
import numpy as np
from PIL import Image
import cv2
from matplotlib import pyplot as plt
import torch
from HIST2ST import Hist2ST
import random
from predict import pk_load, test
from torch.utils.data import DataLoader

batch_id = '151673'


# genes = adata.shape[1] # number of genes
tag='5-7-2-8-4-16-32'
k,p,d1,d2,d3,h,c=map(lambda x:int(x),tag.split('-'))
dropout=0.2
random.seed(12000)
np.random.seed(12000)
torch.manual_seed(12000)
torch.cuda.manual_seed(12000)
torch.cuda.manual_seed_all(12000)  
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

model=Hist2ST(
    depth1=d1, depth2=d2, depth3=d3,
    n_genes=785, 
    kernel_size=k, patch_size=p,
    fig_size=112,
    learning_rate=1e-5,
    heads=h, channel=c, dropout=dropout,
    zinb=0.25, nb=False,
    bake=5, lamb=0.5, 
)

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(f'./model/5-Hist2ST.ckpt', map_location=device))    

testset = pk_load(batch_id,'test',dataset='dlpfc',flatten=False,adj=True,ori=True)
test_loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)