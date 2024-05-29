import time
import math
import json
import argparse
import itertools
import numpy as np
from math import pi
import os, sys, glob
print(os.getcwd())
from tqdm import tqdm
import sparseconvnet as scn

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from util import *
from model_gru import *
from cluster import *
import data_val as data
import pickle

os.environ['CUDA_VISIBLE_DEVICES']='0'

parser = argparse.ArgumentParser()
parser.add_argument('--restore_epoch', type=int, default=32, metavar='N', help='Epoch of model to restore')
parser.add_argument('--exp_name', type=str, default='gru', metavar='N', help='Name of the experiment')
args= parser.parse_args()

DIR = None
def _init_():
	DIR = './validation/' + args.exp_name + '/scenes/'
	print ('Save directory: ', DIR)

	if not os.path.exists('./validation'):
		os.mkdir('./validation')
	if not os.path.exists('./validation/' + args.exp_name):
		os.mkdir('./validation/' + args.exp_name)
	if not os.path.exists(DIR):
		os.mkdir(DIR)
_init_()
DIR = './validation/' + args.exp_name + '/scenes/'
io = IOStream('validation/' + args.exp_name + '/run.log')
save_feat = {}


def batch_val(batch, model, batch_size):
    backbone_model = model['backbone']
    # ref_model = model['refer']

    sem, pc_feat, offset = backbone_model(batch['x'])
    sem = F.softmax(sem, -1)

    IOUs = []
    idx = 0
    for i, num_p in enumerate(batch['num_points']):
        name = batch['names'][i]
        io.cprint(name)

        scene_pcfeat = pc_feat[idx:idx+num_p]
        scene_sem = sem[idx:idx+num_p]
        scene_offset = offset[idx:idx+num_p]
        scene_coords = batch['x'][0][idx:idx+num_p, 0:3].float().cuda()
        scene_coords /= data.scale
        scene_coords = scene_coords - (scene_coords.max(0)[0]*0.5 + scene_coords.min(0)[0]*0.5)


        # Mask out wall and floor 
        m = scene_sem.argmax(-1) > 1
        pred_sem = scene_sem.argmax(-1)
        pred_cen = scene_coords + scene_offset

        scene_data = {}
        scene_data['lang_feat'] = batch['lang_feat'][i].float().cuda()
        scene_data['lang_len'] = batch['lang_len'][i]
 
        grps, grp_feat, grp_cen, _, _ = IterativeSample(scene_pcfeat[m], scene_coords[m], scene_offset[m], scene_sem[m])

        # Clustering for Wall & Floor, here we use mean shift clustering
        grps1 = SampleMSCluster(scene_pcfeat[~m], scene_coords[~m])
        grp_feat1, _ = gather(scene_pcfeat[~m], grps1)
        grp_cen1, _ = gather(scene_coords[~m], grps1)

        save_feat[name] = {
            'scene_pcfeat':scene_pcfeat.cpu(),
            'scene_sem':scene_sem.cpu(),
            'scene_offset':scene_offset.cpu(),
            'scene_coords':scene_coords.cpu(),
            'grps': grps.cpu(),
            'grp_feat': grp_feat.cpu(),
            'grp_cen': grp_cen.cpu(),
            'grps1': grps1.cpu(),
            'grp_feat1': grp_feat1.cpu(),
            'grp_cen1': grp_cen1.cpu()
        }



        idx += num_p
    # IOUs = np.concatenate(IOUs, 0)
    # return IOUs

use_cuda = torch.cuda.is_available()

backbone_model = SCN().cuda()
backbone_model = nn.DataParallel(backbone_model)
models = {'backbone': backbone_model}

training_epoch = checkpoint_restore(models, 'TGNN/GRU/checkpoints/'+args.exp_name+'/'+'models'+'/'+args.exp_name, io, use_cuda, args.restore_epoch)
for m in models:
    models[m].eval()

total_ious = []

for i,batch in enumerate(tqdm(data.val_data_loader)):
    if use_cuda:
        batch['x'][1]=batch['x'][1].cuda()
    with torch.no_grad():
        batch_val(batch, models, data.batch_size)

#save one stage's feat
with open('TGNN/val_feat.pkl', 'wb') as f:   
    pickle.dump(save_feat, f)

