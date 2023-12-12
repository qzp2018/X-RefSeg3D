"""

Modified from: https://github.com/facebookresearch/SparseConvNet/blob/master/examples/ScanNet/unet.py

Training for referring model with bert encoder.

"""

import math
import argparse
import itertools
import numpy as np
import os, sys, glob
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from tqdm import tqdm
import sparseconvnet as scn



import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter  
import data_bert
import data_bert_val as data_valid
import pickle
from util import *
from model_bert import *
import random


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
parser.add_argument('--epochs', type=int, default=64, help='Number of epochs')
parser.add_argument('--exp_name', type=str, default='unet_bert_edgefusion_resedge_PFA_seed1234', metavar='N', help='Name of the experiment')
args= parser.parse_args()

def _init_():
    if not os.path.exists('/BERT/checkpoints'):
        os.makedirs('/BERT/checkpoints')
    if not os.path.exists('/BERT/checkpoints/'+args.exp_name):
        os.makedirs('/BERT/checkpoints/'+args.exp_name)
    if not os.path.exists('/BERT/checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('/BERT/checkpoints/'+args.exp_name+'/'+'models')
        

_init_()
io = IOStream('/BERT/checkpoints/' + args.exp_name + '/run.log')
def compute_relative_coord(coord):
    delta = coord[:, None, :] - coord[None, :, :]  
    distances = torch.norm(delta, dim=-1) 
    max_distance = torch.max(distances)
    N = coord.shape[0]
    relation = torch.zeros(N, N)  
    relation[(delta[..., 2] >= 0) & (delta[..., 1] >= 0) & (delta[..., 0] >= 0)] = 1
    relation[(delta[..., 2] >= 0) & (delta[..., 1] >= 0) & (delta[..., 0] <= 0)] = 2
    relation[(delta[..., 2] >= 0) & (delta[..., 1] <= 0) & (delta[..., 0] >= 0)] = 3
    relation[(delta[..., 2] >= 0) & (delta[..., 1] <= 0) & (delta[..., 0] <= 0)] = 4
    relation[(delta[..., 2] <= 0) & (delta[..., 1] >= 0) & (delta[..., 0] >= 0)] = 5
    relation[(delta[..., 2] <= 0) & (delta[..., 1] >= 0) & (delta[..., 0] <= 0)] = 6
    relation[(delta[..., 2] <= 0) & (delta[..., 1] <= 0) & (delta[..., 0] >= 0)] = 7
    relation[(delta[..., 2] <= 0) & (delta[..., 1] <= 0) & (delta[..., 0] <= 0)] = 8
    relation[(delta[..., 2] == 0) & (delta[..., 1] == 0) & (delta[..., 0] == 0)] = 0
    relation[distances / max_distance >= 0.25] = 0
    return relation

def compute_relative_rel(coord):
    delta = coord[:, None, :] - coord[None, :, :]  
    dis = (torch.norm(delta, dim=-1)).unsqueeze(-1) 
    coord_expand=coord.unsqueeze(1).expand(coord.shape[0],coord.shape[0],coord.shape[-1])
    coord_expand1=coord.unsqueeze(0).expand(coord.shape[0],coord.shape[0],coord.shape[-1])
    rel=torch.cat([coord_expand,coord_expand1,delta,dis],dim=-1)

    return rel



with open('/val_feat.pkl', 'rb') as f:
    feat_data = pickle.load(f)

def batch_val(batch, model, batch_size):
    backbone_model = model['backbone']
    ref_model = model['refer']
    bert_model = model['bert']

    IOUs = []
    idx = 0
    for i, num_p in enumerate(batch['num_points']):
        name = batch['names'][i]
        # io.cprint(name)
        scene_pcfeat = feat_data[name]['scene_pcfeat'].cuda()
        scene_sem = feat_data[name]['scene_sem'].cuda()
        scene_offset = feat_data[name]['scene_offset'].cuda()
        scene_coords = feat_data[name]['scene_coords'].cuda()
        grps = feat_data[name]['grps'].cuda()
        _,pc_ins  = torch.max(grps, dim=1)
        grp_feat = feat_data[name]['grp_feat'].cuda()
        grp_cen = feat_data[name]['grp_cen'].cuda()
        grps1 = feat_data[name]['grps1'].cuda()
        grp_feat1 = feat_data[name]['grp_feat1'].cuda()
        grp_cen1 = feat_data[name]['grp_cen1'].cuda()


        m = scene_sem.argmax(-1) > 1#
        pred_sem = scene_sem.argmax(-1)#
        pred_cen = scene_coords + scene_offset#

        scene_data = {}

        ins_lbl = batch['y_ins'][idx:idx+num_p].cuda()#y instance labels
        
        ins_mask = torch.zeros(num_p, grps.shape[-1]).to(grps.device).long()# [N,C]
        ins_mask[m] = grps
        obj_feat = torch.cat([grp_feat, grp_feat1], 0)
        obj_coord = torch.cat([grp_cen, grp_cen1], 0)

        obj_num, dim = obj_feat.shape
        scene_data['obj_feat'] = obj_feat
        scene_data['obj_coord'] = obj_coord
        input_ids = batch['input_ids'][i].cuda()
        attention_mask = batch['attention_mask'][i].cuda()
        lang_feat = bert_model(input_ids, attention_mask=attention_mask)[0]
        scene_data['lang_feat'] = lang_feat
        scene_data['lang_mask'] = attention_mask
        lrel=compute_relative_coord(obj_coord)
        scene_data['lrel']=lrel
        relative_rel=compute_relative_rel(obj_coord)
        scene_data['relative_rel']=relative_rel
        possible_obj_num = grp_feat.shape[0]
        total_score,ttl_score_mlp = ref_model(scene_data)
        total_score=total_score+ttl_score_mlp
        total_score = total_score[:, 0:possible_obj_num]
        total_score = F.softmax(total_score, -1)
        scores = [total_score.cpu().numpy()]

        pred = ins_mask[:, total_score.argmax(-1)]
        gt = batch['ref_lbl'][i].cuda()
        iou = (pred*gt).sum(0).float()/((pred|gt).sum(0).float()+1e-5)
        IOUs.append(iou.cpu().numpy())

        pc = [scene_coords, batch['x'][1][idx:idx+num_p], pred_cen, ins_lbl.unsqueeze(-1).float()]
        
        pc = torch.cat(pc, -1)
        sen = batch['sentences'][i]
        sen = np.array(sen)
        token = batch['tokens'][i]
        ref_objname = batch['lang_objname'][i]
        # np.save(DIR+name+'_sen', sen)
        # np.save(DIR+name+'_pc', pc.cpu().numpy())
        # np.save(DIR+name+'_ref', pred.cpu().numpy())
        # np.save(DIR+name+'_gt', gt.cpu().numpy())
        # np.save(DIR+name+'_score', np.array(scores))
        idx += num_p
    IOUs = np.concatenate(IOUs, 0)
    return IOUs





def batch_train(batch, model, optimizer, optimizer1, batch_size):
    ref_model = model['refer']
    backbone_model = model['backbone']
    bert_model = model['bert']
    
    with torch.no_grad():
        pc_sem, pc_feat, pc_offset = backbone_model(batch['x'])

    idx = 0
    loss = 0
    train_loss = 0
    total_pred = 0
    total_ttl_correct = 0
    loss4print = {'ttl':0}
    for i, num_p in enumerate(batch['num_points']):
        optimizer.zero_grad()
        optimizer1.zero_grad()
        scene_pcfeat = pc_feat[idx:idx+num_p]
        pc_ins = batch['y_ins'][idx:idx+num_p]
        pc_sem = batch['y'][idx:idx+num_p]
        m = pc_ins > -1 # Remove unlabeled points

        # Get instance mean features & coordinate using groundtruth labels
        obj_feat, obj_id = gather(scene_pcfeat[m], pc_ins[m])  
        
        coord = batch['x'][0][idx:idx+num_p, 0:3].float().cuda()
        coord/=data_bert.scale
        #shift
        coord = coord - (coord.max(0)[0]*0.5+coord.min(0)[0]*0.5)
        obj_coord, _ = gather(coord[m], pc_ins[m])

        # Referring---------------------------------------------
        input_ids = batch['input_ids'][i].cuda()
        attention_mask = batch['attention_mask'][i].cuda()
        lrel=compute_relative_coord(obj_coord)
        scene_data = {}
        relative_rel=compute_relative_rel(obj_coord)
        scene_data['relative_rel']=relative_rel     
        scene_data['obj_feat'] = obj_feat
        scene_data['obj_coord'] = obj_coord
        obj_gt = batch['lang_objID'][i].unsqueeze(-1).cuda() == obj_id.cuda()
        obj_gt = obj_gt.float().argmax(-1)
        lang_feat = bert_model(input_ids, attention_mask=attention_mask)[0]
        scene_data['lang_feat'] = lang_feat
        scene_data['lang_mask'] = attention_mask
        scene_data['lrel']=lrel
            
        ttl_score, ttl_score_mlp = ref_model(scene_data)

        obj_num = obj_coord.shape[0]
        ref_ttl_loss = F.cross_entropy(ttl_score, obj_gt)
        ref_ttl_loss_mlp = F.cross_entropy(ttl_score_mlp, obj_gt)

        ttl_score_sum=ttl_score+ttl_score_mlp                
        total_pred += ttl_score.shape[0]
        total_ttl_correct += (ttl_score_sum.argmax(-1) == obj_gt).sum()
        loss =  (ref_ttl_loss+ref_ttl_loss_mlp)
        loss.backward()
        optimizer.step()
        optimizer1.step()
        loss4print['ttl'] += ref_ttl_loss.item()
        idx += num_p
    loss4print['ttl']/=batch_size 
    return loss4print['ttl'], total_pred, total_ttl_correct

use_cuda = torch.cuda.is_available()
io.cprint(args.exp_name)

# Initialize backbone Sparse 3D-Unet and Text-Guided GNN with Pretrained BERT
backbone_model = SCN().cuda()
backbone_model = nn.DataParallel(backbone_model)
ref_model = RefNetV2(k=10).cuda()
ref_model.relconv = nn.DataParallel(ref_model.relconv)
bert_model = BertModel.from_pretrained('bert-base-uncased').cuda()
bert_model = nn.DataParallel(bert_model)

# Load pretrained instance segmentation model for backbone
models = {}
models['backbone'] = backbone_model
training_epoch = checkpoint_restore(models, '/BERT/checkpoints/model_insseg', io, use_cuda)
models['refer'] = ref_model
models['bert'] = bert_model

training_epoch = 1
training_epochs = args.epochs
io.cprint('Starting with epoch: ' + str(training_epoch))

params = list(bert_model.parameters()) + list(ref_model.parameters())
optimizer = optim.Adam([{'params': ref_model.parameters(), 'initial_lr': args.lr}], lr=args.lr)
optimizer1 = optim.Adam([{'params': bert_model.parameters(), 'initial_lr': 1e-5}], lr=1e-5)
#scheduler = optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.5, last_epoch=training_epoch)
scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer,
                                                        T_max =64)
#scheduler1 = optim.lr_scheduler.StepLR(optimizer1, 10, gamma=0.5, last_epoch=training_epoch)
scheduler1 =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer1,
                                                        T_max =64)
set_random_seed(1234)


best_IOU=0
for epoch in range(training_epoch, training_epochs+1):
    for m in models:
        models[m].train()
    total_loss = {}
    pbar = tqdm(data_bert.train_data_loader)
    total = 0
    ttl_correct = 0
    obj_correct = 0
    for i,batch in enumerate(pbar):
        optimizer.zero_grad()
        optimizer1.zero_grad()
        if use_cuda:
            batch['x'][1]=batch['x'][1].cuda()

        with torch.autograd.set_detect_anomaly(True):
            loss, t, tc = batch_train(batch, models, optimizer, optimizer1, data_bert.batch_size)
        total += t
        ttl_correct += tc
    scheduler.step()
    scheduler1.step()
    writer=SummaryWriter('/BERT/tensorboard_output/'+args.exp_name)   # TODO root
    writer.add_scalar('loss',loss,epoch)
    writer.add_scalar('Correct Objects',float(ttl_correct)/float(total),epoch)
    exp_name='/BERT/checkpoints/'+args.exp_name+'/'+'models'+'/'+args.exp_name
    f='/BERT/checkpoints/'+args.exp_name+'/'+'models'+'/'+'best_model.pth'
    save = {}
    use_cuda=True
    total_ious = []
    for i,batch in enumerate(tqdm(data_valid.val_data_loader)):
        for m in models:
            models[m].eval()

        if use_cuda:
            batch['x'][1]=batch['x'][1].cuda()
        with torch.no_grad():
            IOUs = batch_val(batch, models, data_valid.batch_size)
            total_ious.append(IOUs)
    IOU = total_ious.mean()
    Precision_5= (total_ious > 0.5).sum().astype(float)/total_ious.shape[0]
    Precision_25 = (total_ious > 0.25).sum().astype(float)/total_ious.shape[0]
    outstr =( 'Epoch: {}, Loss: {:.4f}, '.format(epoch, loss) + 'Correct Objects: {:.4f}, '.format(float(ttl_correct)/float(total)) 
             + 'Mean IOU: {:.4f}, '.format(IOU)  + 'P@0.5: {:.4f}, '.format(Precision_5) +'P@0.25: {:.4f}'.format(Precision_25))
    io.cprint(outstr)
    writer.add_scalar('Mean IOU',IOU,epoch)
    if IOU>best_IOU:
        best_IOU=IOU

   #checkpoint_save
    if epoch>=20 and best_IOU==IOU:
        for k in models.keys():
            model = models[k].cpu()
            save[k] = model.state_dict()
            if use_cuda:
                model.cuda()	
            torch.save(save,f)
