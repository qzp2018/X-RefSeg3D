import argparse
import numpy as np
import os, sys, glob
import os 
from tqdm import tqdm
import sparseconvnet as scn
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter  

import data
import data_val as data_valid
from util import *
from model_gru import *
import pickle

from torch.nn.functional import pdist
# 设置随机种子
seed = 3407
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
parser.add_argument('--epochs', type=int, default=48, help='Number of epochs')
parser.add_argument('--exp_name', type=str, default='model_gru', metavar='N', help='Name of the experiment')
args= parser.parse_args()

def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')

_init_()
io = IOStream('checkpoints/' + args.exp_name + '/run.log')


def compute_relative_coord(coord):
    #Calculate the spatial coordinate relationship 
    delta = coord[:, None, :] - coord[None, :, :]  # shape: (N, N, 3)
    distances = torch.norm(delta, dim=-1)  
    max_distance = torch.max(distances) 
    N = coord.shape[0]
    relation = torch.zeros(N, N)  # shape: (N, N)
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
    # compute relation
    delta = coord[:, None, :] - coord[None, :, :]  
    dis = (torch.norm(delta, dim=-1)).unsqueeze(-1) 
    coord_expand=coord.unsqueeze(1).expand(coord.shape[0],coord.shape[0],coord.shape[-1])
    coord_expand1=coord.unsqueeze(0).expand(coord.shape[0],coord.shape[0],coord.shape[-1])

    rel=torch.cat([coord_expand,coord_expand1,delta,dis],dim=-1)
 

    return rel





with open('val_feat.pkl', 'rb') as f:
    feat_data = pickle.load(f)

def batch_val(batch, model, batch_size):
    backbone_model = model['backbone']
    ref_model = model['refer']
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
        m = scene_sem.argmax(-1) > 1
        pred_sem = scene_sem.argmax(-1)
        pred_cen = scene_coords + scene_offset
        scene_data = {}
        scene_data['lang_feat'] = batch['lang_feat'][i].float().cuda()
        scene_data['lang_len'] = batch['lang_len'][i]
        ins_lbl = batch['y_ins'][idx:idx+num_p].cuda()#y instance labels       
        ins_mask = torch.zeros(num_p, grps.shape[-1]).to(grps.device).long()
        ins_mask[m] = grps
        obj_feat = torch.cat([grp_feat, grp_feat1], 0)
        obj_coord = torch.cat([grp_cen, grp_cen1], 0)
        obj_num, dim = obj_feat.shape
        scene_data['obj_feat'] = obj_feat
        scene_data['obj_coord'] = obj_coord
        lrel=compute_relative_coord(obj_coord)
        scene_data['lrel']=lrel
        relative_rel=compute_relative_rel(obj_coord)
        scene_data['relative_rel']=relative_rel
        possible_obj_num = grp_feat.shape[0]
        total_score,ttl_score_mlp = ref_model(scene_data)
        total_score=total_score+ttl_score_mlp
        total_score = total_score[:, 0:possible_obj_num]
        total_score = F.softmax(total_score, -1)#score
        pred = ins_mask[:, total_score.argmax(-1)]#indice
        gt = batch['ref_lbl'][i].cuda()
        iou = (pred*gt).sum(0).float()/((pred|gt).sum(0).float()+1e-5)#compute iou
        IOUs.append(iou.cpu().numpy())
        pc = [scene_coords, batch['x'][1][idx:idx+num_p], pred_cen, ins_lbl.unsqueeze(-1).float()]
        pc = torch.cat(pc, -1)
        sen = batch['sentences'][i]
        sen = np.array(sen)
        idx += num_p
    IOUs = np.concatenate(IOUs, 0)
    return IOUs




with open('train_feat.pkl', 'rb') as f:
    train_data = pickle.load(f)
def batch_train(batch, model, optimizer, batch_size):
    ref_model = model['refer']
    backbone_model = model['backbone']
    
    with torch.no_grad():#backbone_model 
        pc_sem, pc_feat, pc_offset = backbone_model(batch['x'])#pc_feat [N,32]

    idx = 0
    loss = 0
    train_loss = 0
    total_pred = 0
    total_ttl_correct = 0
    loss4print = {'ttl':0}
    for i, num_p in enumerate(batch['num_points']):
        scene_pcfeat = pc_feat[idx:idx+num_p]
        pc_ins = batch['y_ins'][idx:idx+num_p]
        pc_sem = batch['y'][idx:idx+num_p]
        m = pc_ins > -1 # Remove unlabeled points 
        # Get instance mean features & coordinate using groundtruth labels
        obj_feat, obj_id = gather(scene_pcfeat[m], pc_ins[m])  
        coord = batch['x'][0][idx:idx+num_p, 0:3].float().cuda()
        coord/=data.scale#data.scale=50
        #shift
        coord = coord - (coord.max(0)[0]*0.5+coord.min(0)[0]*0.5)
        obj_coord, _ = gather(coord[m], pc_ins[m])#
        # Referring---------------------------------------------
        lang_len = batch['lang_len'][i]
        lang_feat = batch['lang_feat'][i].float().cuda()#[lang_len,80,300]

        if lang_feat.shape[0] < 256:
            rand_idx = np.arange(lang_feat.shape[0])
        else:
            rand_idx = np.random.choice(lang_feat.shape[0], 256, replace=False)
        
        lrel=compute_relative_coord(obj_coord)
        
        scene_data = {}

        relative_rel=compute_relative_rel(obj_coord)
        scene_data['relative_rel']=relative_rel        
        scene_data['obj_feat'] = obj_feat
        scene_data['obj_coord'] = obj_coord
        scene_data['lang_len'] = lang_len[rand_idx]
        scene_data['lang_feat'] = lang_feat[rand_idx]
        scene_data['lrel']=lrel
        
        
        ttl_score, ttl_score_mlp= ref_model(scene_data)
        obj_gt = batch['lang_objID'][i][rand_idx].unsqueeze(-1).cuda() == obj_id.cuda()
        obj_gt = obj_gt.float().argmax(-1)

        total_pred += ttl_score.shape[0]
        ttl_score_sum=ttl_score+ ttl_score_mlp
        total_ttl_correct += (ttl_score_sum.argmax(-1) == obj_gt).sum()

        if torch.isnan(ttl_score).any():
            print (ttl_score)
        ref_ttl_loss = F.cross_entropy(ttl_score, obj_gt)
        ref_ttl_loss_mlp = F.cross_entropy(ttl_score_mlp, obj_gt)
        loss += (ref_ttl_loss+ref_ttl_loss_mlp)

        loss4print['ttl'] += ref_ttl_loss.item()
        idx += num_p
    train_loss += loss/batch_size 
    train_loss.backward()
    optimizer.step()
    for t in loss4print:
        loss4print[t]/=batch_size
    return loss4print['ttl'], total_pred, total_ttl_correct







use_cuda = torch.cuda.is_available()
io.cprint(args.exp_name)
# Initialize backbone Sparse 3D-Unet and Text-Guided GNN
backbone_model = SCN().cuda()
backbone_model = nn.DataParallel(backbone_model)
ref_model = RefNetGRU(k=16).cuda()
ref_model.relconv = nn.DataParallel(ref_model.relconv)

# Load pretrained instance segmentation model for backbone
models = {}
models['backbone'] = backbone_model
training_epoch = checkpoint_restore(models, '/GRU/checkpoints/model_insseg', io, use_cuda)
models['refer'] = ref_model
training_epoch = 1
training_epochs = args.epochs
io.cprint('Starting with epoch: ' + str(training_epoch))
params = ref_model.parameters()
optimizer = optim.Adam(params, lr=args.lr)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer,
                                                        T_max =args.epochs)

best_IOU=0

for epoch in range(training_epoch, training_epochs+1):
    for m in models:
        models[m].train()
    total_loss = {}
    pbar = tqdm(data.train_data_loader)

    total = 0
    ttl_correct = 0
    obj_correct = 0
    for i, batch in enumerate(pbar):
        optimizer.zero_grad()
        if use_cuda:
            batch['x'][1] = batch['x'][1].cuda()#[N,3]
        with torch.autograd.set_detect_anomaly(True):#
            loss, t, tc = batch_train(batch, models, optimizer, data.batch_size)
        total += t
        ttl_correct += tc
    scheduler.step()


    writer=SummaryWriter('/GRU/tensorboard_output/'+args.exp_name)   # TODO root
    writer.add_scalar('loss',loss,epoch)
    writer.add_scalar('Correct Objects',float(ttl_correct)/float(total),epoch)
    exp_name='/GRU/checkpoints/'+args.exp_name+'/'+'models'+'/'+args.exp_name
    f='/GRU/checkpoints/'+args.exp_name+'/'+'models'+'/'+'best_model.pth'
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
        #print ('({}/{}) Mean IOU so far {:.4f}'.format((i+1)*data.batch_size, len(data.loader_list), np.concatenate(total_ious, 0).mean()))
    total_ious = np.concatenate(total_ious, 0)
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

