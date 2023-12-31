"""

Modified from: https://github.com/facebookresearch/SparseConvNet/blob/master/examples/ScanNet/data.py,
               https://github.com/daveredrum/ScanRefer/blob/master/lib/dataset.py

Dataloader for validation (BERT)

"""

import torch, numpy as np, glob, math, torch.utils.data, scipy.ndimage, multiprocessing as mp, time, json, pickle, logging as logging_0

from config_bert_val import *
from transformers import *

GLOVE_PICKLE = '../glove.p'

logging_0.basicConfig(level=logging_0.ERROR)
# 设置随机种子
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

val_3d = {}
def load_data(name):
    idx = name[0].find('scene')
    scene_name = name[0][idx:idx+12]
    data = torch.load(name[0])
    return data, scene_name
for x in torch.utils.data.DataLoader(
        glob.glob('/val/*.pth'),
        collate_fn=load_data, num_workers=mp.cpu_count()):
    val_3d[x[1]] = x[0]
print('Validating examples:', len(val_3d))
loader_list = list(val_3d.keys())

# Load the ScanRefer dataset and BERT tokenizer
scanrefer = json.load(open('/ScanRefer/ScanRefer_filtered_val.json'))
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

lang = {}
for i, data in enumerate(scanrefer):
    scene_id = data['scene_id']
    object_id = data['object_id']
    ann_id = data['ann_id']

    if scene_id not in lang:
        lang[scene_id] = {'idx':[]}
    if object_id not in lang[scene_id]:
        lang[scene_id][object_id] = {}
    
    lang[scene_id]['idx'].append(i)
    
blur0=np.ones((3,1,1)).astype('float32')/3
blur1=np.ones((1,3,1)).astype('float32')/3
blur2=np.ones((1,1,3)).astype('float32')/3
def elastic(x,gran,mag):
    bb=np.abs(x).max(0).astype(np.int32)//gran+3
    noise=[np.random.randn(bb[0],bb[1],bb[2]).astype('float32') for _ in range(3)]
    noise=[scipy.ndimage.filters.convolve(n,blur0,mode='constant',cval=0) for n in noise]
    noise=[scipy.ndimage.filters.convolve(n,blur1,mode='constant',cval=0) for n in noise]
    noise=[scipy.ndimage.filters.convolve(n,blur2,mode='constant',cval=0) for n in noise]
    noise=[scipy.ndimage.filters.convolve(n,blur0,mode='constant',cval=0) for n in noise]
    noise=[scipy.ndimage.filters.convolve(n,blur1,mode='constant',cval=0) for n in noise]
    noise=[scipy.ndimage.filters.convolve(n,blur2,mode='constant',cval=0) for n in noise]
    ax=[np.linspace(-(b-1)*gran,(b-1)*gran,b) for b in bb]
    interp=[scipy.interpolate.RegularGridInterpolator(ax,n,bounds_error=0,fill_value=0) for n in noise]
    def g(x_):
        return np.hstack([i(x_)[:,None] for i in interp])
    return x+g(x)*mag

def trainMerge(tbl):
    locs=[]
    feats=[]
    labels=[]
    ins_labels=[]
    ref_labels=[]
    coords=[]
    num_points=[]
    point_ids=[]    
    scene_names=[]
    batch_ins_names=[]
    batch_lang_objID=[]
    batch_lang_objname=[]
    batch_sentences=[]
    batch_tokens=[]
    batch_input_ids=[]
    batch_attention_mask=[]
    batch_ref_lbl=[]
    for idx,scene_id in enumerate(tbl):
        refer_idxs = lang[scene_id]['idx']
        lang_feat=[]
        lang_len=[]
        lang_objID=[]
        lang_objname=[]
        sentences=[]
        tokens=[]
        input_ids=[]
        attention_mask=[]
        for i in refer_idxs:
            scene_id = scanrefer[i]['scene_id']  
            object_id = scanrefer[i]['object_id']
            ann_id = scanrefer[i]['ann_id']
            object_name = scanrefer[i]['object_name']
         
            lang_objID.append(int(object_id))
            lang_objname.append(object_name)
            sentences.append(scanrefer[i]['description'])
            tokens.append(scanrefer[i]['token'])
            token_dict = tokenizer.encode_plus(scanrefer[i]['description'], add_special_tokens=True, max_length=80, pad_to_max_length=True, return_attention_mask=True,return_tensors='pt',)
            input_ids.append(token_dict['input_ids'])
            attention_mask.append(token_dict['attention_mask'])

        # Obj_num, 
        lang_objID=torch.LongTensor(lang_objID)
        batch_lang_objID.append(lang_objID)
        batch_lang_objname.append(np.array(lang_objname))
        batch_sentences.append(sentences)
        batch_tokens.append(tokens)

        input_ids = torch.cat(input_ids, 0)
        attention_mask = torch.cat(attention_mask, 0)
        batch_input_ids.append(input_ids)
        batch_attention_mask.append(attention_mask)
        
        a,b,c,d=val_3d[scene_id]
        coord = a
        m=np.eye(3)
        m*=scale
        a=np.matmul(a,m)
        if elastic_deformation:
            a=elastic(a,6*scale//50,40*scale/50)
            a=elastic(a,20*scale//50,160*scale/50)
        m=a.min(0)
        M=a.max(0)
        q=M-m
        offset=-m+np.clip(full_scale-M+m-0.001,0,None)*np.random.rand(3)+np.clip(full_scale-M+m+0.001,None,0)*np.random.rand(3)
        a+=offset
        idxs=(a.min(1)>=0)*(a.max(1)<full_scale)
        a=a[idxs]
        b=b[idxs]
        c=c[idxs]
        d=d[idxs]
        coord=coord[idxs]

        a=torch.from_numpy(a).long()
        locs.append(torch.cat([a,torch.LongTensor(a.shape[0],1).fill_(idx)],1))
        feats.append(torch.from_numpy(b)+torch.randn(3)*0.1)
        labels.append(torch.from_numpy(c))
        ins_labels.append(torch.from_numpy(d.astype(int)))
        coords.append(torch.from_numpy(coord))
        num_points.append(a.shape[0])
        scene_names.append(scene_id)

        # Label
        # Num_points, Obj_num
        ref_lbl = (ins_labels[-1].unsqueeze(-1)) == lang_objID
        batch_ref_lbl.append(ref_lbl.long())
    locs=torch.cat(locs,0)
    feats=torch.cat(feats,0)
    labels=torch.cat(labels,0)
    ins_labels=torch.cat(ins_labels,0)
    coords = torch.cat(coords,0)
    batch_data = {'x': [locs,feats], 
                  'y': labels.long(),
                  'id': tbl,
                  'y_ins': ins_labels.long(),
                  'coords': coords,
                  'num_points': num_points,
                  'names': scene_names,
                  'lang_objID': batch_lang_objID,
                  'lang_objname': batch_lang_objname,
                  'input_ids': batch_input_ids,
                  'attention_mask': batch_attention_mask,
                  'sentences': batch_sentences,
                  'tokens': batch_tokens,
                  'ref_lbl': batch_ref_lbl} 
    return batch_data

print (len(loader_list))

val_data_loader = torch.utils.data.DataLoader(
    loader_list, 
    batch_size=batch_size,
    collate_fn=trainMerge,
    num_workers=0, 
    shuffle=True,
    worker_init_fn=lambda x: np.random.seed(x+int(time.time()))
)
