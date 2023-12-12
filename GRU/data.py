"""

Modified from: https://github.com/facebookresearch/SparseConvNet/blob/master/examples/ScanNet/data.py,
               https://github.com/daveredrum/ScanRefer/blob/master/lib/dataset.py

Dataloader for training

"""

import sparseconvnet as scn

import torch, numpy as np, glob, math, torch.utils.data, scipy.ndimage, multiprocessing as mp, time, json, pickle, random

from config import *

GLOVE_PICKLE = '.../glove.p'

# 设置随机种子
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


# Mapping from object names to id
remap = {}
f = open('.../labelids.txt', 'r')
NYU_CLASS_IDS = f.readlines()[2:]
for i, line in enumerate(NYU_CLASS_IDS):
    obj_name = line.strip().split('\t')[-1]
    remap[obj_name] = i

# Load the preprocessed data
train_3d = {}
def load_data(name):
    idx = name[0].find('scene')
    scene_name = name[0][idx:idx+12]
    return torch.load(name[0]), scene_name
for x in torch.utils.data.DataLoader(
        glob.glob('/train/*.pth'),#
        collate_fn=load_data, num_workers=mp.cpu_count()):
    train_3d[x[1]] = x[0]
print('Training examples:', len(train_3d))
loader_list = list(train_3d.keys())

# Load Glove Embeddings
with open(GLOVE_PICKLE, 'rb') as f:
    glove = pickle.load(f)

# Load the ScanRefer dataset
scanrefer = json.load(open('ScanRefer/ScanRefer_filtered_train.json'))

lang = {}
for i, data in enumerate(scanrefer):#
    
    scene_id = data['scene_id']
    scans_objects=json.load(open('/scannet_data/scans/'+scene_id+'/'+scene_id+'.aggregation.json'))
    segGroups=scans_objects['segGroups']


    object_id = data['object_id']
    ann_id = data['ann_id']

    if scene_id not in lang:
        lang[scene_id] = {'idx':[]}
    if object_id not in lang[scene_id]:
        lang[scene_id][object_id] = {}

    lang[scene_id]['object names']={}
    for j in range(len(segGroups)):
        tmp_id=segGroups[j]['id']
        tmp_label=segGroups[j]['label']
        lang[scene_id]['object names'][tmp_id]=[tmp_label]
        


    
    tokens = data['token']
    embeddings = np.zeros((MAX_DES_LEN, 300))
    for token_id in range(MAX_DES_LEN):
        if token_id < len(tokens):
            token = tokens[token_id]
            if token in glove:
                embeddings[token_id] = glove[token]
            else:
                embeddings[token_id] = glove['unk']
        lang[scene_id][object_id][ann_id] = [embeddings, len(tokens)]
    
    lang[scene_id]['idx'].append(i)

#Elastic distortion
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
    num_points=[]
    scene_names=[]
    batch_ins_names=[]
    batch_lang_feat=[]
    batch_lang_len=[]
    batch_lang_objID=[]
    batch_lang_objname=[]
    batch_lang_cls=[]
    batch_sentences=[]  
    batch_tokens=[]

    batch_ref_lbl=[]


    batch_objects_names=[]
    for idx,scene_id in enumerate(tbl):
        scene_dict = lang[scene_id]
        refer_idxs = lang[scene_id]['idx']
        lang_feat=[]
        lang_len=[]
        lang_objID=[]
        lang_objname=[]
        lang_cls=[]
        sentences=[]
        tokens=[]
        objects_names=[lang[scene_id]['object names']]
        for i in refer_idxs:
            scene_id = scanrefer[i]['scene_id']  
            object_id = scanrefer[i]['object_id']
            ann_id = scanrefer[i]['ann_id']
            object_name = scanrefer[i]['object_name']
            object_name = ' '.join(object_name.split('_'))
            
            lang_feat.append(torch.from_numpy(lang[scene_id][object_id][ann_id][0])) 
            lang_len.append(min(MAX_DES_LEN, lang[scene_id][object_id][ann_id][1]))
            lang_objID.append(int(object_id))
            lang_objname.append(object_name)
            sentences.append(scanrefer[i]['description'])
            tokens.append(scanrefer[i]['token'])
            if object_name not in remap:
                lang_cls.append(17)
            else:
                lang_cls.append(remap[object_name])



        # Obj_num, 30, 300
        lang_feat=torch.stack(lang_feat, 0)
        # Obj_num, 
        lang_len = torch.LongTensor(lang_len)
        # Obj_num, 
        lang_objID = torch.LongTensor(lang_objID)
        lang_cls=torch.LongTensor(lang_cls)

        batch_lang_feat.append(lang_feat)
        batch_lang_len.append(lang_len)
        batch_lang_objID.append(lang_objID)
        batch_lang_objname.append(np.array(lang_objname))
        batch_lang_cls.append(lang_cls)
        batch_sentences.append(sentences)
        batch_tokens.append(tokens)

        batch_objects_names.append(objects_names)
        
        a,b,c,d=train_3d[scene_id] 
        m=np.eye(3)+np.random.randn(3,3)*0.1
        m[0][0]*=np.random.randint(0,2)*2-1
        m*=scale
        theta=np.random.rand()*2*math.pi
        rot = np.array([[math.cos(theta),math.sin(theta),0],[-math.sin(theta),math.cos(theta),0],[0,0,1]])
        m = np.matmul(m, rot)
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

        a=torch.from_numpy(a).long()
        locs.append(torch.cat([a,torch.LongTensor(a.shape[0],1).fill_(idx)],1))
        feats.append(torch.from_numpy(b)+torch.randn(3)*0.1)
        labels.append(torch.from_numpy(c))
        ins_labels.append(torch.from_numpy(d.astype(int)))
        num_points.append(a.shape[0])
        scene_names.append(scene_id)

        ref_lbl = (ins_labels[-1].unsqueeze(-1)) == lang_objID
        batch_ref_lbl.append(ref_lbl.long())
    locs=torch.cat(locs,0)
    feats=torch.cat(feats,0)
    labels=torch.cat(labels,0)
    ins_labels=torch.cat(ins_labels,0)
    batch_data = {'x': [locs,feats],
                  'y': labels.long(),
                  'id': tbl,
                  'y_ins': ins_labels.long(),
                  'num_points': num_points,
                  'names': scene_names,
                  'lang_feat': batch_lang_feat,
                  'lang_len': batch_lang_len,
                  'lang_objID': batch_lang_objID,
                  'lang_objname': batch_lang_objname,  
                  'lang_cls': batch_lang_cls,
                  'sentences': batch_sentences,
                  'tokens': batch_tokens,
                  'ref_lbl': batch_ref_lbl,
                  'object_names':batch_objects_names
                   } 
    return batch_data

data_size = len(loader_list)
# batch_size=1
total_iteration = data_size/batch_size
train_data_loader = torch.utils.data.DataLoader(
    loader_list, 
    batch_size=batch_size,
    collate_fn=trainMerge,
    num_workers=0, 
    shuffle=True,
    drop_last=True,
    worker_init_fn=lambda x: np.random.seed(x+int(time.time()))
)

#-------------------------------------------------------------------------------------------

# BRIEF get position label [scannet]
MAX_NUM_OBJ = 132
def _get_token_positive_map(self, anno):
    """Return correspondence of boxes to tokens."""
    # Token start-end span in characters
    caption = ' '.join(anno['utterance'].replace(',', ' ,').split())
    caption = ' ' + caption + ' '
    
    tokens_positive = np.zeros((MAX_NUM_OBJ, 2))
    if isinstance(anno['target'], list):
        cat_names = anno['target']
    else:
        cat_names = [anno['target']]
    if self.detect_intermediate:
        cat_names += anno['anchors']
    for c, cat_name in enumerate(cat_names):
        start_span = caption.find(' ' + cat_name + ' ')
        len_ = len(cat_name)
        if start_span < 0:
            start_span = caption.find(' ' + cat_name)
            len_ = len(caption[start_span+1:].split()[0])
        if start_span < 0:
            start_span = caption.find(cat_name)
            orig_start_span = start_span
            while caption[start_span - 1] != ' ':
                start_span -= 1
            len_ = len(cat_name) + orig_start_span - start_span
            while caption[len_ + start_span] != ' ':
                len_ += 1
        
        end_span = start_span + len_
        assert start_span > -1, caption
        assert end_span > 0, caption
        tokens_positive[c][0] = start_span
        tokens_positive[c][1] = end_span

    # Positive map (for soft token prediction)
    tokenized = self.tokenizer.batch_encode_plus(
        [' '.join(anno['utterance'].replace(',', ' ,').split())],
        padding="longest", return_tensors="pt"
    )

    positive_map = np.zeros((MAX_NUM_OBJ, 256))

    # note: empty in scannet prompt
    modify_positive_map = np.zeros((MAX_NUM_OBJ, 256))
    pron_positive_map = np.zeros((MAX_NUM_OBJ, 256))
    other_entity_positive_map = np.zeros((MAX_NUM_OBJ, 256))
    rel_positive_map = np.zeros((MAX_NUM_OBJ, 256))
    auxi_entity_positive_map = np.zeros((MAX_NUM_OBJ, 256))
    
    # main object component
    gt_map = get_positive_map(tokenized, tokens_positive[:len(cat_names)])
    positive_map[:len(cat_names)] = gt_map
    return tokens_positive, positive_map, modify_positive_map, pron_positive_map, \
        other_entity_positive_map, auxi_entity_positive_map, rel_positive_map



# BRIEF Construct position label(map)
def get_positive_map(tokenized, tokens_positive):
    positive_map = torch.zeros((len(tokens_positive), 256), dtype=torch.float)  # ([positive], 256])
    for j, tok_list in enumerate(tokens_positive):
        (beg, end) = tok_list
        beg = int(beg)
        end = int(end)
        beg_pos = tokenized.char_to_token(beg)
        end_pos = tokenized.char_to_token(end - 1)
        if beg_pos is None:
            try:
                beg_pos = tokenized.char_to_token(beg + 1)
                if beg_pos is None:
                    beg_pos = tokenized.char_to_token(beg + 2)
            except:
                beg_pos = None
        if end_pos is None:
            try:
                end_pos = tokenized.char_to_token(end - 2)
                if end_pos is None:
                    end_pos = tokenized.char_to_token(end - 3)
            except:
                end_pos = None
        if beg_pos is None or end_pos is None:
            continue
        positive_map[j, beg_pos:end_pos + 1].fill_(1)

    positive_map = positive_map / (positive_map.sum(-1)[:, None] + 1e-12)
    return positive_map.numpy()
