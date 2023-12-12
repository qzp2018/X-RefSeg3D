import numpy as np
import sparseconvnet as scn
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.parameter import Parameter
from util import *
from config import *
import math
m = 32
residual_blocks= True
block_reps = 2

dimension = 3
full_scale = 4096

class SCN(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.sparseModel = scn.Sequential().add(
            scn.InputLayer(dimension,full_scale, mode=4)).add(
            scn.SubmanifoldConvolution(dimension, 3, m, 3, False)).add(
            scn.UNet(dimension, block_reps, [m, 2*m, 3*m, 4*m, 5*m, 6*m, 7*m], residual_blocks)).add(
            scn.BatchNormReLU(m)).add(
            #scn.SubmanifoldConvolution(data.dimension, m, 4, 1, False)).add(
            scn.OutputLayer(dimension))
        self.linear = nn.Linear(m, 20)
        self.linear1 = nn.Linear(m, m) 
        self.cen_pred = nn.Sequential(nn.Linear(m, m),nn.LeakyReLU(negative_slope=0.01, inplace=True), nn.Linear(m, 3))
    def forward(self,x):
        fv = self.sparseModel(x)
        y = self.linear(fv)
        fv = self.linear1(fv)
        offset = self.cen_pred(fv)
        #sigma=self.linear1(y)
        #fv = F.normalize(fv, p=2, dim=1)
        #return fv
        return y, fv, offset   #pc_sem, pc_feat, pc_offset

class cross_Attention(nn.Module):
    def __init__(self, input_dim, dim_q, dim_k, n_head):
        super(cross_Attention, self).__init__()
        # dim_q = dim_k
        self.dim_q, self.dim_k, self.n_head = dim_q, dim_k, n_head
         
        if self.dim_k % n_head != 0:
            print("dim_k can't divide n_head")
        if self.dim_q % n_head != 0:
            print("dim_q  can't divide n_head")
        self.n_head=n_head
        self.wq = nn.Linear(input_dim, dim_q)
        self.wk = nn.Linear(input_dim, dim_k)
        self.wo = nn.Linear(dim_k, input_dim)
        self._norm_fact = 1 / math.sqrt(dim_q)
      #  self.init_weights()
        self.dropout = nn.Dropout(p=0.1)
    def init_weights(self):
        nn.init.xavier_uniform_(self.wq.weight)
        nn.init.xavier_uniform_(self.wk.weight)    
        nn.init.constant_(self.wq.bias, 0)
        nn.init.constant_(self.wk.bias, 0)
        nn.init.constant_(self.wo.bias, 0)

    def forward(self, q, k, attn_mask):   
        # q: B Nq input_dim     
        Q = self.wq(q)  # B Nq input_dim ->B Nq dim_q
        K = self.wk(k)  # B Nk dim_k
    
        Q =( Q.reshape(Q.shape[0], self.n_head, Q.shape[1], self.dim_q // self.n_head))
        # B Nq dim_q->  B head  Nq dim_q/head = [batch_size,n_head,n_q,dim_q/n_head]      
        K = K.reshape(K.shape[0], self.n_head, K.shape[1], self.dim_k // self.n_head)
        attention = torch.matmul(Q, K.permute(0, 1, 3, 2)) * self._norm_fact         # batch_size ,n_head, Q.shape[1] , K.shape[1]
        if attn_mask is not None:
            attention =attention +attn_mask  # attn_mask   b n  Q.shape[1] , K.shape[1]
        attention=torch.nn.functional.softmax(attention, dim=-1) 
        attention=self.dropout(attention)
         #B head Nq Nk  *  B head Nk 768/head     -> B head Nq 768/head  -> B Nq dv
        attention = attention.reshape(-1, attention.shape[2], attention.shape[3])
        
        return attention

class NormalizeScale(nn.Module):

    def __init__(self, dim, init_norm=20):
        super(NormalizeScale, self).__init__()
        self.init_norm = init_norm
        self.weight = nn.Parameter(torch.ones(1, dim) * init_norm)

    def forward(self, bottom):
        bottom_normalized = F.normalize(bottom, p=2, dim=2)
        bottom_normalized_scaled = bottom_normalized * self.weight
        return bottom_normalized_scaled




class SentenceEncoderGRU(nn.Module):
    def __init__(self, embedding_size=300, input_size=256, hidden_size=256, output_size=64, bidirectional=True):
        nn.Module.__init__(self)
        self.mlp = nn.Sequential(nn.Linear(embedding_size, input_size), nn.LeakyReLU(negative_slope=0.01, inplace=True))
        self.biLSTM = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=False)
        self.lang_mlp = nn.Linear(hidden_size, output_size)
    def forward(self, lang_feat, lang_len):
        lang_feat = self.mlp(lang_feat)
        total_length = lang_feat.shape[1]
    
        lang_feat = pack_padded_sequence(lang_feat, lang_len, batch_first=True, enforce_sorted=False)#pack_padded_sequence 
        output, hidden = self.biLSTM(lang_feat)
        # batch, sen_len, vec_size
        #lang_feat, _ = pad_packed_sequence(lang_feat, batch_first=True)
        output, _ = pad_packed_sequence(output, batch_first=True, total_length=total_length)
        output = self.lang_mlp(output)
        #weight = torch.zeros(hidden.shape[0], 2).cuda()
        return output, lang_len

class RefNetGRU(nn.Module):
    def __init__(self, k):
        nn.Module.__init__(self)
        self.lang_encoder = SentenceEncoderGRU(300,512,512,256)
        self.relconv = TARelationConvBlock(k)
    def forward(self, scene):
        #lang_feat: num_sentences, max_len, d   
        #lang_len: num_sentences, 
        num_obj, d = scene['obj_feat'].shape#d
        #num_sen, num_obj, d
        num_sen, max_len, _ = scene['lang_feat'].shape

        obj_feat = scene['obj_feat'].unsqueeze(0).expand(num_sen,num_obj,d)
        obj_coord = scene['obj_coord'].unsqueeze(0).expand(num_sen,num_obj,3)
        lrel=scene['lrel']
        #num_sen, 30, d / num_sen 
        lang_feat, lang_len = self.lang_encoder(scene['lang_feat'], scene['lang_len'])
        feat, score_cos,score_mlp = self.relconv(obj_feat, obj_coord, lang_feat, lang_len,lrel)
        return score_cos,score_mlp


class Matching(nn.Module):
    def __init__(self, vis_dim, lang_dim, jemb_dim, jemb_drop_out):
        super(Matching, self).__init__()
        self.vis_emb_fc = nn.Sequential(nn.Linear(vis_dim, jemb_dim),
                                         nn.BatchNorm1d(jemb_dim),
                                         nn.ReLU(),
                                         nn.Dropout(jemb_drop_out),
                                         nn.Linear(jemb_dim, jemb_dim),
                                         nn.BatchNorm1d(jemb_dim))
        self.lang_emb_fc = nn.Sequential(nn.Linear(lang_dim, jemb_dim),
                                         nn.BatchNorm1d(jemb_dim),
                                         nn.ReLU(),
                                         nn.Dropout(jemb_drop_out),
                                         nn.Linear(jemb_dim, jemb_dim),
                                         nn.BatchNorm1d(jemb_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, visual_input, lang_input):

        assert visual_input.size(0) == lang_input.size(0)

        visual_feat = visual_input.view((visual_input.size(0) * visual_input.size(1), -1))
        lang_feat = lang_input
        #vis feat和lang feat过fc
        visual_emb = self.vis_emb_fc(visual_feat)
        lang_emb = self.lang_emb_fc(lang_feat)

        # l2-normalize
        visual_emb_normalized = F.normalize(visual_emb, p=2, dim=1)
        lang_emb_normalized = F.normalize(lang_emb, p=2, dim=1)

        block_visual_emb_normalized = visual_emb_normalized.view((visual_input.size(0), visual_input.size(1), -1))
        block_lang_emb_normalized = lang_emb_normalized.unsqueeze(1).expand((visual_input.size(0),
                                                                             visual_input.size(1),
                                                                             lang_emb_normalized.size(1)))
        cossim = torch.sum(block_lang_emb_normalized * block_visual_emb_normalized, 2)
        logit_scale = self.logit_scale.exp()
        cossim=logit_scale *cossim

        return cossim

class KNN(nn.Module):
    def __init__(self,feat_dim_in,feat_dim_out,num_k):
        super().__init__()
        self.rel_encoder = nn.Sequential(nn.Linear(10, 64), nn.LeakyReLU(negative_slope=0.01, inplace=True), nn.Linear(64, 128))
        self.feat_encoder = nn.Sequential(nn.Linear(feat_dim_in, 256), nn.LeakyReLU(negative_slope=0.01, inplace=True), nn.Linear(256, feat_dim_out))
        self.feat_encoder2 = nn.Sequential(nn.Linear(feat_dim_in, 256), nn.LeakyReLU(negative_slope=0.01, inplace=True), nn.Linear(256, feat_dim_out))
        self.norm = nn.LayerNorm(128)
        self.k=num_k
    
    def forward(self,feat,coord,relation, edge_weight_per_type_per_sent,fusion_edgefeat_per_node):
        feat=self.feat_encoder(feat)
        edge_feat=self.feat_encoder2(fusion_edgefeat_per_node)
        num_sen, num_obj, _ = feat.shape
        k=min(num_obj-1,self.k)
        d = ((coord.unsqueeze(1) - coord.unsqueeze(2))**2).sum(-1)
        indice0 = torch.arange(coord.shape[0]).view(coord.shape[0],1,1).repeat(1, num_obj, k+1)#
        _, indice1 = torch.topk(d, k+1, dim=-1, largest=False)


        weight_matrix=edge_weight_per_type_per_sent[:,relation.long()]
        edge_weight=torch.gather(weight_matrix,2,indice1).unsqueeze(-1)

        coord_expand = coord[indice0, indice1]
        coord_expand1 = coord.unsqueeze(2).expand(coord.shape[0], coord.shape[1], k+1, coord.shape[-1])#
        rel_coord = coord_expand - coord_expand1
        dis = torch.norm(rel_coord, p=2, dim=-1).unsqueeze(-1)
        rel = torch.cat([coord_expand, coord_expand1, rel_coord, dis], -1)
        # num_sen, num_obj, k+1, d
        rel = self.rel_encoder(rel)#Realtion Encoding

        residual_feat=(feat[indice0,indice1]*rel+edge_weight*(edge_feat[indice0,indice1])).sum(2)+feat+edge_feat
        norm_feat= self.norm(residual_feat)
        return norm_feat


class TARelationConvBlock(nn.Module):
    def __init__(self, k):
        nn.Module.__init__(self)
        
        self.knn1=KNN(256,128,k)
        self.gcn=RefNetGCN()
        self.ffn = nn.Linear(128, 1)
        self.matching = Matching(128,
                                 256,
                                 256,#opt['jemb_dim']
                                 0.2)#opt['jemb_drop_out']
    def forward(self, feat, coord, lang_feat, lang_len,lrel):
        feat,edge_weight_per_type_per_sent,fusion_edgefeat_per_node=self.gcn(feat,lang_feat,lang_len)
        feat = F.relu(feat)
        max_lang_feat=torch.max(lang_feat,dim=1)[0]
        feat = self.knn1(feat, coord,lrel, edge_weight_per_type_per_sent,fusion_edgefeat_per_node)
        feat = F.relu(feat)
        score_cos = self.matching(feat, max_lang_feat)#公式8 
        score_mlp=self.ffn(feat).squeeze(-1)
        return feat, score_cos,score_mlp


class RefNetGCN(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        
        self.word_judge = nn.Sequential(nn.Linear(256, 64),#nn.Linear(dim_word_output, opt['dim_hidden_word_judge'])
                                nn.Sigmoid(),
                                nn.Linear(64, 4),#opt['dim_hidden_word_judge'], num_cls_word
                                nn.Softmax(dim=2))
        self.lang_feat_encoder=nn.Sequential(nn.Linear(256, 256), nn.LeakyReLU(negative_slope=0.01, inplace=True),nn.Dropout(0.1),nn.Linear(256, 256))
        # absolute location
        self.edge_gate = nn.Sequential(nn.Linear(256, 64),
                                       nn.LeakyReLU(negative_slope=0.01, inplace=True),
                                       nn.Dropout(0.1),
                                       nn.Linear(64,9),#opt['dim_edge_gate'], opt['num_location_relation']
                                       nn.Softmax(dim=2))
        #GGCN模块
        #cross attn
        self.crossattn=cross_Attention(input_dim=256, dim_q=512, dim_k=512, n_head=1)
        #feat mlp
        self.feat_mlp1 = nn.Sequential(nn.Linear(32, 128), nn.LeakyReLU(negative_slope=0.01, inplace=True),nn.Linear(128, 256))
        self.feat_mlp2 = nn.Sequential(nn.Linear(32, 128), nn.LeakyReLU(negative_slope=0.01, inplace=True),nn.Linear(128, 256))
        self.feat_mlp3 = nn.Sequential(nn.Linear(32, 128), nn.LeakyReLU(negative_slope=0.01, inplace=True),nn.Linear(128, 256))
        self.feat_mlp4 = nn.Sequential(nn.Linear(32, 128), nn.LeakyReLU(negative_slope=0.01, inplace=True),nn.Linear(128, 256))
        #normalizer
        self.word_normalizer = NormalizeScale(256, 20)
 
        
    def forward(self, obj_feat,lang_feat,lang_len):
        num_sen, max_len, _ = lang_feat.shape

        lengths_gpu = lang_len.cuda()
        #context,_=self.lang_encoder(lang_feat, lengths_cpu)

        context_weight = self.word_judge(lang_feat)
        mask = torch.arange(max_len).repeat(num_sen, 1).cuda()
        mask = (mask < lengths_gpu.unsqueeze(-1)).float()
        context_weight=context_weight*mask.unsqueeze(-1)

        vis_feat_per_node1 =self.feat_mlp1(obj_feat)
        vis_feat_per_node2 =self.feat_mlp2(obj_feat)
        vis_feat_per_node3 =self.feat_mlp3(obj_feat)
        #x=self.feat_encoder(x)
        #words = self.word_normalizer(context)  #ht
        words=self.word_normalizer(lang_feat) 
        words=words*mask.unsqueeze(-1)  #gai    
        word_node_weights_expand = (context_weight[:, :, 0]+context_weight[:, :, 1]).unsqueeze(2)

        # obtain note gate
        attn_word_node1=self.crossattn(words,vis_feat_per_node1,attn_mask=None)
        attn_word_node1=F.softmax(attn_word_node1,dim=-1)*mask.unsqueeze(-1) 
        node_weight_per_word1 = attn_word_node1 * word_node_weights_expand 
        word_feat_per_node1 = torch.bmm(node_weight_per_word1.transpose(1, 2), words)  # visual location

        attn_word_node2=self.crossattn(words,vis_feat_per_node2,attn_mask=None)
        attn_word_node2=F.softmax(attn_word_node2,dim=-1)*mask.unsqueeze(-1) 
        node_weight_per_word2 = attn_word_node2 * word_node_weights_expand 
        word_feat_per_node2 = torch.bmm(node_weight_per_word2.transpose(1, 2), words)    # visual location

        attn_word_node3=self.crossattn(words,vis_feat_per_node3,attn_mask=None)
        attn_word_node3=F.softmax(attn_word_node3,dim=-1)*mask.unsqueeze(-1) 
        node_weight_per_word3 = attn_word_node3 * word_node_weights_expand
        word_feat_per_node3 = torch.bmm(node_weight_per_word3.transpose(1, 2), words)   # visual location
                         
        fusion_feat_per_node = word_feat_per_node1*vis_feat_per_node1+ word_feat_per_node2*vis_feat_per_node2+ word_feat_per_node3*vis_feat_per_node3 #每个顶点的文本特征和视觉特征相乘融合在一块  X^m

        attn_word_node4=self.crossattn(words,fusion_feat_per_node ,attn_mask=None)
        attn_word_node4=F.softmax(attn_word_node4,dim=-1)*mask.unsqueeze(-1) 
        word_node_edgeweights_expand = context_weight[:, :, 2].unsqueeze(2)
        node_edgeweight_per_word = attn_word_node4 * word_node_edgeweights_expand 
        word_edgefeat_per_node = torch.bmm(node_edgeweight_per_word.transpose(1, 2), words)   # visual location
        fusion_edgefeat_per_node= word_edgefeat_per_node*fusion_feat_per_node                  
        
        # obtain edge gate
        word_edge_weights_expand = context_weight[:, :, 2].unsqueeze(2).expand(context_weight.size(0),context_weight.size(1), 9)  #每个单词属于关系词的概率    
        edge_type_weight = self.edge_gate(lang_feat)
        edge_weight_per_type_per_word =edge_type_weight*word_edge_weights_expand
        edge_weight_per_type_per_sent = torch.sum(edge_weight_per_type_per_word, 1)       
        return fusion_feat_per_node,edge_weight_per_type_per_sent,fusion_edgefeat_per_node

def gather(feat, lbl):
    uniq_lbl = torch.unique(lbl)
    gather_func = scn.InputLayer(1, uniq_lbl.shape[0], mode=4)
    grp_f = gather_func([lbl.long().unsqueeze(-1), feat])
    grp_idx, sorted_indice = grp_idx.sort()
    grp_f = grp_f.features[sorted_indice]
    return grp_f, grp_idx



def gather_1hot(feat, mask):
    # (obj, )
    obj_size = mask.sum(0)
    mean_f = torch.bmm(mask.unsqueeze(-1).float(), feat.unsqueeze(1))
    # (obj, d)
    mean_f = mean_f.sum(0) / obj_size.float().unsqueeze(-1)
    idx = torch.arange(mask.shape[1]).cuda()
    return mean_f, idx
