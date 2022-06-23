# -*- coding: utf-8 -*-
# @Time    : 2020/9/18 11:33
# @Author  : Hui Wang
# @Email   : hui.wang@ruc.edu.cn

"""
SASRec
################################################

Reference:
    Wang-Cheng Kang et al. "Self-Attentive Sequential Recommendation." in ICDM 2018.

Reference:
    https://github.com/kang205/SASRec

"""
#===============================
import inspect,os,sys
_f_,_f_dir_=inspect.getframeinfo(inspect.currentframe()).filename, os.path.dirname(os.path.abspath(inspect.getframeinfo(inspect.currentframe()).filename)) 
#sys.path.append('D:/prog/exp1-master/')
_main_pth=os.path.realpath(_f_dir_+'/../../../')
if _main_pth not in sys.path:
    sys.path.append(_main_pth) 
import rstat #pptt!!
#===============================
import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import BPRLoss
import torch.nn.functional as F


import numpy as np
#pptt!!
#USE_PRJ_HEAD     =rstat.USE_PRJ_HEAD       
#USE_TOKEN_SHUFFLE=rstat.USE_TOKEN_SHUFFLE
#USE_ADD_NOISE    =rstat.USE_ADD_NOISE


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

class SASCTS(SequentialRecommender):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.

    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, config, dataset):
        super(SASCTS, self).__init__(config, dataset)

        # load parameters info
        self.lambda_cts = rstat.cts_loss_lambda  #0 # 0.06 # 0.02 #home-kitchen #0.06 #yelp; #0.02 #toys-games; #0.1
        self.config = config
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']

        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        print(self.hidden_size, self.inner_size)
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']

        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )



        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        #pptt!!
        self.nns_sim=Similarity(rstat.noise_neg_sample_temp)#pptt!!
        
        self.train_steps=-1
        
        print('--------------token-shuffle:',rstat.USE_TOKEN_SHUFFLE)
        print('--------------add-noise:',rstat.USE_ADD_NOISE)
        print('--------------prj-head:',rstat.USE_PRJ_HEAD)
        
        affine = True
        self.projection_head = torch.nn.ModuleList()
        inner_size = self.hidden_size #self.layers[-1] * 2
        print("------projection_head -- inner size:", inner_size)
        self.projection_head.append(torch.nn.Linear(inner_size, inner_size * 4, bias=False))
        self.projection_head.append(torch.nn.BatchNorm1d(inner_size * 4, eps=1e-12, affine=affine))
        self.projection_head.append(torch.nn.Linear(inner_size * 4, inner_size, bias=False))
        self.projection_head.append(torch.nn.BatchNorm1d(inner_size, eps=1e-12, affine=affine))
        self.mode = 0




        self.batch_size = config['train_batch_size']
        self.mask_default = self.mask_correlated_samples(batch_size=self.batch_size)
        self.CrossEntropy = nn.CrossEntropyLoss()

        # parameters initialization
        self.apply(self._init_weights)
    def device_as(self, t1, t2):
        """
        Moves t1 to the device of t2
        """
        return t1.to(t2.device)

    def calc_similarity_batch(self, a, b):
        representations = torch.cat([a, b], dim=0)
        return F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

    #pptt!!
    def projection_head_map(self, state, mode):
        for i, l in enumerate(self.projection_head): # 0: Linear 1: BN (relu)  2: Linear 3:BN (relu)
            if i % 2 != 0:
                if mode == 0:
                    l.train()   # set BN to train mode: use a learned mean and variance.
                else:
                    l.eval()   # set BN to eval mode: use a accumulated mean and variance.
            state = l(state)
            if i % 2 != 0:
                state = F.relu(state)
        return state
       
    def cts_loss_2(self, proj_1, proj_2, temp):
        mask = (~torch.eye(proj_1.shape[0] * 2, proj_1.shape[0] * 2, dtype=bool)).float()
        batch_size = proj_1.shape[0]

        z_i = F.normalize(proj_1, p=2, dim=1)
        z_j = F.normalize(proj_2, p=2, dim=1)

        similarity_matrix = self.calc_similarity_batch(z_i, z_j)

        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)

        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / temp)

        denominator = self.device_as(mask, similarity_matrix) * torch.exp(similarity_matrix / temp)

        all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / (2 * batch_size)
        return loss

    def cts_loss(self, z_i, z_j, temp, batch_size): #B * D    B * D
        N = 2 * batch_size
    
        z = torch.cat((z_i, z_j), dim=0)   #2B * D  
    
        sim = torch.mm(z, z.T) / temp   # 2B * 2B
    
        sim_i_j = torch.diag(sim, batch_size)    #B*1
        sim_j_i = torch.diag(sim, -batch_size)   #B*1
    
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        if batch_size != self.batch_size:
            mask = self.mask_correlated_samples(batch_size)
        else:
            mask = self.mask_default
        negative_samples = sim[mask].reshape(N, -1)
    
        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)  # N * C
        return logits, labels



    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len):
        is_train=self.training
        
        #if np.random.random()<0.2:
        #    #print('is-train:',is_train)
        #    pass
        
        
        if rstat.USE_TOKEN_SHUFFLE and is_train:#pptt!!
            position_ids=torch.arange(item_seq.size(1),dtype=torch.long,device=item_seq.device)#pptt!!
            position_ids=position_ids.unsqueeze(0).expand_as(item_seq)
            position_ids=position_ids[:,torch.randperm(position_ids.size()[1])] #add token shuffle;
    
        else:
            
            position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)#orig
            position_ids = position_ids.unsqueeze(0).expand_as(item_seq) #orig
            
            
        position_embedding = self.position_embedding(position_ids)
        
        
        if rstat.USE_ADD_NOISE and is_train:#pptt!!
            x=0.01+torch.zeros(position_embedding.shape[0],position_embedding.shape[1],position_embedding.shape[2],dtype=torch.float32,device=item_seq.device) #add noise
            noise=torch.normal(mean=torch.tensor([0.0]).to(item_seq.device),std=x).to(item_seq.device)
        
        
        

        item_emb = self.item_embedding(item_seq)
        
        if rstat.USE_ADD_NOISE and is_train:#pptt!!
            input_emb = item_emb + position_embedding+noise
        else:
            input_emb = item_emb + position_embedding #orig;
            
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq) 

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        
        
        output_l2 = trm_output[-2] #pptt!!
        output_l2 = self.gather_indexes(output_l2, item_seq_len - 1)  #pptt!!      
        
        if is_train:#pptt!!
            return output, output_l2  # [B H]
        else:
            return output
            

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]   #N * L
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output,seq_output_l2 = self.forward(item_seq, item_seq_len)  # N * D
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)

        if rstat.USE_LIKE_FINETUNE:
            if rstat.train_steps>=16: #pptt!!
                #rstat.USE_PRJ_HEAD=False       
                #rstat.USE_TOKEN_SHUFFLE=True
                #rstat.USE_ADD_NOISE=False 
                #self.lambda_cts=0.06
                a=0


        raw_seq_output,raw_seq_output_l2 = self.forward(item_seq, item_seq_len) #pptt!!
        
        if rstat.USE_PRJ_HEAD:
            raw_seq_output = self.projection_head_map(raw_seq_output, self.mode)#pptt!!
    
        
        if self.config['aug'] == 'self':
            cts_seq_output,cts_seq_output_l2 = self.forward(item_seq, item_seq_len)
            if rstat.USE_PRJ_HEAD:
                cts_seq_output = self.projection_head_map(cts_seq_output, 1 - self.mode)#pptt!!
            
        else:
            cts_aug, cts_aug_lengths = interaction['aug'], interaction['aug_lengths']
            cts_seq_output = self.forward(cts_aug, cts_aug_lengths)

        self.mode = 1 - self.mode     #pptt!!  


        cts_nce_logits, cts_nce_labels = self.cts_loss(raw_seq_output, cts_seq_output, temp=1.0,
                                                        batch_size=item_seq_len.shape[0])
        nce_loss = self.loss_fct(cts_nce_logits, cts_nce_labels)


        cts_nce_logits_l2, cts_nce_labels_l2 = self.cts_loss(raw_seq_output_l2, cts_seq_output_l2, temp=1.0,
                                                        batch_size=item_seq_len.shape[0])  #pptt!!
        nce_loss_l2 = self.loss_fct(cts_nce_logits_l2, cts_nce_labels_l2) #pptt!!

        if rstat.USE_NOISE_NEG_SAMPLE: #pptt!!
            nns_batch_size, nns_hidden_size = raw_seq_output.size()
            
            nns_labels = torch.arange( nns_batch_size).long().to(raw_seq_output.device) 
            nns_loss_fct = nn.CrossEntropyLoss()            
        
            nns_z_negative = torch.randn([int(nns_batch_size * 1 ), nns_hidden_size], device=raw_seq_output.device)  # * variation + avg
            nns_z_negative.requires_grad = True        
            
            nns_cos_sim = self.nns_sim(raw_seq_output.unsqueeze(1), cts_seq_output.unsqueeze(0))
        
            nns_loss_fct = nn.CrossEntropyLoss()
        
            for _ in range(rstat.noise_neg_sample_loops): # (cls.model_args.pgd):#!!!!!!!!!
                nns_cos_sim_negative = self.nns_sim(raw_seq_output.unsqueeze(1), nns_z_negative.unsqueeze(0))
                nns_cos_sim_fused = torch.cat([nns_cos_sim, nns_cos_sim_negative], 1)
        
                #c:\Users\11\anaconda3\envs\tch\Lib\site-packages\torch\nn\modules\loss.py; ln1028;
                nns_loss = nns_loss_fct(nns_cos_sim_fused, nns_labels) #common cross-entropy; 
        
                nns_noise_grad = torch.autograd.grad(nns_loss, nns_z_negative, retain_graph=True)[0]
        
                nns_z_negative = nns_z_negative + (nns_noise_grad / torch.norm(nns_noise_grad, dim=-1, keepdim=True)).mul_(1e-3)
                # noise = torch.clamp(noise, 0, 1.0)
                nns_z_negative = torch.where(torch.isnan(nns_z_negative), torch.zeros_like(nns_z_negative), nns_z_negative)        
        
            nns_cos_sim_negative = self.nns_sim(raw_seq_output.unsqueeze(1), nns_z_negative.unsqueeze(0))
            nns_cos_sim_fused = torch.cat([nns_cos_sim, nns_cos_sim_negative], 1)
    
            #c:\Users\11\anaconda3\envs\tch\Lib\site-packages\torch\nn\modules\loss.py; ln1028;
            nns_loss = nns_loss_fct(nns_cos_sim_fused, nns_labels) #common cross-entropy;         
        
            a=0
            loss+=rstat.noise_neg_sample_lambda * nns_loss#!!!!!!!!!!            


#        cts_loss = self.cts_loss_2(raw_seq_output, cts_seq_output, temp=1.0)
        loss += self.lambda_cts * nce_loss
        
        loss += rstat.cts_layer2_loss_lambda * nce_loss_l2 #pptt!!
        
        if np.random.random()<0.02: #0.01:#pptt!!
            print(' %d,%d,%d; %d;  %.4f;%.4f; %.4f; train_steps:%d; loss:%s;'%(rstat.USE_PRJ_HEAD,rstat.USE_TOKEN_SHUFFLE,rstat.USE_ADD_NOISE,
                                                                     rstat.USE_NOISE_NEG_SAMPLE,
                                                            self.lambda_cts, rstat.cts_layer2_loss_lambda,  rstat.noise_neg_sample_lambda,
                                                            rstat.train_steps, str(loss)))
        
        return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores
