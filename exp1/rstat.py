# -*- coding: utf-8 -*-
# coding=utf-8

train_steps=-1    #usless now;

USE_PRJ_HEAD     =False  #wether use project-head;     
USE_TOKEN_SHUFFLE=False  #wether use token-shuffle;
USE_ADD_NOISE    =False  #wether add guassian noise; seems usless;
UAE_ADV_NOISE	 =True   #adversial attack positive sample noise;  do not use token-shuffle;

USE_LIKE_FINETUNE=False  #seems useless;ignore this;

cts_loss_lambda       =0.02  #0.02   #original lambda;
cts_layer2_loss_lambda=0 #0.002 #layer the 2nd-to-last; no gain;

#=================================
USE_NOISE_NEG_SAMPLE=True          #from paper <<DCLR>>;

noise_neg_sample_loops =4 #4   #default 4,seems enough
noise_neg_sample_lambda=0 #0.04 #0.03 #0.02  #if 0: not use nns_vector some-loops training loss; should always be 0??
noise_neg_sample_weight_loss_lambda=2 #0.5 #0.02 #0.02  #weight used to ignore positive when calc cross-entropy; should has some val??
noise_neg_sample_temp  =0.5 #0.05 too small??equal to *20, so 0.5,0.3=>10,6, and softmax like:0.9,0.1, while 0.5,0.3 softmax like:0.7,0.3;

noise_neg_sample_batch_sz_times=1 #2.5 #200,300;   big & small times same effects, seems all no gain

noise_neg_sample_sim_as_positive=0.9/noise_neg_sample_temp  #0.9*20 from DCLR;

#for posi-posi loss (1) & posi-nega loss (multi), after nns-some-train-loops,the posi-nega loss very small 0.003,so simple mean will results very small total loss;
noise_neg_sample_loss_weights=[0.5,0.5] #sum:1 