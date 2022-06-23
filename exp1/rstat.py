# -*- coding: utf-8 -*-
# coding=utf-8

train_steps=-1    #usless now;

USE_PRJ_HEAD     =False  #wether use project-head;     
USE_TOKEN_SHUFFLE=False  #wether use token-shuffle;
USE_ADD_NOISE    =False  #wether add guassian noise;

USE_LIKE_FINETUNE=False  #seems useless;ignore this;

cts_loss_lambda       =0.02 #0.02   #original lambda;
cts_layer2_loss_lambda=0.002		#layer the 2nd-to-last;

USE_NOISE_NEG_SAMPLE=False          #from paper <<DCLR>>;

noise_neg_sample_loops =4 #4
noise_neg_sample_lambda=0.04 #0.03 #0.02
noise_neg_sample_temp  =0.05