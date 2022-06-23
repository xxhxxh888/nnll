# -*- coding: utf-8 -*-
# coding=utf-8
from recbole.quick_start import run_recbole

#nm='Amazon_Toys_and_Games' #  'Amazon_Home_and_Kitchen'
nm='Amazon_Home_and_Kitchen'

parameter_dict = {
   'neg_sampling': None,
}
config_file_list = ['sas_cts.yaml']
run_recbole(model='SASCTS', dataset=nm, config_file_list=config_file_list, config_dict=parameter_dict)
