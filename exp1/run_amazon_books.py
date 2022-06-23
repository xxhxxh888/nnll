# -*- coding: utf-8 -*-
# coding=utf-8
from recbole.quick_start import run_recbole

dataset_nm='amazon-books'
#dataset_nm='ml-1m'

parameter_dict = {
   'neg_sampling': None,
}
config_file_list = ['sas.yaml']
run_recbole(model='SASRec', dataset=dataset_nm, config_file_list=config_file_list, config_dict=parameter_dict)
