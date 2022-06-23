# -*- coding: utf-8 -*-
# coding=utf-8
from recbole.quick_start import run_recbole



parameter_dict = {
   'neg_sampling': None,
}
config_file_list = ['sas.yaml']
run_recbole(model='SASRec', dataset='ml-1m', config_file_list=config_file_list, config_dict=parameter_dict)
