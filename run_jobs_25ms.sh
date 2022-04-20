#!/usr/bin/env bash

# prepare dataset of 25ms

#python ./save_cd_feats_25ms_train.py

#python ./save_cd_feats_25ms_dev.py

#python ./save_cd_feats_25ms_test.py

python countdown.py 7900

# train models on 25ms cd dataset

cfg_path=/home/tinazhang/Models/pytorch-kaldi/cfg/TIMIT_baselines/cd/25ms

python ./run_exp.py $cfg_path/TIMIT_MLP_cd_basic.cfg

python ./run_exp.py $cfg_path/TIMIT_CNN_cd.cfg

python ./run_exp.py $cfg_path/TIMIT_liGRU_cd.cfg

python ./run_exp.py $cfg_path/TIMIT_LSTM_cd.cfg

python ./run_exp.py $cfg_path/TIMIT_LSTM_cd_mulit_gpu.cfg
