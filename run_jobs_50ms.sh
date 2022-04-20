#!/usr/bin/env bash

# train models on 50ms cd dataset

cfg_path=/home/tinazhang/Models/pytorch-kaldi/cfg/TIMIT_baselines/cd/50ms

python ./run_exp.py $cfg_path/TIMIT_MLP_cd_basic.cfg

python ./run_exp.py $cfg_path/TIMIT_CNN_cd.cfg

python ./run_exp.py $cfg_path/TIMIT_liGRU_cd.cfg

python ./run_exp.py $cfg_path/TIMIT_LSTM_cd.cfg

python ./run_exp.py $cfg_path/TIMIT_LSTM_cd_mulit_gpu.cfg


