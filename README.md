# Honours Thesis Project code

## Modification on pytorch-kaldi:
File: compute_cd_correlation.py
- To calculate the Chromatic Derivatives Correlation of given frame
File: save_cd_feats.py
- To generate CDC feature datasets
File: data_io.py
- Line: 263
- redefine the normalization for CDC[i, i] = 1 terms

## Configuration and runs:
CDC runs
- Folder: ./cfg/TIMIT_baselines/cd 
MFCC runs
- Folder: ./cfg/TIMIT_baselines/mfcc
Raw runs
- Folder: ./cfg/TIMIT_baselines/raw

## Pytorch-Kalsi
Follow
https://github.com/mravanelli/pytorch-kaldi
to install Kaldi and Pytorch-Kaldi packages.

