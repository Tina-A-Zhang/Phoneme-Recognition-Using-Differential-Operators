"""
save CD correlation features in ark format
"""

import scipy.io.wavfile
import numpy as np
import os
import math
import tqdm

from data_io import read_vec_int_ark, write_mat
from compute_cd_correlation import cd_correlation


# Run it for all the data chunks (e.g., train, dev, test) => uncomment
#25ms
#lab_folder = "/home/tinazhang/Install/kaldi/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali"
#lab_opts = "ali-to-pdf"
#out_folder = "/home/tinazhang/Install/kaldi/egs/timit/s5/cd_TIMIT_25ms/train"
#wav_lst = "/home/tinazhang/Install/kaldi/egs/timit/s5/cd_TIMIT_25ms/train/train-wav.lst"
#scp_file_out = "/home/tinazhang/Install/kaldi/egs/timit/s5/cd_TIMIT_25ms/train/feats_raw.scp"

lab_folder = "/home/tinazhang/Install/kaldi/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali_test"
lab_opts = "ali-to-pdf"
out_folder = "/home/tinazhang/Install/kaldi/egs/timit/s5/cd_TIMIT_25ms/test"
wav_lst = "/home/tinazhang/Install/kaldi/egs/timit/s5/cd_TIMIT_25ms/test/test-wav.lst"
scp_file_out = "/home/tinazhang/Install/kaldi/egs/timit/s5/cd_TIMIT_25ms/test/feats_raw.scp"

#lab_folder = "/home/tinazhang/Install/kaldi/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali_dev"
#lab_opts = "ali-to-pdf"
#out_folder = "/home/tinazhang/Install/kaldi/egs/timit/s5/cd_TIMIT_25ms/dev"
#wav_lst = "/home/tinazhang/Install/kaldi/egs/timit/s5/cd_TIMIT_25ms/dev/dev-wav.lst"
#scp_file_out = "/home/tinazhang/Install/kaldi/egs/timit/s5/cd_TIMIT_25ms/dev/feats_raw.scp"

#50ms
#lab_folder = "/home/tinazhang/Install/kaldi/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali"
#lab_opts = "ali-to-pdf"
#out_folder = "/home/tinazhang/Install/kaldi/egs/timit/s5/cd_TIMIT_50ms/train"
#wav_lst = "/home/tinazhang/Install/kaldi/egs/timit/s5/cd_TIMIT_50ms/train/train-wav.lst"
#scp_file_out = "/home/tinazhang/Install/kaldi/egs/timit/s5/cd_TIMIT_50ms/train/feats_raw.scp"

#lab_folder = "/home/tinazhang/Install/kaldi/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali_test"
#lab_opts = "ali-to-pdf"
#out_folder = "/home/tinazhang/Install/kaldi/egs/timit/s5/cd_TIMIT_50ms/test"
#wav_lst = "/home/tinazhang/Install/kaldi/egs/timit/s5/cd_TIMIT_50ms/test/test-wav.lst"
#scp_file_out = "/home/tinazhang/Install/kaldi/egs/timit/s5/cd_TIMIT_50ms/test/feats_raw.scp"

#lab_folder = "/home/tinazhang/Install/kaldi/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali_dev"
#lab_opts = "ali-to-pdf"
#out_folder = "/home/tinazhang/Install/kaldi/egs/timit/s5/cd_TIMIT_50ms/dev"
#wav_lst = "/home/tinazhang/Install/kaldi/egs/timit/s5/cd_TIMIT_50ms/dev/dev-wav.lst"
#scp_file_out = "/home/tinazhang/Install/kaldi/egs/timit/s5/cd_TIMIT_50ms/dev/feats_raw.scp"

#100ms
#lab_folder = "/home/tinazhang/Install/kaldi/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali"
#lab_opts = "ali-to-pdf"
#out_folder = "/home/tinazhang/Install/kaldi/egs/timit/s5/cd_TIMIT_100ms/train"
#wav_lst = "/home/tinazhang/Install/kaldi/egs/timit/s5/cd_TIMIT_100ms/train/train-wav.lst"
#scp_file_out = "/home/tinazhang/Install/kaldi/egs/timit/s5/cd_TIMIT_100ms/train/feats_raw.scp"

#lab_folder = "/home/tinazhang/Install/kaldi/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali_test"
#lab_opts = "ali-to-pdf"
#out_folder = "/home/tinazhang/Install/kaldi/egs/timit/s5/cd_TIMIT_100ms/test"
#wav_lst = "/home/tinazhang/Install/kaldi/egs/timit/s5/cd_TIMIT_100ms/test/test-wav.lst"
#scp_file_out = "/home/tinazhang/Install/kaldi/egs/timit/s5/cd_TIMIT_100ms/test/feats_raw.scp"

#lab_folder = "/home/tinazhang/Install/kaldi/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali_dev"
#lab_opts = "ali-to-pdf"
#out_folder = "/home/tinazhang/Install/kaldi/egs/timit/s5/cd_TIMIT_100ms/dev"
#wav_lst = "/home/tinazhang/Install/kaldi/egs/timit/s5/cd_TIMIT_100ms/dev/dev-wav.lst"
#scp_file_out = "/home/tinazhang/Install/kaldi/egs/timit/s5/cd_TIMIT_100ms/dev/feats_raw.scp"

#200ms
#lab_folder = "/home/tinazhang/Install/kaldi/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali"
#lab_opts = "ali-to-pdf"
#out_folder = "/home/tinazhang/Install/kaldi/egs/timit/s5/cd_TIMIT_200ms/train"
#wav_lst = "/home/tinazhang/Install/kaldi/egs/timit/s5/cd_TIMIT_200ms/train/train-wav.lst"
#scp_file_out = "/home/tinazhang/Install/kaldi/egs/timit/s5/cd_TIMIT_200ms/train/feats_raw.scp"

#lab_folder = "/home/tinazhang/Install/kaldi/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali_test"
#lab_opts = "ali-to-pdf"
#out_folder = "/home/tinazhang/Install/kaldi/egs/timit/s5/cd_TIMIT_200ms/test"
#wav_lst = "/home/tinazhang/Install/kaldi/egs/timit/s5/cd_TIMIT_200ms/test/test-wav.lst"
#scp_file_out = "/home/tinazhang/Install/kaldi/egs/timit/s5/cd_TIMIT_200ms/test/feats_raw.scp"

#lab_folder = "/home/tinazhang/Install/kaldi/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali_dev"
#lab_opts = "ali-to-pdf"
#out_folder = "/home/tinazhang/Install/kaldi/egs/timit/s5/cd_TIMIT_200ms/dev"
#wav_lst = "/home/tinazhang/Install/kaldi/egs/timit/s5/cd_TIMIT_200ms/dev/dev-wav.lst"
#scp_file_out = "/home/tinazhang/Install/kaldi/egs/timit/s5/cd_TIMIT_200ms/dev/feats_raw.scp"


sig_fs = 16000  # Hz
sig_wlen = 25  # 50, 100ms for new tests

lab_fs = 16000  # Hz
lab_wlen = 25  # ms
lab_wshift = 10  # ms

sig_wlen_samp = int((sig_fs * sig_wlen) / 1000)
lab_wlen_samp = int((lab_fs * lab_wlen) / 1000)
lab_wshift_samp = int((lab_fs * lab_wshift) / 1000)

#init cd
cd_cor = cd_correlation()

# Create the output folder
try:
    os.stat(out_folder)
except:
    os.makedirs(out_folder)


# Creare the scp file
scp_file = open(scp_file_out, "w")

# reading the labels
lab = {
    k: v
    for k, v in read_vec_int_ark(
        "gunzip -c " + lab_folder + "/ali*.gz | " + lab_opts + " " + lab_folder + "/final.mdl ark:- ark:-|", out_folder
    )
}

# reading the list file
with open(wav_lst) as f:
    sig_lst = f.readlines()

sig_lst = [x.strip() for x in sig_lst]

for sig_file in tqdm.tqdm(sig_lst):
    sig_id = sig_file.split(" ")[0]
    sig_path = sig_file.split(" ")[1]
    # get the CDs
    signal = cd_cor.process(sig_path)

    cnt_fr = 0
    beg_samp = 0
    frame_all = []

    while beg_samp + lab_wlen_samp < signal.shape[1]:
        sample_fr = np.zeros(sig_wlen_samp)
        central_sample_lab = int(((beg_samp + lab_wlen_samp / 2) - 1))
        central_fr_index = int(((sig_wlen_samp / 2) - 1))

        beg_signal_fr = int(central_sample_lab - (sig_wlen_samp / 2))
        end_signal_fr = int(central_sample_lab + (sig_wlen_samp / 2))

        if beg_signal_fr >= 0 and end_signal_fr <= signal.shape[1]:
            feats = cd_cor.get_cor(signal, beg_signal_fr, end_signal_fr)
        else:
            if beg_signal_fr < 0:
                feats = cd_cor.get_cor(signal, 0, end_signal_fr)
            if end_signal_fr > signal.shape[1]:
                feats = cd_cor.get_cor(signal, beg_signal_fr, signal.shape[1])

        frame_all.append(feats)
        cnt_fr = cnt_fr + 1
        beg_samp = beg_samp + lab_wshift_samp

    frame_all = np.asarray(frame_all)

    # Save the matrix into a kaldi ark
    out_file = out_folder + "/" + sig_id + ".ark"
    write_mat(out_folder, out_file, frame_all, key=sig_id)
    #print(sig_id)
    scp_file.write(sig_id + " " + out_folder + "/" + sig_id + ".ark:" + str(len(sig_id) + 1) + "\n")

    N_fr_comp = 1 + math.floor((signal.shape[1] - 400) / 160)
    # print("%s %i %i "%(lab[sig_id].shape[0],N_fr_comp,cnt_fr))

scp_file.close()



