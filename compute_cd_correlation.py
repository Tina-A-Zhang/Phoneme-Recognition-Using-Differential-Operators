'''
Compute Chromatic Derivatives and Its Correlation.
'''
import scipy.io.wavfile
import numpy as np
import os

cd_folder = './CD/'

class cd_correlation:
    def __init__(self):
        self.filter_lpf = np.genfromtxt(os.path.join(cd_folder, 'LPF.csv'), delimiter=',')
        self.filter_bank = np.zeros((48, 129))
        for i in range(48):
            filter_name = 'legendre_0.885_0.985_1.00_129_' + '%d'%i + '.txt'
            filter_path = os.path.join(cd_folder, '0.885_0.985_1.00_129', filter_name)
            self.filter_bank[i, :] = np.loadtxt(filter_path) / 1e15

    def process(self, sound_file_path):
        _, signal = scipy.io.wavfile.read(sound_file_path)
        signal = signal.astype(float)
        lpf_signal = np.convolve(self.filter_lpf, signal, 'same')
        fl_signal = np.zeros((48, len(lpf_signal)))
        for i in range(48):
            fl_signal[i, :] = np.convolve(self.filter_bank[i, :], lpf_signal, 'same')
        return fl_signal

    def get_cor(self, fl_signal, begin, end):
        seg_signal = fl_signal[:, begin:end]
        cov = np.matmul(seg_signal, np.transpose(seg_signal))
        cor = np.zeros_like(cov)
        for i in range(cov.shape[0]):
            for j in range(cov.shape[1]):
                cor[i, j] = 0.5 * cov[i, j] / np.sqrt(cov[i, i] * cov[j, j]) + 0.5
        
        #whole matrix and upper triangle (save half feats)
        # cor: 48x48, effective cor: ~48x48/2
        return cor.flatten()
        #return np.asarray(cor[np.triu_indices_from(cor)])
