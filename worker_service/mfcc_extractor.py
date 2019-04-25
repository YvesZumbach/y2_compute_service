import speechpy
import scipy.io.wavfile as wav
import os

file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Alesis-Sanctuary-QCard-AcoustcBas-C2.wav')
fs, signal = wav.read(file_name)
signal = signal[:,0]

mfcc = speechpy.feature.mfcc(signal, sampling_frequency=fs, frame_length=0.025, frame_stride=0.01,
             num_filters=20, fft_length=512, low_frequency=0, high_frequency=None)
mfcc_feature_cube = speechpy.feature.extract_derivative_feature(mfcc)
print('mfcc feature cube shape=', mfcc_feature_cube.shape)
