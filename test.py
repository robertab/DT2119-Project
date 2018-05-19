import numpy as np
import matplotlib.pyplot as plt
# from scipy import signal
from scipy.io import wavfile
from pysndfile import sndio
from scipy.stats import signaltonoise
# from math import sqrt
from sklearn.decomposition import FastICA


def loadAudio(filename):
    """
    loadAudio: loads audio data from file using pysndfile

    Note that, by default pysndfile converts the samples into floating point
    numbers and rescales them in the range [-1, 1]. This can be avoided by
    specifying the dtype argument in sndio.read(). However, when I imported'
    the data in lab 1 and 2, computed features and trained the HMM models,
    I used the default behaviour in sndio.read() and rescaled the samples
    in the int16 range instead. In order to compute features that are
    compatible with the models, we have to follow the same procedure again.
    This will be simplified in future years.
    """
    sndobj = sndio.read(filename)
    samplingrate = sndobj[1]
    samples = np.array(sndobj[0])*np.iinfo(np.int16).max
    return samples, samplingrate


# **********************************************************************************
EXP1 = "experiment1"
np.random.seed(0)
# Signals to separate. First experiment: 4 different utterances
man_one = EXP1 + '/' + 'man_one_a.wav'
man_three = EXP1 + '/' + 'man_three_a.wav'
woman_four = EXP1 + '/' + 'woman_four_a.wav'
woman_five = EXP1 + '/' + 'woman_five_a.wav'

s1, sample1 = loadAudio(man_one)
s2, sample2 = loadAudio(woman_five)
s3, sample3 = loadAudio(man_three)
s4, sample4 = loadAudio(woman_four)
# Signals to separate. Second experiment: 4 utterances of a same digit
# man_one = EXP1 + '/' + 'man_one_a.wav'
# man_three = EXP1 + '/' + 'man_three_a.wav'
# woman_four = EXP1 + '/' + 'woman_four_a.wav'
# woman_five = EXP1 + '/' + 'woman_five_a.wav'

# s1, sample1 = loadAudio(man_one)
# s2, sample2 = loadAudio(woman_five)
# s3, sample3 = loadAudio(man_three)
# s4, sample4 = loadAudio(woman_four)

# Concatenate both signals. Strip of the shortest
S = np.c_[s1[:s3.shape[0]], s2[:s3.shape[0]], s3, s4[:s3.shape[0]]]
# Add a gaussian noise (zero mean, unit variance)
# S += np.random.normal(size=S.shape)
# # Standarize the data
S = (S - S.mean(0)) / S.std(axis=0)
# # Linear transformation matrix
A = np.array([[1., .5, np.sqrt(.5), .5],
              [.5, 1., .5, np.sqrt(.5)],
              [np.sqrt(.5), .5, 1, .5],
              [.5, np.sqrt(.5), .5, 1.]])

# # Linearly combine signals (Observed signals)
X = np.dot(S, A.T)
# # Compute ICA
ica = FastICA(n_components=4)
reconstructions = ica.fit_transform(X)
# A_ = ica.mixing_

scaled1 = np.int16(reconstructions[:, 0] /
                   np.max(np.abs(reconstructions[:, 0])) * 32767)
wavfile.write("experiment2/r1_exp2.wav", sample1, scaled1)
# Standarize the data for comparison purposes
scaled1 = (scaled1 - scaled1.mean()) / scaled1.std()

scaled2 = np.int16(reconstructions[:, 1] /
                   np.max(np.abs(reconstructions[:, 1])) * 32767)
wavfile.write("experiment2/r2_exp2.wav", sample2, scaled2)
# Standarize the data for comparison purposes
scaled2 = (scaled2 - scaled2.mean()) / scaled2.std()

scaled3 = np.int16(reconstructions[:, 2] /
                   np.max(np.abs(reconstructions[:, 2])) * 32767)
wavfile.write("experiment2/r3_exp2.wav", sample3, scaled3)
# Standarize the data for comparison purposes
scaled3 = (scaled3 - scaled3.mean()) / scaled3.std()

scaled4 = np.int16(reconstructions[:, 3] /
                   np.max(np.abs(reconstructions[:, 3])) * 34767)
wavfile.write("experiment2/r4_exp2.wav", sample4, scaled4)
# Standarize the data for comparison purposes
scaled4 = (scaled4 - scaled4.mean()) / scaled4.std()

# plt.hist(S[:, 0])
# plt.hist(scaled2)
# plt.hist(scaled3)
# plt.hist(scaled4)
# freq1, times1, spectogram1 = signal.spectrogram(scaled2, sample2)
# freq2, times2, spectogram2 = signal.spectrogram(S[:, 1], sample2)
# freq3, times3, spectogram3 = signal.spectrogram(scaled1, sample1)
# freq4, times4, spectogram4 = signal.spectrogram(S[:, 0], sample1)
# print(np.linalg.norm(spectogram1 - spectogram3))
# # print(np.linalg.norm(spectogram1 - spectogram4))
# # print(np.linalg.norm(spectogram2 - spectogram3))
# # print(np.linalg.norm(spectogram2 - spectogram4))
# # plt.figure()
# # plt.subplot(4, 1, 1)
# # plt.pcolormesh(spectogram1)
# # plt.subplot(4, 1, 2)
# # plt.pcolormesh(spectogram2)
# # plt.subplot(4, 1, 3)
# # plt.pcolormesh(spectogram3)
# # plt.subplot(4, 1, 4)
# # plt.pcolormesh(spectogram4)
# # plt.show()
# # rmse_0 = np.mean((scaled1 - S_[:, 0])**2)
# # rmse_1 = np.mean((scaled1 - S_[:, 1])**2)
# # print("RMSE: 0-0: {}, 0-1: {}".format(rmse_0, rmse_1))
# #print("Scaled1 - S0: {}".format(np.linalg.norm(scaled1 - S[:, 0])))
# print("Scaled1 - S1: {}".format(np.linalg.norm(scaled1 - S[:, 1])))
# print("Scaled2 - S0: {}".format(np.linalg.norm(scaled2 - S[:, 0])))
# # print("Scaled2 - S1: {}".format(np.linalg.norm(scaled2 - S[:, 1])))
# # print(np.linalg.norm(scaled1 - S[:, 0]))
# # print("Scaled2")
# # print(np.linalg.norm(scaled2 - S[:, 1]))
# # print(np.linalg.norm(scaled2 - S[:, 0]))
# ***************************** PLOTS ***********************************************
# plt.figure()
# plt.suptitle("Reconstructions / Original sources, no noise", fontsize=16)
# plt.suptitle("Reconstructions / Original sources, ($\mu=0, \sigma^2=1$)", fontsize=16)
# # True sources
# plt.subplot(426)
# plt.plot(S[:, 1], 'b')
# plt.subplot(424)
# plt.plot(S[:, 0], 'b')
# plt.subplot(428)
# plt.plot(S[:, 3], 'b')
# plt.subplot(422)
# plt.plot(S[:, 2], 'b')
# # Reconstructed signals
# plt.subplot(421)
# plt.plot(scaled1, 'r')
# plt.subplot(423)
# plt.plot(scaled2, 'r')
# plt.subplot(425)
# plt.plot(scaled3, 'r')
# plt.subplot(427)
# plt.plot(scaled4, 'r')

# plt.show()

# print(np.linalg.norm(scaled1 - S[:, 1]))

# scaled = [scaled1, scaled2, scaled3, scaled4]
# dist = np.empty((4, 4))
# for i, s in enumerate(scaled):
#     for j in range(S.shape[1]):
#         dist[i, j] = np.linalg.norm(S[:, j] - s)

# print(signaltonoise(s1))
# # X_ = ica.inverse_transform(S_)

# # # plt.contourf(xx, yy, multivariate_normal([0, 0], np.corrcoef(S[:, 1], S_[:, 0])).pdf(pos))
# # # plt.show()
# # # similiarity_measure = np.sum((S[:, 0]*S_[:, 0]))**2 / np.sum(S[:, 0]**2)*np.sum(S_[:, 0]**2)
# # # print(np.corrcoef(S[:, 0], S_[:, 0]))
# # # hej = np.dot(S_, A_.T) + ica.mean_
# # print(np.sqrt(np.mean((X_ - S)**2)))
# # plt.plot(X_[:, 0])
# # plt.plot(X[:, 0])
# # # plt.plot()
# # plt.show()


# # # We can `prove` that the ICA model applies by reverting the unmixing.
# # assert np.allclose(X, np.dot(S_, A_.T) + ica.mean_)

# ###############################################################################
# # Plot results
# # plt.figure()
# # # plt.pcolormesh(S.T)

# # models = [X, S, S_]
# # names = ['Observations (mixed signal)',
# #          'True Sources',
# #          'ICA recovered signals']

# # colors = ['steelblue', 'orange']

# # for ii, (model, name) in enumerate(zip(models, names), 1):
# #     plt.subplot(3, 1, ii)
# #     plt.title(name)
# #     for sig, color in zip(model.T, colors):
# #         plt.plot(sig, color=color)
# # plt.show()
