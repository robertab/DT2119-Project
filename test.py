import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from pysndfile import sndio

from sklearn.decomposition import FastICA, PCA

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


###############################################################################
np.random.seed(0)
# Signals to separate
man_one = 'man_one_a.wav'
woman_one = 'woman_one_a.wav'
s1, sample2 = loadAudio(man_one)
s2, sample2 = loadAudio(woman_one)

# Concatenate both signals. Strip of the shortest
S = np.c_[s1[:s2.shape[0]], s2]
# Add a gaussian noise
S += 0.2 * np.random.normal(size=S.shape)
# Standarize the data
S = (S - S.mean()) / S.std(axis=0)
# Linear transformation matrix
A = np.array([[0.1, 1], [0.5, 2]])
# Linearly combine signals
X = np.dot(S, A.T)
# Compute ICA
ica = FastICA(n_components=2)

S_ = ica.fit_transform(X)
A_ = ica.mixing_

# We can `prove` that the ICA model applies by reverting the unmixing.
assert np.allclose(X, np.dot(S_, A_.T) + ica.mean_)

###############################################################################
# Plot results
plt.figure()

models = [X, S, S_]
names = ['Observations (mixed signal)',
         'True Sources',
         'ICA recovered signals']

colors = ['steelblue', 'orange']

for ii, (model, name) in enumerate(zip(models, names), 1):
    plt.subplot(3, 1, ii)
    plt.title(name)
    for sig, color in zip(model.T, colors):
        plt.plot(sig, color=color)

plt.show()
