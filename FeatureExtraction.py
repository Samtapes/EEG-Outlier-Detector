import numpy as np
from scipy import stats
import scipy.signal
from tqdm import tqdm

def mean(x):
  return np.mean(x, axis=-1)

def mean_d(h1, h2):
  return (mean(h2)[0] - mean(h1)[0]).flatten()

def mean_q(q1, q2, q3, q4):
  v1 = np.concatenate((mean(q1), [mean(mean(q1))] ), axis=-1)
  v2 = mean(q2)
  v3 = np.concatenate((mean(q3), [mean(mean(q3))] ), axis=-1)
  v4 = mean(q4)
  return np.hstack([v1,v2,v3,v4,v1-v2,v1-v3,v1-v4,v2-v3,v2-v4,v3,v4])


def std(x):
  return np.std(x, axis=-1)

def std_d(h1,h2):
  return (std(h2)[0] - std(h1)[0]).flatten()


def ptp(x):
  return np.ptp(x, axis=-1)

def var(x):
  return np.var(x, axis=-1)


def minim(x):
  return np.min(x, axis=-1)

def minim_d(h1,h2):
  return (minim(h2)[0] - minim(h1)[0]).flatten()

def minim_q(q1,q2,q3, q4):
  v1 = np.concatenate((minim(q1), [minim(minim(q1))] ), axis=-1)
  v2 = minim(q2)
  v3 = np.concatenate((minim(q3), [minim(minim(q3))] ), axis=-1)
  v4 = minim(q4)
  return np.hstack([v1,v2,v3,v4,v1-v2,v1-v3,v1-v4,v2-v3,v2-v4,v3,v4])


def maxim(x):
  return np.max(x, axis=-1)

def maxim_d(h1,h2):
  return (maxim(h2)[0] - maxim(h1)[0]).flatten()

def maxim_q(q1,q2,q3, q4):
  v1 = np.concatenate((maxim(q1), [maxim(maxim(q1))] ), axis=-1)
  v2 = maxim(q2)
  v3 = np.concatenate((maxim(q3), [maxim(maxim(q3))] ), axis=-1)
  v4 = maxim(q4)
  return np.hstack([v1,v2,v3,v4,v1-v2,v1-v3,v1-v4,v2-v3,v2-v4,v3,v4])



def argminim(x):
  return np.argmin(x, axis=-1)

def argmaxim(x):
  return np.argmax(x, axis=-1)

def rms(x):
  return np.sqrt(np.mean(x**2, axis=-1))

def abs_diff_signal(x):
  return np.sum(np.abs(np.diff(x, axis=-1)), axis=-1)

def skewness(x):
  return stats.skew(x, axis=-1)

def kurtosis(x):
  return stats.kurtosis(x, axis=-1)

def moments(skw, krt):
  return np.append(skw, krt)

def covariance_matrix(x):
  covM = np.cov(x.T)
  return covM

def eigenvalues(covM):
  return np.linalg.eigvals(covM)

def logcov(covM):
  log_cov = scipy.linalg.logm(covM)
  indx = np.triu_indices(log_cov.shape[0])
  return np.abs(log_cov[indx])

def fft(matrix, period = 1., mains_f = 50., 
				filter_mains = True, filter_DC = True,
				normalise_signals = True,
				ntop = 10, get_power_spectrum = True):

  # Signal properties
  N   = matrix.shape[0] # number of samples
  T = period / N        # Sampling period
	
	# Scale all signals to interval [-1, 1] (if requested)
  if normalise_signals:
    matrix = -1 + 2 * (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))
	
  # Compute the (absolute values of the) FFT
  # Extract only the first half of each FFT vector, since all the information
  # is contained there (by construction the FFT returns a symmetric vector).
  fft_values = np.abs(scipy.fft.fft(matrix, axis = 0))[0:N//2] * 2 / N
	
  # Compute the corresponding frequencies of the FFT components
  freqs = np.linspace(0.0, 1.0 / (2.0 * T), N//2)

  # Remove DC component (if requested)
  if filter_DC:
    fft_values = fft_values[1:]
    freqs = freqs[1:]
		
  # Remove mains frequency component(s) (if requested)
  if filter_mains:
    indx = np.where(np.abs(freqs - mains_f) <= 1)
    fft_values = np.delete(fft_values, indx, axis = 0)
    freqs = np.delete(freqs, indx)
	
  # Extract top N frequencies for each signal
  indx = np.argsort(fft_values, axis = 0)[::-1]
  indx = indx[:ntop]
	
  ret = freqs[indx].flatten(order = 'F')

  if (get_power_spectrum):
    ret = np.hstack([ret, fft_values.flatten(order = 'F')])
  
  return ret


def calculate_features(x):
  h1, h2 = np.split(x, [int(x.shape[0]/2)])
  q1,q2,q3,q4 = np.split(x, [int(0.25*x.shape[0]),
                              int(0.50*x.shape[0]),
                              int(0.75*x.shape[0])
                            ])
  
  skw = skewness(x)
  krt = kurtosis(x)
  covM = covariance_matrix(x)

  
  output = np.concatenate((mean(x), mean_d(h1,h2), mean_q(q1,q2,q3,q4), std(x), std_d(h1,h2), ptp(x), var(x), minim(x), minim_d(h1,h2), minim_q(q1,q2,q3,q4), maxim(x),
                            maxim_d(h1,h2), maxim_q(q1,q2,q3,q4), argminim(x), argmaxim(x), rms(x), abs_diff_signal(x), skw, krt, moments(skw, krt), mean(covM),
                            eigenvalues(covM), fft(x)), axis=-1)

  # eigenvalues(covM), logcov(covM), 
  return output


if __name__ == '__main__':
  data = np.random.rand(2000, 22, 500)
  print(data.shape)

  feature_arr = []

  for i in tqdm(range(data.shape[0])):
    features = calculate_features(data[i])
    feature_arr.append(features)
  print(feature_arr.shape)
