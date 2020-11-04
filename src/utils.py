import numpy as np
from tqdm import tqdm


def window_transform(series, win_len, stride):
    """
    Performs window transform of input timeseries
    """
    n_features = series.shape[1]
    k = (len(series) - win_len) // stride + 1
    assert k >= 1, "series length should be greater then window length"
    ts_array = np.zeros((k, win_len, n_features))
    for i in range(k):
        ts_array[i, :, :] = series[i * stride:i * stride + win_len, :]
    return ts_array


def calculate_correlation_matrix(window):
    n = window.shape[1]
    corr_matrix = np.zeros((n, n))
    for i in range(n):
        x_i = window[:,i]
        for j in range(n):
            x_j = window[:,j]
            corr_matrix[i,j] = np.inner(x_i, x_j)/len(x_i)
    return corr_matrix


def calculate_signature_matrix_dataset(X: np.array, lags=[10, 30, 60], stride=1, num_timesteps=5):
    max_lag = max(lags)
    X_w = window_transform(X, max_lag, stride)
    result = []
    for i in tqdm(range(len(X_w))):
        matrix_list = []
        for lag in sorted(lags, reverse=True):
            current_slice = X_w[i, -lag:, :]
            corr_matrix = calculate_correlation_matrix(current_slice)
            matrix_list.append(np.expand_dims(corr_matrix, axis=2))
        signature_matrix = np.concatenate(matrix_list, axis=2)
        result.append(np.expand_dims(signature_matrix, 0))

    matrix_num = len(result) - num_timesteps + 1

    input_matrix_series = []

    for j in range(matrix_num):
        matrix_series = np.expand_dims(np.concatenate(result[j: j + num_timesteps], axis=0), axis=0)
        input_matrix_series.append(matrix_series)
    input_matrix_series = np.concatenate(input_matrix_series, axis=0)
    target = input_matrix_series[:, -1, :, :]

    return input_matrix_series, target


def generate_harmonics(omega, t0, lmbda=0.3, length=2000):
    alpha = np.random.randint(0,2)
    t = np.arange(length)
    eps = np.random.normal(loc=0, scale=1, size=length)
    result = (1 - alpha)*np.sin((t-t0)/omega) + alpha*np.cos((t-t0)/omega) + eps*lmbda
    return result.reshape(-1,1)

def generate_dataset(n_features=30, seq_len=30000):
    t0= np.random.randint(50, 100, size=seq_len)
    omega = np.random.randint(40, 50, size=seq_len)
    dataset = [generate_harmonics(omega[i], t0[i], lmbda=0.1, length=seq_len) for i in range(n_features)]
    dataset = np.concatenate(dataset, axis=1)
    return dataset