import numpy as np

class NLMSFilter:
    def __init__(self, lr=0.01, sample_size=16, epsilon=1e-8):
        self.lr = lr
        self.sample_size = sample_size
        self.epsilon = epsilon

    def filter(self, input_signal, desired_signal):
        """
        Perform NLMS adaptive filtering on a 1D signal.
        """
        N = len(input_signal)
        weights = np.zeros(self.sample_size)
        output = np.zeros(N)

        for n in range(self.sample_size, N):
            x = input_signal[n - self.sample_size:n][::-1]
            y = np.dot(weights, x)
            output[n] = y
            e = desired_signal[n] - y
            norm = np.dot(x, x) + self.epsilon
            weights += (self.lr * e * x) / norm

        return output
