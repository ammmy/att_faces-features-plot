# simple GA model

import numpy as np

class GA():
    def __init__(self, feature_size, n_g=100, n_g_next=50, mut_rate=0.1, target_class=0):
        n_g_mut = n_g - n_g_next - n_g_next / 2
        n_mut = int(feature_size * mut_rate)
        label = np.zeros(n_g) + target_class
        g = np.random.rand(n_g, feature_size)
        self.feature_size, self.n_g, self.n_g_next, self.mut_rate, self.target_class = feature_size, n_g, n_g_next, mut_rate, target_class
        self.n_g_mut, self.n_mut, self.label, self.g = n_g_mut, n_mut, label, g

    def evaluation(self, scores):
        self.scores = scores

    def selection(self):
        self.g[:self.n_g_next] = self.g[self.scores.argsort()[::-1][:self.n_g_next]]

    def crossover(self):
        pairs = np.arange(self.n_g_next)
        np.random.shuffle(pairs)
        pairs = pairs.reshape((2, -1))
        self.g[self.n_g_next:self.n_g_next + self.n_g_next / 2] = self.g[pairs[0]]
        for i, p in enumerate(pairs[1]):
            ridx = np.arange(self.feature_size)
            np.random.shuffle(ridx)
            ridx = ridx.reshape((2, -1))[0]
            self.g[self.n_g_next + i][ridx] = self.g[p][ridx]

    def mutation(self):
        targets = np.arange(self.n_g - self.n_g_mut)
        np.random.shuffle(targets)
        targets = targets[:self.n_g_mut]
        for i, t in enumerate(targets):
            ridx = np.arange(self.feature_size)
            np.random.shuffle(ridx)
            ridx = ridx[:self.n_mut]
            rval = np.random.rand(ridx.shape[0])
            self.g[self.n_g - self.n_g_mut + i][ridx] = rval

    def step(self, scores):
        self.evaluation(scores)
        self.selection()
        self.crossover()
        self.mutation()

