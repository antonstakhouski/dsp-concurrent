#!/usr/bin/env python3

import numpy as np
import os
import matplotlib.image as mpimg
from functools import reduce
import random
import scipy.spatial.distance as dist


class ConcurrentNetwok:
    def __init__(self):
        self.side = 6
        self.n = 3
        self.entry_layer = np.zeros(self.side ** 2)
        self.neurons = np.zeros(self.n)
        self.weights = np.zeros((self.side ** 2, self.n))
        self.beta = 10

        self.m = 5
        self.test_images = np.zeros((self.m, self.side, self.side))
        self.test_dir = "original/"
        self.rec_dir = "noized/"
        self.out_dir = "res/"

    def load_test_images(self):
        for _, _, files in os.walk(self.test_dir):
            i = 0
            for _file in files:
                f = mpimg.imread(self.test_dir + _file)[:, :, 0]
                self.test_images[i] = f

                # normalize image vectors
                s = np.sqrt(reduce((lambda x, y: x + y), self.test_images[i].ravel()))
                for y in range(0, self.side):
                    for x in range(0, self.side):
                        self.test_images[i, y, x] = self.test_images[i, y, x] / s
                i += 1

    def out_value(self, j):
        return dist.cosine(self.entry_layer, self.weights[:, j])

    def init_weights(self):
        # init weights with random values
        for i in range(0, len(self.weights)):
            for j in range(0, len(self.weights[0])):
                self.weights[i, j] = random.randint(0, 10)

        # normalize weight vectors
        for i in range(0, len(self.weights)):
            s = np.sqrt(reduce((lambda x, y: x + (y ** 2)), self.weights[i], 0))
            self.weights[i] = list(map((lambda x: x / s), self.weights[i]))

    def train(self):
        self.init_weights()

        for image in self.test_images:
            self.entry_layer = image.ravel()

            # find out values
            for i in range(0, len(self.neurons)):
                self.neurons[i] = self.out_value(i)

            # find neuron - winner
            maximum = 0
            max_pos = 0
            for i in range(0, len(self.neurons)):
                if self.neurons[i] >= maximum:
                    maximum = self.neurons[i]
                    max_pos = i

            # powerup synaptic connections
            for i in range(0, len(self.weights)):
                self.weights[i, max_pos] = self.weights[i, max_pos] +\
                                           self.beta * (self.entry_layer[i] - self.weights[i, max_pos])

    def play(self, image):
        neurons_t = np.array(image.ravel())
        neurons_t1 = np.zeros(self.n)

        while True:
            for i in range(0, self.n):
                value = 0
                for j in range(0, self.n):
                    value += self.weights[i][j] * neurons_t[j]
                neurons_t1[i] = self.activate(value)

            converged = True
            for i in range(0, self.n):
                if neurons_t[i] != neurons_t1[i]:
                    converged = False
                    break
            if converged:
                break

            neurons_t = neurons_t1

        res = np.zeros((self.side, self.side, 3))
        for i in range(0, self.side):
            for j in range(0, self.side):
                value = neurons_t1[self.side * i + j]
                if value < 0:
                    value = 0
                res[i, j] = value
        return res

    def recognize(self):
        for _, _, files in os.walk(self.rec_dir):
            for _file in files:
                f = mpimg.imread(self.rec_dir + _file)[:, :, 0]
                mpimg.imsave(self.out_dir + _file, self.play(f))

    def run(self):
        self.load_test_images()
        self.train()
        #  self.recognize()


net = ConcurrentNetwok()
net.run()
