#!/usr/bin/env python3

import numpy as np
import os
import matplotlib.image as mpimg
import random
import scipy.spatial.distance as dist
import operator
import sys


class ConcurrentNetwok:
    def __init__(self, n=5):
        self.side = 6
        self.n = n
        self.x = np.zeros(self.side ** 2)
        self.y = np.zeros(self.n)
        self.w = np.zeros((self.side ** 2, self.n))
        self.b = 0.4
        self.victories = np.zeros((self.n))
        self.max_cup = 0.8

        self.m = 5
        self.test_images = np.zeros((self.m, self.side ** 2))
        self.test_dir = "original/"
        self.rec_dir = "noized/"
        self.out_dir = "res/"

    def load_test_images(self):
        for _, _, files in os.walk(self.test_dir):
            i = 0
            for _file in files:
                f = mpimg.imread(self.test_dir + _file)[:, :, 0]
                self.test_images[i] = f.ravel()

                # normalize image vectors
                self.normalize_input(self.test_images[i])
                i += 1

    def out_value(self, j):
        return dist.cosine(self.x, self.w[:, j])

    def init_w(self):
        # init w with random values
        for i in range(0, len(self.w)):
            for j in range(0, len(self.w[0])):
                self.w[i, j] = random.randint(0, 10)

        # normalize weight vectors
        array = np.zeros((len(self.w[0])))
        for i in range(0, len(self.w[0])):
            array[i] += sum(self.w[:, i])
        for i in range(0, len(self.w)):
            for j in range(0, len(self.w[0])):
                self.w[i, j] = self.w[i, j] / array[j]

    def normalize_input(self, array):
        # normalize image vectors
        s = sum(array)
        for i in range(0, len(array)):
            array[i] = array[i] / s

    def find_winner(self):
        minimum = 0
        min_pos = 0
        for j in range(0, len(self.y)):
            value = self.victories[j] * dist.euclidean(self.x, self.w[:, j])
            if j == 0:
                minimum = value
            if value <= minimum:
                minimum = value
                min_pos = j
        self.victories[min_pos] += 1
        return min_pos

    def find_cluster(self):
        minimum = 0
        min_pos = 0
        for j in range(0, len(self.y)):
            value = dist.euclidean(self.x, self.w[:, j])
            if j == 0:
                minimum = value
            if value <= minimum:
                minimum = value
                min_pos = j
        return min_pos

    def calc_neurons(self):
        for i in range(0, len(self.y)):
            self.y[i] = self.out_value(i)

    def train(self):
        self.init_w()

        while True:
            maximum = 0
            for image in self.test_images:
                self.x = image.copy()

                self.calc_neurons()
                winner_pos = self.find_winner()

                # powerup synaptic connections
                self.w_new = self.w.copy()
                for i in range(0, len(self.w)):
                    self.w_new[i, winner_pos] = (self.w[i, winner_pos] + self.b * (self.x[i] - self.w[i, winner_pos])) /\
                            np.linalg.norm(list(map(operator.add, self.w[:, winner_pos], list(map((lambda x: self.b * x), list(map(operator.sub, self.x, self.w[:, winner_pos])))))))
                self.w = self.w_new.copy()

                value = dist.euclidean(self.x, self.w[:, winner_pos])
                if value >= maximum:
                    maximum = value

            if maximum <= self.max_cup:
                break

    def play(self, image):
        self.x = np.array(image)
        self.normalize_input(self.x)
        self.calc_neurons()
        return self.find_cluster()

    def recognize(self):
        for _, _, files in os.walk(self.rec_dir):
            for _file in files:
                f = mpimg.imread(self.rec_dir + _file)[:, :, 0]
                cluster = self.play(f.ravel())
                res = np.zeros((f.shape[0], f.shape[1], 3))
                res[:, :, 0] = f
                res[:, :, 1] = f
                res[:, :, 2] = f
                mpimg.imsave(self.out_dir + str(cluster) + "/" + _file, res)

    def run(self):
        self.load_test_images()
        self.train()
        self.victories = np.zeros((self.n))
        self.recognize()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        net = ConcurrentNetwok(int(sys.argv[1]))
    else:
        net = ConcurrentNetwok()
    net.run()
