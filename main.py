#!/usr/bin/env python3

import numpy as np
import os
import matplotlib.image as mpimg
import random
import scipy.spatial.distance as dist
import operator
import sys


class ConcurrentNetwok:
    def __init__(self, n):
        self.side = 6
        self.n = n
        self.x = np.zeros(self.side ** 2)
        self.y = np.zeros(self.n)
        self.w = np.zeros((self.side ** 2, self.n))
        self.b = 10
        self.victories = np.zeros((self.n))
        self.max_cup = 0.3

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
                s = np.linalg.norm(self.test_images[i].ravel())
                for y in range(0, self.side):
                    for x in range(0, self.side):
                        self.test_images[i, y, x] = self.test_images[i, y, x] / s
                i += 1

    def out_value(self, j):
        return dist.cosine(self.x, self.w[:, j])

    def init_w(self):
        # init w with random values
        for i in range(0, len(self.w)):
            for j in range(0, len(self.w[0])):
                self.w[i, j] = random.randint(0, 10)

        # normalize weight vectors
        for i in range(0, len(self.w)):
            s = np.linalg.norm(self.w[i])
            self.w[i] = list(map((lambda x: x / s), self.w[i]))

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

    def calc_neurons(self):
        for i in range(0, len(self.y)):
            self.y[i] = self.out_value(i)

    def train(self):
        self.init_w()

        break_flag = False
        while True:
            for image in self.test_images:
                self.x = image.ravel()

                self.calc_neurons()
                winner_pos = self.find_winner()

                # powerup synaptic connections
                self.w_new = self.w
                for i in range(0, len(self.w)):
                    self.w_new[i, winner_pos] = (self.w[i, winner_pos] + self.b * (self.x[i] - self.w[i, winner_pos])) /\
                            np.linalg.norm(list(map(operator.add, self.w[:, winner_pos], list(map((lambda x: self.b * x), list(map(operator.sub, self.x, self.w[:, winner_pos])))))))
                self.w = self.w_new

                maximum = 0
                for i in range(0, len(self.w)):
                    value = dist.euclidean(self.x[i], self.w[i, winner_pos])
                    if value >= maximum:
                        maximum = value

                if maximum <= self.max_cup:
                    break_flag = True

            if break_flag:
                break

    def play(self, image):
        self.x = np.array(image.ravel())
        self.calc_neurons()
        return self.find_winner()

    def recognize(self):
        for _, _, files in os.walk(self.rec_dir):
            for _file in files:
                f = mpimg.imread(self.rec_dir + _file)[:, :, 0]
                cluster = self.play(f)
                res = np.zeros((f.shape[0], f.shape[1], 3))
                res[:, :, 0] = f
                res[:, :, 1] = f
                res[:, :, 2] = f
                mpimg.imsave(self.out_dir + str(cluster) + "/" + _file, res)

    def run(self):
        self.load_test_images()
        self.train()
        self.recognize()


if __name__ == "__main__":
    net = ConcurrentNetwok(int(sys.argv[1]))
    net.run()
