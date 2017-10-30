#!/usr/bin/env python
import matplotlib.image as mpimg


class Analyz:
    def __init__(self):
        self.src_dir = "original/"
        self.dst_dir = "res/"
        self.noize_levels = [10, 20, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]

    def run(self):
        for letter in ["e", "o", "sh"]:
            orig = mpimg.imread(self.src_dir + letter + ".png")[:, :, 0]
            for lvl in self.noize_levels:
                counter = 0
                for i in range(0, 100):
                    res = mpimg.imread(self.dst_dir + letter +
                                       str(lvl) + "_" + str(i) + ".png")[:, :, 0]
                    if (orig == res).all():
                        counter += 1
                print("{} at {} noize level: {}%".format(letter, lvl, counter))
                pass
            print("-------")


analyz = Analyz()
analyz.run()
