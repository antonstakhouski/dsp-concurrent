#!/usr/bin/env python
import os


class Analyz:
    def __init__(self):
        self.src_dir = "original/"
        self.dst_dir = "res/"
        self.letters = []
        self.noize_levels = [10, 20, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]

    def get_letters(self):
        for _, _, files in os.walk(self.src_dir):
            for _file in files:
                self.letters.append(_file.replace(".png", ""))

    def run(self):
        self.get_letters()
        res = dict(map((lambda x: [x, 0]), self.letters))
        for dirs, _, files in os.walk(self.dst_dir):
            letters = dict(map((lambda x: [x, 0]), self.letters))

            files_cnt = 0
            for _file in files:
                files_cnt += 1
                letters[_file[0]] += 1

            for k in letters:
                if files_cnt and res[k] < (letters[k] / files_cnt):
                    res[k] = letters[k] / files_cnt
        for k in letters:
            res[k] *= 100
        print(res)


analyz = Analyz()
analyz.run()
