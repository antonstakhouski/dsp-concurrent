#!/bin/bash
rm -f noized/*
./make_noize.py
./run.sh 5
./stat.py
