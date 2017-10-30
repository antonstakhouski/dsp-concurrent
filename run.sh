#!/bin/bash

rm -rf res/*

for (( i = 0; i < $1; i++ ))
do
    mkdir "res/$i"
done

python3 main.py $1
