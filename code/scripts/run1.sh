#!/bin/sh

python hyperoptim.py -c "python ensemble.py -n 3 -m allcnn --dataset cifar10 --augment --lrs '[[2,0.1],[4,0.02],[6,0.004],[10,0.001]]' -B 10 --frac 0.5 -d 0.0" -p '{"s":[44]}' -j 1 --dist -r

python hyperoptim.py -c "python ensemble.py -n 6 -m allcnn --dataset cifar10 --augment --lrs '[[2,0.1],[4,0.02],[6,0.004],[10,0.001]]' -B 10 --frac 0.25 -d 0.0" -p '{"s":[44]}' -j 1 --dist -r

