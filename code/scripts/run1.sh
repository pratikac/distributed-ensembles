#!/bin/sh

python hyperoptim.py -c "python ensemble.py -n 3 -m allcnn --dataset cifar10 --augment --lrs '[[60,0.1],[120,0.02],[160,0.004],[200,0.001]]' -B 200 --frac 0.5 -L 0 -d 0.0" -p '{"s":[42,43,44]}' -j 1 --dist -r

python hyperoptim.py -c "python ensemble.py -n 6 -m allcnn --dataset cifar10 --augment --lrs '[[60,0.1],[120,0.02],[160,0.004],[200,0.001]]' -B 200 --frac 0.25 -L 0 -d 0.0" -p '{"s":[42,43,44]}' -j 1 --dist -r
