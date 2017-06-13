#!/bin/sh

python hyperoptim.py -c "python ensemble.py -m wrn2810 --dataset cifar100 --lrs '[[1,0.1],[2,0.02],[3,0.004],[5,0.001]]' -B 5 --augment -d 0.25 -n 6 --frac 0.25 -b 64" -p '{"s":[42]}' -j 1 -r --dist
python hyperoptim.py -c "python ensemble.py -m wrn2810 --dataset cifar10 --lrs '[[1,0.1],[2,0.02],[3,0.004],[5,0.001]]' -B 5 --augment -d 0.25 -n 6 --frac 0.25 -b 64" -p '{"s":[42]}' -j 1 -r --dist

python hyperoptim.py -c "python ensemble.py -m wrn168 --dataset svhn --lrs '[[1,0.1],[2,0.02],[3,0.004],[5,0.001]]' -B 5 -d 0.4 -n 3" -p '{"s":[42]}' -j 1 -r --dist
python hyperoptim.py -c "python ensemble.py -m wrn168 --dataset svhn --lrs '[[1,0.1],[2,0.02],[3,0.004],[5,0.001]]' -B 6 -d 0.4 -n 1" -p '{"s":[42]}' -j 1 -r --dist
