#!/bin/sh

# wrn desgd cifar100
python hyperoptim.py -c "python ensemble.py -m wrn2810 --dataset cifar100 --lrs '[[1,0.1],[2,0.02],[3,0.004],[5,0.001]]' -B 5 --augment -d 0.25 -n 3" -p '{"s":[44]}' -j 1 -r --dist
python hyperoptim.py -c "python ensemble.py -m wrn2810 --dataset cifar100 --lrs '[[1,0.1],[2,0.02],[3,0.004],[5,0.001]]' -B 5 --augment -d 0.25 -n 3 --frac 0.5" -p '{"s":[43]}' -j 1 -r --dist
python hyperoptim.py -c "python ensemble.py -m wrn2810 --dataset cifar100 --lrs '[[1,0.1],[2,0.02],[3,0.004],[5,0.001]]' -B 5 --augment -d 0.25 -n 6 --frac 0.25" -p '{"s":[43]}' -j 1 -r --dist

# wrn esgd  cifar100
python hyperoptim.py -c "python ensemble.py -m wrn2810 --dataset cifar100 --lrs '[[2,0.1],[4,0.02],[6,0.004],[10,0.001]]' -B 10 --augment -d 0.25 -n 1" -p '{"s":[42], "frac":[0.5,0.25]}' -j 3 -r

# wrn desgd  cifar10
python hyperoptim.py -c "python ensemble.py -m wrn2810 --dataset cifar10 --lrs '[[1,0.1],[2,0.02],[3,0.004],[5,0.001]]' -B 5 --augment -d 0.25 -n 3" -p '{"s":[44]}' -j 1 -r --dist
python hyperoptim.py -c "python ensemble.py -m wrn2810 --dataset cifar10 --lrs '[[1,0.1],[2,0.02],[3,0.004],[5,0.001]]' -B 5 --augment -d 0.25 -n 3 --frac 0.5" -p '{"s":[43]}' -j 1 -r --dist
python hyperoptim.py -c "python ensemble.py -m wrn2810 --dataset cifar10 --lrs '[[1,0.1],[2,0.02],[3,0.004],[5,0.001]]' -B 5 --augment -d 0.25 -n 6 --frac 0.25" -p '{"s":[43]}' -j 1 -r --dist

# wrn esgd cifar10
python hyperoptim.py -c "python ensemble.py -m wrn2810 --dataset cifar10 --lrs '[[2,0.1],[4,0.02],[6,0.004],[10,0.001]]' -B 10 --augment -d 0.25 -n 1" -p '{"s":[42], "frac":[1.0,0.5,0.25]}' -j 3 -r