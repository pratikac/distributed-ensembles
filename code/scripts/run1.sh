#!/bin/sh

python hyperoptim.py -c "python ptb.py -m ptbl -n 1 --lrs '[[4,20],[5,5],[6,1.25],[7,0.3],[8,0.08],[10,0.02]]' -B 10 --g0 0.001" -p '{"s":[42,43,44]}' -j 3 -r
