#!/bin/bash
icc -O2 -qopt-report=1 -qopt-report-phase=vec -g -Wall -march=native -fopenmp  -shared -std=c++14 -fPIC `python2 -m pybind11 --includes` laplacian_pybind.cpp -o laplacian_pybind`python2-config --extension-suffix` -I/home/local/USHERBROOKE/gaga2313/src/pybind11/include/pybind11 -I/usr/include/python2.7 -lpython2.7 -I/usr/include/eigen3

