# Optimisations algorithms for solving unconstrained convex problems
## Introduction
This repository contains the implementation of the following algorithms:
- Gradient descent
- Gradient descent with momentum
- Nesterov accelerated gradient descent

## How to setup the environment
refer to setup.ps1 for Windows users 
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
and setup.sh for Linux/MacOS users
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## How to run the tests (Pytest)
The tests can be run with the following command:
```
python -m pytest
```

## Usage
The algorithms are implemented in the `optimisation_algorithms.py` file. The `main.py` file contains an example of usage of the algorithms. The `main.py` file can be run with the following command:
```
python main.py
```