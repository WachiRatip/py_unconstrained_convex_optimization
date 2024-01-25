# Optimisations algorithms for solving unconstrained convex problems

## Table of contents
- [Introduction](#introduction)
- [How to setup the environment](#how-to-setup-the-environment)
- [How to run the tests (Pytest)](#how-to-run-the-tests-pytest)
- [Usage](#usage)
- [References](#references)
- [License](#license)

## Introduction
[Unconstrained convex problems](https://neos-guide.org/guide/types/#unconstrained) can be solved using gradient descent. However, gradient descent can be slow to converge. In order to speed up the convergence, gradient descent can be modified by adding a momentum term. This repository contains the implementation of the following algorithms:
- Gradient descent
- Gradient descent with momentum
- Nesterov accelerated gradient descent

The algorithms are implemented in Python and [Numpy](https://numpy.org/) and tested using [Pytest](https://docs.pytest.org/en/7.4.x/index.html).


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

## References
- [Gradient descent](https://en.wikipedia.org/wiki/Gradient_descent)
- [Gradient descent with momentum](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Momentum)


## License
[MIT](https://choosealicense.com/licenses/mit/)