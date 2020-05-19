# SDMM

Library for running Gaussian mixture models and spatio-directional Mixture Models using the [Enoki library](https://github.com/mitsuba-renderer/enoki).
It is a complete rewrite from the original Eigen code.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

This library depends on other projects, and they have to be available first.

```
git submodule update --init --recursive
sudo apt install libboost-all-dev libspdlog-dev libfmt-dev
```

### Compiling

```
mkdir build
cd build
cmake ..
make -j
```

## Running the tests

TODO: Explain how to run the tests for this system.

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Authors

**Copyright (c) 2020 by Ana Dodik.** 

## License

This project is licensed under the (TODO) Licence - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments


I would like to thank my advisors, Thomas Müller, Cengiz Öztireli, and Marios Papas, whose insightful comments and ideas significantly improved the quality of the work.
