# CSE598 Project: Improving Performance for Distributed SGD using Ray

- `MNIST/mnist.py` runs the MNIST model
- `CIFAR10/cifar10.py` runs the CIFAR-10 model
- `Fashion-MNIST/fashion_mnist.py` runs the Fashion-MNIST model

The programs allow the user to specify the following parameters:
- batch size
- number of workers
- number of workers that send updates to parameter server in each iteration
- learning rate
- staleness tolerance of gradients among workers
- interval for workers to pull latest weights from parameter server
