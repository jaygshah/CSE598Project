# CSE598 Project: Improving Performance for Distributed SGD using Ray

- `MNIST/mnist.py` runs the MNIST model
- `CIFAR10/cifar10.py` runs the CIFAR-10 model
- `Fashion-MNIST/fashion_mnist.py` runs the Fashion-MNIST model

The programs allow the user to specify the following parameters:
- batch size (`batch_size`)
- number of workers (`num_workers`)
- number of workers that send updates to the parameter server in each iteration
  (`num_workers_ps_update`)
- learning rate (`lr`)
- staleness tolerance of gradients among workers (`staleness_tolerance`)
- interval for workers to pull latest weights from parameter server
  (`pull_weights_interval_rule`)
