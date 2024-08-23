Loading a Dataset in PyTorch
=============================

PyTorch offers two data primitives—`torch.utils.data.DataLoader` and `torch.utils.data.Dataset`— which 
facilitate the use of both pre-loaded datasets and custom data.

Pre-loaded Datasets
********************

The `Fashion-MNIST` dataset is an example of a pre-loaded curated dataset. It can be loaded using the following parameters:

- `root` specifies the path where the training or test data is stored.
- `train` indicates whether to load the training or test dataset.
- `download=True` will download the data from the internet if it's not available at the specified `root`.
- `transform` and `target_transform` define the transformations applied to the features and labels, respectively.

Load the training data:

.. code-block:: python
    :linenos:

    training_data = datasets.FashionMNIST(
        root="data",         # root directory of data
        train=True,          # load training dataset
        download=True,       # download the data if unvailable at the `root`
        transform=ToTensor() # transformations applied to the features and labels
    )

Load the testing data:

.. code-block:: python
    :linenos:

    training_data = datasets.FashionMNIST(
        root="data",         # root directory of data
        train=False,         # load testing dataset
        download=True,       # download the data if unvailable at the `root`
        transform=ToTensor() # transformations applied to the features and labels
    )