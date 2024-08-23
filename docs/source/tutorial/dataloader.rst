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

Custom Dataset
***************

What if working with a custom dataset? To illustrate this, we will download a dataset and set it up for
use in PyTorch training.

.. admonition:: Explanation
   :class: attention

   The data used for this demonstration is relatively *clean*. In a practical use case, significant 
   time will likely be spent on cleaning and preparing the data.

The data:

    #. There are **3 classes**: pizza, steak, and sushi.
    #. The data is split into *train* and *test* datasets.
    #. Both *train* and *test* datasets are further organized into 3 directories, each corresponding to one of the classes.

.. admonition:: Explanation
   :class: attention

   In practice, it is our responsibility to divide the data into training and testing sets and 
   further categorize it into different classes.

Transformation on the data
**************************************

Transform functions in the PyTorch library simplify the application of various data enhancement/manipulation techniques 
to your input data. These functions enable you to apply multiple changes simultaneously.


.. code-block:: python
    :linenos:

    data_transform = transforms.Compose([
        transforms.Resize(size=(64, 64)), # Resize the images to 64x64
        transforms.RandomHorizontalFlip(p=0.5), # Horizontally flip image with a 0.5 probability
        transforms.ToTensor() # convert to tensor of shape (C x H x W) in the range [0.0, 1.0] 
])

.. admonition:: Explanation
   :class: attention

    A Tensor Image is a tensor with a shape of (C, H, W), where C represents the number of channels, 
    and H and W denote the image's height and width. Typically, an image consists of three color 
    channels: red, green, and blue (RGB).

    **Note**: PyTorch uses the [C, H, W] format by default, while Matplotlib uses [H, W, C].

Loading Image Data Using ImageFolder
***********************************

`ImageFolder` is a generic data loader where images are expected to be organized into separate directories,
each corresponding to a different class.

.. code-block:: python
    :linenos:

    train_data = datasets.ImageFolder(root=train_dir, # root of the train images
                    transform=data_transform, # transforms to perform on each image
                    target_transform=None # transforms to perform on labels (eg: 1-hot encoding)
                ) 

    test_data = datasets.ImageFolder(root=test_dir, # root of the test images
                                 transform=data_transform # transforms to perform on each image
                )


DataLoader
**********

In PyTorch, `DataLoader` is a built-in class that offers an efficient and flexible method for loading 
data into a model for training or inference. It is especially beneficial for managing large datasets that 
may not fit into memory and for carrying out data augmentation and preprocessing. 
Data loader combines a dataset and a sampler, and provides an iterable over the given dataset.


.. code-block:: python
    :linenos:

    from torch.utils.data import DataLoader

    train_dataloader = DataLoader(dataset=train_data, # dataset from which to load the data
                              batch_size=8, # samples per batch to load
                              num_workers=1, # subprocesses to use for data loading
                              shuffle=True) # reshuffled the data at every epoch

    test_dataloader = DataLoader(dataset=test_data, # dataset from which to load the data
                             batch_size=8, # samples per batch to load 
                             num_workers=1, # subprocesses to use for data loading
                             shuffle=False) # don't shuffle testing data

.. admonition:: Explanation
   :class: attention

   Each tensor will be of size [8, 3, 64, 64] -> [batch_size, channels, height, width].