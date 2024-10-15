Loading a Dataset in PyTorch
=============================

.. admonition:: Overview
   :class: Overview

    * **Tutorial:** 20 min

        **Objectives:**
            #. Learn how to use pre-loaded data in PyTorch.
            #. Learn how to use custom data in PyTorch.
            #. Learn how to use custom dataloader in PyTorch.

PyTorch offers two data primitives—`torch.utils.data.DataLoader` and `torch.utils.data.Dataset`— which facilitate the use of both pre-loaded datasets and custom data. 
Dataset is an abstract class that represents a dataset. It defines how the data should be accessed and loaded, allowing users to specify how to retrieve 
individual data points. DataLoader wraps around a Dataset and provides iterable functionality, handling batching, shuffling, and loading data in 
parallel using multiprocessing.

.. list-table:: Differences Between Dataset and DataLoader
   :header-rows: 1

   * - Feature
     - Dataset
     - DataLoader
   * - Purpose
     - Defines how individual data samples are loaded.
     - Provides batch loading and efficient data iteration.
   * - Customizable
     - Users implement custom loading logic (e.g., loading images, preprocessing).
     - Handles batching, shuffling, and parallel data loading.
   * - Methods
     - Requires ``__len__()`` and ``__getitem__()`` methods.
     - Takes a Dataset as input and provides data batches.
   * - Functionality
     - Accesses individual data points (samples).
     - Loads data in batches and supports multiprocessing.
   * - Parallelization
     - Not parallelized (loads one item at a time).
     - Supports parallel data loading (``num_workers``).


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


Writing a custom DataLoader
****************************

The DataLoader works in conjunction with a Dataset class that defines how to access and preprocess data. 

1. Initialization (`__init__``): Loads the dataset from a file (e.g., CSV) or another source. Performs any necessary preprocessing, such as normalization or 
feature extraction.

2. Length (`__len__``): Returns the number of samples in the dataset, which helps the DataLoader know how many batches to create.

3. Item Retrieval (`__getitem__``): Retrieves a sample from the dataset given an index. This method is called by the DataLoader to get individual data points 
for batching.

We will use the Pima Indians Diabetes dataset for the demonstration. The Pima Indians Diabetes dataset is a popular dataset in the field of machine learning 
and statistics, particularly for those working on classification problems. 

#. **Source**: The dataset was created by the National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK) and is available in the UCI Machine Learning Repository.
#. **Purpose**: The dataset is used to predict the onset of diabetes within five years based on diagnostic measures.
#. **Features**: The dataset contains 768 samples, each with 8 features. 

The features are:

#. Pregnancies: Number of times pregnant.
#. Glucose: Plasma glucose concentration (mg/dL) a 2 hours in an oral glucose tolerance test.
#. Blood Pressure: Diastolic blood pressure (mm Hg) at the time of screening.
#. Skin Thickness: Triceps skinfold thickness (mm) measured at the back of the upper arm.
#. Insulin: 2-Hour serum insulin (mu U/ml).
#. BMI: Body mass index.
#. Diabetes Pedigree Function: A function that scores likelihood of diabetes based on family history.
#. Age: Age of the individual (years).

**Outcome**: Whether or not the individual has diabetes (1 for positive, 0 for negative).

.. code-block:: python
    :linenos:

    column_names = [ 'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness','Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

    class PimaDataset(Dataset):

        def __init__(self, csv_file):
            # Load the CSV file without header and assign column names
            self.data = pd.read_csv(csv_file, header=None, names=column_names)
            self.features = self.data.drop('Outcome', axis=1).values
            self.labels = self.data['Outcome'].values

            # Convert to PyTorch tensors
            self.features_tensor = torch.tensor(self.features, dtype=torch.float32)
            self.labels_tensor = torch.tensor(self.labels, dtype=torch.long)

            # Calculate mean and std
            self.mean = self.features_tensor.mean(dim=0)
            self.std = self.features_tensor.std(dim=0)

            # Normalize the features
            self.features_tensor = (self.features_tensor - self.mean) / self.std

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            feature = self.features_tensor[idx]
            label = self.labels_tensor[idx]
            return feature, label


.. admonition:: Exercise
   :class: todo

    Try the notebook *dataloader.ipynb*.

.. admonition:: Key Points
   :class: hint

    #. PyTorch provides pre-loaded datasets that can be used directly.
    #. Custom datasets can also be utilized in PyTorch.
    #. We can create custom dataloaders in PyTorch.