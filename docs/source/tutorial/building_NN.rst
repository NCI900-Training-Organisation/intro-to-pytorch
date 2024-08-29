Building a Neural Network
=========================

.. admonition:: Overview
   :class: Overview

    * **Tutorial:** 30 min
    * **Exercises:** 15 min

        **Objectives:**
            #. Learn how to use pre-loaded data in PyTorch.
            #. Learn how to use custom data in PyTorch.

Neural networks are computational models inspired by the human brain, designed to recognize patterns and
make decisions based on data. They consist of interconnected layers of nodes, or "neurons," which process
and transform input information. Through training, neural networks learn to improve their accuracy in tasks like image recognition, language processing, and more.
Neural networks comprise of layers that perform operations on data.

Defining a Model
****************

A model can be defined as a sequence of layers. 
The first layer shoukd have have the  the correct number of input features.
we can have ha how many ever internal layers as we need.
if the number of interbal layers are big the computation time will be height
if the numbet of internal layer is small, the accuracy of the model maybe low.
usually we have an  activation function after each layer


The model expects rows of data with 8 variables (the first argument at the first layer set to 8)
The first hidden layer has 12 neurons, followed by a ReLU activation function
The second hidden layer has 8 neurons, followed by another ReLU activation function
The output layer has one neuron, followed by a sigmoid activation function

.. admonition:: Explanation
   :class: attention

   Each tensor will be of size [8, 3, 64, 64] -> [batch_size, channels, height, width].


.. admonition:: Exercise
   :class: todo

    Try the notebook *dataloader.ipynb*.

.. admonition:: Key Points
   :class: hint

    #. PyTorch provides pre-loaded datasets that can be used directly.
    #. Custom datasets can also be utilized in PyTorch.