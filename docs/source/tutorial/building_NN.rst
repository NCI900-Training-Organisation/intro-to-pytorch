Building a Neural Network
=========================

.. admonition:: Overview
   :class: Overview

    * **Tutorial:** 30 min
    * **Exercises:** 30 min

        **Objectives:**
            #. Learn how to implement a neural network in PyTorch.
            #. Learn the differ modules that go into building a neural network in PyTorch.

Dataset
*******
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

Defining the Model
*******************

When designing the model, we have to keep the following points in mind:

#. The input features in the input layer must match the input features in the dataset.
#. A high number of layers can increase computation time, while too few layers may result in poor predictions.
#. Each layer should be followed by an activation function.

In this example, we will use a 3-layer neural network:

#. The input layer expects 8 features.
#. The first hidden layer has 12 neurons, followed by a ReLU activation function.
#. The second hidden layer has 8 neurons, followed by another ReLU activation function.
#. The output layer has one neuron, followed by a sigmoid activation function.

The **sigmoid function** outputs values between 0 and 1, which is exactly what we need.

Sequential vs. Class-Based Models
***********************************

In PyTorch, neural networks can be defined using different approaches, and two common ones are the `Sequential` model and the `class-based model`.

The `Sequential` model is a simple, linear stack of layers where each layer has a single input and output. It is useful for straightforward feedforward 
networks where layers are applied in a sequential order.

**Characteristics:**

#. **Ease of Use:** It is easier to use for simple architectures where layers are applied in a linear fashion.
#. **Defined Using:** `torch.nn.Sequential`.

**Limitations:**

#. **Flexibility:** Limited flexibility for more complex architectures (e.g., networks with multiple inputs/outputs, shared layers, or non-sequential data flow).
#. **Custom Behavior:** Difficult to implement custom forward passes or dynamic architectures.


The `class-based`` model allows you to define a network by subclassing `torch.nn.Module`. This approach provides greater flexibility and control, making it 
suitable for complex models and custom behaviors.

**Characteristics:**

#. **Flexibility:** Offers full control over the network architecture, including complex data flows, multiple inputs/outputs, and custom forward methods.
#. **Defined Using:** Subclass of `torch.nn.Module`.


**Advantages:**

#. **Custom Forward Pass:** You can define complex forward passes and control data flow through the network
#. **Dynamic Behavior:** Allows for dynamic computations, such as conditional layers or operations.


Choosing between the two depends on the complexity of the network you need to build and your specific requirements for flexibility and control.

Loss function
*************

Each model needs a loss function. In this case we will use the Binary Cross-Entropy (BCE) Loss. It Measures the performance of a classification model whose 
output is a probability value between 0 and 1. It calculates the difference between the predicted probabilities and the actual binary labels (0 or 1) and
penalizes the model more when the predictions are further from the true labels.

.. math::

   BCELoss(y', y) = −[ylog(y')+(1 − y)log(1 − y')]

Where, y' is the predicted output and y is the actual otput.

Optmizer
*********

Optimizer's main role is to update the model's parameters based on the gradients computed during backpropagation.

1. **Parameter Updates**: Optimizers adjust the weights and biases of the neural network to reduce the loss. This involves applying algorithms that modify 
the parameters to minimize the difference between the predicted outputs and the actual targets.

2. **Learning Rate Management**: Most optimizers include mechanisms to adjust the learning rate, either statically or dynamically, to control how large 
the parameter updates are.

In this example we use an optimizer called Adaptive Moment Estimation (Adam). This computes an adaptive learning rates for each parameter by considering 
both the mean and the variance of the gradients.

Training the Model
*******************

Training a neural network involves epochs and batches, which define how data is fed to the model:

#. **Epoch:** A full pass through the entire training dataset.
#. **Batch:** A subset of samples processed at a time, with gradient descent performed after each batch.

In practice, the dataset is divided into batches, and each batch is processed sequentially in a training loop. Completing all batches constitutes one epoch. 
The process is repeated for multiple epochs to refine the model.

Batch size is constrained by system memory (GPU memory), and computational demands scale with batch size. More epochs and batches lead to better model 
performance but increase training time. The optimal number of epochs and batch size is often determined through experimentation.

1. **optimizer.zero_grad()**: During training, gradients accumulate by default in PyTorch. This means that if you don't clear them, gradients from multiple 
backward passes (from different batches) will be added together, which can lead to incorrect updates to the model parameters. By calling optimizer.zero_grad(),
you ensure that gradients from previous steps are reset to zero, preventing them from affecting the current update.

2. **loss.backward()**:  Calculates the gradients of the loss with respect to each parameter of the model. This is done using backpropagation, a key algorithm 
for training neural networks.

3. **optimizer.step()**: Used to update the model's parameters based on the gradients computed during during the backward pass (loss.backward()).

Model Evaluation
******************

Ideally, we should split the data into separate training and testing datasets, or use a distinct dataset for evaluation. For simplicity, we are testing the 
model on the same data used for training.


.. admonition:: Exercise
   :class: todo

    Try the notebook *building_NN.ipynb*.

.. admonition:: Key Points
   :class: hint

   #. PyTorch offers *Sequential* models for simple linear stacks and *Class-based* models for complex, customizable architectures.
   #. Training involves epochs and batches with functions like `optimizer.zero_grad()`, `loss.backward()`, and `optimizer.step()`
   #. Ideally, data should be split into training and testing sets.