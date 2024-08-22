Tensor
------

.. admonition:: Overview
   :class: Overview

    * **Tutorial:** 15 min
    * **Exercises:** 15 min

        **Objectives:**
            #. Learn about tensors.
            #. Learn the differences between a tensor and numpy array.




Tensors are specialized data structures used in PyTorch to represent model inputs, outputs, and parameters. While they are conceptually similar to 
arrays and matrices, they offer additional features such as support for hardware accelerators like GPUs and automatic differentiation.

Creating a Tensor
*****************

A tensor can be created in multiple ways:

1. Directly from data

.. code-block:: python
    :linenos:

    data = [[1, 2],[3, 4]]
    x_tensor= torch.tensor(data)

2. From Numpy

.. code-block:: python
    :linenos:

    x_np = np.array(data)
    x_tensor = torch.from_numpy(x_np)

3. From another Tensor

.. code-block:: python
    :linenos:

    x_tensor = torch.ones_like(x_data)
    y_tensor = torch.rand_like(x_data, dtype=torch.float) 


.. admonition:: Explanation
   :class: attention

   **torch.rand_like()** returns a tensor with the same size as input that but filled with random numbers from the interval [0,1).


Tensor Attributes
*****************

.. code-block:: python
    :linenos:

    print(f"Shape of tensor: {y_tensor.shape}")
    print(f"Datatype of tensor: {y_tensor.dtype}")
    print(f"Device tensor is stored on: {y_tensor.device}")


*Automatic differentiation* is one of the main characteristics that differentiate numpy arrays from tensors. 

.. math::

    z = x * w + b

    g1 = \frac{\partial loss}{\partial w} 

    g2 = \frac{\partial loss}{\partial b} 

.. code-block:: python
    :linenos:

    x = torch.ones(5)  # input tensor
    y = torch.zeros(3)  # expected output

    w = torch.randn(5, 3, requires_grad=True)
    b = torch.randn(3, requires_grad=True)

    z = torch.matmul(x, w)+b

    loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)


.. admonition:: Exercise
   :class: todo

    Try the notebook *tensors.ipynb*.

.. admonition:: Key Points
   :class: hint

    #. Tensors can be created in different 




