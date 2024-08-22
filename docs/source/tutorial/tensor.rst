Tensor
------

.. admonition:: Overview
   :class: Overview

    * **Tutorial:** 15 min
    * **Exercises:** 15 min

        **Objectives:**
            #. Learn about tensors.
            #. Learn the differences between a tensor and NumPy array.




Tensors are specialized data structures used in PyTorch to represent model inputs, outputs, and parameters. While they are conceptually similar to 
arrays and matrices, they offer additional features such as support for hardware accelerators like GPUs and 
automatic differentiation.

Creating a Tensor
*****************

A tensor can be created in multiple ways:

1. Directly from data

.. code-block:: python
    :linenos:

    data = [[1, 2],[3, 4]]
    x_tensor= torch.tensor(data)

2. From NumPy

.. code-block:: python
    :linenos:

    x_np = np.array(data)
    x_tensor = torch.from_NumPy(x_np)

3. From another Tensor

.. code-block:: python
    :linenos:

    x_tensor = torch.ones_like(x_data)
    y_tensor = torch.rand_like(x_data, dtype=torch.float) 


.. admonition:: Explanation
   :class: attention

   **torch.rand_like()** returns a tensor with the same size as input that but filled with random numbers 
   from the interval [0,1).


Operations on Tensors
*********************

Tensors can perform almost all operations a NumPy array can perform

1.  indexing and slicing

.. code-block:: python
    :linenos:

    tensor = torch.ones(4, 4)
    print(f"First row: {tensor[0]}")
    print(f"First column: {tensor[:, 0]}")
    print(f"Last column: {tensor[..., -1]}")
    tensor[:,1] = 0
    print(tensor)

2. Concatenate multiple tensors

.. code-block:: python
    :linenos:

    t_cat = torch.cat([tensor, tensor, tensor], dim=1)
    print(t_cat)


3. Arithmetic Operations

.. code-block:: python
    :linenos:

    x = torch.ones(4, 4)

    # Transpose
    x_t = tensor.T

    # Matrix Multiplication
    y1 = tensor @ tensor.T
    y2 = tensor.matmul(tensor.T)

    y3 = torch.rand_like(y1)
    torch.matmul(tensor, tensor.T, out=y3)


    # Element-wise multiplication
    z1 = tensor * tensor
    z2 = tensor.mul(tensor)

    z3 = torch.rand_like(tensor)
    torch.mul(tensor, tensor, out=z3)

3. In-place Operations

.. code-block:: python
    :linenos:

    x = torch.ones(4, 4)

    # Transpose
    x.t_()

    # Copy
    y = torch.rand_like(x)
    x.copy_(y)

NumPy and Tensor
****************

Tensors on the **CPU** and NumPy arrays can share memory locations, so modifying one will also affect 
the other.

.. code-block:: python
    :linenos:

    x_t = torch.ones(5) 
    x_n = t.numpy() # tensor to numpy
    print(f"t: {x_t}")
    print(f"n: {x_n}")

    x_t.add_(1)

    print(f"t: {x_t}")
    print(f"n: {x_n}")

    y_n = np.ones(5)
    y_t = torch.from_numpy(n) # numpy to tensor

    np.add(n, 1, out=n)

    print(f"t: {t}")
    print(f"n: {n}")


Moving Tensor to GPU
*********************



Tensor Attributes
*****************

.. code-block:: python
    :linenos:

    print(f"Shape of tensor: {y_tensor.shape}")
    print(f"Datatype of tensor: {y_tensor.dtype}")
    print(f"Device tensor is stored on: {y_tensor.device}")


*Automatic differentiation* is a key feature that distinguishes tensors from NumPy arrays. This capability
is particularly useful in neural networks, where model weights are adjusted during backpropagation based 
on the gradient of the loss function with respect to each parameter. Tensors support automatic gradient 
computation for any computational graph. For example, consider the computational graph of a one-layer 
neural network:


.. image:: ../figs/loss.png

In this context, **w** and **b** are the parameters that need to be optimized. Therefore, we compute 
the gradients of the loss function with respect to these variables.

.. math::

    z = x * w + b

    g1 = \frac{\partial loss}{\partial w} 

    g2 = \frac{\partial loss}{\partial b} 

Tensors make this process quite straightforward:

.. code-block:: python
    :linenos:

    x = torch.ones(5)  # input tensor
    y = torch.zeros(3)  # expected output

    w = torch.randn(5, 3, requires_grad=True)
    b = torch.randn(3, requires_grad=True)

    z = torch.matmul(x, w)+b

    loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

    loss.backward()
    print(w.grad)
    print(b.grad)



.. admonition:: Exercise
   :class: todo

    Try the notebook *tensors.ipynb*.

.. admonition:: Key Points
   :class: hint

    #. In PyTorch, we can create tensors using various techniques.   
    #. Automatic differentiation is simple with tensors in PyTorch.




