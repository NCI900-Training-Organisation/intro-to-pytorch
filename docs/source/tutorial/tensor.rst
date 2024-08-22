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


.. admonition:: Explanations
   :class: note

   torch.rand_like returns a tensor with the same size as input that but filled with random numbers from the interval [0,1).


**Exercise** : Try the notebook *tensor.ipynb*.

.. admonition:: Key Points
   :class: hint

    #. Processes are isolated with separate memory spaces, while threads share the same memory space within a process.
    #. Processes have higher creation and management overhead due to separate resources and memory, whereas threads are lighter and cheaper to manage.
    #. Threads can communicate easily and efficiently since they share memory, while processes require more complex and resource-intensive Inter-Process Communication (IPC) mechanisms.
    #. Locks can be used for synchronization.




