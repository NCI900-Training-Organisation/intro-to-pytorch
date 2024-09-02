Multi-GPU Training using Data Parallelism
=========================================

.. admonition:: Overview
   :class: Overview

    * **Tutorial:** 15 min
    * **Exercises:** 10 min

        **Objectives:**
            #. Learn how to use multiple GPUs in training using data parallelism. 

By default, PyTorch will use only one GPU. However, you can easily leverage multiple GPUs by running your model in parallel using `DataParallel`.

DataParallel
*************

Whenever you have multiple GPUs, you can wrap your model with `nn.DataParallel`. Then, you can move your model to the GPUs using `model.to(device)`.

.. code-block:: python
    :linenos:
    
    if torch.cuda.device_count() > 1:
        class_model = nn.DataParallel(class_model)
    class_model.to(device)

Then we can use the model as usual and pytorch will distribute the data across multiple GPUs.


Detailed Working 
*****************

`nn.DataParallel` splits the input data across the available GPUs, performing computations in parallel, and then aggregating the results. 

1. **Splitting the Input Data**

- **Batch Splitting**: `nn.DataParallel` splits each mini-batch of data into smaller chunks, with each chunk sent to a different GPU.

- **Replication**: The model is replicated on each GPU, ensuring that each GPU has a copy of the model.

2. **Parallel Computation**

- **Forward Pass**: Each GPU performs a forward pass on its respective chunk of the data. Since the model is replicated on each GPU, the computations are done independently for each chunk.

- **Backward Pass**: During backpropagation, gradients are computed separately on each GPU.

3. **Aggregation of Results**

- **Concatenation of Outputs**: After the forward pass, `nn.DataParallel` gathers the outputs from all GPUs and concatenates them along the batch dimension. This is necessary to maintain the correct order of the outputs.

- **Gradient Aggregation**: During backpropagation, `nn.DataParallel` aggregates the gradients from each GPU. It does this by summing the gradients computed by each GPU, which are then used to update the model parameters.

4. **Synchronizing Parameters**

- **Parameter Updates**: After gradients are aggregated, the model parameters are updated on the primary GPU. The updated parameters are then synchronized and broadcasted to all other GPUs.


Limitations
***********

`nn.DataParallel` in PyTorch allows you to distribute data across multiple GPUs for parallel processing, but it has some limitations:

#. **Single-process bottleneck**: nn.DataParallel uses a single process that sends data to each GPU, collects the results, and aggregates them. This can become a bottleneck, especially with a large number of GPUs.
#. **Limited scalability**: As the number of GPUs increases, the performance gains from nn.DataParallel diminish due to the overhead of distributing data and collecting results.
#. **Less efficient memory usage**: nn.DataParallel replicates the entire model on each GPU, which can lead to inefficient memory usage, especially with large models.
#. **Inflexible device placement**: nn.DataParallel requires all GPUs to be on the same machine. It doesn't support distributed training across multiple nodes


When using nn.DataParallel in PyTorch, `class_model.parameters()).device` often returns `cuda:0`, even if multiple GPUs are used. This happens because 
nn.DataParallel replicates the model across multiple GPUs but keeps the original model's parameters on the primary GPU (cuda:0). The `nn.DataParallel` wrapper 
itself does not move parameters to different GPUs; it only distributes the input data to the GPUs and then aggregates the results. The underlying parameters of the model are still located on the primary device.
It's always a good idea to use `nvidia-smi` to check that the GPU utilization is as expected.


.. admonition:: Exercise
   :class: todo

    Try the notebook *multi_GPU.ipynb*.


.. admonition:: Key Points
   :class: hint

    #. We can use `nn.DataParallel` to utilize multiple GPUs for training.
    #. The training is limited to a single node and cannot span across multiple nodes.

