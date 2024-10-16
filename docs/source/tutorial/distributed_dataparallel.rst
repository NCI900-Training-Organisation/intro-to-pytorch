Distributed Data Parallelism
=============================

.. admonition:: Overview
   :class: Overview

    * **Tutorial:** 20 min
    * **Exercises:** 10 min

        **Objectives:**
            #. Learn how to use multiple GPUs in training using distributed data parallelism. 
            #. Train the model using PBS a job script.


Components of a distributed data parallel model:

- **Master Node:** The primary GPU responsible for synchronization, model replication, loading models, and logging.
- **Process Group:** When training or testing a model across K GPUs, these K processes form a group.
- **Rank:** Each process within the process group is identified by a rank, ranging from 0 to K-1 (similar to MPI).
- **World Size:** The total number of processes in the group, which equals the number of GPUs (similar to MPI).

Advantage over DataParallel
****************************

- **Scalability:** DataParallel operates as a single-process, multi-threaded approach and only works on a single machine, whereas, DistributedDataParallel (DDP) uses a multi-process approach and supports both single- and multi-machine training. DataParallel is often slower than DDP, even on a single machine, due to *GIL* contention across threads, the overhead of replicating the model per iteration, and the extra steps involved in scattering inputs and gathering outputs.
  
- **Model Parallelism:** If your model is too large to fit on a single GPU, you need to use model parallelism to distribute it across multiple GPUs. DistributedDataParallel supports model parallelism, while DataParallel does not. When combining DDP with model parallelism, each DDP process utilizes model parallelism, and all processes together perform data parallelism.


Process Group
*************

In DistributedDataParallel (DDP), a *Process Group* is a collection of processes that can communicate with each other during distributed training. 

.. code-block:: python
    :linenos:

    import torch.distributed as dist
    
    def setup(rank, world_size):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

.. admonition:: Explanation
   :class: attention

   Here, `nccl` is the backend that determines how communication between processes is handled. 
   NCCL is optimized for multi-GPU communication and is recommended for use with NVIDIA GPUs.
   Other common backends are `gloo`, a CPU-based backend, and `mpi`
   the where MPI (Message Passing Interface) based backend.

Splitting the Dataloader
************************

To split the data across multiple GPUs we use `DistributedSampler`.

.. code-block:: python
    :linenos:

    def prepare(rank, world_size, batch_size=32, pin_memory=False, num_workers=0):
        dataset = PimaDataset(datapath)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    
        dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=False, shuffle=False, sampler=sampler)
    
        return dataloader


.. admonition:: Explanation
   :class: attention

    - `num_replicas` - Is typically the number of processes in the distributed training job.
    - `rank` - Each process is assigned a rank which ensures that each process only accesses the data corresponding to its rank.
    - `drop_last` -   When working with datasets in distributed training, it is common for the total number of samples in the dataset to not be perfectly divisible by the product of the batch size and the number of replicas. When `drop_last` is set to *True*, the last batch that is not full will be dropped. 

and a distributed `DataLoader`.

**DistributedSampler** is used to divide the dataset among multiple processes (workers). It ensures 
that each process only gets its portion of the data. When we create the **dataloader** we pass the 
DistributedSampler object to the dataloader object (*sampler=sampler*). This ensures each process gets
a unique subset of data for distributed training.


.. admonition:: Explanation
   :class: attention

    - `num_workers` - Number of subprocesses to use for data loading.
    - `pin_memory` - Pinned (or Page-locked) memory is a region of host memory that is "locked" in physical RAM and cannot be paged out to disk by the operating system. This ensures that the memory remains in RAM and is directly accessible for operations like data transfer between the CPU and GPU. Page-locking excessive amounts of memory with cudaMallocHost() may degrade system performance, since it reduces the amount of memory available to the system for paging. As a result, this function is best used sparingly to allocate staging areas for data exchange between host and device.

    .. image:: ../figs/pinning.png



Wrapping a Model in DDP
**********************

DistributedDataParallel (DDP) is a PyTorch wrapper that helps to parallelize training across multiple GPUs and minimizes communication overhead and 
synchronizes gradients automatically.


.. code-block:: python
    :linenos:

    model_ddp = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

.. admonition:: Explanation
   :class: attention

    - `model`: The neural network (`torch.nn.Module`) that you want to train. Before wrapping it with DDP, it should be placed on the appropriate device (GPU) using model.to(device).
    - `device-ids`: Specifies the GPU device(s) to which this process's model should be mapped. The rank typically corresponds to the index of the current process within the distributed setup, and in a single-node setup with multiple GPUs, rank is often the GPU ID. For example, if rank=0, it means this process will use GPU 0.
    - `output_device` : Specifies the device where the output of the model should be stored.
    - `find_unused_parameters` : DDP assumes all model parameters are used in every forward pass, and it synchronizes their gradients accordingly. Setting `find_unused_parameters=True`` ensures that DDP will only synchronize the gradients of parameters that are actually used, preventing errors and unnecessary communication overhead.

.. admonition:: Exercise
   :class: todo

    1. Examine the program *src/distributed_data_parallel.py*. What the changes from data_parallel.ipynb?
    2. Examine the job script *job_scripts/distributed_data_parallel.pbs*.
    3. Run the program using the job script *job_scripts/distributed_data_parallel.pbs*.

    .. code-block:: console
        :linenos:

        cd job_scripts
        qsub distributed_data_parallel.pbs


.. admonition:: Key Points
   :class: hint

    #. We can use distributed data parallelism to use multiple GPUs on the same node.