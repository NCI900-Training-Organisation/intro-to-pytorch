Distributed Data Parallelism
=============================

.. admonition:: Overview
   :class: Overview

    * **Tutorial:** 15 min
    * **Exercises:** 10 min

        **Objectives:**
            #. Learn how to use multiple GPUs in training using data parallelism. 


Components of a distributed data parallel model:

- **Master Node:** The primary GPU responsible for synchronization, model replication, loading models, and logging.
- **Process Group:** When training or testing a model across K GPUs, these K processes form a group.
- **Rank:** Each process within the process group is identified by a rank, ranging from 0 to K-1 (similar to MPI).
- **World Size:** The total number of processes in the group, which equals the number of GPUs (similar to MPI).

Advantage over DataParallel
****************************

- **Scalability:** DataParallel operates as a single-process, multi-threaded approach and only works on a single machine, whereas
DistributedDataParallel (DDP) uses a multi-process approach and supports both single- and multi-machine training. DataParallel is often slower than DDP, 
even on a single machine, due to *GIL* contention across threads, the overhead of replicating the model per iteration, and the extra steps involved in 
scattering inputs and gathering outputs.
  
- **Model Parallelism:** If your model is too large to fit on a single GPU, you need to use model parallelism to distribute it across multiple GPUs. 
DistributedDataParallel supports model parallelism, while DataParallel does not. When combining DDP with model parallelism, each DDP process utilizes model 
parallelism, and all processes together perform data parallelism.


Process Group
*************

.. code-block:: python
    :linenos:

    import torch.distributed as dist
    
    def setup(rank, world_size):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

