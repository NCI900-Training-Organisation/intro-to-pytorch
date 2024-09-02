Distributed Data Parallelism
=============================

.. admonition:: Overview
   :class: Overview

    * **Tutorial:** 20 min
    * **Exercises:** 10 min

        **Objectives:**
            #. Learn how to use multiple GPUs in multiple nodes using Torchrun.


To run the provided code on multiple nodes using torchrun (previously torch.distributed.launch), we need to make a few modifications to the 
single node code:

- Environment Variables for Multi-Node Training*: Set environment variables like MASTER_ADDR, MASTER_PORT, WORLD_SIZE, and RANK using command-line arguments when launching the script with torchrun.
- Modifications to the setup function*: The setup function should be updated to handle the environment variables for multi-node training.
- Remove Hardcoded MASTER_ADDR and MASTER_PORT*: These should be passed dynamically when using torchrun.
- `main`function: Remove the use of mp.spawn and instead rely on torchrun to handle the spawning of processes across nodes.


PBS Script
**********

As Gadi uses the PBS job scheduler we can use the same to run the training on multiple nodes. Here we are requesting 2 nodes, each with 4 GPUs.

.. code-block:: console
    :linenos:

    #!/bin/bash

    #PBS -P vp91
    #PBS -q gpuvolta

    #PBS -l ncpus=96
    #PBS -l ngpus=8
    #PBS -l mem=10GB
    #PBS -l walltime=00:05:00 

    #PBS -N multinode

    module load python3/3.11.0  
    module load cuda/12.3.2
    module load nccl/2.19.4

    . /scratch/vp91/Training-Venv/pytorch/bin/activate

    python3 /scratch/vp91/$USER/intro-to-pytorch/src/distributed_data_parallel.py

    # Get the list of allocated nodes
    NODES=$(cat $PBS_NODEFILE | uniq)
    NODE_ARR=($NODES)

    # Define the master node (usually the first node in the list)
    MASTER_ADDR=${NODE_ARR[0]}
    MASTER_PORT=12355  # Set an appropriate port for communication

    NNODES=2
    NPROC_PER_NODE=4
    WORLD_SIZE=$(($NNODES * $NPROC_PER_NODE))

    torchrun --nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE \
             --node_rank=$PBS_NODEID --master_addr=$MASTER_ADDR \
             --master_port=$MASTER_PORT /scratch/vp91/$USER/intro-to-pytorch/src/multinode_torchrun.py


.. admonition:: Explanation
   :class: attention

    `MASTER_ADDR`: The IP address or hostname of the master node, which is typically the first node allocated by PBS.
    `MASTER_PORT`: A port for inter-node communication. Ensure it is open and unused.
    `NNODES`: The number of nodes requested.
    `NPROC_PER_NODE`: The number of GPUs per node.
    `WORLD_SIZE`: The total number of processes (NNODES * NPROC_PER_NODE).


.. admonition:: Explanation
   :class: attention

    The rendezvous backend in PyTorch is a key component of the distributed training setup. It is
    responsible for coordinating the initialization of multiple processes that may be running across different 
    nodes in a distributed system. This process is crucial for ensuring that all distributed processes are aware 
    of each other and can start training in a synchronized manner.

    - `RDZV_BACKEND`: The backend used for the rendezvous process (c10d is default for PyTorch).
    - `RDZV_ENDPOINT`: The network address of the rendezvous server, combining `MASTER_ADDR` and `MASTER_PORT`.
    - `RDZV_ID`: Provides a unique identifier for each distributed training job. This is essential when multiple distributed jobs are running on the same set of nodes.


Alternative Options
********************

Alternatively, if you can SSH into the indvdiual nodes we can do the following.

On the first node (rank 0):

.. code-block:: console
    :linenos:

    torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 --master_addr="<Node1 IP>" --master_port=12355 /scratch/vp91/$USER/intro-to-pytorch/src/multinode_torchrun.py

On the second node (rank 1):


.. code-block:: console
    :linenos:

    torchrun --nnodes=2 --nproc_per_node=4 --node_rank=1 --master_addr="<Node1 IP>" --master_port=12355 /scratch/vp91/$USER/intro-to-pytorch/src/multinode_torchrun.py


.. admonition:: Exercise
   :class: todo

    1. Examine the program *src/ multinode_torchrun.py*. What are the changes from *src/distributed_data_parallel.py*?
    2. Examine the job script *job_scripts/multinode_torchrun.pbs*.
    3. Run the program using the job script *job_scripts/multinode_torchrun.pbs*.

    .. code-block:: console
        :linenos:

        cd job_scripts
        qsub multinode_torchrun.pbs


.. admonition:: Key Points
   :class: hint

    #. We can use Torchrun to use multiple GPUs in multiple nodes.
    #. We can use PBS script to launch multi-node trainings.