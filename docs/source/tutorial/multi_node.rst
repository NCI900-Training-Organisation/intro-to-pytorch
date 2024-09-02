Distributed Data Parallelism
=============================

.. admonition:: Overview
   :class: Overview

    * **Tutorial:** 20 min
    * **Exercises:** 10 min

        **Objectives:**
            #. Learn how to use multiple GPUs, in multiple nodes using Torchrun.


To run the provided code on multiple nodes using torchrun (previously torch.distributed.launch), we need to make a few modifications to the 
single node code:

- Environment Variables for Multi-Node Training*: Set environment variables like MASTER_ADDR, MASTER_PORT, WORLD_SIZE, and RANK using command-line arguments when launching the script with torchrun.
- Modifications to the setup function*: The setup function should be updated to handle the environment variables for multi-node training.
- Remove Hardcoded MASTER_ADDR and MASTER_PORT*: These should be passed dynamically when using torchrun.
- `main`function: Remove the use of mp.spawn and instead rely on torchrun to handle the spawning of processes across nodes.


PBS Script
**********

As Gadi uses the PBS job scheduler we can use it to run the training on multiple nodes. Here we are requesting 2 nodes, each with 4 GPUs.

.. code-block:: console
    :linenos:

    #!/bin/bash

    #PBS -P vp91
    #PBS -q gpuvolta

    #PBS -l ncpus=96
    #PBS -l ngpus=8
    #PBS -l mem=10GB
    #PBS -l walltime=00:20:00 

    #PBS -N multinode

    module load python3/3.11.0  
    module load cuda/12.3.2

    . /scratch/vp91/Training-Venv/pytorch/bin/activate

    # Set variables
    if [[ $PBS_NCPUS -ge $PBS_NCI_NCPUS_PER_NODE ]]
    then
    NNODES=$((PBS_NCPUS / PBS_NCI_NCPUS_PER_NODE))
    else
    NNODES=1
    fi
 
    PROC_PER_NODE=$((PBS_NGPUS / NNODES))
 
    MASTER_ADDR=$(cat $PBS_NODEFILE | head -n 1)
 
    # Launch script
    LAUNCH_SCRIPT=/scratch/vp91/jxj900/intro-to-pytorch/job_scripts/multinode_torchrun.sh
 
    # Set execute permission
    chmod u+x ${LAUNCH_SCRIPT}
 
    # Run PyTorch application
    for inode in $(seq 1 $PBS_NCI_NCPUS_PER_NODE $PBS_NCPUS); do
        pbsdsh -n $inode ${LAUNCH_SCRIPT} ${NNODES} ${PROC_PER_NODE} ${MASTER_ADDR} &
    done
    wait

Here, `pbsdsh` launches the `multinode_torchrun.sh` script simultaneously on all nodes. The `multinode_torchrun.sh` script contains the following:

.. code-block:: console
    :linenos:

    #!/bin/bash
 
 
    module load python3/3.11.0  
    module load cuda/12.3.2

    . /scratch/vp91/Training-Venv/pytorch/bin/activate
 
    # Application script
    APPLICATION_SCRIPT=/scratch/vp91/jxj900/intro-to-pytorch/src/distributed_data_parallel.py
 
    # Set execute permission
    chmod u+x ${APPLICATION_SCRIPT}
 
    # Run PyTorch application
    torchrun --nnodes=${1} --nproc_per_node=${2} --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=${3}:29400 ${APPLICATION_SCRIPT}


Where `torchrun` will launch the training program `distributed_data_parallel.py` on each node and
use all the 4 GPUs on each node.


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

Alternatively, if you can SSH into the individual nodes, you can proceed with the following steps.

On the first node (rank 0):

.. code-block:: console
    :linenos:

    torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 --master_addr="<Node1 IP>" --master_port=12355 /scratch/vp91/$USER/intro-to-pytorch/src/multinode_torchrun.py

On the second node (rank 1):


.. code-block:: console
    :linenos:

    torchrun --nnodes=2 --nproc_per_node=4 --node_rank=1 --master_addr="<Node1 IP>" --master_port=12355 /scratch/vp91/$USER/intro-to-pytorch/src/multinode_torchrun.py


.. admonition:: Explanation
   :class: attention

   If you have a `SLURM scheduler, <https://youtu.be/KaAJtI1T2x4>`_, things are a bit easier since the *srun* command can launch the PyTorch job directly 
   from the job script, eliminating the need for an additional shell script.


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