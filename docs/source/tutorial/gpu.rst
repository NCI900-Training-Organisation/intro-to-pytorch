GPU Parallelism 
---------------

.. admonition:: Overview
   :class: Overview

    * **Tutorial:** 30 min
    * **Exercises:** 10 min

        **Objectives:**
            #. Learn the general architecture of GPUs.
            #. Learn about the execution model of an NVIDIA GPU. 
            #. Learn about thread indexing in GPUs.
            #. Learn about streams in NVIDIA GPU.
            #. Learn about data movements in GPUs.

Gadi only has NVIDIA GPUs. So when we say GPUs we mean NVIDIA GPUs. Nevertheless, many concepts discussed here 
are the same across different vendors_. While the CPU is optimized to do a single operation as fast as it 
can (low latency operation), the GPU is optimized to do a large number of slow operations (high throughput 
operation). GPUs are composed of multiple Streaming Multiprocessors (SMs), an on-chip L2 cache, and 
high-bandwidth DRAM. The SMs execute operations and the data and code are accessed from DRAM through the L2 cache.

.. image:: ../figs/SM.png

Each SM is organized into CUDA cores capable of doing specialized operations.

.. image:: ../figs/cuda_cores.png

GPU Execution Model
*******************

Each GPU kernels are launched with a set of threads. The threads can be organized into blocks, and the blocks 
can be organized into a grid. The maximum number of threads a block can have will depend on the GPU generation. 

.. image:: ../figs/blocks.png

A block can be executed only in one SM, but an SM can have multiple blocks simultaneously. The maximum number 
of blocks an SM can host will depend on the GPU generation. Since an SM can execute multiple thread blocks 
concurrently, it is always a good idea to launch a kernel with blocks several times higher than the number 
of SMs. 

.. image:: ../figs/wave.png

**Wave** is the number of thread blocks that run concurrently. So if we have 12 SMs and we launch a kernel 
with 8 blocks, with an occupancy of 1 block per SM, there will be two waves.


Thread Indexing
***************

Threads, blocks, and grids are organized in three dimensions: x, y, and z. For simplicity, we will use only 
two dimensions.

**Dimensions**:

1.  *gridDim.x* — blocks in the x dimension of the grid 
2.  *gridDim.y* — blocks in the y dimension of the grid 
3.  *blockDim.x* — threads in the x dimension of the block 
4.  *blockDim.y* — threads in the y dimension of the block 

**Indexing**: 

1.  *blockIdx.x* — block index in x dimension 
2.  *blockIdx.y* — block index in y dimension 
3.  *threadIdx.x* — thread index in x dimension 
4.  *threadIdx.y* — thread index in y dimension 

How do we assign a unique thread id to each thread?
***************************************************

.. image:: ../figs/thread_index.drawio.png


1. Find the blockId --> 
.. code-block:: console
    blockId  = (gridDim.y * blockIdx.x) + blockIdx.y

2. Using the blockId, find the threadId 
.. code-block:: console
    threadId = [(blockDim.x * blockDim.y) * blockId] + [(blockDim.y * threadIdx.x) + threadIdx.y]

Warps and Warp Schedulers
*************************

While we can arrange the threads in any order, the SM schedules the threads as **Warps**, and each warp 
contains 32 threads. For example, if you launch a block with 256 threads, those 256 threads are arranged as 
8 warps (256/8). All the threads in the same warp can only execute the same instruction at a given time. 
For example, if we have a program

.. code-block:: console
    a = b + c
    d = x * y

*All* the threads in the warp should finish executing the addition operation, only then can the threads 
execute the multiplication operation. Depending on the generation of the GPU, it may contain more than one 
warp scheduler. For instance, in the *Fermi GPU*, each SM features two warp schedulers and two instruction 
dispatch units. This allows two warps to be issued and executed concurrently. It is always a good idea to 
consider the warp size (32) and the maximum number of concurrent warps possible when deciding the block size.

.. image:: ../figs/warp.png

Data Movement in GPUs
*********************

.. image:: ../figs/gpu-node.png

The are two types of data movement in GPUs:

1.  Host-to-Device data movement (H2D): Move data from the host memory to the GPU memory.
2.  Device-to-Device data movement (D2D): Move data from the memory of one GPU to another.

H2D transfer happens through the PCIe switch and D2D transfer happens through NVLink. This makes D2D 
transfers more faster than H2D transfers.

Streams
*******

Streams are used to manage and optimize parallel computing tasks. It is a sequence of operations (or tasks) 
that are executed on the GPU. Streams allow for the parallel execution of these tasks, enabling more efficient 
use of the GPU's resources. Each stream can execute kernels and manage data transfers between the CPU and GPU.

.. image:: ../figs/streams.png

The main advantages of using streams are:

1. **Concurrency**: Streams allow for concurrent execution of multiple tasks. This means you can have multiple 
streams, each handling different tasks, and the GPU can process these tasks in parallel. For example, one stream 
might be handling a computation kernel, while another is performing data transfer.

2. **Asynchronous Execution**: Operations within a stream are executed in the order they are issued. 
However, different streams can operate asynchronously with respect to each other. This means that tasks 
in one stream can proceed independently of tasks in another stream, leading to potential performance improvements.

3. **Data Transfers**: Streams can be used to overlap data transfers between the CPU and GPU with kernel 
executions. For example, while one stream is using data, another stream can be used to transfer new data 
to the GPU, which helps in hiding the latency of data transfers.

Exercise
*********

1. What speedup is achieved with GPU parallelism?
    
.. code-block:: console
    :linenos:
    
    qsub 4_gpu.pbs

2. Does using a separate stream affect GPU performance?

.. code-block:: console
    :linenos:
    
    qsub 5_gpu_streams.pbs
    

.. admonition:: Key Points
   :class: hint

    #. Thread indexing play an important role in GPU performance.
    #. Warp size play an important role in GPU performance.
    #. Streams can improve asynchronous parallelism in GPUs.