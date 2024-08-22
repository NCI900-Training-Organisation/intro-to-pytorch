Multi-core Parallelism
----------------------

.. admonition:: Overview
   :class: Overview

    * **Tutorial:** 20 min
    * **Exercises:** 5 min

        **Objectives:**
            #. Learn about multi-core parallelism.
            #. Learn about NUMA regions. 

Multi-core parallelism is a technique used to enhance computing performance by utilizing multiple processor 
cores within a NUMA node or across NUMA nodes. It involves using multiple processing cores 
within a CPU to perform different tasks or parts of a task simultaneously. Each core can execute its own 
thread, allowing for more efficient processing and faster computation.

A core is an individual processing unit within a CPU. Multi-core processors have multiple cores, each capable 
of executing instructions independently.

.. image:: ../figs/multicorePrallelism.drawio.png



1. **Multiple Cores**: A multi-core processor contains multiple independent processing units (cores) on a 
single chip. Each core can execute its own instructions simultaneously, allowing for parallel execution of tasks.

2. **Parallel Execution**: In multi-core systems, different threads or processes can run concurrently on separate
 cores. For example, if an application is designed to perform multiple tasks at once these tasks can be 
 distributed across different cores to improve performance.

3. **Thread Management**: Applications manage the distribution of tasks across cores using threads. Multi-threaded
 applications can divide their workload into smaller threads, which can then be scheduled to run on different 
 cores. This approach improves efficiency by making use of the available cores.

4. **Load Balancing**: Efficient multi-core parallelism requires effective load balancing, where tasks are evenly
 distributed among cores to avoid bottlenecks. Runtime environments often include scheduling algorithms to ensure
 that all cores are utilized effectively and that no single core becomes a performance bottleneck.

5. **Inter-Core Communication**: Cores in a multi-core system may need to communicate with each other to 
coordinate tasks or share data. This communication is facilitated through shared memory or interconnects, and 
efficient handling of inter-core communication is crucial for maintaining performance.

6. **Scalability**: Multi-core parallelism improves scalability, allowing applications to take advantage of 
increasing core counts. As more cores become available, applications and workloads that are designed for 
parallelism can scale up their performance proportionally.

7. **Challenges**: Multi-core parallelism introduces challenges such as ensuring proper synchronization between 
threads, managing shared resources, and avoiding issues like race conditions and deadlocks. Effective multi-core 
programming requires careful design to handle these complexities.

Non-Uniform Memory Access (NUMA)
********************************

NUMA is a computer architecture design used in multi-core and multi-processor systems
where the memory access time depends on the location of the memory relative to the processor. In NUMA systems, 
the memory is divided into multiple regions, each associated with one or more processors. Each processor has 
fast access to its local memory region but slower access to memory nodes associated with other processors. 
This creates a "non-uniform" access pattern compared to Uniform Memory Access (UMA) systems, where all memory 
access times are the same regardless of the processor's location.

Exercise
*********

1. How does performance with multiple cores compare to performance with a single core?

.. code-block:: console
    :linenos:

    qsub 3_multi_core.pbs

2. How many NUMA regions does the compute node have? (Hint: You can find this information by using the `lstopo` command in the 3_multi_core.pbs script.)

.. admonition:: Key Points
   :class: hint

    #. Multi-core parallelism can improve the peformance of the application.
    #. The effectiveness of the parallelism depends on the problem size.