Basics of Parallelism
--------------------

.. admonition:: Overview
   :class: Overview

    * **Tutorial:** 15 min
    * **Exercises:** 15 min

        **Objectives:**
            #. Learn about the difference between threads and process
            #. Learn how to synchronize between threads.
            #. Learn how to synchronize between processes.


Process
********

A process is an instance of a program in execution. A process is responsible for executing a program's 
instructions and providing the environment in which the program operates (such as memory and I/O devices).

#. **Program Code**: The instructions of the program that the process is executing.
#. **Process Stack**: Contains temporary data such as method/function parameters, return addresses, and local variables.
#. **Heap**: A region of memory used for dynamic memory allocation during the process's execution.
#. **Data Section**: Contains global and static variables used by the process.
#. **Process Control Block (PCB)**: A data structure maintained by the operating system that holds information about the process, including its state, program counter, CPU registers, and memory management information.

We can launch multiple process and the same time and the processes are isolated from each other.
Each process manages its own resources, including memory and CPU time and one application can
have more than one process. Process communicates between each other using Inter-Process Communication (IPC) 
mechanisms.

The OS can manage multiple process at the same time and the OS can switch the process executing in a CPU.
This involves saving the current state of the process that is being paused (the "old" process) and restoring 
the state of the process that is being resumed (the "new" process). The state of a process is 
captured in its *Process Control Block (PCB)*. This idea of switching between processes is called
*Context Switching*.

Threads
*******

A thread is the smallest unit of execution within a process. A process can contain multiple threads that 
share the same resources but execute independently. While each process is isolated from processes, threads 
within the same process share the same memory space and resources but they execute independently.

*Concurrency* refers to the ability to run multiple threads simultaneously. Threads can be managed by 
the operating system to run on different CPU cores, doing different computations, thereby 
improving performance. If two concurrent threads (or processes) can be run simultaneously we can say 
they are *parallel*.

Challenges with Threads
***********************

#. **Synchronization**: As threads share resources, they need mechanisms to synchronize access to prevent conflicts and ensure data consistency. Common synchronization tools include mutexes, semaphores, and locks.
#. **Deadlock**: A situation where two or more threads are waiting indefinitely for resources held by each other, leading to a standstill.
#. **Race Conditions**: Occur when the outcome depends on the unpredictable timing of thread execution, potentially causing inconsistent results.


Synchronization in programming is the coordination of concurrent threads or processes to ensure they operate 
correctly when accessing shared resources. It prevents issues such as race conditions and data corruption by 
managing access to shared resources, ensuring that only one thread or process can modify the resource at a time. 
Synchronization mechanisms, like locks, semaphores, and mutexes, help maintain consistency and order in a 
multithreaded or multiprocess environment.

The Room and the Key: An analogy
*********************************

**The Room**: Think of a room that represents a shared resource or a critical section of code in a program. 
This room can only be used by one person at a time to ensure that things don't get messed up.

**The Lock**: The lock is like a physical lock that controls access to the room. Only one person can hold the 
key of the lock at any given time.

**Entering the Room**: When a person (a thread) wants to use the room (access the shared resource), 
they need to get the key (acquire the lock). If no one else is using the room, the person can take the key, 
enter the room, and use it as needed.

**Occupied Room**: If someone is already inside the room and using it, other people who want to use the room 
must wait outside. They cannot enter until the current occupant leaves and returns the key.

**Exiting the Room**: Once the person is done using the room, they leave and return the key (release the lock). 
This allows another person to take the key and use the room.

**Preventing Conflicts**: The lock ensures that only one person is in the room at any time. This prevents 
conflicts or issues that might arise if multiple people were trying to use the room simultaneously.

Exercise
*********

1. What occurs when locks aren't used??

.. code-block:: console
    :linenos:

    module load python3/3.11.0
    python3 threads.py

2. How do threads differ from processes?

.. code-block:: console
    :linenos:

    module load python3/3.11.0
    python3 process.py



.. admonition:: Key Points
   :class: hint

    #. Processes are isolated with separate memory spaces, while threads share the same memory space within a process.
    #. Processes have higher creation and management overhead due to separate resources and memory, whereas threads are lighter and cheaper to manage.
    #. Threads can communicate easily and efficiently since they share memory, while processes require more complex and resource-intensive Inter-Process Communication (IPC) mechanisms.
    #. Locks can be used for synchronization.




