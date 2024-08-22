Requesting a Job
****************

.. admonition:: Overview
   :class: Overview

    * **Tutorial:** 15 min
    * **Exercises:** 5 min

        **Objectives:**
            #. Learn how to write a PBS job script for Gadi.
            #. Learn how to launch a job in Gadi.

1.  Which project are you using?
2.  Which job queue are you planning to use?
3.  How many CPU cores are required for your task?
4.  How many GPUs do you need?
5.  What is the estimated runtime of your program?
6.  Which modules are necessary to execute the program?
7.  What script or command will you use to run the program?


.. code-block:: console
    :linenos:

    #!/bin/bash

    #PBS -P vp91 
    #PBS -q normal
    #PBS -l ncpus=48
    #PBS -l mem=10GB
    #PBS -l walltime=00:02:00
    #PBS -N testScript

    module load python3/3.11.0
    module load papi/7.0.1

    . /scratch/vp91/Training-Venv/intro-parallel-prog/bin/activate

    which python

#. **P** - Gadi project (sometimes called account) to use
#. **q** - Gadi queue to use
#. **ncpus** - Total number of cores requested
#. **ngpus** - Total number of GPUs requested
#. **mem** - Total memory requested
#. **l** - Total wall time for which the resources are provisioned
#. **N** - Name of the job 


For more PBS Directives please check the `Gadi document <https://opus.nci.org.au/display/Help/PBS+Directives+Explained>`_ and for more details on the 
different Gadi queues please check out the corresponding `Gadi document <https://opus.nci.org.au/display/Help/Queue+Structure>`_ .

All the Python code are available in the directory `python/src`` while all the job scripts are available in the 
directory `python/jobScripts`. To submit a job use 
the command

.. code-block:: console
    :linenos:

    qsub 0_test_script.pbs

and to know the status of your job use the command

.. code-block:: console
    :linenos:

    qstat <jobid>

To know get the details about the job use the command

.. code-block:: console
    :linenos:

    qstat -swx <jobid>

.. admonition:: Key Points
   :class: hint

    #. Multiple PBS directives are available request a job.
    #. Gadi uses some custom directives.
    #. There are two modes to request a job - batched and interactive.

