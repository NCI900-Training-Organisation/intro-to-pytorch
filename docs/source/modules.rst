Modules
=======

.. note::
 1.  python3/3.11.0
 2.  papi/7.0.1
 3.  openmpi/4.0.1
 4.  cuda/12.3.2

Modules are how we manage software in most HPC machines. We can see all the available modules using the command

.. code-block:: console
    :linenos:
    
    module avail

If we want load a module *python3/3.11.0* we can use the command

.. code-block:: console
    :linenos:

    module load python3/3.11.0

If we want to unload the same module use the command

.. code-block:: console
    :linenos:
    
    module unload python3/3.11.0

We can unload all the modules using the command

.. code-block:: console
    :linenos:
    
    module purge