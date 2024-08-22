Python Virtual Environment
==========================

.. note::
 1.  python-papi
 2.  numpy
 3.  codetiming
 4.  numba
 5.  mpi4py

In this workshop, we will use a Python virtual environment to manage all the required Python packages. A Python virtual environment is an isolated 
workspace that allows you to manage project-specific dependencies without affecting the global Python installation or other projects. By creating a 
virtual environment, you can install and manage libraries and packages independently, ensuring that each project has its own set of dependencies and 
avoiding version conflicts. This isolation helps maintain consistent and reproducible development environments.

We've already set up the Python virtual environment for this workshop, so you *don't need* to install one separately. However, the following 
commands will guide you on how to create one if necessary.

To get started with Python virtual environment load the Python module you want to use. In this workshop, we will be using *python3/3.11.0*.

.. code-block:: console
    :linenos:

    module load python3/3.11.0 cuda/12.3.2

Create the Python virtual environment.

.. code-block:: console
    :linenos:

    python3 -m venv my_env

Activate the Python virtual environment.

.. code-block:: console
    :linenos:

    source my_env/bin/activate

Install all the required Python packages.

.. code-block:: console
    :linenos:

    python3 -m pip install python-papi numpy codetiming numba mpi4py

You can deactivate the virtual environment once you are done with it.

.. code-block:: console
    :linenos:

    deactivate
 