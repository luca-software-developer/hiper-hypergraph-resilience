Installation
============

Installing HIPER
----------------

HIPER can be installed locally using pip. Navigate to the project root directory
and run one of the following commands:

Development Installation (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For development work, install HIPER in editable mode:

.. code-block:: bash

   pip install -e .

This installs the package in editable mode, which means any changes you make to
the source code will be immediately available without needing to reinstall the
package. This is ideal for development and testing.

Standard Installation
^^^^^^^^^^^^^^^^^^^^^

For a standard installation:

.. code-block:: bash

   pip install .

This creates a standard installation in your Python environment's site-packages
directory.

Verifying Installation
----------------------

After installation, you can verify that HIPER is correctly installed by running:

.. code-block:: bash

   python -c "import hiper; print(hiper.__version__)"

This should print the version number without any errors.

You can also test basic functionality:

.. code-block:: python

   from hiper import Hypernetwork

   # Create a simple hypernetwork
   hn = Hypernetwork()
   hn.add_hyperedge(0, [1, 2, 3])

   # Check basic properties
   print(f"Order: {hn.order()}, Size: {hn.size()}")
   # Output: Order: 3, Size: 1

Uninstalling
------------

To uninstall HIPER from your Python environment:

.. code-block:: bash

   pip uninstall hiper