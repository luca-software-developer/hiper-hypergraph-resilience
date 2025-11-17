Contributing
============

Thank you for your interest in contributing to HIPER!

Development Setup
-----------------

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/luca-software-developer/hiper-hypergraph-resilience.git
      cd hiper-hypergraph-resilience

2. Create a virtual environment:

   .. code-block:: bash

      python -m venv .venv
      source .venv/bin/activate  # On Windows: .venv\Scripts\activate

3. Install dependencies:

   .. code-block:: bash

      pip install -r requirements.txt

4. Install development dependencies:

   .. code-block:: bash

      pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints

Building Documentation
----------------------

To build the documentation locally:

.. code-block:: bash

   cd docs
   sphinx-build -b html . _build/html

The generated documentation will be in ``docs/_build/html/``.

To rebuild after making changes:

.. code-block:: bash

   sphinx-build -b html . _build/html

Running Tests
-------------

Run the test suite:

.. code-block:: bash

   python -m pytest

Run specific test files:

.. code-block:: bash

   python test_performance_single.py
   python test_simulation.py

Code Style
----------

* Follow PEP 8 style guidelines
* Use meaningful variable and function names
* Add docstrings to all public functions and classes
* Keep functions focused and small
* Write tests for new features

Docstring Format
----------------

Use Google-style docstrings:

.. code-block:: python

   def example_function(param1: int, param2: str) -> bool:
       """
       Brief description of the function.

       More detailed explanation if needed.

       Args:
           param1: Description of param1
           param2: Description of param2

       Returns:
           Description of return value

       Raises:
           ValueError: When param1 is negative

       Example:
           >>> example_function(5, "test")
           True
       """
       pass

Submitting Changes
------------------

1. Create a new branch for your feature:

   .. code-block:: bash

      git checkout -b feature/your-feature-name

2. Make your changes and commit:

   .. code-block:: bash

      git add .
      git commit -m "Add: Brief description of changes"

3. Push to your fork:

   .. code-block:: bash

      git push origin feature/your-feature-name

4. Create a Pull Request on GitHub

Reporting Issues
----------------

When reporting issues, please include:

* Python version
* HIPER version
* Operating system
* Steps to reproduce the issue
* Expected behavior
* Actual behavior
* Error messages or stack traces

Contact
-------

For questions or discussions, please open an issue on GitHub.
