Contributing
============

Contributions are welcome! Whether you want to add new equipment cost
correlations, fix a bug, improve documentation, or add a new feature,
please follow the guidelines below.

Getting started
---------------

1. **Fork** the repository on GitHub.
2. **Clone** your fork locally:

   .. code-block:: bash

      git clone https://github.com/<your-username>/OpenPyTEA.git
      cd OpenPyTEA

3. **Install** the package in editable mode with dev dependencies:

   .. code-block:: bash

      pip install -e ".[test]"
      # or with uv:
      uv sync

4. Create a **feature branch**:

   .. code-block:: bash

      git checkout -b feat/my-new-feature

5. Make your changes, add tests, and open a **pull request** against ``main``.

Adding equipment cost correlations
-----------------------------------

The cost database lives in ``src/openpytea/data/cost_correlations.csv``.
Each row defines one correlation:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Column
     - Description
   * - ``key``
     - Unique identifier (e.g., ``"compressor_centrifugal"``)
   * - ``category``
     - Display category (e.g., ``"Compressors (Centrifugal)"``)
   * - ``type``
     - Sub-type (e.g., ``"Centrifugal"``)
   * - ``form``
     - Correlation form: ``"offset power-law"``, ``"log-log quadratic"``,
       ``"ln-ln quadratic"``, ``"power-sizing"``, ``"2-var power-law"``, or
       ``"exponential"``
   * - ``s_lower``
     - Minimum valid size parameter
   * - ``s_upper``
     - Maximum size parameter for a single unit
   * - ``s2_lower``, ``s2_upper``
     - Valid range for the second size parameter (``s2``), used by
       two-parameter forms such as ``"2-var power-law"``. Unlike
       ``s_upper``, ``s2_upper`` is a hard bound only — exceeding it
       raises ``ValueError`` rather than triggering parallelization.
   * - ``upper_parallel``
     - Maximum total size (triggers parallelization above ``s_upper``)
   * - ``a``, ``b``, ``n``
     - Offset power-law coefficients (:math:`C = a + b \cdot S^n`), used by
       the ``"offset power-law"`` form.
   * - ``a``, ``b``, ``n``, ``n2``
     - 2-var power-law coefficients
       (:math:`C = a + b \cdot S_1^{n} \cdot S_2^{n2}`), used by the
       ``"2-var power-law"`` form. ``a`` is typically 0 unless the
       correlation has a genuine offset term. Evaluating this form
       requires passing both size parameters, e.g.
       ``Equipment(..., param=(S1, S2))`` or
       ``CostCorrelationDB.evaluate(key, S1, S2)``.
   * - ``k1``, ``k2``, ``k3``, ``k4``, ``k5``
     - Log-log quadratic coefficients
       (:math:`\log C = k_1 + k_2\log S + k_3(\log S)^2 + k_4(\log S)^3 + k_5(\log S)^4`),
       used by the ``"log-log quadratic"`` form. ``k4`` and ``k5`` are
       optional and default to 0 for correlations that only need up to the
       quadratic term (all current ``"ln-ln quadratic"`` correlations do).
       The same columns are reused by the ``"ln-ln quadratic"`` form
       (:math:`C = \exp\{k_1 + k_2\ln S + k_3(\ln S)^2 + k_4(\ln S)^3 + k_5(\ln S)^4\}`),
       which takes natural rather than base-10 logarithms and also
       supports optional ``k4``/``k5`` cubic/quartic terms.
   * - ``s0``, ``c0``, ``f``
     - Power-sizing reference size, reference cost, and exponent
       (:math:`C = C_0 \cdot (S / S_0)^f`)
   * - ``a``, ``b``
     - Exponential coefficients (:math:`C = a \cdot \exp(b \cdot S)`), used
       by the ``"exponential"`` form.
   * - ``cost_year``
     - Reference year for the correlation (used for CEPCI adjustment)

Please cite the source of any new correlation in the PR description.

Running tests
-------------

.. code-block:: bash

   pytest tests/

Adding docstrings
-----------------

OpenPyTEA uses **NumPy-style** docstrings. New public functions and classes
must include a docstring with at minimum a summary line, ``Parameters``,
``Returns``, and at least one ``Examples`` block.

.. code-block:: python

   def my_function(x: float, y: int = 10) -> float:
       """
       Brief one-line summary.

       Extended description (optional).

       Parameters
       ----------
       x : float
           Description of x.
       y : int, optional
           Description of y. Default is 10.

       Returns
       -------
       float
           Description of return value.

       Examples
       --------
       >>> my_function(3.5)
       35.0
       """

Code style
----------

* Formatting: ``black`` (line length 60 for ``src/``, 88 elsewhere)
* Linting: ``ruff`` — run ``ruff check src/`` before committing
* Type hints are encouraged for all public APIs

Reporting bugs
--------------

Please open an issue on `GitHub Issues
<https://github.com/PBTamarona/OpenPyTEA/issues>`_ with:

* A minimal reproducible example
* The OpenPyTEA version (``import openpytea; print(openpytea.__version__)``)
* Your Python version and OS

For citation and license information, see :doc:`citation`.
