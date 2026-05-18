Contributing to Canapy
======================

If you are reading this, **thank you very much for considering contributing**, and welcome to the Canapy project!
**Whether you feel like a developer or not is not important. There are many ways to help:**

- `I would like to submit a bug report or suggest a new feature <#submitting-a-bug-report-or-a-feature-request>`_
- `I would like to improve code, style, or add features <#contributing-code>`_
- `I can help answer questions and solve issues <https://github.com/birds-canopy/canapy/issues>`_
- `I have a suggestion to improve or complete documentation <#documentation>`_
- `I want to cite Canapy in my work <https://github.com/birds-canopy/canapy#cite>`_
- `I use the project and want to add a star <https://github.com/birds-canopy/canapy>`_

**All of the above are equally important contributions to us.**

If you have any question about this document or contribution in general, do not hesitate to contact
axel.arnaud<@>inria.fr, main developer, or xavier.hinaut<@>inria.fr, head of the project.

This document is based on and inspired by the `ReservoirPy project contributing guidelines
<https://github.com/reservoirpy/reservoirpy/blob/master/CONTRIBUTING.rst>`_.


Submitting a Bug Report or a Feature Request
--------------------------------------------

We use GitHub issues to track all bugs and feature requests; feel free to `open
an issue <https://github.com/birds-canopy/canapy/issues>`_ if you have found a
bug or wish to see a feature implemented.

It is recommended to check that your issue complies with the following rules before submitting:

- Verify that your issue is not being currently addressed by other
  `issues <https://github.com/birds-canopy/canapy/issues?q=>`_
  or `pull requests <https://github.com/birds-canopy/canapy/pulls?q=>`_.

- If you are submitting a bug report, we strongly encourage you to follow the
  guidelines below.


How to Write a Good Bug Report
-------------------------------

When you submit an issue to `GitHub <https://github.com/birds-canopy/canapy/issues>`_,
please do your best to follow these guidelines to make it easier to provide you with good feedback:

- The ideal bug report contains a **short reproducible code snippet**, this way
  anyone can try to reproduce the bug easily (see `this
  <https://stackoverflow.com/help/mcve>`_ for more details). If your snippet is
  longer than around 50 lines, please link to a `gist <https://gist.github.com>`_
  or a GitHub repo.

- If not feasible to include a reproducible snippet, please be specific about
  **which functions or classes are involved and what the input data looks like**
  (species, audio format, annotation format, etc.).

- If an exception is raised, please **provide the full traceback**.

- Please include your **operating system type and version number**, as well as
  your **Python, canapy, numpy, reservoirpy, and librosa versions**. This
  information can be found by running::

    >>> import canapy
    >>> print("canapy", canapy.__version__)

- Please ensure all **code snippets and error messages are formatted in
  appropriate code blocks**. See `Creating and highlighting code blocks
  <https://help.github.com/articles/creating-and-highlighting-code-blocks>`_
  for more details.


Contributing Code
-----------------

To avoid duplicating work, it is highly advised that you search through the
`issue tracker <https://github.com/birds-canopy/canapy/issues>`_ and the
`PR list <https://github.com/birds-canopy/canapy/pulls>`_ before starting.
If in doubt about duplicated work, or if you want to work on a non-trivial
feature, it's recommended to first open an issue in the `issue tracker
<https://github.com/birds-canopy/canapy/issues>`_ to get feedback from core
developers.


GitHub Workflow
~~~~~~~~~~~~~~~

The preferred way to contribute to Canapy is to fork the `main repository
<https://github.com/birds-canopy/canapy/>`__ on GitHub, then submit a
"pull request" (PR).

1. `Create an account <https://github.com/join>`_ on GitHub if you do not
   already have one.

2. Fork the `project repository <https://github.com/birds-canopy/canapy>`__:
   click on the 'Fork' button near the top of the page. This creates a copy
   of the code under your account on GitHub. For more details on how to fork
   a repository see `this guide <https://help.github.com/articles/fork-a-repo/>`_.

3. Clone your fork of the Canapy repo from your GitHub account to your local
   disk::

       $ git clone git@github.com:YourLogin/canapy.git
       $ cd canapy

4. Install Canapy in editable mode with development dependencies::

       $ pip install -e ".[dev]"

5. Add the ``upstream`` remote. This saves a reference to the main Canapy
   repository, which you can use to keep your repository synchronized with
   the latest changes::

       $ git remote add upstream https://github.com/birds-canopy/canapy.git

6. Synchronize your ``master`` branch with the upstream master branch::

       $ git checkout master
       $ git pull upstream master

7. Create a feature branch to hold your development changes::

       $ git checkout -b my_feature

   Always use a feature branch. Never work directly on the ``master`` branch!

8. Develop the feature on your feature branch. When you're done editing,
   add changed files and commit::

       $ git add modified_files
       $ git commit -m "Add my feature"

   Your commit message should respect the `good commit messages guidelines
   <https://git-scm.com/book/en/v2/Distributed-Git-Contributing-to-a-Project>`_.
   Then push the changes to your GitHub account::

       $ git push -u origin my-feature

9. Follow `these instructions
   <https://help.github.com/articles/creating-a-pull-request-from-a-fork>`_
   to create a pull request from your fork. You may want to send an email
   to the core developers for visibility (see the introduction of this document).

To keep your local feature branch up to date with the main repository::

    $ git fetch upstream
    $ git rebase upstream/master

If it's been a while since you last updated, merging may be easier::

    $ git fetch upstream
    $ git merge upstream/master

Refer to the `Git documentation on resolving merge conflicts
<https://help.github.com/articles/resolving-a-merge-conflict-using-the-command-line/>`_
if needed. The `Git documentation <https://git-scm.com/documentation>`_ and
http://try.github.io are excellent resources to get started with git.


Pull Request Checklist
~~~~~~~~~~~~~~~~~~~~~~

In order to ease the reviewing process, we recommend that your contribution
complies with the following rules. The **bolded** ones are especially important:

1. **Give your pull request a helpful title** that summarises what your
   contribution does. "Fix #<ISSUE NUMBER>" alone is never a good title.

2. **Submit your code with associated unit tests**. New features should have
   their own test methods in the ``canapy/tests/`` directory, in a file named
   ``test_<module>.py`` where ``<module>`` corresponds to the code you modified.

3. **Make sure your code passes all unit tests**. First run the tests related
   to your changes, for example::

       $ pytest canapy/tests/test_corpus.py

   Then run the full test suite to ensure nothing is broken::

       $ pytest canapy/tests/

4. **Make sure your PR follows Python style guidelines**, `PEP8
   <https://www.python.org/dev/peps/pep-0008>`_. You can check for violations
   with ``flake8``::

       $ pip install flake8
       $ flake8 --ignore=D,W503,W504 canapy

   Please avoid reformatting parts of the file that your PR doesn't change.

5. **Make sure your PR follows Canapy's coding style and API**, as described
   in the `Coding Style Guidelines`_ section below.

6. **Make sure your code is properly documented**, and that the documentation
   renders correctly (see the `Documentation`_ section).

7. If your PR resolves one or more issues, `use keywords to link them
   <https://github.com/blog/1506-closing-issues-via-pull-requests/>`_
   (e.g., ``Fixes #1234``). Upon merging, those issues will automatically
   be closed by GitHub.

8. **Each PR needs to be accepted by a core developer** before being merged.

.. note::

   The current state of the Canapy codebase is not fully compliant with all
   of these guidelines, but enforcing them on new contributions will move the
   overall code quality in the right direction.


Coding Style Guidelines
-----------------------

In addition to `PEP8 <https://www.python.org/dev/peps/pep-0008>`_, Canapy
follows these guidelines:

1. Use underscores to separate words in non-class names: ``n_samples``
   rather than ``nsamples``, ``spec_radius`` rather than ``specradius``.

2. Use understandable function and variable names. Names like ``res``, ``tmp``,
   or ``aaa`` are generally bad, whereas ``corpus``, ``annotator``,
   ``n_frames``, or ``feature_matrix`` read well.

3. Use domain-consistent names that match the Canapy architecture: ``corpus``
   for a data corpus object, ``annotator`` for a model, ``transforms`` for
   preprocessing steps.

4. Avoid comments in the code — explanations belong in docstrings. This keeps
   the documentation and the code in sync, and forces writing cleaner code.

5. Avoid multiple statements on one line. Prefer a line return after control
   flow statements (``if``/``for``).

6. **Do not use** ``import *`` **in any case**. It makes the code harder to
   read and prevents static analysis tools from finding bugs.

7. Avoid the use of ``import ... as`` and of ``from ... import foo, bar``.
   Do not rename modules or their functions, as this creates objects living
   in several namespaces, which creates confusion and slows down code reviews.

8. Use double quotes ``"`` and not single quotes ``'`` for strings.

9. If you need several lines for a function call, use this syntax::

       my_function_with_a_long_name(
           my_param_1=value_1, my_param_2=value_2)

   and not the aligned style, which breaks if the function name changes.

These guidelines may be revised over time; the only constraint is that they
remain consistent throughout the codebase. To propose a change, submit a PR
to this contributing file along with the corresponding changes in the codebase.


Documentation
-------------

We are glad to accept any sort of documentation: function docstrings,
reStructuredText documents (like this one), tutorials, etc. reStructuredText
documents live in the source code repository under the ``docs/`` directory.


Building the Documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~

Install Sphinx and the required extensions::

    $ pip install sphinx

To build the documentation, run the following from the root ``canapy/`` folder::

    $ sphinx-build docs/ docs/html


Writing Docstrings
~~~~~~~~~~~~~~~~~~

Canapy uses the **NumPy docstring standard**. Please review the `NumPy docstring
guide <https://numpydoc.readthedocs.io/en/latest/format.html>`_ before writing
docstrings.

The most important sections for functions are:

1. **Summary** — one line, starting with a capital letter, ending with a period.
   Use a verb in the imperative mood (e.g. *Compute* rather than *Computes*).

2. **Description** — a more detailed multi-line description, separated from
   the summary by a blank line.

3. **Parameters** — a formatted list of arguments with types and descriptions.

4. **Returns** — a formatted list of returned objects with types and descriptions.

Here is a minimal template::

    def my_function(corpus, n_frames=100):
        """Compute something from a corpus.

        Longer description of what the function does and why.

        Parameters
        ----------
        corpus : Corpus
            The annotated audio corpus to process.
        n_frames : int, optional
            Number of frames to consider. Default: 100.

        Returns
        -------
        result : np.ndarray of shape (n_frames, n_features)
            Description of the returned array.

        Examples
        --------
        >>> result = my_function(corpus, n_frames=50)
        """

When editing reStructuredText (``.rst``) files, try to keep line length under
80 characters (exceptions include links and tables).


Code Review Guidelines
----------------------

Reviewing code contributed to the project as PRs is a crucial component of
Canapy's development. We encourage anyone to start reviewing code from other
developers. The code review process is highly educational for everyone involved.

Here are the key aspects that should be covered in any code review:

- Do we want this in the library? Is it likely to be used? Is it in the scope
  of Canapy? Will the cost of maintaining the new feature be worth its benefits?

- Is the code consistent with Canapy's API? Are public functions, classes, and
  parameters well named and intuitively designed?

- Are all public functions and classes documented clearly, with correct
  parameter types and return types?

- Is every public function tested? Do the tests validate that the code does
  what the documentation says it does? If the change is a bug fix, is a
  non-regression test included?

- Do the tests pass in the continuous integration build?

- Is the code easy to read and low on redundancy? Should variable names be
  improved for clarity or consistency?

- Could the code be rewritten to run more efficiently for relevant inputs
  (e.g. large corpora or long audio files)?

- Will the new code add any new dependencies? (this is unlikely to be accepted
  without strong justification)

- Does the documentation render properly?

- Upon merging, use the ``Rebase and Merge`` option to keep git history clean.


Issues for New Contributors
---------------------------

New contributors should look for the following tags when browsing issues.
We strongly recommend starting with "easy" issues to become familiar with
the contribution workflow before tackling more complex features.

**good first issue**
   A great way to start contributing to Canapy. These issues can be resolved
   without deep prior knowledge of the codebase. See the `good first issues list
   <https://github.com/birds-canopy/canapy/labels/good%20first%20issue>`_.

**help wanted**
   Issues that need contributors, regardless of difficulty. Also used to mark
   pull requests that have been abandoned and are available to be picked up.
   See the `help wanted list
   <https://github.com/birds-canopy/canapy/labels/help%20wanted>`_.
