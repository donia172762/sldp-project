"""Detection of AI-generated speech with traditional speech features.

Spoken Language Processing, Birzeit University, Summer 2026.

Phase 1 (this package) is the data foundation:

* :mod:`src.config`   - YAML configuration, path resolution, seeding
* :mod:`src.audio`    - the shared audio front end used by every later stage
* :mod:`src.manifest` - corpus parsing, splits, and leakage checks
* :mod:`src.qc`       - a read-only quality pass with report-ready figures

Phases 2 and 3 will add ``src/features.py`` (MFCC / LFCC / spectral / temporal
descriptors), ``src/models.py`` (SVM, Random Forest, CNN) and
``src/evaluate.py`` (accuracy, precision, recall, F1, confusion matrix,
ROC-AUC, EER). Those modules will consume manifests and call
:func:`src.audio.load_and_preprocess`; nothing in Phase 1 needs to change to
accommodate them.
"""

__version__ = "0.1.0"
