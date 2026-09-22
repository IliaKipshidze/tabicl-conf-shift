# nanoTabPFN model provenance

`model.py` is adapted from the AutoML group's nanoTabPFN `model.py` at
commit [`530670098e4befabe80825a3eefed408a926e34a`](https://github.com/automl/nanoTabPFN/blob/530670098e4befabe80825a3eefed408a926e34a/model.py).
The SHA-256 of that upstream source file is
`5a0f6ed071addb27b5be3ca361d759aa77e651cac3a9ae484bb62c8227c21168`.

The upstream project publishes this code under Apache License 2.0. Its license
is reproduced in `LICENSE-NANOTABPFN`; the surrounding TabICL project retains
its own license. The only changes here are formatting, concise comments and
docstrings, and omission of the NumPy-dependent sklearn-style classifier
wrapper. The model architecture, operations, module/parameter names, and
forward signature are retained. No paper-era weights are included.
