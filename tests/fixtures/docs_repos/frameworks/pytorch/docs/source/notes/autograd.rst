Autograd mechanics
==================

.. currentmodule:: torch.autograd

.. autoclass:: torch.autograd.Function

This note explains :func:`~torch.autograd.grad` and **how** ``backward`` works for every tensor
that records operations in the graph, so that gradients flow through the whole computation graph.

.. note:: Gradients accumulate in the leaf ``.grad`` attribute unless they are cleared first.
