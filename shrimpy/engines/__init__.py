"""Acquisition engines for the microscopes shrimPy drives.

- :mod:`shrimpy.engines.base_engine` — ``BaseEngine``, the shared MDA engine
- :mod:`shrimpy.engines.mantis_engine` — ``MantisEngine`` (implemented)
- :mod:`shrimpy.engines.isim_engine` — ``ISIMEngine`` (placeholder)
- :mod:`shrimpy.engines.dragonfly_engine` — ``DragonflyEngine`` (placeholder)

The engine classes are deliberately *not* re-exported here: importing an engine
pulls in the shared smart-microscopy packages (DynaTrack, and its optional
dependencies), and the CLI controls when that happens. Import the module you
need, e.g. ``from shrimpy.engines.mantis_engine import MantisEngine``.
"""
