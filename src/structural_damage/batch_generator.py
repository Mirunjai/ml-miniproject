"""Batch generator re-export for backward compatibility.

The BatchGenerator class is defined in structural_damage.dataset.
This module provides a convenient import alias so that tests and
other code can import from structural_damage.batch_generator.
"""

from structural_damage.dataset import BatchGenerator

__all__ = ["BatchGenerator"]
