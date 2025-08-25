"""
utils/norms/registry.py
-----------------------
Flexible loader for norm classes.

Two spec styles:

  1) Short form (assumes the package is `utils.norms`):
        "growth_reserve_local:GrowthReserveLocal"

  2) Fully-qualified module path:
        "myproj.norms.growth_reserve_local:GrowthReserveLocal"
"""

from __future__ import annotations
import importlib
from typing import Any


def load_norm(spec: str, default_pkg: str = "utils.norms") -> Any:
    """
    Args:
        spec:
          - "module:ClassName"  → module resolved under `default_pkg`
          - "pkg.module:ClassName" → fully-qualified path
        default_pkg:
          Package base used by the short form.

    Returns:
        Instance of the requested class (expects zero-arg constructor).

    Raises:
        ValueError with a helpful message if anything goes wrong.
    """
    if ":" not in spec:
        raise ValueError(
            f"Norm spec must look like 'module:ClassName' or 'pkg.module:ClassName' (got {spec!r})."
        )

    module_part, class_name = spec.split(":", 1)

    # If user passed dots we treat it as fully-qualified, else prefix default_pkg.
    module_path = module_part if "." in module_part else f"{default_pkg}.{module_part}"

    try:
        module = importlib.import_module(module_path)
    except Exception as e:
        raise ValueError(f"Could not import module '{module_path}': {e}") from e

    try:
        klass = getattr(module, class_name)
    except AttributeError as e:
        raise ValueError(f"Module '{module_path}' has no class '{class_name}'.") from e

    try:
        return klass()  # zero-arg constructor
    except Exception as e:
        raise ValueError(f"Failed to instantiate {module_path}.{class_name}(): {e}") from e
