pgml.config
===========

Explicit, documented modelling defaults.

All default constants and default model selections live in ``defaults.yaml``,
ordered by component with a value, units, and a short description.

Resolution precedence:

    **explicit (user)** > **config (this package)** > **converter (source library)**

This means a user-supplied value always wins; a config default overrides
whatever a converter would infer.

.. automodule:: pgml.config
   :members:
   :show-inheritance:
