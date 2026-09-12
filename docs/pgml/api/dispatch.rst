pgml.dispatch
=============

Storage dispatch and state-of-charge integration, outside the per-snapshot solve.

A :class:`~pgml.schemas.grid_schema.Storage` element is a signed ``(P, Q)`` injection at any
single power-flow snapshot (``p_nom_w > 0`` discharges).  What couples timesteps is the state
of charge and the dispatch decision, and this module resolves both into a realized per-step
active-power sequence that the solver consumes as an ``operating_point`` — the same split
pandapower and OpenDSS make.

The dispatch RULE is the caller's ordinary Python: a profile, a threshold, a price signal, a
sampled behaviour trace.  It carries no gradient, and the caller supplies the REQUESTED power
sequence.  :func:`~pgml.dispatch.integrate_soc` then realizes that request under the
state-of-charge reserve, the energy capacity and the power rating with torch, so a gradient
with respect to the realized setpoint value is available — and, where a step is not clamped by
a limit, through the state-of-charge recurrence as well.  Never through the decision logic.

:func:`~pgml.dispatch.dispatch_storage` wraps one
:class:`~pgml.schemas.grid_schema.Storage` element's own energy-state fields
(``energy_capacity_wh``, ``soc``, ``soc_min``, ``soc_max``, ``efficiency_charge``,
``efficiency_discharge``, ``p_rated_w``), and
:func:`~pgml.dispatch.storage_operating_point` turns the realized sequence into the per-step
operating-point mapping a batched solve takes.  See
:doc:`/pgml/modeling/der-pv-storage` for the model and its cross-tool comparison.

.. automodule:: pgml.dispatch
   :members:
   :show-inheritance:
