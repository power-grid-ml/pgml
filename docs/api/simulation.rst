pgml.simulation
===============

High-level simulation entry point — the front door for external consumers.

:func:`~pgml.simulation.simulate` dispatches to the differentiable solvers and returns a
:class:`~pgml.simulation.SolvedState` — a complete, lazily-derived snapshot of the solved
grid.  :func:`~pgml.simulation.simulate_serializable` is the convenience wrapper for REST /
file-persistence callers that want a JSON-ready :class:`~pgml.simulation.ResultBundle`
instead.

.. rubric:: Design split

- :class:`~pgml.simulation.SimulationConfig` is the **serializable** definition of WHAT to
  simulate (calculation type, harmonic orders, slack, symmetry, tolerances).  It is a
  pydantic model — a clean JSON body for a REST handler.
- ``device`` and ``dtype`` are **execution** concerns (WHERE to run / precision), passed as
  keyword arguments to :func:`~pgml.simulation.simulate`, not part of the config.

.. rubric:: Differentiability

Gradients flow from ``grid`` parameters through all :class:`~pgml.simulation.SolvedState`
tensor accessors (``node_voltages``, ``voltage``, ``branch_currents``, ``branch_flows``,
``thd``).  :meth:`~pgml.simulation.SolvedState.to_result_set` detaches tensors when
materialising the serializable :class:`~pgml.simulation.ResultBundle` — use the tensor
accessors for autograd.

.. rubric:: Alternative entry points

For batched training-data generation use :func:`pgml.scenarios.run_scenarios`; for raw
differentiable tensors at minimal overhead use :mod:`pgml.solver` directly.

Usage example
-------------

Run a harmonic power flow and read node voltages::

    import pgml

    state = pgml.simulate(grid)            # harmonic, orders 1 3 5 7 9 11 13
    V = state.node_voltages()              # [H, N] complex, differentiable
    thd_bus2 = state.thd(node_id=2, phase=Phase.A)

Serialise for REST / persistence::

    bundle = pgml.simulate_serializable(grid)
    json_body = bundle.model_dump_json()

Custom config (power flow only, 64-bit complex)::

    from pgml import simulate, SimulationConfig

    cfg = SimulationConfig(calculation="power_flow")
    state = simulate(grid, cfg, dtype="complex128", device="cpu")

.. automodule:: pgml.simulation
   :members:
   :show-inheritance:
