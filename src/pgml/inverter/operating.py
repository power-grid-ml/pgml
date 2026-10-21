"""Operating state of an inverter: what the bridge, the relay and the DC side do.

The fundamental power ``(P, Q)`` of a device comes from the grid's own power flow. The
harmonic model needs what the power flow does not carry: whether the bridge switches,
whether the AC relay is closed, and the DC-link voltage, which sets the modulation index
and the dead-time and PWM residues. :class:`InverterState` holds these per scenario.

Discrete states (from the knowledge base note on operating-state dependence):

=================  ===========  =========  ==========================================
state              bridge_on    connected  model
=================  ===========  =========  ==========================================
disconnected       no           no         EMC stages on the grid side of the relay
standby / night    no           no         same (filter behind the open relay)
bridge idle        no           yes        passive LCL with the bridge shorted
MPPT, curtailed    yes          yes        full closed-loop model at the solved P, Q
Q at night         yes          yes        full model at ``P = 0``
=================  ===========  =========  ==========================================

:class:`PVArray` maps irradiance and cell temperature to the maximum-power point. The
power follows the linear irradiance law ``P = P_stc (G / 1000) (1 + gamma (T - 25))``
that OpenDSS's PVSystem also uses (irradiance times ``Pmpp`` times a temperature
factor). The MPP voltage of a single-stage inverter is its DC-link voltage; the
logarithmic irradiance dependence ``V_mpp ~ 1 + c ln(G / 1000)`` follows from the
diode law of a cell (the open-circuit voltage grows with the logarithm of the
photocurrent) and ``c`` is a module parameter to be fitted to a datasheet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from ._common import Scalar, as_real


@dataclass(frozen=True)
class InverterState:
    """Per-scenario state of an inverter (each field a float, bool or ``[*batch]`` tensor).

    Attributes
    ----------
    u_dc_v:
        DC-link voltage.
    bridge_on:
        Whether the bridge switches (and the control acts).
    connected:
        Whether the AC relay is closed. With the relay open only the EMC stages on the
        grid side remain at the terminals (``emc_on_grid_side`` of the model).
    carrier_angle_rad:
        Carrier phase of the PWM; carriers of different units are not synchronised, so
        this is a free parameter of a deterministic (coherent) evaluation.
    """

    u_dc_v: Scalar
    bridge_on: object = True
    connected: object = True
    carrier_angle_rad: Scalar = 0.0

    def masks(self, rdt, device) -> tuple[Tensor, Tensor]:
        """``(bridge_on, connected)`` as bool tensors; the bridge needs the relay closed."""
        on = torch.as_tensor(self.bridge_on, device=device).to(torch.bool)
        conn = torch.as_tensor(self.connected, device=device).to(torch.bool)
        return on & conn, conn


@dataclass(frozen=True)
class PVArray:
    """Maximum-power point of a PV array.

    Attributes
    ----------
    p_stc_w, v_mpp_stc_v:
        MPP power and voltage at standard test conditions (1000 W/m², 25 °C).
    gamma_p_per_k:
        Temperature coefficient of the MPP power [1/K].
    beta_v_per_k:
        Temperature coefficient of the MPP voltage [1/K].
    v_log_irradiance:
        ``c`` in ``V_mpp ~ 1 + c ln(G / 1000)``.
    noct_c:
        Nominal operating cell temperature for :meth:`cell_temperature`.
    """

    p_stc_w: Scalar
    v_mpp_stc_v: Scalar
    gamma_p_per_k: Scalar = -0.0037
    beta_v_per_k: Scalar = -0.0030
    v_log_irradiance: Scalar = 0.05
    noct_c: float = 45.0

    def cell_temperature(self, irradiance_w_m2: Tensor, ambient_c: Tensor) -> Tensor:
        """NOCT estimate ``T_cell = T_amb + (NOCT - 20) G / 800``."""
        return ambient_c + (self.noct_c - 20.0) * irradiance_w_m2 / 800.0

    def mpp(
        self, irradiance_w_m2: Tensor, cell_temperature_c: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
        """``(P_mpp, V_mpp)`` at the given irradiance and cell temperature."""
        g = irradiance_w_m2
        rdt = g.dtype
        dev = g.device
        t = (
            cell_temperature_c
            if cell_temperature_c is not None
            else torch.full_like(g, 25.0)
        )
        dt = t - 25.0
        p = (
            as_real(self.p_stc_w, rdt, dev)
            * (g / 1000.0)
            * (1.0 + as_real(self.gamma_p_per_k, rdt, dev) * dt)
        )
        ratio = torch.clamp(g / 1000.0, min=1e-6)
        v = (
            as_real(self.v_mpp_stc_v, rdt, dev)
            * (1.0 + as_real(self.beta_v_per_k, rdt, dev) * dt)
            * (1.0 + as_real(self.v_log_irradiance, rdt, dev) * torch.log(ratio))
        )
        return torch.clamp(p, min=0.0), v
