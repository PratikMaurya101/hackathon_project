
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import math
 
 
 
# ─────────────────────────────────────────────
# 1. CONSTANTS & PHYSICAL PARAMETERS
# ─────────────────────────────────────────────
 
N_ZONES       = 5
STATE_DIM     = 31      # [T×5, RH×5, CO2×5, flow×5, Tout, RHout, Qsol, Occ×5, h, day, E_cum]
ACTION_DIM    = 15       # [Tset×5, m_dot×5, Tsupply, damper×5, chiller]
GAMMA         = 0.99
TAU           = 0.005
ACTOR_LR      = 1e-4
CRITIC_LR     = 1e-4
BATCH_SIZE    = 64
BUFFER_SIZE   = int(1e5)
MAX_EPISODES  = 5000
STEPS_PER_EP  = 288      # 24 h at 5-min intervals
LAMBDA_PHYS   = 0.01
LAMBDA_ENERGY = 1.0
LAMBDA_PSYCHRO= 0.1
LAMBDA_COMFORT= 0.05
GRAD_CLIP     = 1.0
 
# Actuator physical bounds  [lower, upper]
ACT_BOUNDS = {
    "Tset":    (20.0, 24.0),   # zone setpoint °C   (×5 zones)
    "m_dot":   (0.0,  10.0),   # mass flow kg/s     (×5 zones)
    "Tsupply": (12.0, 20.0),   # supply air temp °C
    "damper":  (0.0,   1.0),   # OA damper fraction (×5 zones ... but we use 1 for clarity)
    "chiller": (0.0,   1.0),   # chiller load
}
 
# Flatten bounds into arrays matching ACTION_DIM=15
LOWER = np.array(
    [20.0]*5 +   # Tset ×5
    [0.0]*5  +   # m_dot ×5
    [12.0]   +   # Tsupply
    [0.0]*3  +   # damper ×3 (simplified to fit dim=15)
    [0.0]        # chiller
, dtype=np.float32)
 
UPPER = np.array(
    [24.0]*5 +
    [10.0]*5 +
    [20.0]   +
    [1.0]*3  +
    [1.0]
, dtype=np.float32)
 
# Rate limits per control step (5 min = 300 s)
RATE_MAX = np.array(
    [0.5]*5  +   # Tset: max 0.5 °C per step
    [1.0]*5  +   # m_dot: max 1.0 kg/s per step
    [1.0]    +   # Tsupply: max 1.0 °C per step
    [0.2]*3  +   # damper: max 0.2 per step
    [0.2]        # chiller: max 0.2 per step
, dtype=np.float32)
 
# PMV comfort corridor threshold
PMV_THRESHOLD = 0.5
 
# Psychrometric constants
P_BAR   = 101325.0   # barometric pressure [Pa]
CP      = 1006.0     # specific heat dry air [J/kg·K]
CPV     = 1860.0     # specific heat water vapour [J/kg·K]
H_FG    = 2.5e6      # latent heat [J/kg]
 
 
# ─────────────────────────────────────────────
# 2. PSYCHROMETRIC HELPERS
# ─────────────────────────────────────────────
 
def saturation_pressure(T_K: torch.Tensor) -> torch.Tensor:
    """Magnus-Tetens saturation vapour pressure [Pa]. T in Kelvin."""
    T_C = T_K - 273.15
    return 610.94 * torch.exp(17.625 * T_C / (T_C + 243.04))
 
 
def relative_humidity(T_K: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    """Compute RH from temperature and humidity ratio."""
    pws = saturation_pressure(T_K)
    pw  = omega * P_BAR / (0.62198 + omega)
    return pw / (pws + 1e-8)
 
 
def psychrometric_barrier(phi: torch.Tensor,
                           omega: torch.Tensor,
                           k: float = 10.0) -> torch.Tensor:
    """
    Soft barrier penalising infeasible psychrometric states.
    phi  ∈ [0,1],  omega ≥ 0
    Returns scalar barrier loss (sum over batch).
    """
    B = (
        torch.nn.functional.softplus( k * (phi   - 1.0)) +
        torch.nn.functional.softplus( k * (0.0   - phi )) +
        torch.nn.functional.softplus( k * (0.0   - omega))
    )
    return B.mean()
 
 
# ─────────────────────────────────────────────
# 3. ORNSTEIN-UHLENBECK NOISE
# ─────────────────────────────────────────────
 
class OUNoise:
    """Ornstein-Uhlenbeck process for temporally correlated exploration."""
 
    def __init__(self, size: int, mu: float = 0.0,
                 theta: float = 0.15, sigma: float = 0.1):
        self.size  = size
        self.mu    = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()
 
    def reset(self):
        self.state = self.mu.copy()
 
    def sample(self) -> np.ndarray:
        dx = self.theta * (self.mu - self.state) + \
             self.sigma * np.random.randn(self.size)
        self.state += dx
        return self.state.astype(np.float32)
 
 
# ─────────────────────────────────────────────
# 4. REPLAY BUFFER
# ─────────────────────────────────────────────
 
class ReplayBuffer:
    """Fixed-size circular replay buffer."""
 
    def __init__(self, capacity: int = BUFFER_SIZE):
        self.buf = deque(maxlen=capacity)
 
    def push(self, state, action_raw, reward, next_state, done):
        self.buf.append((
            state.astype(np.float32),
            action_raw.astype(np.float32),
            float(reward),
            next_state.astype(np.float32),
            float(done)
        ))
 
    def sample(self, batch_size: int = BATCH_SIZE):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = zip(*batch)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return (
            torch.tensor(np.array(s),  dtype=torch.float32).to(device),
            torch.tensor(np.array(a),  dtype=torch.float32).to(device),
            torch.tensor(np.array(r),  dtype=torch.float32).unsqueeze(1).to(device),
            torch.tensor(np.array(s2), dtype=torch.float32).to(device),
            torch.tensor(np.array(d),  dtype=torch.float32).unsqueeze(1).to(device),
        )
 
    def __len__(self):
        return len(self.buf)
 
 
# ─────────────────────────────────────────────
# 5. DIFFERENTIABLE CONSTRAINT LAYER  Π_phys
# ─────────────────────────────────────────────
 
class PhysicsProjection(nn.Module):
    """
    Differentiable thermodynamic projection layer.
 
    Step 1 — Range mapping:   â = ℓ + (u−ℓ)/2 · (1 + tanh(a_raw))
    Step 2 — Rate limiting:   a = a_prev + clamp_soft(â − a_prev, r_max·Δt)
    Step 3 — Psychrometric barrier is added to the actor loss (not projected here).
    """
 
    def __init__(self):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lower    = torch.tensor(LOWER,    device=device)
        self.upper    = torch.tensor(UPPER,    device=device)
        self.rate_max = torch.tensor(RATE_MAX, device=device)
 
    def range_map(self, a_raw: torch.Tensor) -> torch.Tensor:
        """Scale tanh output to physical actuator bounds."""
        l, u = self.lower, self.upper
        return l + (u - l) / 2.0 * (1.0 + torch.tanh(a_raw))
 
    def rate_limit(self, a_hat: torch.Tensor,
                   a_prev: torch.Tensor) -> torch.Tensor:
        """Soft clamp on per-step rate of change."""
        delta   = a_hat - a_prev
        clamped = self.rate_max * torch.tanh(delta / (self.rate_max + 1e-8))
        return a_prev + clamped
 
    def forward(self, a_raw: torch.Tensor,
                a_prev: torch.Tensor) -> torch.Tensor:
        a_hat = self.range_map(a_raw)
        a_phy = self.rate_limit(a_hat, a_prev)
        return a_phy
 
 
# ─────────────────────────────────────────────
# 6. ACTOR NETWORK  μ_θ
# ─────────────────────────────────────────────
 
class Actor(nn.Module):
    """
    MLP actor: s → a_raw ∈ [−1,1]^ACTION_DIM  (tanh output).
    Hidden layers [256, 512, 256] as in the paper.
    """
 
    def __init__(self, state_dim: int = STATE_DIM,
                 action_dim: int = ACTION_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 512),       nn.ReLU(),
            nn.Linear(512, 256),       nn.ReLU(),
            nn.Linear(256, action_dim),nn.Tanh(),
        )
        self._init_weights()
 
    def _init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=0.01)
                nn.init.zeros_(layer.bias)
 
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)
 
 
# ─────────────────────────────────────────────
# 7. CRITIC NETWORK  Q_φ
# ─────────────────────────────────────────────
 
class Critic(nn.Module):
    """
    MLP critic: (s, a_phys) → Q-value  ∈ ℝ.
    State and action are concatenated before the first layer.
    """
 
    def __init__(self, state_dim: int = STATE_DIM,
                 action_dim: int = ACTION_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 512),                     nn.ReLU(),
            nn.Linear(512, 256),                     nn.ReLU(),
            nn.Linear(256, 1),
        )
        self._init_weights()
 
    def _init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                nn.init.zeros_(layer.bias)
 
    def forward(self, state: torch.Tensor,
                action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1))
 
 
# ─────────────────────────────────────────────
# 8. PHYSICS REGULARIZATION LOSS
# ─────────────────────────────────────────────
 
class PhysicsLoss(nn.Module):
    """
    L_phys = λ1·L_energy + λ2·L_psychro + λ3·L_comfort
 
    L_energy:  normalized MSE between predicted and observed temperature rates
    L_psychro: psychrometric barrier on predicted (φ, ω)
    L_comfort: Huber loss on PMV exceeding the comfort corridor ±0.5
    """
 
    def __init__(self,
                 lambda1: float = LAMBDA_ENERGY,
                 lambda2: float = LAMBDA_PSYCHRO,
                 lambda3: float = LAMBDA_COMFORT):
        super().__init__()
        self.l1 = lambda1
        self.l2 = lambda2
        self.l3 = lambda3
 
    def energy_loss(self, T_pred: torch.Tensor,
                    T_obs: torch.Tensor) -> torch.Tensor:
        """Normalized MSE on temperature rate residuals."""
        residual = T_pred - T_obs
        return (residual ** 2).mean()
 
    def comfort_loss(self, pmv: torch.Tensor) -> torch.Tensor:
        """Soft Huber penalty outside PMV corridor [−0.5, 0.5]."""
        excess = torch.clamp(pmv.abs() - PMV_THRESHOLD, min=0.0)
        return nn.functional.huber_loss(excess,
                                        torch.zeros_like(excess),
                                        delta=0.1)
 
    def forward(self,
                T_pred:  torch.Tensor,
                T_obs:   torch.Tensor,
                phi_pred: torch.Tensor,
                omega_pred: torch.Tensor,
                pmv:     torch.Tensor) -> torch.Tensor:
 
        L_e = self.energy_loss(T_pred, T_obs)
        L_p = psychrometric_barrier(phi_pred, omega_pred)
        L_c = self.comfort_loss(pmv)
 
        return self.l1 * L_e + self.l2 * L_p + self.l3 * L_c
 
 
# ─────────────────────────────────────────────
# 9. MULTI-ZONE RC THERMAL SIMULATOR
# ─────────────────────────────────────────────
 
class RCThermalSimulator:
    """
    Simplified 5-zone resistance-capacitance thermal simulator.
    State: s ∈ ℝ^47
    Action: a ∈ ℝ^15 (feasible, after projection)
 
    Physics per zone i:
        C_i · dT_i/dt = Σ_j (T_j−T_i)/R_ij + Q_HVAC_i + (T_out−T_i)/R_out_i
    """
 
    def __init__(self,
                 n_zones: int = N_ZONES,
                 dt_ctrl: float = 300.0,   # 5-min control step [s]
                 dt_ode:  float = 60.0):   # 1-min integration step [s]
        self.n    = n_zones
        self.dt_c = dt_ctrl
        self.dt_o = dt_ode
        self.n_substeps = int(dt_ctrl / dt_ode)
 
        rng = np.random.default_rng(42)
 
        # Zone capacitances [J/K]
        self.C = rng.uniform(0.8e6, 1.4e6, n_zones).astype(np.float32)
 
        # Inter-zone conductances [W/K] — sparse upper triangle
        self.UA_zone = np.zeros((n_zones, n_zones), dtype=np.float32)
        pairs = [(0,1),(0,4),(1,2),(1,4),(2,3),(2,4),(3,4)]
        for i, j in pairs:
            v = rng.uniform(40, 140)
            self.UA_zone[i, j] = v
            self.UA_zone[j, i] = v
 
        # Envelope conductances [W/K]
        self.UA_out = rng.uniform(120, 350, n_zones).astype(np.float32)
 
        # Reward weights
        self.alpha = 1.0   # energy
        self.beta  = 0.3   # comfort
        self.gamma = 0.2   # peak demand
        self.delta = 0.1   # IAQ
 
        self.t_step    = 0
        self.T         = None
        self.T_out     = None
        self.a_prev    = LOWER.copy()
        self.E_cumul   = 0.0
        self.P_ref     = 50.0   # kW reference for normalisation
 
    # ── helpers ──────────────────────────────
 
    def _outdoor_temp(self) -> float:
        """Synthetic diurnal + seasonal outdoor temperature."""
        h   = (self.t_step * 5 / 60) % 24
        day = (self.t_step * 5 / 1440) % 365
        return 10.0 + 8.0 * math.sin(2*math.pi*(day-80)/365) \
                    + 5.0 * math.sin(2*math.pi*(h-14)/24)
 
    def _solar_gain(self, zone: int) -> float:
        """Simplified solar gain [W] per zone based on orientation."""
        h = (self.t_step * 5 / 60) % 24
        peak_hour = [12, 10, 12, 14, 12][zone]
        if 6 <= h <= 18:
            return max(0, 500 * math.cos(math.pi*(h - peak_hour)/6))
        return 0.0
 
    def _occupancy(self, zone: int) -> float:
        """Binary occupancy: 1 during office hours (08-18), else 0."""
        h = (self.t_step * 5 / 60) % 24
        return 1.0 if (8 <= h < 18) else 0.0
 
    def _internal_gains(self, zone: int) -> float:
        """Lighting + equipment + occupant gains [W]."""
        occ = self._occupancy(zone)
        return occ * (10 + 10 + 22 * 75 / 10)   # W/m² × area simplified
 
    def _pmv(self, T_zone: float) -> float:
        """
        Simplified PMV approximation for sedentary office occupants.
        Full ISO 7730 requires iterative solution; this linear form is
        adequate for reward shaping (met=1.1, clo=0.5, v_air=0.1 m/s).
        """
        T_neutral = 22.0
        return 0.303 * math.exp(-0.036 * 1.1) * (
            58.15 * (1.1 - 0.35)
            - 0.1 * (T_zone - 22.0) + 0.7 * (T_zone - T_neutral)
        ) * 0.08  # scale to ≈ ±1.5 range
 
    # ── ODE integration ───────────────────────
 
    def _dT_dt(self, T: np.ndarray, Q_hvac: np.ndarray) -> np.ndarray:
        dT = np.zeros(self.n, dtype=np.float32)
        for i in range(self.n):
            q_zone  = np.sum(self.UA_zone[i] * (T - T[i]))
            q_env   = self.UA_out[i] * (self.T_out - T[i])
            q_int   = self._internal_gains(i)
            q_sol   = self._solar_gain(i)
            dT[i]   = (q_zone + q_env + Q_hvac[i] + q_int + q_sol) / self.C[i]
        return dT
 
    def _integrate_implicit(self, T: np.ndarray, Q_hvac: np.ndarray,
                             dt: float) -> np.ndarray:
        """
        Semi-implicit (backward Euler) update of zone temperatures.
 
        Linear conduction/HVAC terms are solved implicitly so the
        integrator stays unconditionally stable even when the (possibly
        untrained / noisy) actor commands large mass flows or sets large
        UA-equivalent couplings. Nonlinear/exogenous terms (solar,
        internal gains) are kept explicit since they don't depend on T.
 
        For each zone i:
            C_i * (T_new_i - T_i) / dt =
                Σ_j UA_ij * (T_new_j - T_new_i)   [treated w/ old neighbor T for simplicity]
              + UA_out_i * (T_out - T_new_i)
              + Q_hvac_i + Q_int_i + Q_sol_i
 
        Solving for T_new_i (zone-decoupled implicit, neighbors lagged by
        one step — still far more stable than explicit Euler because the
        zone's own large self-coupling terms, e.g. HVAC convective duty
        and envelope UA, are on the implicit side):
 
            T_new_i = (C_i/dt * T_i + UA_total_i_neighbors*T_avg_neighbors
                       + UA_out_i*T_out + Q_hvac_i + Q_int_i + Q_sol_i)
                      / (C_i/dt + UA_self_i)
 
        where UA_self_i = Σ_j UA_ij + UA_out_i is the total conductance
        draining/feeding zone i (the term responsible for instability).
        """
        T_new = T.copy()
        UA_self = self.UA_zone.sum(axis=1) + self.UA_out   # [n]
        for i in range(self.n):
            q_neighbors = np.sum(self.UA_zone[i] * T)        # uses old T (explicit neighbor coupling)
            q_int = self._internal_gains(i)
            q_sol = self._solar_gain(i)
            numerator = (self.C[i] / dt) * T[i] + q_neighbors \
                        + self.UA_out[i] * self.T_out \
                        + Q_hvac[i] + q_int + q_sol
            denominator = (self.C[i] / dt) + UA_self[i]
            T_new[i] = numerator / denominator
        return T_new
 
    def _hvac_power(self, action: np.ndarray) -> tuple:
        """
        Compute HVAC heat transfer and electrical power.

        Returns:
            Q_hvac  : np.ndarray [W per zone]
            P_elec  : kW
            P_cool  : kW
            P_heat  : kW
            P_fan   : kW
        """

        T_sup   = action[10]
        m_dot   = action[5:10]
        chiller = action[14]

        Q_hvac = np.array([
            m_dot[i] * CP * (T_sup - self.T[i])
            for i in range(self.n)
        ], dtype=np.float32)

        Q_HVAC_MAX = 50000.0
        Q_hvac = np.clip(Q_hvac, -Q_HVAC_MAX, Q_HVAC_MAX)

        COP = max(2.0, 4.12 - 1.5 * chiller)
        HEATING_COP = 3.5

        P_cool = sum(max(0, -q) for q in Q_hvac) / (COP * 1000)
        P_heat = sum(max(0, q) for q in Q_hvac) / (HEATING_COP * 1000)
        P_fan = sum(m_dot) * 100 / 1000
        P_elec = P_cool + P_heat + P_fan

        return (
            Q_hvac,
            P_elec,
            P_cool,
            P_heat,
            P_fan
        )

    # ── Gym-style API ──────────────────────────
 
    def reset(self) -> np.ndarray:
        self.t_step  = 0
        self.E_cumul = 0.0
        self.T       = np.random.uniform(20.0, 24.0, self.n).astype(np.float32)
        self.T_out   = self._outdoor_temp()
        self.a_prev  = LOWER.copy()
        return self._build_state()
 
    def step(self, action: np.ndarray) -> tuple:
        """
        Advance simulator by one 5-min control interval.
        Returns (next_state, reward, done, info).
        """
        self.T_out  = self._outdoor_temp()
        Q_hvac, P_elec, P_cool, P_heat, P_fan = \
    self._hvac_power(action)
        # Semi-implicit (backward Euler) integration at 60-s substeps.
        # Unconditionally stable regardless of action magnitude, unlike
        # the previous forward-Euler scheme which could diverge when
        # large HVAC mass flows pushed the per-zone time constant below
        # the substep size.
        T = self.T.copy()
        for _ in range(self.n_substeps):
            T = self._integrate_implicit(T, Q_hvac, self.dt_o)
        self.T = T
 
        # Defensive guard: clip to a physically sane range and replace
        # any residual non-finite values, so a single bad transition
        # can never poison the replay buffer / network weights with NaNs.
        self.T = np.clip(self.T, -20.0, 60.0)
        if not np.all(np.isfinite(self.T)):
            self.T = np.nan_to_num(self.T, nan=22.0, posinf=60.0, neginf=-20.0)
 
        # Energy accounting
        dt_h = self.dt_c / 3600

        E_step = P_elec * dt_h

        E_cool = P_cool * dt_h
        E_heat = P_heat * dt_h
        E_fan  = P_fan  * dt_h  # kWh
        self.E_cumul += E_step
 
        # Reward components
        E_REF = 100.0
        r_energy  = -self.alpha * (E_step / E_REF)
        pmv_vals  = [self._pmv(self.T[i]) for i in range(self.n)]
        r_comfort = -self.beta * sum(max(0, abs(p) - 0.5) for p in pmv_vals)
        r_peak    = -self.gamma * max(0, P_elec - self.P_ref) / self.P_ref
        occ       = [self._occupancy(i) for i in range(self.n)]
        r_iaq     = self.delta * sum(occ) / self.n
        reward    = r_energy + r_comfort + r_peak + r_iaq
 
        self.t_step += 1
        done  = (self.t_step >= STEPS_PER_EP)
        self.a_prev = action.copy()
        return (
            self._build_state(),
            reward,
            done,
            {
                "E_step": E_step,
                "E_cool": E_cool,
                "E_heat": E_heat,
                "E_fan": E_fan,
                "P_elec": P_elec,
                "P_cool": P_cool,
                "P_heat": P_heat,
                "P_fan": P_fan,
                "pmv": pmv_vals
            }
        )
 
    def _build_state(self) -> np.ndarray:
        """Assemble 47-dim state vector."""
        RH_zones = [0.5] * self.n     # simplified: assume 50% RH
        CO2      = [600.0 + 200.0 * self._occupancy(i) for i in range(self.n)]
        flows    = self.a_prev[5:10].tolist()
        h_norm   = (self.t_step % 288) / 288.0
        day_norm = (self.t_step // 288 % 365) / 365.0
        state    = (
            self.T.tolist()  +       # 5  zone temps
            RH_zones         +       # 5  RH
            CO2              +       # 5  CO2
            flows            +       # 5  current flows
            [self.T_out]     +       # 1  outdoor temp
            [0.6]            +       # 1  outdoor RH
            [max(0, self._solar_gain(0))] +  # 1 solar irr
            [self._occupancy(i) for i in range(self.n)] +  # 5 occupancy
            [h_norm, day_norm] +     # 2  time features
            [self.E_cumul / 100.0]   # 1  cumulative energy (normalised)
        )
        return np.array(state, dtype=np.float32)
 
 
# ─────────────────────────────────────────────
# 10. TC-DDPG AGENT
# ─────────────────────────────────────────────
 
class TCDDPG:
    """
    Thermodynamically-Constrained Deep Deterministic Policy Gradient.
 
    Key differences from vanilla DDPG:
      • PhysicsProjection layer applied to all actor outputs
      • PhysicsLoss added to actor gradient
      • Both critic targets and actor paths use projected actions
    """
 
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[TC-DDPG] Using device: {self.device}")
 
        # Networks
        self.actor        = Actor().to(self.device)
        self.actor_target = Actor().to(self.device)
        self.critic       = Critic().to(self.device)
        self.critic_target= Critic().to(self.device)
 
        # Hard-copy weights to targets
        self.actor_target .load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
 
        # Optimizers
        self.opt_actor  = optim.Adam(self.actor.parameters(),  lr=ACTOR_LR)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)
 
        # Modules
        self.projection  = PhysicsProjection().to(self.device)
        self.phys_loss   = PhysicsLoss().to(self.device)
        self.buffer      = ReplayBuffer()
        self.noise       = OUNoise(ACTION_DIM)
        self.update_step = 0
 
    # ── projection helper ─────────────────────
 
    def _project(self, a_raw: torch.Tensor,
                 a_prev: torch.Tensor) -> torch.Tensor:
        return self.projection(a_raw, a_prev)
 
    # ── action selection ──────────────────────
 
    @torch.no_grad()
    def select_action(self, state: np.ndarray,
                      a_prev: np.ndarray,
                      explore: bool = True) -> tuple:
        """Return (a_raw, a_feasible) numpy arrays."""
        s      = torch.tensor(state,  dtype=torch.float32).unsqueeze(0).to(self.device)
        ap     = torch.tensor(a_prev, dtype=torch.float32).unsqueeze(0).to(self.device)
        a_raw  = self.actor(s)
 
        if explore:
            noise  = torch.tensor(self.noise.sample(),
                                  dtype=torch.float32).unsqueeze(0).to(self.device)
            a_raw  = torch.clamp(a_raw + noise, -1.0, 1.0)
 
        a_phy  = self._project(a_raw, ap)
        return a_raw.cpu().numpy().squeeze(), a_phy.cpu().numpy().squeeze()
 
    # ── soft update ───────────────────────────
 
    def _soft_update(self, live: nn.Module, target: nn.Module):
        for p, pt in zip(live.parameters(), target.parameters()):
            pt.data.copy_(TAU * p.data + (1 - TAU) * pt.data)
 
    # ── placeholder physics tensors ───────────
    # In production these come from the RC model's one-step prediction;
    # here we approximate from the batch states and actions.
 
    def _physics_tensors(self, S: torch.Tensor,
                         A: torch.Tensor) -> tuple:
        """
        Derive approximate physics quantities from state/action batch.
        Returns (T_pred, T_obs, phi_pred, omega_pred, pmv).
        """
        # Zone temps from state [first 5 dims]
        T_obs   = S[:, :N_ZONES]                          # [B, 5]
        Tsup    = A[:, 10:11].expand(-1, N_ZONES)         # supply temp
        m_dot   = A[:, 5:10]                              # mass flow
        # Simple 1-step temperature prediction
        dT_hvac = m_dot * CP * (Tsup - T_obs) / 1e6      # very coarse
        T_pred  = T_obs + dT_hvac
 
        # Psychrometric: RH from state [5:10], approx omega
        RH      = S[:, N_ZONES:2*N_ZONES].clamp(0.0, 1.0)
        T_K     = T_obs[:, 0:1] + 273.15
        pws     = saturation_pressure(T_K)
        pw      = RH[:, 0:1] * pws
        omega   = 0.62198 * pw / (P_BAR - pw + 1e-8)
        phi     = RH[:, 0:1]
 
        # PMV approximation
        pmv = 0.3 * (T_obs.mean(dim=1, keepdim=True) - 22.0)
 
        return T_pred, T_obs, phi, omega, pmv
 
    # ── update step ───────────────────────────
 
    def update(self):
        if len(self.buffer) < BATCH_SIZE:
            return None, None
 
        S, A_raw, R, S2, D = self.buffer.sample()
 
        # Project stored raw actions
        # (a_prev not tracked per-transition; use zero as approximation
        #  — in production store a_prev alongside a_raw in the buffer)
        a_prev_dummy = torch.zeros_like(A_raw)
        A  = self._project(A_raw, a_prev_dummy)
 
        # ── Critic update ────────────────────
        with torch.no_grad():
            A2_raw = self.actor_target(S2)
            A2     = self._project(A2_raw, torch.zeros_like(A2_raw))
            y      = R + GAMMA * self.critic_target(S2, A2) * (1 - D)
 
        L_critic = nn.functional.mse_loss(self.critic(S, A), y)
        self.opt_critic.zero_grad()
        L_critic.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), GRAD_CLIP)
        self.opt_critic.step()
 
        # ── Actor update ─────────────────────
        A_hat = self._project(self.actor(S), a_prev_dummy)
        Q_val = self.critic(S, A_hat)
 
        # Physics regularization
        T_pred, T_obs, phi, omega, pmv = self._physics_tensors(S, A_hat)
        L_phys  = self.phys_loss(T_pred, T_obs, phi, omega, pmv)
        L_actor = -Q_val.mean() + LAMBDA_PHYS * L_phys
 
        self.opt_actor.zero_grad()
        L_actor.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), GRAD_CLIP)
        self.opt_actor.step()
 
        # ── Soft target update ────────────────
        self._soft_update(self.actor,  self.actor_target)
        self._soft_update(self.critic, self.critic_target)
 
        return L_critic.item(), L_actor.item()
 
    # ── evaluation ────────────────────────────
 
    @torch.no_grad()
    def evaluate(self, env: RCThermalSimulator, n_episodes: int = 5) -> dict:
        """Run noise-free evaluation rollouts and return metrics."""
        total_energy, total_comfort_viol, total_steps = 0.0, 0.0, 0
        for _ in range(n_episodes):
            s       = env.reset()
            a_prev  = np.zeros(ACTION_DIM, dtype=np.float32)
            done    = False
            while not done:
                _, a = self.select_action(s, a_prev, explore=False)
                s, _, done, info = env.step(a)
                a_prev = a
                total_energy      += info["P_elec"] * (300 / 3600)
                total_comfort_viol+= sum(max(0, abs(p) - 0.5) for p in info["pmv"])
                total_steps       += 1
        n = n_episodes
        return {
            "energy_kWh":        total_energy / n,
            "comfort_viol_h":    total_comfort_viol / n / 12,  # steps → hours
            "avg_steps":         total_steps / n,
        }
 
    # ── checkpoint ────────────────────────────
 
    def save(self, path: str = "tc_ddpg_checkpoint.pt"):
        torch.save({
            "actor":         self.actor.state_dict(),
            "critic":        self.critic.state_dict(),
            "actor_target":  self.actor_target.state_dict(),
            "critic_target": self.critic_target.state_dict(),
        }, path)
        print(f"[TC-DDPG] Checkpoint saved → {path}")
 
    def load(self, path: str = "tc_ddpg_checkpoint.pt"):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        print(f"[TC-DDPG] Checkpoint loaded ← {path}")
 
 
# ─────────────────────────────────────────────
# 11. TRAINING LOOP
# ─────────────────────────────────────────────
 
def train(n_episodes:   int   = MAX_EPISODES,
          eval_every:   int   = 50,
          patience:     int   = 20,
          save_path:    str   = "tc_ddpg_best.pt") -> dict:
    """
    Full TC-DDPG training loop.
 
    Args:
        n_episodes:  maximum training episodes
        eval_every:  evaluate every K episodes
        patience:    early-stopping patience (eval rounds without improvement)
        save_path:   path for best checkpoint
 
    Returns:
        history dict with episode rewards, losses, and eval metrics
    """
    env   = RCThermalSimulator()
    agent = TCDDPG()
 
    history = {
        "ep_reward": [], "L_critic": [], "L_actor": [],
        "eval_energy": [], "eval_comfort": [],
    }
 
    best_eval_energy = float("inf")
    patience_counter = 0
 
    print(f"\n{'='*60}")
    print("  TC-DDPG Training Started")
    print(f"  Episodes: {n_episodes}  |  Steps/ep: {STEPS_PER_EP}")
    print(f"  λ_phys={LAMBDA_PHYS}  γ={GAMMA}  τ={TAU}")
    print(f"{'='*60}\n")
 
    for episode in range(1, n_episodes + 1):
 
        state   = env.reset()
        a_prev  = np.zeros(ACTION_DIM, dtype=np.float32)
        agent.noise.reset()
 
        ep_reward = 0.0
        ep_lc, ep_la, ep_steps = 0.0, 0.0, 0
 
        done = False
        while not done:
 
            # Select action with exploration noise
            a_raw, a_phys = agent.select_action(state, a_prev, explore=True)
 
            # Step environment with feasible action
            next_state, reward, done, _ = env.step(a_phys)
 
            # Store RAW action (projection is deterministic; re-applied on each sample)
            agent.buffer.push(state, a_raw, reward, next_state, done)
 
            # Update networks
            lc, la = agent.update()
            if lc is not None:
                ep_lc += lc
                ep_la += la
 
            state  = next_state
            a_prev = a_phys
            ep_reward += reward
            ep_steps  += 1
 
        history["ep_reward"].append(ep_reward)
        history["L_critic" ].append(ep_lc / max(ep_steps, 1))
        history["L_actor"  ].append(ep_la / max(ep_steps, 1))
 
        # ── Periodic evaluation ───────────────
        if episode % eval_every == 0:
            metrics = agent.evaluate(env, n_episodes=5)
            history["eval_energy" ].append(metrics["energy_kWh"])
            history["eval_comfort"].append(metrics["comfort_viol_h"])
 
            print(f"  Ep {episode:4d}/{n_episodes}"
                  f"  R={ep_reward:7.2f}"
                  f"  E={metrics['energy_kWh']:6.2f} kWh"
                  f"  Comfort viol={metrics['comfort_viol_h']:5.2f} h"
                  f"  Lc={ep_lc/ep_steps:.4f}"
                  f"  La={ep_la/ep_steps:.4f}")
 
            # Early stopping on energy (primary KPI)
            if metrics["energy_kWh"] < best_eval_energy:
                best_eval_energy = metrics["energy_kWh"]
                patience_counter = 0
                agent.save(save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n[TC-DDPG] Early stopping at episode {episode}"
                          f" (patience={patience})")
                    break
 
    print(f"\n[TC-DDPG] Training complete. Best eval energy: {best_eval_energy:.2f} kWh")
    return history
 
 
# ─────────────────────────────────────────────
# 12. ENTRY POINT
# ─────────────────────────────────────────────
 
if __name__ == "__main__":
    import argparse
 
    parser = argparse.ArgumentParser(description="TC-DDPG HVAC training")
    parser.add_argument("--episodes",   type=int,   default=500,
                        help="Number of training episodes (default 500 for demo)")
    parser.add_argument("--eval-every", type=int,   default=50,
                        help="Evaluate every N episodes")
    parser.add_argument("--patience",   type=int,   default=10,
                        help="Early-stopping patience (eval rounds)")
    parser.add_argument("--save",       type=str,   default="tc_ddpg_best.pt",
                        help="Path for best checkpoint")
    parser.add_argument("--seed",       type=int,   default=42,
                        help="Random seed")
    args = parser.parse_args()
 
    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
 
    history = train(
        n_episodes   = args.episodes,
        eval_every   = args.eval_every,
        patience     = args.patience,
        save_path    = args.save,
    )
 
    print("\nFinal 10-episode average reward:",
          np.mean(history["ep_reward"][-10:]))
   