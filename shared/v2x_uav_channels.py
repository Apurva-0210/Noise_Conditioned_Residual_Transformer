"""
shared/v2x_uav_channels.py
────────────────────────────────────────────────────────────────────────
Application-Specific Wireless Channel Simulators:
  1. UAV / Air-to-Ground (A2G) Massive MIMO Channels (3GPP TR 38.811 / TR 36.777)
     - Altitude-dependent Rician K-factor & elevation geometry
     - 3D spatial steering vectors (elevation theta, azimuth phi)
     - 3D UAV velocity vectors & aerodynamic rotor micro-wobble
  2. V2X / V2V (Vehicle-to-Vehicle) High-Mobility Channels (3GPP TR 37.885)
     - Dual-mobility Doppler spread (Tx vehicle + Rx vehicle moving simultaneously)
     - Extreme relative velocities (up to 280 km/h in opposite lanes)
     - Highway LoS convoy vs. Urban intersection non-stationary scatterers

Outputs standardized tensors of shape (TIME_STEPS, Nr, Nt, 2) in float32.
"""

import numpy as np

TIME_STEPS = 10
SPEED_OF_LIGHT = 3e8


def steering_vector_ula(n_antennas, angle_rad, d_over_lambda=0.5):
    """
    Computes steering vector for a Uniform Linear Array (ULA).
    a(theta) = [1, exp(j 2pi d/lambda sin(theta)), ..., exp(j 2pi d/lambda (N-1) sin(theta))]^T / sqrt(N)
    """
    idx = np.arange(n_antennas)
    phase = 2.0 * np.pi * d_over_lambda * idx * np.sin(angle_rad)
    return np.exp(1j * phase) / np.sqrt(n_antennas)


# ═════════════════════════════════════════════════════════════════════════
# 1. UAV / AIR-TO-GROUND (A2G) SIMULATOR (3GPP TR 38.811 / TR 36.777)
# ═════════════════════════════════════════════════════════════════════════

def gen_uav_channel(Nr=64, Nt=64, h_uav_m=120.0, d_2d_m=300.0,
                    v_uav_kmh=80.0, fc=28e9, num_steps=TIME_STEPS,
                    num_paths=15, rng=None):
    """
    Generates a 3D UAV Air-to-Ground Massive MIMO channel snapshot sequence.

    Physics:
      - Elevation angle: theta_elev = arctan(h_uav / d_2d)
      - Altitude-dependent Rician K: K(dB) = K0 + alpha * theta_elev
      - LoS path has geometric 3D steering vector + 3D Doppler
      - Aerodynamic rotor wobble adds micro-Doppler phase modulation
      - NLoS paths represent ground-level clutter scattering
    """
    if rng is None:
        rng = np.random.default_rng()

    # Geometry
    theta_elev = np.arctan2(h_uav_m, d_2d_m)  # elevation angle in radians
    phi_azim = rng.uniform(-np.pi / 3, np.pi / 3)  # BS azimuth angle

    # Rician K-factor increases with elevation (clearer line-of-sight)
    # At theta = 10 deg -> ~2 dB, at theta = 60 deg -> ~15 dB
    k_db = float(np.clip(2.0 + 15.0 * (theta_elev / (np.pi / 3)), 0.0, 20.0))
    k_factor = 10.0 ** (k_db / 10.0)

    # Wavelength & Max Doppler
    lam = SPEED_OF_LIGHT / fc
    v_uav = v_uav_kmh / 3.6
    fd_max = (v_uav / lam)

    # 3D Doppler component for LoS path
    cos_uav_motion = np.cos(theta_elev) * np.cos(rng.uniform(0, 2 * np.pi))
    fd_los = fd_max * cos_uav_motion

    # Aerodynamic rotor wobble (micro-Doppler vibration)
    f_wobble = rng.uniform(20.0, 80.0)  # rotor mechanical vibration Hz
    wobble_amp = rng.uniform(0.1, 0.3)  # phase modulation index

    # Time vector (sampling interval dt = 0.5 ms)
    dt = 0.5e-3
    t = np.arange(num_steps) * dt

    # 1. LoS Component
    # Ground BS array response (ULA azimuth)
    a_bs_los = steering_vector_ula(Nt, phi_azim)
    # UAV array response
    phi_uav_los = rng.uniform(-np.pi / 2, np.pi / 2)
    a_uav_los = steering_vector_ula(Nr, phi_uav_los)
    H_los_static = np.outer(a_uav_los, np.conj(a_bs_los))  # (Nr, Nt)

    # 2. NLoS Ground Multipath Component
    # Several scattered paths reflected from ground buildings
    path_powers = rng.exponential(1.0, size=num_paths)
    path_powers /= np.sum(path_powers)

    nlos_paths = []
    for p in range(num_paths):
        p_phi_bs = rng.uniform(-np.pi / 2, np.pi / 2)
        p_phi_uav = rng.uniform(-np.pi, np.pi)
        a_bs_p = steering_vector_ula(Nt, p_phi_bs)
        a_uav_p = steering_vector_ula(Nr, p_phi_uav)
        p_mat = np.outer(a_uav_p, np.conj(a_bs_p))

        # Random ground scatterer Doppler
        p_fd = fd_max * np.cos(rng.uniform(0, 2 * np.pi))
        p_phase0 = rng.uniform(0, 2 * np.pi)
        nlos_paths.append((np.sqrt(path_powers[p]), p_mat, p_fd, p_phase0))

    # Synthesize over time steps
    H_seq = []
    los_weight = np.sqrt(k_factor / (k_factor + 1.0))
    nlos_weight = np.sqrt(1.0 / (k_factor + 1.0))

    for step in range(num_steps):
        ti = t[step]
        # LoS time phase with rotor wobble
        phi_los_t = 2.0 * np.pi * fd_los * ti + wobble_amp * np.sin(2.0 * np.pi * f_wobble * ti)
        H_t = los_weight * H_los_static * np.exp(1j * phi_los_t)

        # Add NLoS multipath
        H_nlos_t = np.zeros((Nr, Nt), dtype=np.complex128)
        for p_amp, p_mat, p_fd, p_phase0 in nlos_paths:
            phi_p = 2.0 * np.pi * p_fd * ti + p_phase0
            H_nlos_t += p_amp * p_mat * np.exp(1j * phi_p)

        H_t += nlos_weight * H_nlos_t
        H_seq.append(np.stack([H_t.real, H_t.imag], axis=-1))

    H_arr = np.array(H_seq, dtype=np.float32)

    # Normalize channel power to unit Frobenius norm per frame
    pwr = np.mean(H_arr ** 2)
    if pwr > 1e-12:
        H_arr = H_arr / np.sqrt(pwr)

    return H_arr


def gen_uav_random(Nr=64, Nt=64):
    """Generates random UAV scenario across altitude (50-300m) and velocity (40-120 km/h)."""
    h_uav = float(np.random.uniform(50.0, 300.0))
    d_2d = float(np.random.uniform(100.0, 800.0))
    v_uav = float(np.random.uniform(40.0, 120.0))
    fc = float(np.random.choice([28e9, 39e9, 60e9]))
    return gen_uav_channel(Nr, Nt, h_uav_m=h_uav, d_2d_m=d_2d, v_uav_kmh=v_uav, fc=fc)


# ═════════════════════════════════════════════════════════════════════════
# 2. V2X / V2V (VEHICLE-TO-VEHICLE) SIMULATOR (3GPP TR 37.885)
# ═════════════════════════════════════════════════════════════════════════

def gen_v2v_channel(Nr=64, Nt=64, scenario='highway',
                    v1_kmh=120.0, v2_kmh=120.0, fc=28e9,
                    num_steps=TIME_STEPS, rng=None):
    """
    Generates a Vehicle-to-Vehicle (V2V) high-mobility Massive MIMO channel sequence.

    Physics:
      - Dual mobility: Both Tx vehicle and Rx vehicle move simultaneously.
      - Relative velocity in opposite directions: v_rel = v1 + v2 (up to 240-280 km/h).
      - Dual-Doppler cascade: fd_path = fd1 * cos(phi1) + fd2 * cos(phi2).
      - Scenarios:
        - 'highway': LoS dominant convoy, low angular spread, extreme Doppler.
        - 'urban': High angular spread from street canyons, intermittent blocking.
    """
    if rng is None:
        rng = np.random.default_rng()

    lam = SPEED_OF_LIGHT / fc
    v1 = v1_kmh / 3.6  # Tx velocity in m/s
    v2 = v2_kmh / 3.6  # Rx velocity in m/s
    fd1_max = v1 / lam
    fd2_max = v2 / lam

    dt = 0.5e-3
    t = np.arange(num_steps) * dt

    if scenario == 'highway':
        # Highway: Tx and Rx in opposite lanes or convoy
        # Strong direct path + roadside barrier reflections
        k_factor = 10.0 ** (float(rng.uniform(3.0, 10.0)) / 10.0)
        num_paths = 12
        # Tx heading 0 deg, Rx heading 180 deg (opposite lanes)
        heading_diff = np.pi if rng.uniform() > 0.3 else 0.0
    else:  # 'urban'
        # Urban Intersection: dense multipath, lower K-factor
        k_factor = 10.0 ** (float(rng.uniform(-3.0, 3.0)) / 10.0)
        num_paths = 24
        heading_diff = rng.uniform(-np.pi / 2, np.pi / 2)

    # 1. Direct / Dominant Path
    phi_tx_los = rng.uniform(-np.pi / 6, np.pi / 6)
    phi_rx_los = phi_tx_los + heading_diff + rng.uniform(-0.1, 0.1)

    a_tx_los = steering_vector_ula(Nt, phi_tx_los)
    a_rx_los = steering_vector_ula(Nr, phi_rx_los)
    H_los_static = np.outer(a_rx_los, np.conj(a_tx_los))

    # Dual Doppler on dominant path
    fd_los_tx = fd1_max * np.cos(phi_tx_los)
    fd_los_rx = fd2_max * np.cos(phi_rx_los)
    fd_los_total = fd_los_tx + fd_los_rx

    # 2. Multipath Scatterers (Roadside infrastructure, other vehicles)
    path_powers = rng.exponential(1.0, size=num_paths)
    path_powers /= np.sum(path_powers)

    paths = []
    for p in range(num_paths):
        p_phi_tx = rng.uniform(-np.pi / 2, np.pi / 2)
        p_phi_rx = rng.uniform(-np.pi / 2, np.pi / 2)
        a_tx_p = steering_vector_ula(Nt, p_phi_tx)
        a_rx_p = steering_vector_ula(Nr, p_phi_rx)
        p_mat = np.outer(a_rx_p, np.conj(a_tx_p))

        # Dual Doppler: independent departure and arrival angles relative to velocity vectors
        p_fd1 = fd1_max * np.cos(p_phi_tx)
        p_fd2 = fd2_max * np.cos(p_phi_rx)
        p_fd_total = p_fd1 + p_fd2
        p_phase0 = rng.uniform(0, 2 * np.pi)
        paths.append((np.sqrt(path_powers[p]), p_mat, p_fd_total, p_phase0))

    # Synthesize over time steps
    H_seq = []
    los_weight = np.sqrt(k_factor / (k_factor + 1.0))
    nlos_weight = np.sqrt(1.0 / (k_factor + 1.0))

    for step in range(num_steps):
        ti = t[step]
        # LoS phase evolution
        phi_los_t = 2.0 * np.pi * fd_los_total * ti
        H_t = los_weight * H_los_static * np.exp(1j * phi_los_t)

        # NLoS sum
        H_nlos_t = np.zeros((Nr, Nt), dtype=np.complex128)
        for p_amp, p_mat, p_fd, p_phase0 in paths:
            phi_p = 2.0 * np.pi * p_fd * ti + p_phase0
            H_nlos_t += p_amp * p_mat * np.exp(1j * phi_p)

        H_t += nlos_weight * H_nlos_t
        H_seq.append(np.stack([H_t.real, H_t.imag], axis=-1))

    H_arr = np.array(H_seq, dtype=np.float32)

    # Normalize channel power
    pwr = np.mean(H_arr ** 2)
    if pwr > 1e-12:
        H_arr = H_arr / np.sqrt(pwr)

    return H_arr


def gen_v2v_random(Nr=64, Nt=64):
    """Generates random V2V scenario (highway or urban with 80-140 km/h speeds)."""
    scenario = 'highway' if np.random.uniform() > 0.5 else 'urban'
    v1 = float(np.random.uniform(80.0, 140.0))
    v2 = float(np.random.uniform(80.0, 140.0))
    fc = float(np.random.choice([5.9e9, 28e9]))
    return gen_v2v_channel(Nr, Nt, scenario=scenario, v1_kmh=v1, v2_kmh=v2, fc=fc)
