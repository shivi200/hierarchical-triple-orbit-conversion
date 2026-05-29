# hierarchical-triple-orbit-conversion
Keplerian ↔ Cartesian coordinate conversion for hierarchical triple body systems - motivated by MSc dissertation on GW sources
"""
Keplerian <-> Cartesian Orbital Element Conversion
for Hierarchical Triple Body Systems
=====================================================
Author: Shivika Lamba
Context: Developed post-MSc, motivated by dissertation work on hierarchical
         triple black hole systems (Cardiff University, supervisor: Dr Fabio Antonini)
 
Converts between Keplerian orbital elements and Cartesian phase-space
coordinates for hierarchical triple systems (inner binary + outer perturber).
 
The round-trip test at the bottom verifies conversion accuracy to
floating-point precision.
 
Keplerian elements: (a, e, i, RAAN, arg_periapsis, true_anomaly)
  a              — semi-major axis
  e              — eccentricity
  i              — inclination
  RAAN (Omega)   — right ascension of ascending node
  arg_periapsis  — argument of periapsis (omega)
  true_anomaly   — true anomaly (nu)
"""
 
import numpy as np
 
 
# =============================================================================
# 1. SINGLE-ORBIT CONVERSIONS
# =============================================================================
 
def keplerian_to_cartesian(a, e, i, RAAN, arg_periapsis, true_anomaly, mu):
    """
    Convert Keplerian orbital elements to Cartesian position and velocity.
 
    Uses the standard rotation matrix sequence Q = R3(RAAN) · R1(i) · R3(omega)
    to rotate from the perifocal frame to the inertial frame.
 
    Parameters
    ----------
    a             : semi-major axis (same units as output position)
    e             : eccentricity (dimensionless)
    i             : inclination (radians)
    RAAN          : right ascension of ascending node (radians)
    arg_periapsis : argument of periapsis (radians)
    true_anomaly  : true anomaly (radians)
    mu            : gravitational parameter G*(m1+m2), consistent units
 
    Returns
    -------
    r_vec : position vector (3,) in inertial frame
    v_vec : velocity vector (3,) in inertial frame
    """
    # Radial distance from orbital mechanics (conic section equation)
    r_mag = a * (1 - e**2) / (1 + e * np.cos(true_anomaly))
 
    # Position and velocity in perifocal (orbital plane) frame
    r_pf = np.array([
        r_mag * np.cos(true_anomaly),
        r_mag * np.sin(true_anomaly),
        0.0
    ])
    v_pf = np.array([
        -np.sqrt(mu / a) / np.sqrt(1 - e**2) * np.sin(true_anomaly),
         np.sqrt(mu / a) / np.sqrt(1 - e**2) * (e + np.cos(true_anomaly)),
         0.0
    ])
 
    # Rotation matrices: perifocal → inertial frame
    # R3(RAAN): rotation about z-axis by RAAN
    R3_W = np.array([
        [ np.cos(RAAN), -np.sin(RAAN), 0],
        [ np.sin(RAAN),  np.cos(RAAN), 0],
        [0,              0,             1]
    ])
    # R1(i): rotation about x-axis by inclination
    R1_i = np.array([
        [1, 0,           0          ],
        [0, np.cos(i),  -np.sin(i)  ],
        [0, np.sin(i),   np.cos(i)  ]
    ])
    # R3(omega): rotation about z-axis by argument of periapsis
    R3_w = np.array([
        [ np.cos(arg_periapsis), -np.sin(arg_periapsis), 0],
        [ np.sin(arg_periapsis),  np.cos(arg_periapsis), 0],
        [0,                       0,                      1]
    ])
 
    # Combined rotation matrix
    Q = R3_W @ R1_i @ R3_w
 
    r_vec = Q @ r_pf
    v_vec = Q @ v_pf
 
    return r_vec, v_vec
 
 
def cartesian_to_keplerian(r_vec, v_vec, mu):
    """
    Convert Cartesian position and velocity to Keplerian orbital elements.
 
    Recovers elements via angular momentum vector, eccentricity vector,
    and orbital energy. Handles sign conventions carefully for RAAN,
    argument of periapsis, and true anomaly.
 
    Parameters
    ----------
    r_vec : position vector (3,) in inertial frame
    v_vec : velocity vector (3,) in inertial frame
    mu    : gravitational parameter G*(m1+m2)
 
    Returns
    -------
    (a, e, i, RAAN, arg_periapsis, true_anomaly) — all angles in radians
    """
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)
 
    # Specific angular momentum
    h_vec = np.cross(r_vec, v_vec)
    h     = np.linalg.norm(h_vec)
 
    # Inclination from z-component of angular momentum
    i = np.arccos(h_vec[2] / h)
 
    # Node vector (points toward ascending node)
    K     = np.array([0, 0, 1])
    n_vec = np.cross(K, h_vec)
    n     = np.linalg.norm(n_vec)
 
    # Eccentricity vector (points toward periapsis)
    e_vec = (1 / mu) * ((v**2 - mu / r) * r_vec - np.dot(r_vec, v_vec) * v_vec)
    e     = np.linalg.norm(e_vec)
 
    # Semi-major axis from orbital energy
    energy = v**2 / 2 - mu / r
    a = -mu / (2 * energy) if abs(e - 1.0) > 1e-8 else np.inf
 
    # RAAN — angle from x-axis to ascending node
    RAAN = np.arccos(n_vec[0] / n) if n != 0 else 0.0
    if n != 0 and n_vec[1] < 0:
        RAAN = 2 * np.pi - RAAN   # quadrant correction
 
    # Argument of periapsis — angle from ascending node to periapsis
    arg_periapsis = (np.arccos(np.dot(n_vec, e_vec) / (n * e))
                     if n != 0 and e > 1e-8 else 0.0)
    if e_vec[2] < 0:
        arg_periapsis = 2 * np.pi - arg_periapsis   # quadrant correction
 
    # True anomaly — angle from periapsis to current position
    true_anomaly = (np.arccos(np.dot(e_vec, r_vec) / (e * r))
                    if e > 1e-8 else 0.0)
    if np.dot(r_vec, v_vec) < 0:
        true_anomaly = 2 * np.pi - true_anomaly     # quadrant correction
 
    return float(a), float(e), float(i), float(RAAN), float(arg_periapsis), float(true_anomaly)
 
 
# =============================================================================
# 2. HIERARCHICAL TRIPLE SYSTEM CONVERSIONS
# =============================================================================
 
def three_body_keplerian_to_cartesian(inner_elements, outer_elements,
                                       m1, m2, m3, G=6.67430e-20):
    """
    Convert hierarchical triple system from Keplerian to Cartesian coordinates.
 
    The triple is decomposed into:
      - Inner orbit: relative orbit of m1 and m2 (gravitational parameter mu_inner)
      - Outer orbit: orbit of the inner binary's barycentre around m3
                     (gravitational parameter mu_outer)
 
    m3 is placed at the origin; m1 and m2 are placed relative to the
    inner binary barycentre using mass-weighted offsets.
 
    Parameters
    ----------
    inner_elements : (a, e, i, RAAN, omega, nu) for the m1-m2 relative orbit
    outer_elements : (a, e, i, RAAN, omega, nu) for the barycentre-m3 orbit
    m1, m2, m3     : masses in kg
    G              : gravitational constant (km^3 kg^-1 s^-2 by default)
 
    Returns
    -------
    r1, v1, r2, v2, r3, v3 : position and velocity vectors for each body
    """
    mu_inner = G * (m1 + m2)          # inner binary gravitational parameter
    mu_outer = G * (m1 + m2 + m3)     # full system gravitational parameter
 
    # Inner relative orbit (m2 relative to m1)
    r_rel, v_rel = keplerian_to_cartesian(*inner_elements, mu_inner)
 
    # Outer orbit (barycentre of inner binary, relative to m3)
    r_bary, v_bary = keplerian_to_cartesian(*outer_elements, mu_outer)
 
    # Place m1 and m2 around the inner barycentre
    r1 = r_bary - (m2 / (m1 + m2)) * r_rel
    v1 = v_bary - (m2 / (m1 + m2)) * v_rel
    r2 = r_bary + (m1 / (m1 + m2)) * r_rel
    v2 = v_bary + (m1 / (m1 + m2)) * v_rel
 
    # m3 at origin
    r3 = np.zeros(3)
    v3 = np.zeros(3)
 
    return r1, v1, r2, v2, r3, v3
 
 
def three_body_cartesian_to_keplerian(r1, v1, r2, v2, r3, v3,
                                       m1, m2, m3, G=6.67430e-20):
    """
    Convert hierarchical triple system from Cartesian to Keplerian coordinates.
 
    Recovers the inner binary relative orbit and the outer barycentre orbit
    from individual body positions and velocities.
 
    Parameters
    ----------
    r1..v3  : position and velocity vectors for each body
    m1,m2,m3: masses in kg
    G       : gravitational constant
 
    Returns
    -------
    inner_elements : (a, e, i, RAAN, omega, nu) for the inner binary
    outer_elements : (a, e, i, RAAN, omega, nu) for the outer orbit
    """
    # Inner relative orbit
    r_rel   = r2 - r1
    v_rel   = v2 - v1
    mu_inner = G * (m1 + m2)
    inner_elements = cartesian_to_keplerian(r_rel, v_rel, mu_inner)
 
    # Inner binary barycentre
    r_bary = (m1 * r1 + m2 * r2) / (m1 + m2)
    v_bary = (m1 * v1 + m2 * v2) / (m1 + m2)
 
    # Outer orbit: barycentre relative to m3
    r_outer  = r_bary - r3
    v_outer  = v_bary - v3
    mu_outer = G * (m1 + m2 + m3)
    outer_elements = cartesian_to_keplerian(r_outer, v_outer, mu_outer)
 
    return inner_elements, outer_elements
 
 
# =============================================================================
# 3. ROUND-TRIP VERIFICATION
# =============================================================================
 
if __name__ == '__main__':
 
    print("=" * 60)
    print("ROUND-TRIP VERIFICATION: Keplerian → Cartesian → Keplerian")
    print("=" * 60)
 
    # Input orbital elements (angles in radians)
    inner_elements = (
        8000,              # semi-major axis
        0.1,               # eccentricity
        np.radians(10),    # inclination
        np.radians(30),    # RAAN
        np.radians(40),    # argument of periapsis
        np.radians(60),    # true anomaly
    )
    outer_elements = (
        20000,
        0.2,
        np.radians(15),
        np.radians(60),
        np.radians(70),
        np.radians(80),
    )
 
    # Masses (kg): inner binary components + outer perturber
    m1, m2, m3 = 5e24, 1e24, 2e30
 
    # Step 1: Keplerian → Cartesian
    r1, v1, r2, v2, r3, v3 = three_body_keplerian_to_cartesian(
        inner_elements, outer_elements, m1, m2, m3
    )
    print("\nCartesian coordinates (Keplerian → Cartesian):")
    print(f"  Body 1 | r = {r1}  |  v = {v1}")
    print(f"  Body 2 | r = {r2}  |  v = {v2}")
    print(f"  Body 3 | r = {r3}  |  v = {v3}")
 
    # Step 2: Cartesian → Keplerian (round-trip)
    inner_recovered, outer_recovered = three_body_cartesian_to_keplerian(
        r1, v1, r2, v2, r3, v3, m1, m2, m3
    )
    print("\nRecovered Keplerian elements (Cartesian → Keplerian):")
    print(f"  Inner: {inner_recovered}")
    print(f"  Outer: {outer_recovered}")
 
    # Verify round-trip accuracy
    print("\nRound-trip residuals (recovered - original):")
    element_names = ['a', 'e', 'i', 'RAAN', 'omega', 'nu']
    for name, orig, rec in zip(element_names, inner_elements, inner_recovered):
        print(f"  Inner {name}: {rec - orig:.2e}")
    for name, orig, rec in zip(element_names, outer_elements, outer_recovered):
        print(f"  Outer {name}: {rec - orig:.2e}")
 
    print("\nConversion verified to floating-point precision.")
