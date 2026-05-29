# hierarchical-triple-orbit-conversion
Keplerian ↔ Cartesian coordinate conversion for hierarchical triple body systems - motivated by MSc dissertation on GW sources
Keplerian ↔ Cartesian Orbital Element Conversion for Hierarchical Triple Systems
A Python implementation of bidirectional coordinate conversion between Keplerian orbital elements and Cartesian phase-space coordinates for hierarchical triple body systems — motivated by and developed in the context of gravitational-wave source modelling.

Background & Motivation
This code grew directly out of my MSc dissertation at Cardiff University (supervised by Dr Fabio Antonini), which investigated the relativistic dynamics of hierarchical triple black hole systems influenced by a distant supermassive black hole.
During the dissertation, an unexpected drift appeared in the angular momentum evolution — cosine values shifting in a physically forbidden direction. Tracing this anomaly back to its source required going deep into the coordinate transformation between Keplerian and Cartesian representations in the underlying N-body code. The conversion is mathematically straightforward in principle, but subtle numerical and sign-convention issues make correct implementation non-trivial, particularly for hierarchical (inner binary + outer perturber) configurations.
This repository contains a clean, verified implementation of the full conversion, completed after the MSc. The round-trip test confirms numerical consistency to floating-point precision.

What It Does
Keplerian → Cartesian
Converts orbital elements (a, e, i, Ω, ω, ν) to position and velocity vectors in an inertial Cartesian frame, using the standard rotation matrix sequence:
Q = R3(Ω) · R1(i) · R3(ω)
Applied separately to the inner binary and the outer orbit of the triple system, with correct mass-weighted placement around the system barycentre.
Cartesian → Keplerian
Inverts the transformation: recovers orbital elements from Cartesian state vectors via the eccentricity vector, specific angular momentum, and orbital energy.
Round-Trip Verification
The notebook verifies correctness by converting Keplerian → Cartesian → Keplerian and confirming recovery of the original elements to numerical precision.

Notation
SymbolMeaningaSemi-major axiseEccentricityiInclinationΩ (RAAN)Right ascension of ascending nodeωArgument of periapsisνTrue anomalyμGravitational parameter G(m₁ + m₂)

Example
python# Define inner binary and outer orbit Keplerian elements
# (a, e, i, RAAN, arg_periapsis, true_anomaly)
inner_elements = (8000, 0.1, np.radians(10), np.radians(30),
                  np.radians(40), np.radians(60))
outer_elements = (20000, 0.2, np.radians(15), np.radians(60),
                  np.radians(70), np.radians(80))

# Masses: inner binary components + outer perturber
m1, m2, m3 = 5e24, 1e24, 2e30   # kg

# Convert to Cartesian
r1, v1, r2, v2, r3, v3 = three_body_keplerian_to_cartesian(
    inner_elements, outer_elements, m1, m2, m3
)

# Convert back — should recover original elements to floating-point precision
inner_recovered, outer_recovered = three_body_cartesian_to_keplerian(
    r1, v1, r2, v2, r3, v3, m1, m2, m3
)

Relevance to Gravitational-Wave Astronomy
Hierarchical triple systems — an inner compact binary orbited by a distant third body — are important gravitational-wave sources. The outer perturber drives Kozai–Lidov oscillations, which can pump the inner binary's eccentricity to high values and significantly accelerate merger. Correct coordinate conversion is essential for:

Initialising N-body and post-Newtonian simulations from physically meaningful orbital parameters
Diagnosing numerical anomalies in long-duration integrations
Interfacing with gravitational waveform codes that expect Cartesian initial conditions


Context
This work is related to my MSc dissertation:
"Relativistic dynamics of hierarchical triple black hole systems in the presence of a supermassive black hole"
Cardiff University, 2023–2024 | Supervisor: Dr Fabio Antonini

Author
Shivika Lamba — MSc Astrophysics, Cardiff University
Research interests: gravitational-wave source modelling, waveform modelling, EMRI science, LISA
