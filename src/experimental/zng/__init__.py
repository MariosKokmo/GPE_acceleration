"""
Zaremba-Nikuni-Griffin (ZNG) finite-temperature framework.

Files
-----
zng_library.py  : pure physics functions — grid operations, potentials, source
                  term, initial-state sampling.  No simulation state.
monte_carlo.py  : test-particle dynamics — leapfrog integrator, C12 stochastic
                  collisions, C22 stub.  No simulation state.
zng_BEC.py      : ZNGBEC model class that wires the above together, inheriting
                  all ground-state, I/O, and snapshotting from BaseBEC.
"""
