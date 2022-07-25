# PYCCT (Python Compositionally Constrained Thermodynamics)

This package consists of tools for modeling equilibrium defect thermodynamics in solid-state semiconducting and insulating compounds based on first principles calculations.


## Installation
This package and its basic dependencies can be installed locally with pip, called within the package root directory containing setup.py:

```bash
pip install -e ./
```

## Method Overview
### Key Quantities
For finite temperature $T$, the equilibrium fractional concentrations ($c_{d,q} = N_\mathrm{defects}/N_\mathrm{sites}$) of point defects $d$ in charge states $q$ are approximated by

$$\frac{c_{d,q}}{1-c_{d,q}} = \theta_{d,q} e^{-\Delta E^f_{d,q} / k_\mathrm{B}T} \ .$$

In the dilute limit, $c_{d,q} \ll 1$, this reduces to the typical Arrhenius form

$$c_{d,q} \approx \theta_{d,q} e^{-\Delta E^f_{d,q} / k_\mathrm{B}T} \ .$$

Entropic contributions to the Gibbs free energy of the system lead to the degeneracy prefactors, $\theta_{d,q} = \theta^\mathrm{mix}_d \theta^\mathrm{conf}_d \theta^\mathrm{el}_{d,q}$, which capture, respectively, lattice mixing, configurational (e.g., orientational or internal structural degrees of freedom) degeneracy, and electronic state degeneracy.  Defect formation energies $\Delta E^f_{i,q}$ can be derived from plane-wave Density Functional Theory (pwDFT) using the so-called *supercell method*,

$$\Delta E^f_{i,q} = E_{d,q} - E_\mathrm{bulk} + \sum_i n_i(\mu_i^0 + \Delta \mu_i) + q(E_\mathrm{VBM} + E_\mathrm{Fermi} + \Delta V) + E_\mathrm{corr} \ ,$$

where $E_\mathrm{bulk}$ is the ground state energy of the perfect bulk crystal and $E_{d,q}$ is the ground state energy of the system containing the defect. This definition of the formation energy assumes a (semi-)grand canonical picture for defect formation, where the energetic cost of exchanging $i$ chemical components with external resevoirs is captured by the set of chemcial potentials $\mu_i = \mu_i^0 + \Delta \mu_i$ measured with respect to a consistent choice of standard references $\mu_i^0$.  Correspondingly, $n_i$ counts the number of chemical species added ($n_i < 0$) or removed ($n_i > 0$) from the system in forming the defect. Analogously, the energy required to add or remove charge from the defect is captured by the Fermi level $E_\mathrm{Fermi}$ with respect to the bulk valence band maximum $E_\mathrm{VBM}$. Note that the periodic boundary conditions in pwDFT incur finite size effects which should be removed to find an adequate description of defect formation in the thermodynamic limit. These are accounted for in the correction term $E_\mathrm{corr}$. In particular, spurious, long-range electrostatic interactions between periodic images of charged defects constitute a primary source of error in calculating $\Delta E^f_{i,q}$. Furthermore, in pwDFT a neutralizing background (jellium) is included to tame divergences in the periodic Hartree potential for charged systems. This induces an asymptotic offset to the physical, aperiodic defect system and thus requires an additional *potential alignment* correction $\Delta V$ (see e.g., [Phys. Rev. B 86, 045112 (2012)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.86.045112)).

The condition of charge neutrality can be used to determine the Fermi level self-consistently with the charged defect population, 

$$p - n + \sum_{d,q} q c_{d,q} = 0 \ ,$$

where the net free carrier concnetration can be formally obtained by

$$n - p = \int_{-\infty}^{\infty} dE \frac{\nu(E)}{1 + \exp\left[ (E-E_\mathrm{Fermi}) / k_\mathrm{B}T \right]} - N_e \ , $$

with $\nu(E)$ the electronic density of states (DOS) and $N_e$ the number of electrons in the neutral bulk cell. For computational efficiency, we perform the Fermi integrals using an effective mass appoximation for the DOS.

For more thorough coverage of the theory involved, see e.g., [Rev. Mod. Phys. 86, 253 (2014)](https://journals.aps.org/rmp/abstract/10.1103/RevModPhys.86.253) and [IEEE JPV 5, 1188 (2015)](https://ieeexplore.ieee.org/document/7110503).

### Basic Usage 
Given a set of defect formation energies, electronic structure parameters, and standard chemical potential references provided by the user, the DefectThermodynamics class allows for the calculation of defect concentrations and self-consistent Fermi level as outlined above. For speed and simplicity, we have retained a grand canonical ensemble picture; in addition to temperature, the user is also required to specify a set of chemical potentials $\Delta \mu_i$ relative to their chosen standard references $\mu_i^0$. In principle, these should be constrained by the stability of the parent bulk material against the formation of secondary phases. 

While the basic computations performed by the DefectThermodynamics class are relatively simple, they can be straighforwardly combined with high-dimensional search and constrained optimization techniques to yield more advanced functionality. Several examples are provided in the examples/CdTe.ipynb notebook.