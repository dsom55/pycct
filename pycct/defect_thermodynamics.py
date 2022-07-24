import json
from typing import Tuple, List, Dict, Union

import numpy as np
from scipy import optimize
from scipy import integrate
from scipy.constants import physical_constants
import pandas as pd

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')


class DefectThermodynamics(object):
    def __init__(self,
                defect_energies: np.ndarray,
                corrections: np.ndarray,
                entropy_prefactors: np.ndarray,
                particle_numbers: np.ndarray,
                reference_chemical_potentials: np.ndarray,
                species_order: Dict[str, int],
                defect_order: Dict[str, int],
                charge_states: np.ndarray,
                compound_stoichiometry: np.ndarray,
                bulk_energy: float,
                bulk_vbm: float,
                band_gap: float,
                bulk_volume_per_site: float,
                n_sites: float,
                electron_eff_mass: float, 
                hole_eff_mass: float) -> None:
        '''
        Provides methods for calculating finite-temperature defect concentrations 
        and self-consistent Fermi level from ab initio parameters.

        Note: To facilitate efficient array broadcasting, defect energies are represented by dense arrays. 
              This means that some array entries will exist which may represent unphysical charge states 
              for a given defect. These are excluded (masked) from subsequent calculations by setting the 
              corresponding entropy_prefactors to zero.

        See DefectThermodynamics.from_json() method for example usage of input args.

        Unless otherwise specified, the following units are assumed:
            Energy: eV
            Temperature: K
            Length: Angstroms (e.g., bulk_volume_per_site)
            Concentration: cm^{-3}

        Args:
            defect_energies: Dense array containing ground state energies of defects across all possible charge_states
                             Shape (len(defect_order), len(charge_states))
            corrections: Dense array containing formation energy corrections for defects across all possible charge_states
                         Shape (len(defect_order), len(charge_states))
            entropy_prefactors: Dense array containing entropy prefactors (number of microstates) of defects across all possible charge_states
                                Shape (len(defect_order), len(charge_states))
            particle_numbers: Dense array containing change in number of atomic species in forming defect across all possible charge_states
                              Convention: Species removed (n > 0) and species added (n < 0)
                              Shape (len(defect_order), len(charge_states))
            reference_chemical_potentials: Standard chemical potential references for all chemical species in system.
                                           Order should match species_order.
                                           Shape (len(species_order),)
            species_order: Dictionary where keys are chemical labels and values are indices indicating order
            defect_order: Dictionary where keys are chemical labels and values are indices indicating order
            charge_states: Integer array of all possible charge states spanning [min(q), max(q)] across all defects.
            compound_stoichiometry: Array defining chemical composition of the perfect host compound.
                                    Shape (len(species_order),)
            bulk_energy: Ground state energy of bulk structure (perfect host/parent structure) with n_sites
            bulk_vbm: Valence band maximum of bulk structure (perfect host/parent structure)
            band_gap: Band gap of bulk structure (perfect host/parent structure)
            bulk_volume_per_site: Volume / n_sites of bulk structure (perfect host/parent structure)
                                  Units [Angstroms^3 / N_sites]
            n_sites: Number of sites in bulk supercell structure (perfect host/parent structure) from which defect supercells are derived
            electron_eff_mass: Electron effective mass (bulk parent structure)
                               Atomic Units
            hole_eff_mass: Hole effective mass (bulk parent structure)
                           Atomic Units
        '''
        self.species_order = species_order
        self.defect_order = defect_order
        
        self.nspecies = len(species_order)
        self.ndefects = len(defect_order)
        self.ncharge_states = len(charge_states)
        self.charge_states = np.repeat(charge_states[None,:], 
        							   self.ndefects, axis=0)

        self.reference_chemical_potentials = reference_chemical_potentials
        self.particle_numbers = particle_numbers
        self.entropy_prefactors = entropy_prefactors
        self.defect_energies = defect_energies
        self.corrections = corrections
                
        self.calculated = np.where( entropy_prefactors > 0 )
        self.uncalculated = np.where( entropy_prefactors == 0 )
        
        self.kB = physical_constants['Boltzmann constant in eV/K'][0]
        self.band_gap = band_gap
        self.bulk_energy = bulk_energy
        self.bulk_vbm = bulk_vbm
        self.bulk_volume_per_site = bulk_volume_per_site
        
        self.compound_stoichiometry = compound_stoichiometry
        self.reference_atomic_fractions = compound_stoichiometry/np.sum(compound_stoichiometry)
        self.n_sites = n_sites
        
        bulk_energy_per_fu = bulk_energy/(n_sites/np.sum(compound_stoichiometry))
        chem_refs_tot = np.dot(reference_chemical_potentials, compound_stoichiometry)        
        self.compound_formation_enthalpy = bulk_energy_per_fu - chem_refs_tot
    
        self.set_electronic_params(electron_eff_mass, hole_eff_mass)
    
    def set_electronic_params(self, 
                            electron_eff_mass: float,  
                            hole_eff_mass: float) -> None:
        '''
        Sets additonal electronic structure parameters for calculating free carrier concentrations
        using an effective mass approximation.

        Args:
            electron_eff_mass: The electron effective mass of the bulk material in atomic units.
            hole_eff_mass: The hole effective mass of the bulk material in atomic units.
        '''
        h_bar = physical_constants['Planck constant over 2 pi in eV s'][0]
        m_e = physical_constants['electron mass energy equivalent in MeV'][0] * 1e6
        c = physical_constants['speed of light in vacuum'][0]
        
        eff_mass_c = electron_eff_mass * m_e
        eff_mass_v = hole_eff_mass * m_e

        ### Units: eV^{-3/2} * cm^{-3}
        self.pref_c = (2 * eff_mass_c)**(1.5) / (2 * np.pi**2 * (c * h_bar)**3) * 1e-6
        self.pref_v = (2 * eff_mass_v)**(1.5) / (2 * np.pi**2 * (c * h_bar)**3) * 1e-6
        
    def charge_carriers(self, 
                        Efermi: float, 
                        temperature: float) -> Tuple[float, float]:
        '''
        Calculates the free electron and hole concentrations using an effective mass approximation
        and numerical Fermi integrals.
        Args:
            Efermi: The Fermi level
            temperature: The temperature
        Returns:
            Free hole amd electron concentrations
        '''
        KT = self.kB * temperature
        
        eta_Fc = (Efermi - self.band_gap)/KT
        eta_Fv = -Efermi/KT
        nred = lambda x: KT**(1.5) * self.pref_c * np.sqrt(x) * np.exp( -(x - eta_Fc) ) / (1 + np.exp( -(x - eta_Fc) ))
        pred = lambda x: KT**(1.5) * self.pref_v * np.sqrt(x) * np.exp( -(x - eta_Fv) ) / (1 + np.exp( -(x - eta_Fv) ))

        electron_conc = integrate.quad(nred, 0, np.inf)[0]
        hole_conc = integrate.quad(pred, 0, np.inf)[0]

        return hole_conc, electron_conc
    
    def formation_energies(self, 
                        chemical_potentials: Union[np.ndarray, List[float]], 
                        Efermi: float) -> np.ndarray:
        '''
        Calculates defect formation energies from ab initio parameters

        Args:
            chemical_potentials: Array of chemical potentials. The order must match the order supplied 
                                 in the self.species_order and self.reference_chemical_potentials.
            Efermi: The Fermi level
        Returns:
            Array of formation energies
        '''
        chem_terms = self.particle_numbers @ (self.reference_chemical_potentials + 
                                              np.array(chemical_potentials))
        chem_terms = np.repeat(chem_terms[:,None], 
                               self.ncharge_states, axis=1)
        form_es = self.defect_energies - self.bulk_energy + \
                  chem_terms + self.charge_states*(self.bulk_vbm + Efermi) + \
                  self.corrections
        form_es[self.uncalculated] = np.nan
        return form_es
        
    def stable_defect_formation_energies(self, 
                                        chemical_potentials: Union[np.ndarray, List[float]], 
                                        Fermi_levels: np.ndarray) -> np.ndarray:
        '''
        Finds the lowest energy defect charge state for Fermi levels within the band gap.

        Args:
            chemical potentials: Array of chemical potentials. The order must match the order supplied 
                                 in the self.species_order and self.reference_chemical_potentials.
            Fermi_levels: Array of Fermi levels to evaluate formation energies.
        Returns: 
            Array of lowest defect formation energies by charge state.
            Shape (n_defects, len(Fermi_levels))
        '''
        chem_terms = self.particle_numbers @ (self.reference_chemical_potentials + 
                                              np.array(chemical_potentials))
        chem_terms = np.repeat(chem_terms[:, None], 
                               self.ncharge_states, axis=1)
        form_es = self.defect_energies - self.bulk_energy + \
                  chem_terms + self.charge_states*(self.bulk_vbm) + \
                  self.corrections
        form_es[self.uncalculated] = np.nan
        form_es_shape = form_es.shape
        form_es = np.repeat(form_es[:,:,None], 
        					len(Fermi_levels), axis=-1)
        charge_states = np.repeat(self.charge_states[:,:,None], 
        						  len(Fermi_levels), axis=-1)
        Fermi_levels = np.tile(Fermi_levels, form_es_shape + (1,))
        form_es += charge_states * Fermi_levels
        return np.nanmin(form_es, axis=1)
    
    def formation_energy_plot(self, 
                            chemical_potentials: Union[np.ndarray, List[float]]) -> plt:
        '''
        Plots of stable defect formation energies versus Fermi level.

        Args:
            chemical potentials: Array of chemical potentials. The order must match the order supplied 
                                 in the self.species_order and self.reference_chemical_potentials.
        Returns: 
            (matplotlib.pyplot) Plot of stable defect formation energies versus Fermi level.
        '''
        light_grey = '#EEEEEE'
        font = {'size': 16}
        linewidth = 2.5

        f, ax = plt.subplots(1, 1, figsize=(8,6))

        flevels = np.linspace(0, self.band_gap, 100)
        form_es = self.stable_defect_formation_energies(chemical_potentials, flevels)

        for defect_label, defect_index in self.defect_order.items():
            ax.plot(flevels, form_es[defect_index], label=defect_label,
                    linestyle='-', linewidth=linewidth)

        ax.axvline(0.0, linestyle="--", linewidth=linewidth, color='k')
        ax.axvline(self.band_gap, linestyle="--", linewidth=linewidth, color='k')
        
        ax.set_xlabel('Fermi Level [eV]')
        ax.set_ylabel('Defect Formation Energy [eV]')
        ax.legend(loc=(1,0.1), frameon=True, facecolor=light_grey, 
                  prop={'size': 14}, ncol=1)

        plt.rc('font', **font)
        plt.tight_layout()
        return plt
        
    def defect_concentrations(self,
                            chemical_potentials: Union[np.ndarray, List[float]], 
                            Efermi: float, 
                            temperature: float,
                            nondilute: bool = False) -> np.ndarray:
        '''
        Calculates defect concentrations in the dilute approximation.

        Args:
            chemical potentials: Array of chemical potentials. The order must match the order supplied 
                                 in the self.species_order and self.reference_chemical_potentials.
            Efermi: The Fermi level
            temperature: The temperature
            nondilute: Boolean flag indicating whether to employ nondilute approximation
        Returns:
            Array of defect concentrations
        '''
        KT = self.kB * temperature
        form_es = self.formation_energies(chemical_potentials, Efermi)
        concentrations = self.entropy_prefactors * np.exp(-form_es / KT)
        if nondilute:
            concentrations /= 1 + concentrations
        ### Note unit conversion of density from 1/N_sites to cm^{-3}
        concentrations *= (1e24 / self.bulk_volume_per_site)
        concentrations[self.uncalculated] = 0
        return concentrations

    def atomic_fractions(self,
                        chemical_potentials: Union[np.ndarray, List[float]], 
                        Efermi: float, 
                        temperature: float,
                        nondilute: bool = False) -> np.ndarray:
        '''
        Calculates atomic fractions from defect concentrations

        Args:
            chemical potentials: Array of chemical potentials. The order must match the order supplied 
                                 in the self.species_order and self.reference_chemical_potentials.
            Efermi: The Fermi level
            temperature: The temperature
            nondilute: Boolean flag indicating whether to employ nondilute approximation 
                       in calculating defect concentrations
        Returns:
            Array of defect concentrations
        '''
        KT = self.kB * temperature
        form_es = self.formation_energies(chemical_potentials, Efermi)
        ### Note units of 1/N_sites
        concentrations = self.entropy_prefactors * np.exp(-form_es / KT)
        if nondilute:
            concentrations /= 1 + concentrations
        concentrations[self.uncalculated] = 0
        total_concentrations = np.sum(concentrations, axis=-1)
        atomic_fractions = self.reference_atomic_fractions - total_concentrations @ self.particle_numbers
        atomic_fractions /= 1 - total_concentrations @ np.sum(self.particle_numbers, axis=-1)
        return atomic_fractions

    def redistribute_defect_charge_states(self, 
                                        defect_concentrations: np.ndarray, 
                                        new_Efermi: float, 
                                        new_temperature: float,
                                        nondilute: bool = False) -> np.ndarray:
        '''
        Redistributes defect concentrations among available charge states for a new Fermi level and temperature.
        Assumes a frozen-in approximation, where total defect concentrations are fixed by some prior equilibrium
        calculation.

        Args:
            defect_concentrations: np.ndarray, 
            new_Efermi: The new Fermi level 
            new_temperature: The new temprature
            nondilute: Boolean flag indicating whether to employ nondilute approximation 
                       in calculating defect concentrations
        Returns: 
            Array of redistributed defect concentrations.
        '''
        new_defect_concentrations = self.defect_concentrations(np.zeros(self.nspecies), 
                                                    		   new_Efermi, 
                                                    		   new_temperature,
                                                               nondilute)
        total_concentration = np.sum(defect_concentrations, axis=1)
        total_concentration = np.repeat(total_concentration[:,None], 
                                        self.ncharge_states, axis=1)
        total_new_concentration = np.sum(new_defect_concentrations, axis=1)
        total_new_concentration = np.repeat(total_new_concentration[:,None], 
                                            self.ncharge_states, axis=1)
        return total_concentration * (new_defect_concentrations / total_new_concentration)

    def defect_concentrations_dataframe(self, 
                                        chemical_potentials: Union[np.ndarray, List[float]], 
                                        Efermi: float, 
                                        temperature: float,
                                        nondilute: bool = False) -> pd.DataFrame:
        '''
        Convenience method to provide more readable dataframe of defect concentrations.

        Args:
            chemical potentials: Array of chemical potentials. The order must match the order supplied 
                                 in the self.species_order and self.reference_chemical_potentials.
            Efermi: The Fermi level
            temperature: The temperature
            nondilute: Boolean flag indicating whether to employ nondilute approximation 
                       in calculating defect concentrations
        Returns:
            DataFrame of defect concentrations.
        '''
        concentrations = self.defect_concentrations(chemical_potentials, 
                                                    Efermi, 
                                                    temperature,
                                                    nondilute)
        defect_dict = {}
        for defect_label, defect_index in self.defect_order.items():
            defect_dict[defect_label] = concentrations[defect_index]
        defect_dict['charge'] = np.arange(self.charge_states.min(), self.charge_states.max()+1, 1)
        defect_df = pd.DataFrame(defect_dict)
        defect_df.set_index('charge', inplace=True)
        return defect_df

    def formation_energies_dataframe(self, 
                                    chemical_potentials: Union[np.ndarray, List[float]], 
                                    Efermi: float) -> pd.DataFrame:
        '''
        Convenience method to provide more readable dataframe of defect formation energies.

        Args:
            chemical potentials: Array of chemical potentials. The order must match the order supplied 
                                 in the self.species_order and self.reference_chemical_potentials.
            Efermi: The Fermi level
            temperature: The temperature
        Returns:
            DataFrame of defect formation energies.
        '''
        formation_energies = self.formation_energies(chemical_potentials, Efermi)
        
        defect_dict = {}
        for defect_label, defect_index in self.defect_order.items():
            defect_dict[defect_label] = formation_energies[defect_index]
        defect_dict['charge'] = np.arange(self.charge_states.min(), self.charge_states.max()+1, 1)
        defect_df = pd.DataFrame(defect_dict)
        defect_df.set_index('charge', inplace=True)
        return defect_df
    
    def solve_for_fermi_energy(self, 
                            chemical_potentials: Union[np.ndarray, List[float]], 
                            temperature: float,
                            nondilute: bool = False) -> float:
        '''
        Finds the Fermi level that satisfies charge neutrality (p - n + \sum_{d,q} q*c_{d,q} == 0)

        Args:
            chemical potentials: Array of chemical potentials. The order must match the order supplied 
                                 in the self.species_order and self.reference_chemical_potentials.
            temperature: The temperature
            nondilute: Boolean flag indicating whether to employ nondilute approximation 
                       in calculating defect concentrations
        Returns:
            The self-consistent Fermi level
        '''
        def _get_total_q(ef):
            concentrations = self.defect_concentrations(chemical_potentials, 
                                                        ef, 
                                                        temperature,
                                                        nondilute)
            qd_tot = np.sum((concentrations * self.charge_states))
            nhole, nelectron = self.charge_carriers(ef, temperature)
            qd_tot += (nhole - nelectron)
            return qd_tot

        return optimize.brentq(_get_total_q, -1.0, self.band_gap + 1.0)

    def solve_for_non_equilibrium_fermi_energy(self, 
                                    chemical_potentials: Union[np.ndarray, List[float]], 
                                    temperature: float, 
                                    high_temperature: float,
                                    nondilute: bool = False) -> float:
        '''
        Finds the Fermi level that satisfies charge neutrality (p - n + \sum_{d,q} q*c_{d,q} == 0) 
        following a quench from an equilibrated state at high temperature. Assumes a frozen-in approximation,
        where total defect concentrations are fixed by the prior equilibrium calculation. 

        Args:
            chemical potentials: Array of chemical potentials. The order must match the order supplied 
                                 in the self.species_order and self.reference_chemical_potentials.
            temperature: New temperature (after quench)
            high_temperature: Initial equilibration temperature (before quench)
            nondilute: Boolean flag indicating whether to employ nondilute approximation 
                       in calculating defect concentrations
        Returns:
            The self-consistent, non-equilibrium Fermi level
        '''
        high_temp_fermi_level = self.solve_for_fermi_energy(chemical_potentials, 
                                                            high_temperature,
                                                            nondilute)
        concentrations = self.defect_concentrations(chemical_potentials, 
                                                    high_temp_fermi_level, 
                                                    high_temperature,
                                                    nondilute)

        def _get_total_q(ef):
            new_concentrations = self.redistribute_defect_charge_states(concentrations, ef, temperature)
            qd_tot = np.sum((new_concentrations * self.charge_states))
            nhole, nelectron = self.charge_carriers(ef, temperature)
            qd_tot += (nhole - nelectron)
            return qd_tot

        return optimize.brentq(_get_total_q, -1.0, self.band_gap + 1.0)
    
    @classmethod
    def from_json(cls, 
                filename: str, 
                excluded_defects: List[str] = []):
        '''
        Instantiates a DefectThermodynamics instance from an external .json parameter file.
        This is the preferred method for creating class instances.

        Args:
            filename: A .json parameter file
            excluded_defects: A list of defects to exclude from subsequent calculations
        '''
        with open(filename) as json_file:
            input_params = json.load(json_file)

        bulk_volume_per_site = input_params['bulk_volume_per_site']
        n_sites = input_params['n_sites']

        bulk_energy = input_params['bulk_energy']
        bulk_vbm = input_params['bulk_vbm']
        band_gap = input_params['band_gap']
        electron_eff_mass = input_params['electron_effective_mass']
        hole_eff_mass = input_params['hole_effective_mass']

        species_order = { sym:i for i,sym in enumerate(input_params['species_order']) }
        defect_dict = { sym:i for i,sym in enumerate(input_params['defect_order'])
        				if sym not in excluded_defects }
        defect_order = { sym:i for i,sym in enumerate(defect_dict.keys()) }

        nspecies = len(species_order)
        ndefects = len(defect_order)
        ncharge_states = len(input_params['charge_order'])
        charge_states = np.array(input_params['charge_order'])

        compound_stoichiometry = np.zeros(nspecies)
        reference_chemical_potentials = np.zeros(nspecies)
        particle_numbers = np.zeros((ndefects, nspecies))
        entropy_prefactors = np.ones((ndefects, ncharge_states))
        defect_energies = np.zeros((ndefects, ncharge_states))
        corrections = np.zeros((ndefects, ncharge_states))
        mask = np.zeros((ndefects, ncharge_states), dtype='int')

        for species, index in species_order.items():
            reference_chemical_potentials[index] = input_params['reference_chemical_potentials'][species]
            compound_stoichiometry[index] = input_params['compound_stoichiometry'][species]

        for defect, dindex in defect_order.items():
            defect_dict = input_params['defect_references'][defect]

            for species, sindex in species_order.items():
                ni_alpha = defect_dict['particles_removed'][species]
                particle_numbers[dindex][sindex] = ni_alpha

            for charge_dict in defect_dict['charge_states']:
                charge = charge_dict['charge']
                cindex = charge - min(charge_states)
                entropy_prefactors[dindex][cindex] = defect_dict['mixing_prefactor'] * \
                                                     defect_dict['configurational_prefactor'] * \
                                                     charge_dict['electronic_prefactor']
                defect_energies[dindex][cindex] = charge_dict['defect_energy']
                corrections[dindex][cindex] = charge_dict['charge_correction'] + \
                                              charge * charge_dict['potential_alignment']
                mask[dindex][cindex] = 1
        
        entropy_prefactors[mask == 0] = 0

        return cls(defect_energies,
                 corrections,
                 entropy_prefactors,
                 particle_numbers,
                 reference_chemical_potentials,
                 species_order,
                 defect_order,
                 charge_states,
                 compound_stoichiometry,
                 bulk_energy,
                 bulk_vbm,
                 band_gap,
                 bulk_volume_per_site,
                 n_sites,
                 electron_eff_mass, 
                 hole_eff_mass)