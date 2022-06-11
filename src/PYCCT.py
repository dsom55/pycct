import json
import copy

import numpy as np
from scipy import optimize
from scipy import integrate
from scipy.constants import physical_constants
import sympy as sp
import pandas as pd


class Constraint_Calculator(object):
    def __init__(self,
                 temperature,
                 ref_afs,
                 compound_stoich,
                 compound_formation_energy,
                 prefactors,
                 fermi_prefactors,
                 particle_numbers,
                 Egap,
                 electron_eff_mass,
                 hole_eff_mass,
                 volume):

        self.ref_afs = ref_afs
        self.compound_stoich = compound_stoich
        self.compound_formation_energy = compound_formation_energy
        self.prefactors = prefactors
        self.fermi_prefactors = fermi_prefactors
        self.particle_numbers = particle_numbers

        self.temperature = temperature
        kB = physical_constants['Boltzmann constant in eV/K'][0]
        self.KT = kB * self.temperature
        self.Egap = Egap
        self.volume = volume

        self.set_additional_params(electron_eff_mass, hole_eff_mass)
        self.construct_constraints()

    def set_additional_params(self, electron_eff_mass, hole_eff_mass):
        h_bar = physical_constants['Planck constant over 2 pi in eV s'][0]
        m_e = physical_constants['electron mass energy equivalent in MeV'][0]*1e6

        self.nspecies = len(self.ref_afs)
        self.ndefects = len(self.prefactors)

        self.ivars = sp.symbols("x0:%d"%(self.nspecies-1))
        self.dvars = sp.symbols("y0:%d"%(self.nspecies+2))

        self.atomic_fracs = sp.symbols("k0:%d"%(self.nspecies))
        self.fugacities = sp.symbols("z0:%d"%(self.nspecies))

        eff_mass_c = electron_eff_mass * m_e
        eff_mass_v = hole_eff_mass * m_e

        self.pref_c = self.volume * (1/(2 * np.pi**2)) * (2 * eff_mass_c / h_bar**2)**(1.5)
        self.pref_v = self.volume * (1/(2 * np.pi**2)) * (2 * eff_mass_v / h_bar**2)**(1.5)

        self.n_intrinsic = np.sqrt( self.pref_c * self.pref_v ) * np.exp(-self.Egap/(2*self.KT))
        self.Ef_intrinsic = self.Egap/2 + 0.75 * self.KT * np.log(hole_eff_mass / electron_eff_mass)

        self.compound_fugacity = np.exp(-self.compound_formation_energy/self.KT)

        self.carrier_frac = sp.symbols("r0")
        self.carrier_frac_data = 1 #np.exp((-self.Ef_intrinsic)/self.KT)

    def construct_constraints(self):
        constraints = []
        af_sum = 0
        fg_prod = 1
        q_sum = 0
        for alpha, af in enumerate(self.atomic_fracs):
            af_sum += af
            fg_prod *= self.fugacities[alpha] ** self.compound_stoich[alpha]

            if alpha < len(self.atomic_fracs) - 1:
                numerator = self.ref_afs[alpha]
                denomenator = 1
                nsum = 0
                dsum = 0
                for i, ni in enumerate(self.particle_numbers):

                    fprod = 1
                    for beta in range(len(self.atomic_fracs)):
                        fprod *= self.fugacities[beta] ** ni[beta]

                    fermi_sum = 0
                    for j, fermi_fact in enumerate(self.fermi_prefactors[i]):
                        q = j - 3
                        if True: #self.prefactors[i] * fermi_fact > 1e-10:
                            fermi_sum += fermi_fact * self.carrier_frac**(+q)
                            q_sum += q * self.prefactors[i] * fermi_fact * fprod * self.carrier_frac**(1+q) / self.n_intrinsic

                    nsum += self.prefactors[i] * ni[alpha] * fermi_sum * fprod
                    dsum += self.prefactors[i] * np.sum(ni) * fermi_sum * fprod

                numerator -= nsum
                denomenator -= dsum

                constraints.append( af * denomenator - numerator )

        constraints.append( 1.0 - af_sum )
        constraints.append( self.compound_fugacity - fg_prod )
        constraints.append( 1.0 - self.carrier_frac**2 + q_sum )

        self.initial_constraints = constraints

    def composition_constraints(self, y, x):
        return [cnt(y, x) for cnt in self.constraints]

    def solve_constraints(self, atomic_frac_data, fugacity_data, fix_dict):
        self.atomic_frac_data = copy.deepcopy(atomic_frac_data)
        self.fugacity_data = copy.deepcopy(fugacity_data)

        find_dict = {
            "fugacities": list(set(range(self.nspecies)) - set(fix_dict['fugacities'])),
            "atomic_fractions": list(set(range(self.nspecies)) - set(fix_dict['atomic_fractions']))
            }

        fix = [ self.fugacities[i] for i in fix_dict['fugacities'] ] + [ self.atomic_fracs[i] for i in fix_dict['atomic_fractions'] ]
        fix_values = [ self.fugacity_data[i] for i in fix_dict['fugacities'] ] + [ self.atomic_frac_data[i] for i in fix_dict['atomic_fractions'] ]

        find = [  self.fugacities[i] for i in find_dict['fugacities'] ] + [  self.atomic_fracs[i] for i in find_dict['atomic_fractions'] ] + [ self.carrier_frac ]
        find_guess = [  self.fugacity_data[i] for i in find_dict['fugacities'] ] + [  self.atomic_frac_data[i] for i in find_dict['atomic_fractions'] ] + [ self.carrier_frac_data ]

        self.find = find
        self.find_guess = find_guess

        ### Recover indices for output array
        fug_ind = []
        for i,zi in enumerate(self.fugacities):
            try:
                ind = find.index(zi)
                fug_ind.append([i,ind])
            except:
                continue

        af_ind = []
        for i,ki in enumerate(self.atomic_fracs):
            try:
                ind = find.index(ki)
                af_ind.append([i,ind])
            except:
                continue

        ### Transpose for broadcasting of sol.x
        self.fug_ind = np.array(fug_ind).T
        self.af_ind = np.array(af_ind).T
        self.carrier_ind = -1

        ### Set fixed and unknown variables in contraint equations
        constraints = copy.deepcopy(self.initial_constraints)
        for j,cnt in enumerate(constraints):
            cnt = cnt.subs( [ (fx, self.ivars[i]) for i,fx in enumerate(fix) ] )
            cnt = cnt.subs( [ (fd, self.dvars[i]) for i,fd in enumerate(find) ] )
            constraints[j] = sp.lambdify( [list(self.dvars)] + [list(self.ivars)] , cnt, "numpy" )

        self.constraints = constraints

        sol = optimize.root(self.composition_constraints, find_guess, args=(fix_values), method='lm')

#         cons = [{'type': 'eq', 'fun': lambda x: cst(x, fix_values)} for cst in constraints[:-1]]
#         bnds = [(0,None) for i in find_dict['fugacities']] + [(0,1) for i in find_dict['atomic_fractions']] + [(0,None)]
#         chg_neut = lambda x,y: constraints[-1](x,y)
#         sol = optimize.minimize(chg_neut, find_guess, args=(fix_values), method='trust-constr', bounds=bnds, constraints=cons)

        self.fugacity_data[ self.fug_ind[0] ] = sol.x[ self.fug_ind[1] ]
        self.atomic_frac_data[ self.af_ind[0] ] = sol.x[ self.af_ind[1] ]
        self.carrier_frac_data = sol.x[ self.carrier_ind ]

    def fermi_level(self):
        nred = lambda x, Ef: self.pref_c * np.sqrt(x - self.Egap) / (1 + np.exp( (x - Ef)/self.KT ))
        n_eqn = lambda Ef: self.carrier_frac_data * self.n_intrinsic - integrate.quad(nred, self.Egap, np.inf, args=(Ef))[0]

        pred = lambda x, Ef: self.pref_v * np.sqrt(-x) / (1 + np.exp( (Ef - x)/self.KT ))
        p_eqn = lambda Ef: (self.n_intrinsic / self.carrier_frac_data) - integrate.quad(pred, -np.inf, 0, args=(Ef))[0]

        try:
            Efermi = optimize.brenth(n_eqn, 0, self.Egap)
        except:
            Efermi = optimize.brenth(p_eqn, 0, self.Egap)

        return Efermi

    def charge_carriers(self, Efermi):
        nred = lambda x: self.pref_c * np.sqrt(x - self.Egap) / (1 + np.exp( (x - Efermi)/self.KT ))
        pred = lambda x: self.pref_v * np.sqrt(-x) / (1 + np.exp( (Efermi - x)/self.KT ))

        self.hole_conc = integrate.quad(pred, -np.inf, 0)[0]
        self.electron_conc = integrate.quad(nred, self.Egap, np.inf)[0]
        return self.hole_conc, self.electron_conc

    def defect_concentrations(self):
        defect_conc = np.zeros(self.fermi_prefactors.shape)
        for i, ni in enumerate(self.particle_numbers):
            fprod = 1
            for beta in range(len(self.atomic_fracs)):
                fprod *= self.fugacity_data[beta] ** ni[beta]

            for j, fermi_fact in enumerate(self.fermi_prefactors[i]):
                q = j - 3
                defect_conc[i][j] = self.prefactors[i] * fermi_fact * fprod * self.carrier_frac_data**(+q)

        return defect_conc

    def charge_balance(self):
        qsum = 0
        defect_concentrations = self.defect_concentrations()
        for i, dconc in enumerate(defect_concentrations):
            for j, dc in enumerate(dconc):
                q = j - 3
                qsum += q * self.carrier_frac_data * dc / self.n_intrinsic
        return 1 - self.carrier_frac_data**2 + qsum
#         return qsum

    def chemical_potentials(self):
        return -self.KT * np.log(self.fugacity_data)

    def atomic_fractions(self):
        return self.atomic_frac_data

    def fugacity_list(self):
        return self.fugacity_data

    def carrier_fraction(self):
        return self.carrier_frac_data

    @classmethod
    def from_json(cls, filename, temperature):
        with open(filename) as json_file:
            input_params = json.load(json_file)

        kB = physical_constants['Boltzmann constant in eV/K'][0]
        h_bar = physical_constants['Planck constant over 2 pi in eV s'][0]
        m_e = physical_constants['electron mass energy equivalent in MeV'][0]*1e6
        KT = kB * temperature

        volume = input_params['volume']

        Egap = input_params['bandgap']
        electron_eff_mass = input_params['electron_effective_mass']
        hole_eff_mass = input_params['hole_effective_mass']

        eff_mass_c = electron_eff_mass * m_e
        eff_mass_v = hole_eff_mass * m_e

        pref_c = volume * (1/(2 * np.pi**2)) * (2 * eff_mass_c / h_bar**2)**(1.5)
        pref_v = volume * (1/(2 * np.pi**2)) * (2 * eff_mass_v / h_bar**2)**(1.5)

        n_intrinsic = np.sqrt( pref_c * pref_v ) * np.exp(-Egap/(2*KT))
        Ef_intrinsic = Egap/2 + 0.75 * KT * np.log(hole_eff_mass / electron_eff_mass)

        species_order = { sym:i for i,sym in enumerate(input_params['species_order']) }
        defect_order = { sym:i for i,sym in enumerate(input_params['defect_order']) }

        nspecies = len(input_params['species_order'])
        ndefects = len(input_params['defect_order'])
        ncharges_states = len(input_params['charge_order'])

        compound_formation_energy = 0
        ref_afs = np.zeros(nspecies)
        compound_stoich = np.ones(nspecies)

        prefactors = np.ones(ndefects)
        fermi_prefactors = np.zeros((ndefects, ncharges_states))
        particle_numbers = np.zeros((ndefects, nspecies))

        for species, index in species_order.items():
            ref_afs[index] = input_params['reference_atomic_fractions'][species]
            compound_stoich[index] = input_params['compound_stoichiometry'][species]
            compound_formation_energy += input_params['compound_stoichiometry'][species] * input_params['reference_chemical_potentials'][species]

        for defect, dindex in defect_order.items():
            defect_dict = input_params['defect_references'][defect]

            for species, sindex in species_order.items():
                ni_alpha = defect_dict['particles_removed'][species]
                mu_alpha_ref = input_params['reference_chemical_potentials'][species]

                particle_numbers[dindex][sindex] = ni_alpha
                prefactors[dindex] *= np.exp(ni_alpha * mu_alpha_ref / KT)

            for charge_dict in defect_dict['charge_states']:
                charge = charge_dict['charge']
                cindex = charge + 3
                Eform_ref = charge_dict['reference_formation_energy']

                fermi_prefactors[dindex][cindex] = defect_dict['mixing_entropy'] * defect_dict['configurational_entropy'] * charge_dict['electronic_entropy']
                fermi_prefactors[dindex][cindex] *= np.exp(-( Eform_ref + charge * Ef_intrinsic ) / KT)

                if charge == 0:
                    prefactors[dindex] *= np.exp(-Eform_ref / KT)
                    fermi_prefactors[dindex][cindex] *= np.exp(Eform_ref / KT)


        return cls(temperature,
                   ref_afs,
                   compound_stoich,
                   compound_formation_energy,
                   prefactors,
                   fermi_prefactors,
                   particle_numbers,
                   Egap,
                   electron_eff_mass,
                   hole_eff_mass,
                   volume)



