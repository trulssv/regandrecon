import pandas as pd
import numpy as np
import json
from typing import List, Union
import os
from data.parameters import CONVERTDICT, ATOMIC_COMPOSITION_TABLE, TISSUETYPES


class TissueAtomicCompositionTable:

    def __init__(self):
        self.path = os.path.join('data', ATOMIC_COMPOSITION_TABLE)
        self.conversion_dict_path = os.path.join('data', CONVERTDICT)

        self.table = self.load_table(self.path)
        self.conversion_dict = self.load_conversion_dict(self.conversion_dict_path)
        
    def load_table(self, path=None):
        """ Load the atomic composition table """
        if path is None:
            path = self.path
        table = pd.read_csv(path)
        table.set_index('Tissue', inplace=True)
        table.index = table.index.str.lower()
        return table
    
    def load_conversion_dict(self, path=None):
        """ Load the conversion dictionary """
        if path is None:
            path = self.conversion_dict_path

        with open(path) as f:
            convert_dict = json.load(f)
        return convert_dict
    

class Atom:
    def __init__(self, name, density=None):
        self.name = name
        self.load_nist_mass_att_coeff()
        if density is None:
            self.load_nist_density()
        else:
            self.density = density

    def load_nist_mass_att_coeff(self):
        """ Get the mass attenuation """
        try:
            data = json.load(open(f'data/nist/{self.name}.json'))
        except FileNotFoundError:
            raise FileNotFoundError(f"Missing NIST attenuation data for atom '{self.name}'")
        self.E, self.mu_mass = np.array(data['energy']), np.array(data['mu'])
    
    def load_nist_density(self):
        """ Get the density of the atom """
        try:
            self.density = json.load(open(f'data/nist/density.json'))[self.name]
        except FileNotFoundError:
            raise FileNotFoundError(f"Missing NIST density data for atom '{self.name}'")
    
    def mass_att_coeff(self, kev):
        """ Get the mass attenuation at E = kev """
        mu_mass = np.exp(np.interp(np.log(kev), np.log(self.E), np.log(self.mu_mass))) # interpolate data in log space for smoother result
        return mu_mass
    
    def linear_att_coeff(self, kev):
        """ Get the linear attenuation coefficient at E = kev """
        mu_mass = self.mass_att_coeff(kev)
        return mu_mass * self.density
    
    def __str__(self):
        return self.name

class TissueType:
    def __init__(self, name):
        self.name = name
        self.tissue_types = TISSUETYPES

    @property
    def type(self):
        for tissue_type, tissue_list in self.tissue_types.items():
            tissue_list = [t.lower() for t in tissue_list]  # Convert to lowercase for case-insensitive comparison
            if self.name.lower() in tissue_list:
                self._type = tissue_type
                break
        else:
            self._type = 'soft_tissue'

        if self._type == 'bone':
            return Bone(self.name)
        elif self._type == 'fat':
            return Fat(self.name)
        else:
            return SoftTissue(self.name)
        
class Bone(TissueType):
    def __init__(self, name):
        super().__init__(name)

class SoftTissue(TissueType):
    def __init__(self, name):
        super().__init__(name)

class Fat(TissueType):
    def __init__(self, name):
        super().__init__(name)
 


class Tissue:

    def __init__(self, name, atomic_composition_table: TissueAtomicCompositionTable=None):
        self.name = name
        self.segmentation_name = name
        self.is_lesion = 'lesion' in name or 'cyst' in name

        if atomic_composition_table is None:
            atomic_composition_table = TissueAtomicCompositionTable()
        self.atomic_composition_table = atomic_composition_table

        self.atomic_composition_table_candidates = self.get_atomic_composition_table_candidates(atomic_composition_table)
        self.atomic_composition, self.density = self.get_atomic_composition(atomic_composition_table)

    @property
    def atoms(self):
        return self.atomic_composition.keys()
    
    @property
    def fractions(self):
        return self.atomic_composition.values()
    
    def get_atomic_composition_table_candidates(self, table):
        """ Get possible atomic composition tissues for the given segmentation label """
        
        # if name is in the table, return it
        if self.segmentation_name.lower() in table.table.index:
            return [self.segmentation_name]
        # else return subname from convert_dict
        else:
            atomic_composition_table_subname = table.conversion_dict.get(self.segmentation_name)

        if atomic_composition_table_subname is None:
            raise KeyError(f"Atomic composition of tissue not found for label {self.segmentation_name}")
        
        # use subname to find matching full names in atomic composition table
        candidates = []
        for tissue_candidate in table.table.index:
            if atomic_composition_table_subname.lower() in tissue_candidate.lower():
                candidates.append(tissue_candidate)

        return candidates
    
    def _generate_candidate_weights(self):
        """ Generate random weights for the candidates
        drawn from Dirichlet distribution (sum to 1) """
        n_candidates = len(self.atomic_composition_table_candidates)
        self._candidate_weights = np.random.dirichlet(np.ones(n_candidates))
        return self._candidate_weights

    @property
    def candidate_weights(self):
        if hasattr(self, '_candidate_weights'):
            return self._candidate_weights
        else:
            return self._generate_candidate_weights()
    
    def get_atomic_composition(self, table):
        """ Get the weighted atomic composition and density """
        density = 0
        atomic_composition = None

        # sum together the weighted contributions of each candidate
        for weight, candidate in zip(self.candidate_weights, self.atomic_composition_table_candidates):
            row = table.table.loc[candidate]
            candidate_density = row.iloc[0]  # g/cm^3
            candidate_atomic_composition = row.iloc[1:].astype(float) / 100  # relative fractions

            if atomic_composition is None:
                atomic_composition = candidate_atomic_composition * weight
            else:
                atomic_composition += candidate_atomic_composition * weight

            density += candidate_density * weight

        return atomic_composition, density
    
    @property
    def type(self):
        """ Return tissue type """
        return TissueType(self.atomic_composition_table_candidates[0]).type
    
    def mass_att_coeff(self, kev, skip_atom: Union[Atom, List[Atom]] = []):

        # Extract the names of the atoms to be skipped
        if skip_atom is None:
            skip_atom_names = []
        else:
            skip_atom_names = [str(skip_atom)] if not isinstance(skip_atom, list) else [str(atom) for atom in skip_atom]

        # Filter out the atoms to be skipped
        filtered_atomic_composition = {atom: fraction for atom, fraction in self.atomic_composition.items() if atom not in skip_atom_names}

        # Normalize the remaining fractions so they sum to 1
        total_fraction = sum(filtered_atomic_composition.values())
        normalized_atomic_composition = {atom: fraction / total_fraction for atom, fraction in filtered_atomic_composition.items()}
        
        att_coeff = 0
        for atom_name, fraction in normalized_atomic_composition.items():
            atom = Atom(atom_name)
            att_coeff += atom.mass_att_coeff(kev) * fraction

        return att_coeff
    
    def linear_att_coeff(self, kev, skip_atom: Union[Atom, List[Atom]] = []):
        return self.mass_att_coeff(kev, skip_atom) * self.density





if __name__ == '__main__':
    import matplotlib.pyplot as plt

    atomic_composition_table = TissueAtomicCompositionTable()
    t = Tissue('kidney_left', atomic_composition_table)
    print(t.atomic_composition_table_candidates)
    print(t.candidate_weights)
    print(t.density)
    print(t.atomic_composition)
    print(t.type)
    print(isinstance(t.type, Bone))
    E = np.linspace(20, 140, 100)
    mu = t.linear_att_coeff(E, skip_atom=Atom('H'))
    
    plt.semilogy(E, mu)
    plt.show()
