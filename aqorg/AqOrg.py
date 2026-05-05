from IPython.display import SVG
from rdkit import Chem
from rdkit.Chem import rdDepictor, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
import pandas as pd
import math
import sigfig
from collections import defaultdict

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", SyntaxWarning)
    import pubchempy as pcp
    
import os
from chemparse import parse_formula
from datetime import datetime
from wormutils import Error_Handler, find_HKF, import_package_file


def find_sigfigs(x):
    
    '''
    Get the number of significant digits in a string representing a number up to
    eight digits long.

    Parameters
    ----------
    x : str
        A string denoting a number. This can include scientific notation.
    
    Examples
    --------
    >>> find_sigfigs("5.220")
    4
    
    This also takes into account scientific notation.
    
    >>> find_sigfigs("1.23e+3")
    3
    
    Insignificant zeros are ignored.
    
    >>> find_sigfigs("4000")
    1
    
    A decimal point denotes that zeros are significant.
    
    >>> find_sigfigs("4000.")
    4
    '''
    
    x = str(x)
    
    # change all the 'E' to 'e'
    x = x.lower()
    if ('-' == x[0]):
        x = x[1:]
    if ('e' in x):
        # return the length of the numbers before the 'e'
        myStr = x.split('e')
        return len(myStr[0]) - 1  # to compenstate for the decimal point
    else:
        # put it in e format and return the result of that
        ### NOTE: because of the 8 below, it may do crazy things when it parses 9 sigfigs
        n = ('%.*e' % (8, float(x))).split('e')
        # remove and count the number of removed user added zeroes. (these are sig figs)
        if '.' in x:
            s = x.replace('.', '')
            #number of zeroes to add back in
            l = len(s) - len(s.rstrip('0'))
            #strip off the python added zeroes and add back in the ones the user added
            n[0] = n[0].rstrip('0') + ''.join(['0' for num in range(l)])
        else:
            #the user had no trailing zeroes so just strip them all
            n[0] = n[0].rstrip('0')
        #pass it back to the beginning to be parsed
    return find_sigfigs('e'.join(n))


class Estimate():
    
    """
    Estimate thermodynamic properties of an aqueous organic molecule.
    
    Parameters
    ----------
    name : str, optional
        Name of the aqueous organic molecule that will have its thermodynamic
        properties estimated.

    smiles : str, optional
        A SMILES string representing the molecule.
                       
    show : bool, default True
        Show a diagram of the molecule?
    
    fig_dim : list, default [400, 150]
        X and Y dimensions in pixels of the figure if show=True or if save=True
    
    ig_group_data : str, optional
        Path of a CSV containing custom ideal gas group contribution data.

    hyd_group_data : str, optional
        Path of a CSV containing custom hydration property group contribution data.

    aq_group_data : str, optional
        Path of a CSV containing custom aqueous group contribution data.
    
    test : bool, default False
        Perform a simple group matching test instead of estimating properties?

    state : str, default "aq"
        Can be "aq" or "gas". Estimate the properties of an aqueous molecule or
        an ideal gas?

    ig_method : str, default "Joback"
        Method used to estimate ideal gas properties.

    save : bool, default False
        Save molecular structure figures as PNG and SVG?
    
    **kwargs : numeric or str, optional
        Known standard state partial molal thermodynamic properties at 298.15 K
        and 1 bar. These will not be estimated, but instead will be used to
        estimate other properties and parameters. Valid **kwargs include:
        
        - Gh : Gibbs free energy change of hydration, kJ/mol.
        - Hh : Enthalpy change of hydration, kJ/mol.
        - Sh : Entropy change of hydration, J/mol/K.
        - Cph : Heat capacity change of hydration, J/mol/K.
        - V : Volume change of hydration, cm3/mol.
        - Gh_err : Error associated with Gh (default 0 kJ/mol).
        - Hh_err : Error associated with Hh (default 0 kJ/mol).
        - Sh_err : Error associated with Sh (default 0 J/mol/K).
        - Cph_err : Error associated with Cph (default 0 J/mol/K).
        - V_err : error associated with V (default 0 cm3/mol).
        - Gig : Ideal gas Gibbs free energy of formation, kJ/mol.
        - Hig : Ideal gas enthalpy of formation, kJ/mol.
        - Sig : Ideal gas entropy, J/mol/K.
        - Cpig : Ideal gas isobaric heat capacity, J/mol/K.
        - Gaq : Aqueous Gibbs free energy of formation, kJ/mol.
        - Haq : Aqueous enthalpy of formation, kJ/mol.
        - Saq : Aqueous entropy, J/mol/K.
        - Cpaq : Aqueous isobaric heat capacity, J/mol/K.
    
    Attributes
    ----------
    pcp_compound : pcp.get_compounds()
        PubChemPy compound object.
        
    smiles : str
        Canonical SMILES string.
        
    formula : str
        Molecular formula.
        
    formula_dict : dict
        Dictionary of element abundance in the molecular formula.
        
    element_data : pd.DataFrame()
        Table of element data adapted from Jeff Dick's CHNOSZ package for R.
        
    Selements : numeric
        Sum of the contributions of the entropies of the elements according to
        Cox, J. D., Wagman, D. D., & Medvedev, V. A. (1989). CODATA key values
        for thermodynamics. Chem/Mats-Sci/E.
        
    note : str
        Notes and warnings associated with the estimation.
        
    charge : numeric
        The charge of the molecule.
        
    OBIGT : pd.DataFrame()
        Table of estimated thermodynamic properties and parameters. The format
        is styled after Jeff Dick's OBIGT thermodynamic table in the CHNOSZ
        package (see https://chnosz.net/manual/thermo.html).

    hide_traceback : bool, default True
        Hide traceback message when encountering errors handled by this function?
        When True, error messages handled by this class will be short and to
        the point.
    
    """
    
    def __init__(self, name=None, smiles=None, show=True, fig_dim=[400, 150], ig_group_data=None, hyd_group_data=None, aq_group_data=None,
                       test=False, state='aq', ig_method="Joback", aq_method="hyd+ig", round_sf=True,
                       save=False, hide_traceback=True, substitute_groups=None, assign_groups_to_atoms=None, **kwargs):
                       # E_units="J" # not implemented... tricky because groups
                                     # are in both kJ and J units.

        self.err_handler = Error_Handler(clean=hide_traceback)
        
        self.name = name
        self.smiles = smiles
        self.state = state
        self.ig_method = ig_method
        self.aq_method = aq_method
        self.show = show
        self.fig_dim = fig_dim
        
        # valid kwargs
        self.Gh = None
        self.Hh = None
        self.Sh = None
        self.Cph = None
        self.V = None
        self.Gh_err = 0
        self.Hh_err = 0
        self.Sh_err = 0
        self.Cph_err = 0
        self.V_err = 0
        self.Gig = None
        self.Hig = None
        self.Sig = None
        self.Cpig_a = None
        self.Cpig_b = None
        self.Cpig_c = None
        self.Cpig_d = None
        self.Cpig = None
        self.Gaq = None
        self.Haq = None
        self.Saq = None
        self.Cpaq = None

        if "pcp_compound" not in list(kwargs.keys()):
            self.pcp_compound = None

        for key, value in kwargs.items():
            self.__setattr__(key, value)
        
        self.ig_group_data = ig_group_data
        self.hyd_group_data = hyd_group_data
        self.aq_group_data = aq_group_data
        self.substitute_groups = substitute_groups
        self.assign_groups_to_atoms = assign_groups_to_atoms if assign_groups_to_atoms is not None else {}

        if self.state == "gas" and ig_group_data != None:
            self.group_data = ig_group_data
        elif self.state == "hyd" and hyd_group_data != None:
            self.group_data = hyd_group_data
        elif self.state == "aq" and aq_group_data != None:
            self.group_data = aq_group_data
        else:
            self.group_data = None


        self.load_group_data()
        # Only apply substitutions when this Estimate directly uses its group data
        # for property estimation. For state="aq" with aq_method="hyd+ig", the parent
        # loads the aq CSV for validation only; substitutions are applied to the
        # gas/hyd sub-Estimates instead.
        if not (self.state == "aq" and self.aq_method != "aqueous"):
            self._apply_substitute_groups()
        self.get_mol_smiles_formula_formula_dict()
        
        if "-" in self.formula_dict.keys() or "+" in self.formula_dict.keys():
            self.err_handler.raise_exception(self.name + " cannot be estimated because it has a net charge.")

        if test:
            if self.state == "aq" and self.aq_method != "aqueous":
                # Test against the gas and hyd group data separately,
                # mirroring how the actual estimation works.
                ig_sub = self.substitute_groups if self.ig_method != "Joback" else None
                ig_assign = self.assign_groups_to_atoms if self.ig_method != "Joback" else None
                Estimate(smiles=self.smiles, name=self.name, state="gas",
                         test=True, ig_method=self.ig_method,
                         ig_group_data=self.ig_group_data,
                         substitute_groups=ig_sub,
                         assign_groups_to_atoms=ig_assign,
                         show=False, fig_dim=self.fig_dim)
                Estimate(smiles=self.smiles, name=self.name, state="hyd",
                         test=True, hyd_group_data=self.hyd_group_data,
                         substitute_groups=self.substitute_groups,
                         assign_groups_to_atoms=self.assign_groups_to_atoms,
                         show=False, fig_dim=self.fig_dim)
            else:
                self.__test_group_match()
        else:
            # load properties of the elements
            # Cox, J. D., Wagman, D. D., and Medvedev, V. A., CODATA Key Values
            # for Thermodynamics, Hemisphere Publishing Corp., New York, 1989.
            # Compiled into a CSV by Jeffrey Dick for CHNOSZ

            with import_package_file(__name__, 'data/element.csv', as_file=True) as path:
                element_data = pd.read_csv(path, index_col="element")
            
            self.element_data = element_data.loc[element_data['source'] == "CWM89"]
            
            self.Selements = self.__entropy()
            self.note = ""
            self.charge = 0 # TODO: allow charge!
            
            # Skip __set_groups() for aq with hyd+ig: the parent's aq group data
            # is not used for estimation; sub-Estimates handle their own matching.
            if not (state == 'aq' and self.aq_method != "aqueous"):
                self.__set_groups()

            if state == 'gas':  # calculates self.Xig properties
                if self.ig_method == "Joback":
                    self.__est_joback()
                    self.group_contributions = self._build_group_contributions_df(
                        ['Gig', 'Hig'],
                        method_constants={'Gig': 53.88, 'Hig': 68.29})
                else:
                    self.__est_ig(round_sf=round_sf)
                    self.group_contributions = self._build_group_contributions_df(
                        ['Gig', 'Hig', 'Cpig'])
            elif state == 'hyd':  # calculates self.Xh properties
                self.__est_hyd(round_sf=round_sf)
                self.group_contributions = self._build_group_contributions_df(
                    ['Gh', 'Hh', 'Cph', 'V'])
            elif state == 'aq':  # calculates self.Xaq properties
                if self.aq_method != "aqueous":
                    # ideal gas and hydration properties summed to get aqueous properties
                    ig_all_precomputed = all(v is not None for v in [self.Gig, self.Hig, self.Sig, self.Cpig])
                    hyd_all_precomputed = all(v is not None for v in [self.Gh, self.Hh, self.Sh, self.Cph, self.V])

                    if ig_all_precomputed:
                        self.ig_group_contributions = None
                    else:
                        if self.ig_method == "Joback":
                            ig_props = Estimate(name, smiles=smiles, state='gas', ig_method="Joback", show=False, ig_group_data=self.ig_group_data, round_sf=False, fig_dim=self.fig_dim,
                                                **{"pcp_compound":self.pcp_compound, "Gig":self.Gig, "Hig":self.Hig, "Sig":self.Sig, "Cpig":self.Cpig})
                        else:
                            ig_props = Estimate(smiles=self.smiles, name=self.name, state="gas", show=False, ig_method=self.ig_method, ig_group_data=self.ig_group_data, round_sf=False, fig_dim=self.fig_dim,
                                                substitute_groups=self.substitute_groups,
                                                assign_groups_to_atoms=self.assign_groups_to_atoms,
                                                **{"pcp_compound":self.pcp_compound, "Gig":self.Gig, "Hig":self.Hig, "Sig":self.Sig, "Cpig":self.Cpig})
                        self.Gig = ig_props.Gig
                        self.Hig = ig_props.Hig
                        self.Sig = ig_props.Sig
                        self.Cpig = ig_props.Cpig
                        self.ig_group_contributions = ig_props.group_contributions

                    if hyd_all_precomputed:
                        self.hyd_group_contributions = None
                    else:
                        hyd_props = Estimate(smiles=self.smiles, name=self.name, state="hyd", show=False, hyd_group_data=self.hyd_group_data, round_sf=False, fig_dim=self.fig_dim,
                                             substitute_groups=self.substitute_groups,
                                             assign_groups_to_atoms=self.assign_groups_to_atoms,
                                             **{"pcp_compound":self.pcp_compound, "Gh":self.Gh, "Hh":self.Hh, "Sh":self.Sh, "Cph":self.Cph, "V":self.V})
                        self.Gh = hyd_props.Gh
                        self.Hh = hyd_props.Hh
                        self.Sh = hyd_props.Sh
                        self.Cph = hyd_props.Cph
                        self.V = hyd_props.V
                        self.hyd_group_contributions = hyd_props.group_contributions

                self.__est_aq(round_sf=round_sf)

                if self.aq_method == "aqueous":
                    self.group_contributions = self._build_group_contributions_df(
                        ['Gaq', 'Haq', 'Cpaq', 'V'])

                self.OBIGT = self.__convert_to_OBIGT()
            else:
                self.err_handler.raise_exception("State must be 'aq', 'hyd', or 'gas'.")

        if self.show:
            self.display_molecule(save=save)


    def get_mol_smiles_formula_formula_dict(self):
        if not isinstance(self.smiles, str):
            if self.pcp_compound == None:
                # look up compound on PubChem
                self.pcp_compound = pcp.get_compounds(self.name, "name")
            if len(self.pcp_compound) == 0:
                self.err_handler.raise_exception("Could not find '" + self.name + "' in PubChem's online database.")
            self.smiles = self.pcp_compound[0].connectivity_smiles 
            self.formula = self.pcp_compound[0].molecular_formula
            self.mol = Chem.MolFromSmiles(self.smiles)
        else:
            self.mol = Chem.MolFromSmiles(self.smiles)
            self.formula = CalcMolFormula(self.mol)
            
        self.formula_dict = parse_formula(self.formula)
    
    def load_group_data(self):
        # load group contribution data
        if not isinstance(self.group_data, pd.DataFrame):
            if self.state == "aq":
                if self.group_data is None:
                    # load default aqueous group data
                    with import_package_file(__name__, 'data/aq_group_contribution_data.csv', as_file=True) as path:
                        self.group_data = pd.read_csv(path, dtype=str)
                else:
                    # load custom aqueous group data
                    self.group_data = pd.read_csv(self.group_data, dtype=str)
            elif self.state == "hyd":
                if self.group_data is None:
                    # load default hydration group data
                    with import_package_file(__name__, 'data/hyd_group_contribution_data.csv', as_file=True) as path:
                        self.group_data = pd.read_csv(path, dtype=str)
                else:
                    # load custom hydration group data
                    self.group_data = pd.read_csv(self.group_data, dtype=str)
            elif self.state == "gas":
                if self.ig_method == "Joback":
                    with import_package_file(__name__, 'data/joback_groups.csv', as_file=True) as path:
                        self.group_data = pd.read_csv(path, dtype=str)
                else:
                    # load custom gas group
                    self.group_data = pd.read_csv(self.group_data, dtype=str)
            else:
                self.err_handler.raise_exception("State is unrecognized. Must be either 'aq', 'hyd', or 'gas'.")

        self.group_data['elem'] = self.group_data['elem'].fillna('')
        self.pattern_dict = pd.Series(self.group_data["elem"].values,
                                      index=self.group_data["smarts"]).to_dict()
        self.group_data = self.group_data.set_index("smarts")


    def _apply_substitute_groups(self):
        if self.substitute_groups is None:
            return
        for new_smarts, existing_smarts in self.substitute_groups.items():
            if existing_smarts not in self.group_data.index:
                print(f"Warning: substitute group '{existing_smarts}' not found in group data. Skipping.")
                continue
            # If this group already exists in group_data and the user has
            # explicitly assigned it to atoms, preserve its original
            # properties so that assign_groups_to_atoms uses the CSV values.
            if new_smarts in self.group_data.index and new_smarts in self.assign_groups_to_atoms:
                continue
            self.group_data.loc[new_smarts] = self.group_data.loc[existing_smarts]
            self.pattern_dict[new_smarts] = self.pattern_dict[existing_smarts]


    def _build_group_contributions_df(self, props, method_constants=None):
        """Build a DataFrame showing group contributions for each estimated property.

        Parameters
        ----------
        props : list
            Property column names to include (e.g. ['Gig', 'Hig', 'Cpig']).
        method_constants : dict, optional
            Extra per-property offsets added to the total (e.g. Joback constants).
        """
        rows = []
        for group in self.groups:
            count = self.group_matches.loc[self.name, group]
            try:
                count = count.item()
            except (AttributeError, ValueError):
                pass
            count = int(count)
            if count == 0:
                continue
            row = {'group': group, 'count': count}
            for prop in props:
                try:
                    val = float(self.group_data.loc[group, prop])
                except:
                    val = float('nan')
                row[prop] = val
                row[prop + '_contribution'] = count * val
            rows.append(row)

        # Add Yo row if present in group data
        if 'Yo' in self.group_data.index:
            yo_row = {'group': 'Yo', 'count': ''}
            for prop in props:
                try:
                    val = float(self.group_data.loc['Yo', prop])
                except:
                    val = 0.0
                yo_row[prop] = val
                yo_row[prop + '_contribution'] = val
            rows.append(yo_row)

        # Add method constants row (e.g. Joback offsets)
        if method_constants is not None:
            const_row = {'group': 'Method constant', 'count': ''}
            for prop in props:
                val = method_constants.get(prop, 0.0)
                const_row[prop] = val
                const_row[prop + '_contribution'] = val
            rows.append(const_row)

        df = pd.DataFrame(rows)

        # Add Total row
        total = {'group': 'Total', 'count': ''}
        for prop in props:
            total[prop] = ''
            total[prop + '_contribution'] = df[prop + '_contribution'].sum()
        df = pd.concat([df, pd.DataFrame([total])], ignore_index=True)

        return df


    def __set_groups(self):
        
        self.group_matches = pd.DataFrame(self.match_groups(), index=[self.name])

        # remove columns with no matches
        self.group_matches = self.group_matches.loc[:, (self.group_matches.sum(axis=0) != 0)]
        
        # get a list of relevent groups
        self.groups = [grp for grp in self.group_matches.columns if grp != "formula"]
        

    def __entropy(self, unit="J/mol/K"):
        
        """
        Calculate the standard molal entropy of elements in a molecule.
        """

        entropies = [(self.element_data.loc[elem, "s"]/self.element_data.loc[elem, "n"])*self.formula_dict[elem] for elem in list(self.formula_dict.keys())]
        if unit == "J/mol/K":
            unit_conv = 4.184
        elif unit == "cal/mol/K":
            unit_conv = 1
        else:
            print("Warning in entropy: specified unit", unit,
                  "is not recognized. Returning entropy in J/mol/K")
            unit_conv = 4.184
            
        return sum(entropies)*unit_conv

    @staticmethod
    def dict_to_formula(formula_dict):
        """
        Convert a formula dictionary into a formula string.
        Example:
        ```dict_to_formula({"C":1, "H":1, "O":3, "-":1})```
        will output "HCO3-"

        Parameters
        ----------
        formula_dict : dict
            Dictionary of elements and charge (as keys) and their quantities
            (as values). Meant to be able to reverse the dictionary output of
            chemparse.parse_formula back into a formula string.
            For example, {"C":1, "H":1, "O":3, "-":1} representing HCO3-
            
        Returns
        ----------
        str
            A chemical formula. E.g., "HCO3-"
        """
        
        formula_string = ""
        for key in formula_dict.keys():
            if abs(formula_dict[key]) == 1:
                v = ""
            else:
                v = formula_dict[key]
                if (v).is_integer():
                    v = int(v)

            formula_string = formula_string + str(key) + str(v)
        return formula_string

    
    def match_groups(self):
        """
        Match SMARTS strings to a molecule and get a dictionary of group matches.
        This function is meant to be used internally by `Estimate`.
        """

        patterns = self.pattern_dict.keys()
        mol = Chem.MolFromSmiles(self.smiles)

        match_dict = dict(zip(patterns, [0]*len(patterns))) # initialize match_dict
        for pattern in patterns:
            if pattern != "Yo": # never match material point
                try:
                    match_dict[pattern] = len(mol.GetSubstructMatches(Chem.MolFromSmarts(pattern)))
                except:
                    print("Warning in match_groups(): problem",
                          "identifying SMARTS group", pattern,
                          ". Skipping this group.")

        # Apply assign_groups_to_atoms overrides if provided
        if self.assign_groups_to_atoms:
            # Build atom-to-smarts mapping (same logic as print_atom_group_matches)
            atom_to_smarts = {}
            for smarts in self.pattern_dict.keys():
                if smarts == "Yo":
                    continue
                query = Chem.MolFromSmarts(smarts)
                if query is None:
                    continue
                for match in mol.GetSubstructMatches(query):
                    core_idx = match[1] if len(match) > 1 else match[0]
                    if core_idx not in atom_to_smarts:
                        atom_to_smarts[core_idx] = smarts

            # Apply display substitution so dictionary keys match atom table
            if self.substitute_groups:
                for core_idx in atom_to_smarts:
                    matched_smarts = atom_to_smarts[core_idx]
                    if matched_smarts in self.substitute_groups:
                        atom_to_smarts[core_idx] = self.substitute_groups[matched_smarts]

            # Validate and apply overrides
            for smarts, atom_indices in self.assign_groups_to_atoms.items():
                if smarts not in self.group_data.index:
                    self.err_handler.raise_exception(
                        "assign_groups_to_atoms: group '" + smarts +
                        "' not found in group data.")
                for idx in atom_indices:
                    if idx < 0 or idx >= mol.GetNumAtoms():
                        self.err_handler.raise_exception(
                            "assign_groups_to_atoms: atom index " + str(idx) +
                            " is out of range for " + self.name + ".")
                    atom_to_smarts[idx] = smarts

            # Rebuild match_dict from the modified atom-to-smarts mapping
            match_dict = dict(zip(patterns, [0]*len(patterns)))
            for smarts in atom_to_smarts.values():
                if smarts in match_dict:
                    match_dict[smarts] += 1

        ### check that total formula of groups matches that of the molecule

        # create a dictionary of element matches
        total_formula_dict = {}
        for match in match_dict.keys():
            this_match = parse_formula(self.pattern_dict[match])
            for element in this_match.keys():
                this_match[element] *= match_dict[match]
                if element in total_formula_dict:
                    total_formula_dict[element] += this_match[element]
                else:
                    total_formula_dict[str(element)] = 0
                    total_formula_dict[element] += this_match[element]

        # remove keys of elements with a value of 0 (e.g. "H":0.0)
        for key in list(total_formula_dict.keys()):
            if total_formula_dict[key] == 0.0:
                total_formula_dict.pop(key, None)

        # retrieve individual charges that contribute to net charge
        all_charges = [a.GetFormalCharge() for a in self.mol.GetAtoms()]
        chargedict = {}
        if any(x != 0 for x in all_charges):
            pos_charge = sum([charge for charge in all_charges if charge > 0])
            neg_charge = abs(sum([charge for charge in all_charges if charge < 0]))
            if pos_charge > 0:
                chargedict['+']=float(pos_charge)
            if neg_charge > 0:
                chargedict['-']=float(neg_charge)
        else:
            chargedict = {}

        # perform the comparison
        test_dict = parse_formula(self.formula)
        test_dict.update(chargedict)
        if total_formula_dict != test_dict:
            mssg = "The formula of " + self.name + \
                " does not equal the the elemental composition of the " + \
                "matched groups (state='" + self.state + "'). This could be because the database " + \
                "is missing representative groups.\nFormula of " + \
                self.name + ":\n"
            pcp_dict = parse_formula(self.formula)
            pcp_dict.update(chargedict)
            mssg = mssg + str(pcp_dict) + "\nTotal formula of group matches:\n" + \
                str(total_formula_dict)
            match_dict_incomplete = {k:v for k,v in zip(match_dict.keys(), match_dict.values()) if v!= 0}
            mssg = mssg + "\nIncomplete group matches:\n" + \
                str(match_dict_incomplete)
            self.display_highlighted_molecule(match_dict_incomplete)
            self.display_highlighted_molecule(match_dict_incomplete, show_atom_indices=True)
            self.print_atom_group_matches(match_dict_incomplete)
            self.suggest_SMARTS(match_dict_incomplete)
            self.err_handler.raise_exception(mssg)

        # add molecular formula to match dictionary
        match_dict["formula"] = self.dict_to_formula(total_formula_dict)

        return match_dict

    @staticmethod
    def get_group_smarts(mol, atom_idx):
        atom = mol.GetAtomWithIdx(atom_idx)
        sym = atom.GetSymbol().lower() if atom.GetIsAromatic() else atom.GetSymbol()
        h = atom.GetTotalNumHs()
        ring_str = 'R' if atom.IsInRing() else 'R0'
        core = f'[{sym}X{atom.GetDegree() + h}H{h}{ring_str}]'

        neighbor_smarts = []
        for neighbor in atom.GetNeighbors():
            ns = neighbor.GetSymbol().lower() if neighbor.GetIsAromatic() else neighbor.GetSymbol()
            nh = neighbor.GetTotalNumHs()
            nd = neighbor.GetDegree() + nh
            nr = 'R' if neighbor.IsInRing() else 'R0'
            neighbor_smarts.append(f'[{ns}X{nd}H{nh}{nr}]')

        neighbor_smarts.sort()

        # Build branched SMARTS so match[1] is always the core atom
        # and all neighbors are bonded to the core (star topology).
        # e.g. for 4 neighbors: [N0]-[core](-[N1])(-[N2])-[N3]
        if len(neighbor_smarts) >= 2:
            branches = ''.join(f'(-{n})' for n in neighbor_smarts[1:-1])
            return f'{neighbor_smarts[0]}-{core}{branches}-{neighbor_smarts[-1]}'
        elif len(neighbor_smarts) == 1:
            return f'{neighbor_smarts[0]}-{core}'
        else:
            return core


    def _build_atom_to_smarts(self):
        """Build atom-to-SMARTS mapping, applying substitute_groups and
        assign_groups_to_atoms overrides."""
        mol = Chem.MolFromSmiles(self.smiles)

        # Use the same matching order as match_groups(): iterate all SMARTS
        # from pattern_dict so the first pattern to claim a core atom wins,
        # consistent with the counting in match_groups().
        atom_to_smarts = {}
        for smarts in self.pattern_dict.keys():
            if smarts == "Yo":
                continue
            query = Chem.MolFromSmarts(smarts)
            if query is None:
                continue
            for match in mol.GetSubstructMatches(query):
                core_idx = match[1] if len(match) > 1 else match[0]
                if core_idx not in atom_to_smarts:
                    atom_to_smarts[core_idx] = smarts

        # For substituted groups, display the target group name (the group
        # whose properties are being used) instead of the matching SMARTS.
        if self.substitute_groups:
            for core_idx in atom_to_smarts:
                matched_smarts = atom_to_smarts[core_idx]
                if matched_smarts in self.substitute_groups:
                    atom_to_smarts[core_idx] = self.substitute_groups[matched_smarts]

        # Apply assign_groups_to_atoms overrides
        if self.assign_groups_to_atoms:
            for smarts, atom_indices in self.assign_groups_to_atoms.items():
                for idx in atom_indices:
                    atom_to_smarts[idx] = smarts

        return atom_to_smarts

    def print_atom_group_matches(self, match_dict):
        """Print which SMARTS group from match_dict matched each atom."""
        mol = Chem.MolFromSmiles(self.smiles)
        atom_to_smarts = self._build_atom_to_smarts()

        state_labels = {"gas": "Ideal gas", "hyd": "Hydration", "aq": "Aqueous"}
        label = state_labels.get(self.state, "Atom")
        print(f"{label} atom group matches:")
        print(f"  {'Atom':<6} {'Symbol':<8} {'SMARTS group'}")
        print(f"  {'----':<6} {'------':<8} {'------------'}")
        for idx in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(idx)
            symbol = atom.GetSymbol()
            if idx in atom_to_smarts:
                print(f"  {idx:<6} {symbol:<8} {atom_to_smarts[idx]}")
            else:
                print(f"  {idx:<6} {symbol:<8} (unmatched)")

    def suggest_SMARTS(self, match_dict):
        mol = Chem.MolFromSmiles(self.smiles)

        # Use _build_atom_to_smarts to get the full atom mapping
        # (includes assign_groups_to_atoms overrides)
        matched = set(self._build_atom_to_smarts().keys())

        unmatched_groups = defaultdict(list)
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            if idx not in matched:
                smarts = self.get_group_smarts(mol, idx)
                unmatched_groups[smarts].append(idx)

        print("Unmatched atom SMARTS suggestions:")
        for smarts, indices in unmatched_groups.items():
            print(f"  '{smarts}' → atoms {indices}")


    def display_highlighted_molecule(self, match_dict, show_atom_indices=False):

        mol = Chem.MolFromSmiles(self.smiles)

        # Use _build_atom_to_smarts to get the full atom mapping
        # (includes assign_groups_to_atoms overrides)
        atom_to_smarts = self._build_atom_to_smarts()

        matched_atoms = list(atom_to_smarts.keys())
        matched_set = set(matched_atoms)
        all_atoms = set(range(mol.GetNumAtoms()))
        unmatched_atoms = list(all_atoms - matched_set)

        # Build per-atom color map
        green = (0.6, 1.0, 0.6)
        red = (1.0, 0.7, 0.7)
        atom_colors = {}
        for idx in matched_atoms:
            atom_colors[idx] = green
        for idx in unmatched_atoms:
            atom_colors[idx] = red

        # Only highlight bonds where both atoms share the same highlight color
        highlight_bonds = []
        bond_colors = {}
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtomIdx()
            a2 = bond.GetEndAtomIdx()
            both_matched = a1 in matched_set and a2 in matched_set
            both_unmatched = a1 not in matched_set and a2 not in matched_set
            if both_matched:
                highlight_bonds.append(bond.GetIdx())
                bond_colors[bond.GetIdx()] = green
            elif both_unmatched:
                highlight_bonds.append(bond.GetIdx())
                bond_colors[bond.GetIdx()] = red

        highlight_atoms = matched_atoms + unmatched_atoms

        if show_atom_indices:
            for atom in mol.GetAtoms():
                atom.SetProp('molAtomMapNumber', str(atom.GetIdx()))

        d2d = rdMolDraw2D.MolDraw2DSVG(self.fig_dim[0], self.fig_dim[1])
        d2d.DrawMolecule(
            mol,
            highlightAtoms=highlight_atoms,
            highlightAtomColors=atom_colors,
            highlightBonds=highlight_bonds,
            highlightBondColors=bond_colors,
        )
        d2d.FinishDrawing()

        import re
        svg = d2d.GetDrawingText()
        # Replace width and height attributes in the SVG tag
        svg = re.sub(r'width=\'[^\']*\'', f'width=\'{self.fig_dim[0]}px\'', svg)
        svg = re.sub(r'height=\'[^\']*\'', f'height=\'{self.fig_dim[1]}px\'', svg)

        display(SVG(svg))


    def display_molecule(self, show=True, save=False, highlights_list=None):
        """
        Display a molecule in a Jupyter notebook or save it as an SVG and PNG.
        This function is meant to be used internally by `Estimate`.
        """
        mol_smiles = Chem.MolFromSmiles(self.smiles)
        
        mc = Chem.Mol(mol_smiles.ToBinary())
        
        if not mc.GetNumConformers():
            #Compute 2D coordinates
            rdDepictor.Compute2DCoords(mc)
        # init the drawer with the size
        drawer = rdMolDraw2D.MolDraw2DSVG(self.fig_dim[0], self.fig_dim[1])
        #draw the molcule
        drawer.DrawMolecule(mc)
        drawer.FinishDrawing()
        # get the SVG string
        svg = drawer.GetDrawingText()

        if show:
            # fix the svg string and display it
            display(SVG(svg.replace('svg:','')))

        if save:
            os.makedirs("mol_svg", exist_ok=True)
            os.makedirs("mol_png", exist_ok=True)
            Draw.MolToFile(mol_smiles, "mol_svg/"+self.name+".svg" )
            Draw.MolToFile(mol_smiles, "mol_png/"+self.name+".png")


    def __test_group_match(self):
        match_dict = self.match_groups()
        match_dict_nonzero = {key:value for key,value in zip(match_dict.keys(), match_dict.values()) if value !=0 and key != "formula"}
        self.display_highlighted_molecule(match_dict_nonzero, show_atom_indices=True)
        self.print_atom_group_matches(match_dict_nonzero)
        print("\nGroup counts:")
        print(f"  {'SMARTS group':<40} {'Count'}")
        print(f"  {'------------':<40} {'-----'}")
        for group, count in match_dict_nonzero.items():
            print(f"  {group:<40} {count}")
        return match_dict_nonzero
        

    def __est_ig(self, props=["Gig", "Hig", "Sig", "Cpig"], round_sf=True):

        for prop in props:
            err_str = prop + "_err"

            # if property is already defined, skip estimating it
            if prop in dir(self):
                if not self.__getattribute__(prop) is None:
                    continue

            if prop == "Sig":
                # calculate Sig
                self.Sig = ((self.Gig - self.Hig)/-298.15)*1000 + self.Selements

            else:
                self.__sum_props(prop)

        # TODO: employ round_sf for ideal gas estimation
        
    
    def __est_joback(self):
        
        # Estimate standard state ideal gas properties of a molecule using the Joback
        # method. (Joback K. G., Reid R. C., "Estimation of Pure-Component Properties
        # from Group-Contributions", Chem. Eng. Commun., 57, 233–243, 1987.)

        # values to be added to final estimate of each property
        joback_props = {"Gig":53.88, "Hig":68.29, # kJ/mol
                        "Cpig_a":-37.93, "Cpig_b":0.210, # j/mol/K
                        "Cpig_c":-3.91*10**-4, "Cpig_d":2.06*10**-7} # j/mol/K
        
        for prop in joback_props.keys():
            mol_prop = 0
            error_groups = []

            # if property is already defined, skip estimating it
            if prop in dir(self):
                if not self.__getattribute__(prop) is None:
                    continue

            for group in self.groups:

                try:
                    contains_group = self.group_matches.loc[self.name, group][0] != 0
                except:
                    contains_group = self.group_matches.loc[self.name, group] != 0

                # if this molecule contains this group...
                if contains_group:
                    try:
                        # add number of groups multiplied by its contribution
                        mol_prop += self.group_matches.loc[self.name, group] * float(self.group_data.loc[group, prop])
                    except:
                        error_groups.append(group)
                        
                if len(error_groups) == 0:
                    self.__setattr__(prop, mol_prop+joback_props[prop])

        if len(error_groups) > 0:
            self.err_handler.raise_exception("" + self.name + " encountered errors with group(s): "
                ""+str(error_groups) + ". Are these groups assigned "
                "ideal gas properties in the Joback data file?")
        
        if self.__getattribute__("Cpig") is None:
            # calculate Cpig
            T=298.15
            self.Cpig = self.Cpig_a + self.Cpig_b*T + self.Cpig_c*T**2 +\
                        self.Cpig_d*T**3
        
        if self.__getattribute__("Sig") is None:
            # calculate Sig
            self.Sig = ((self.Gig - self.Hig)/-298.15)*1000 + self.Selements

        # TODO: employ round_sf for ideal gas estimation
    
    
    def __est_hyd(self, props=["Gh", "Hh", "Sh", "Cph", "V"], round_sf=True):
        for prop in props:
            err_str = prop + "_err"

            # if property is already defined, skip estimating it
            if prop in dir(self):
                if not self.__getattribute__(prop) is None:
                    continue

            # derive Sh, entropy of hydration, in J/mol K
            if prop == "Sh":
 
                # Entropy calculated from S = (G-H)/(-Tref)
                mol_prop = (float(self.Gh) - float(self.Hh))/(-298.15)
                mol_prop = mol_prop*1000 # convert kJ/molK to J/molK

                # propagate error from Gh and Hh to estimate Sh error.
                # equation used: Sh_err = Sh*sqrt((Gh_err/Gh)**2 + (Hh_err/Hh)**2)
                Gh_err_float = float(self.Gh_err)/float(self.Gh)
                Hh_err_float = float(self.Hh_err)/float(self.Hh)
                mol_err = abs(mol_prop)*math.sqrt(Gh_err_float**2 + Hh_err_float**2)

                if round_sf:
                    # check whether Gh or Hh as the fewest sigfigs
                    sf = min([find_sigfigs(self.Gh), find_sigfigs(self.Hh)])

                    # round Sh to this number of sigfigs
                    mol_prop_formatted = sigfig.round(str(mol_prop), sigfigs=sf)

                    # check how many decimal places Sh has after sigfig rounding
                    if "." in mol_prop_formatted:
                        this_split = mol_prop_formatted.split(".")
                        n_dec = len(this_split[len(this_split)-1])
                    else:
                        n_dec = 0

                    # assign Sh and Sh_err
                    self.__setattr__(prop, float(mol_prop))
                    self.__setattr__(err_str, round(float(mol_err), n_dec))
                    self.__setattr__(prop+"_n_dec", n_dec)
                    self.__setattr__(prop+"_formatted", format(mol_prop, '.'+str(n_dec)+'f'))
                    self.__setattr__(prop+"_err_formatted", format(mol_err, '.'+str(n_dec)+'f'))
                
                else:
                    self.__setattr__(prop, float(mol_prop))
                    self.__setattr__(err_str, float(mol_err))
                    self.__setattr__(prop+"_n_dec", None)
                    self.__setattr__(prop+"_formatted", None)
                    self.__setattr__(prop+"_err_formatted", None)


                continue

            else:
                self.__sum_props(prop, round_sf=round_sf)


    def __sum_props(self, prop, round_sf=True):
            if prop != None:
                if self.__getattribute__(prop) == None:
                    err_str = prop + "_err"

                    # For all properties except for Sh:
                    # initialize variables and lists
                    mol_prop = 0
                    mol_err = 999
                    prop_errs = []
                    n_dec = 999
                    error_groups = []

                    for group in self.groups:

                        try:
                            contains_group = self.group_matches.loc[self.name, group][0] != 0
                        except:
                            contains_group = self.group_matches.loc[self.name, group] != 0

                        # if this molecule contains this group...
                        if contains_group:

                            try:

                                # add number of groups multiplied by its contribution
                                mol_prop += self.group_matches.loc[self.name, group] * float(self.group_data.loc[group, prop])

                                if not math.isnan(float(self.group_data.loc[group, prop])):
                                    if "." in self.group_data.loc[group, prop]:
                                        this_split = self.group_data.loc[group, prop].split(".")
                                        n_dec_group = len(this_split[len(this_split)-1])
                                    else:
                                        n_dec_group = 0
        
                                    if n_dec_group < n_dec:
                                        n_dec = n_dec_group
            
                                    # handle group std errors
                                    try:
                                        float(self.group_data.loc[group, err_str]) # assert that this group's error is numeric
                                        prop_errs.append(self.group_data.loc[group, err_str]) # append error
                                    except:
                                        # if group's error is non-numeric, pass
                                        pass

                            except:
                                error_groups.append(group)

                    if len(error_groups) == 0:

                        # add Y0
                        mol_prop += float(self.group_data.loc["Yo", prop])

                        # propagate error of summed groups: sqrt(a**2 + b**2 + ...)
                        mol_err = round(math.sqrt(sum([float(err)**2 for err in prop_errs])), n_dec)

                        if round_sf:
                            self.__setattr__(prop, round(float(mol_prop), n_dec)) 
                            self.__setattr__(err_str, round(float(mol_err), n_dec))
                            self.__setattr__(prop+"_n_dec", n_dec)
                            self.__setattr__(prop+"_formatted", format(mol_prop, '.'+str(n_dec)+'f'))
                            self.__setattr__(prop+"_err_formatted", format(mol_err, '.'+str(n_dec)+'f'))
                        else:
                            self.__setattr__(prop, float(mol_prop)) 
                            self.__setattr__(err_str, float(mol_err))
                            self.__setattr__(prop+"_n_dec", None)
                            self.__setattr__(prop+"_formatted", None)
                            self.__setattr__(prop+"_err_formatted", None)

                    else:
                        msg = self.name + " encountered errors with group(s): " +\
                            str(error_groups) + ". Are these groups assigned "+\
                            "{} properties in the data file?".format(self.state)
                        self.err_handler.raise_exception(msg)


    def __est_aq(self, props=["Gaq", "Haq", "Saq", "Cpaq", "V"], round_sf=True):

        if self.aq_method == "aqueous":
            # building from aqueous groups only
            for prop in props:
                err_str = prop + "_err"

                # if property is already defined, skip estimating it
                if prop in dir(self):
                    if not self.__getattribute__(prop) is None:
                        continue

                # derive Saq, eaqueous third law entropy, in J/mol K
                if prop == "Saq":
                    # needs sig figs and error propagation
                    mol_prop = ((float(self.Gaq) - float(self.Haq))/-298.15)*1000 + self.Selements

                    # propagate error from Gaq and Haq to estimate Saq error.
                    # equation used: Saq_err = Saq*sqrt((Gaq_err/Gaq)**2 + (Haq_err/Haq)**2)
                    Gaq_err_float = float(self.Gaq_err)/float(self.Gaq)
                    Haq_err_float = float(self.Haq_err)/float(self.Haq)
                    mol_err = abs(mol_prop)*math.sqrt(Gaq_err_float**2 + Haq_err_float**2)

                    if round_sf:
                        # check whether Gaq or Haq as the fewest sigfigs
                        sf = min([find_sigfigs(self.Gaq), find_sigfigs(self.Haq)])

                        # round Saq to this number of sigfigs
                        Saq_formatted = sigfig.round(str(mol_prop), sigfigs=sf)

                        # check how many decimal places Sh has after sigfig rounding
                        if "." in Saq_formatted:
                            this_split = Saq_formatted.split(".")
                            n_dec = len(this_split[len(this_split)-1])
                        else:
                            n_dec = 0

                        # assign Saq_err
                        self.__setattr__("Saq", round(float(mol_prop), n_dec))
                        self.__setattr__("Saq_err", round(float(mol_err), n_dec))
                        self.__setattr__("Saq_n_dec", n_dec)
                        self.__setattr__("Saq_formatted", format(mol_prop, '.'+str(n_dec)+'f'))
                        self.__setattr__("Saq_err_formatted", format(mol_err, '.'+str(n_dec)+'f'))
                    else:
                        self.__setattr__("Saq", float(mol_prop))
                        self.__setattr__("Saq_err", float(mol_err))
                        self.__setattr__("Saq_n_dec", None)
                        self.__setattr__("Saq_formatted", None)
                        self.__setattr__("Saq_err_formatted", None)

                else:
                    self.__sum_props(prop, round_sf=round_sf)
                

        else:
            # Summing hydration and ideal gas properties to get aqueous properties

            if self.Gig != None and self.Gh != None:
                self.Gaq = float(self.Gig) + float(self.Gh)
            else:
                self.Gaq = float("NaN")

            if self.Hig != None and self.Hh != None:
                self.Haq = float(self.Hig) + float(self.Hh)
            else:
                self.Haq = float("NaN")

            try:
                if self.Saq == None:
                    self.Saq = ((float(self.Gaq) - float(self.Haq))/-298.15)*1000 + self.Selements
            except:
                self.Saq = float("NaN")
            
            if self.Cpig != None and self.Cph != None:
                self.Cpaq = self.Cpig + float(self.Cph)
            else:
                self.Cpaq = float("NaN")


        # # calculate HKF parameters
        try:

            if self.Gh is None:
                self.Gh = float("NaN")

            # find_HKF requires calories
            hkf_dict, eq = find_HKF(Gh=float(self.Gh)*1000/4.184,
                                    V=float(self.V),
                                    Cp=float(self.Cpaq)/4.184,
                                    Gf=float(self.Gaq)*1000/4.184,
                                    Hf=float(self.Haq)*1000/4.184,
                                    Saq=float(self.Saq)/4.184,
                                    Z=float(self.charge),
                                    organic=True)

            properties_to_convert = ["G", "H", "S", "Cp", "a1", "a2", "a3", "a4", "c1", "c2", "omega"]
            for k,v in zip(hkf_dict.keys(), hkf_dict.values()):
                if k in properties_to_convert:
                    hkf_dict[k] = v*4.184
                else:
                    hkf_dict[k] = v

            for param in ["a1", "a2", "a3", "a4", "c1", "c2", "omega"]:
                self.__setattr__(param, hkf_dict[param])

        except:
            print("Could not calculate HKF parameters for", self.name)
            pass

    # convert dataframe into an OBIGT table with an option to write to a csv file.
    def __convert_to_OBIGT(self):


        df_prop = {'name':[self.name],
                   'abbrv':[self.formula],
                   'formula':[self.formula],
                   'state':['aq'],
                   'ref1':['AqOrg'],
                   'ref2':['GrpAdd'],
                   'date':[datetime.now().strftime("%d/%m/%Y %H:%M:%S")],
                   'model':['HKF'],
                   'E_units':['J'],
                   'G':[float(self.Gaq)*1000],
                   'H':[float(self.Haq)*1000],
                   'S':[float(self.Saq)],
                   'Cp':[float(self.Cpaq)],
                   'V':[float(self.V)]}
        try:
            # if HKF parameters could be estimated
            df_hkf = {'a1.a':[float(self.a1)],
                      'a2.b':[float(self.a2)],
                      'a3.c':[float(self.a3)],
                      'a4.d':[float(self.a4)],
                      'c1.e':[float(self.c1)],
                      'c2.f':[float(self.c2)],
                      'omega.lambda':[float(self.omega)],
                      'z.T':[self.charge]}
        except:
            df_hkf = {'a1.a':[float("NaN")],
                      'a2.b':[float("NaN")],
                      'a3.c':[float("NaN")],
                      'a4.d':[float("NaN")],
                      'c1.e':[float("NaN")],
                      'c2.f':[float("NaN")],
                      'omega.lambda':[float("NaN")],
                      'z.T':[self.charge]}

        df_prop.update(df_hkf)
        
        df = pd.DataFrame(df_prop)
        
        return df


### This function is a shortcut to the Joback method and should never be used inside of one of the other functions/classes above

def Joback(name, smiles=None, group_data=None):
    """
    Estimate standard state ideal gas properties of a molecule using the Joback
    method. (Joback K. G., Reid R. C., "Estimation of Pure-Component Properties
    from Group-Contributions", Chem. Eng. Commun., 57, 233–243, 1987.)
    
    Parameters
    ----------
    name : str
        Name of the molecule for which to estimate ideal gas properties.

    smiles : str, optional
        A SMILES string representing the molecule.

    group_data : str, optional
        Path of a CSV containing custom Joback group property data.
        
    Returns
    ----------
    dict
        A dictionary containing standard state ideal gas properties estimated
        with the Joback method:
        
        - Gig : Ideal gas Gibbs free energy of formation, kJ/mol.
        - Hig : Ideal gas enthalpy of formation, kJ/mol.
        - Sig : Ideal gas entropy, J/mol/K.
        - Cpig : Ideal gas isobaric heat capacity, J/mol/K.
    """
    ig_est = Estimate(name, smiles=smiles, state='gas', ig_method="Joback", show=False, ig_group_data=group_data)
    
    return {"Gig":ig_est.Gig, "Hig":ig_est.Hig, "Sig":ig_est.Sig, "Cpig":ig_est.Cpig}