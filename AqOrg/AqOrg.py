from IPython.display import SVG
from rdkit import Chem
from rdkit.Chem import rdDepictor, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
import pandas as pd
import math
import sigfig
import pubchempy as pcp
import os
from chemparse import parse_formula
from datetime import datetime
from WORMutils import Error_Handler, find_HKF, import_package_file

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
    
    group_data : str, optional
        Path of a CSV containing custom group contribution data. If `state="aq"`
        then the CSV should contain hydration properties. If `state="gas"` then
        the CSV should contain ideal gas properties.
    
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
    
    def __init__(self, name=None, smiles=None, show=True, group_data=None,
                       test=False, state='aq', ig_method="Joback",
                       save=False, hide_traceback=True, **kwargs):
                       # E_units="J" # not implemented... tricky because groups
                                     # are in both kJ and J units.

        self.err_handler = Error_Handler(clean=hide_traceback)
        
        self.name = name
        self.smiles = smiles
        self.state = state
        
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

        for key, value in kwargs.items():
            self.__setattr__(key, value)
        
        self.group_data = group_data
        self.load_group_data()
        self.get_mol_smiles_formula_formula_dict()
        
        if "-" in self.formula_dict.keys() or "+" in self.formula_dict.keys():
            self.err_handler.raise_exception(self.name + " cannot be estimated because it has a net charge.")

        if show:
            self.display_molecule(save=save)
            
        if test:
            print(self.__test_group_match())            
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
            self.charge = 0 # !
            
            if state == 'aq':
                self.OBIGT = self.__estimate_hydration()
            elif state == 'gas':
                if ig_method == "Joback":
                    self.__estimate_joback()
                else:
                    self.__estimate_ig()
            else:
                self.err_handler.raise_exception("State must be 'aq' or 'gas'.")

    def get_mol_smiles_formula_formula_dict(self):
        if not isinstance(self.smiles, str):
            # look up compound on PubChem
            self.pcp_compound = pcp.get_compounds(self.name, "name")
            if len(self.pcp_compound) == 0:
                self.err_handler.raise_exception("Could not find '" + self.name + "' in PubChem's online database.")
            self.smiles = self.pcp_compound[0].canonical_smiles
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
                with import_package_file(__name__, 'data/group_contribution_data.csv', as_file=True) as path:
                    self.group_data = pd.read_csv(path, dtype=str)
            elif self.state == "gas":
                if self.ig_method == "Joback":
                    with import_package_file(__name__, 'data/joback_groups.csv', as_file=True) as path:
                        self.group_data = pd.read_csv(path, dtype=str)
            else:
                self.err_handler.raise_exception("State is unrecognized. Must be either 'aq' or 'gas'.")
        
        self.group_data['elem'] = self.group_data['elem'].fillna('')
        self.pattern_dict = pd.Series(self.group_data["elem"].values,
                                      index=self.group_data["smarts"]).to_dict()
        self.group_data = self.group_data.set_index("smarts")

        
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
                "matched groups. This could be because the database " + \
                "is missing representative groups.\nFormula of " + \
                self.name + ":\n"
            pcp_dict = parse_formula(self.formula)
            pcp_dict.update(chargedict)
            mssg = mssg + str(pcp_dict) + "\nTotal formula of group matches:\n" + \
                str(total_formula_dict)
            mssg = mssg + "\nIncomplete group matches:\n" + \
                str({k:v for k,v in zip(match_dict.keys(), match_dict.values()) if v!= 0})
            self.err_handler.raise_exception(mssg)
        
        # add molecular formula to match dictionary
        match_dict["formula"] = self.dict_to_formula(total_formula_dict)
        
        return match_dict


    def display_molecule(self, show=True, save=False):
        """
        Display a molecule in a Jupyter notebook or save it as an SVG and PNG.
        This function is meant to be used internally by `Estimate`.
        """
        mol_smiles = Chem.MolFromSmiles(self.smiles)
        
        mc = Chem.Mol(mol_smiles.ToBinary())
        molSize=(450, 150)
        
        if not mc.GetNumConformers():
            #Compute 2D coordinates
            rdDepictor.Compute2DCoords(mc)
        # init the drawer with the size
        drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0],molSize[1])
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
        return {key:value for key,value in zip(match_dict.keys(), match_dict.values()) if value !=0}
        

    def __est_ig(self):
        props = ["Gig", "Hig", "Cpig"]
        
        for prop in props:
            err_str = prop + "_err"

            # if property is already defined, skip estimating it
            if prop in dir(self):
                if not self.__getattribute__(prop) is None:
                    continue
            
            mol_prop = 0
            error_groups = []
            mol_err = 999
            prop_errs = []
            n_dec = 999

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
    
                # add Y0
                mol_prop += float(self.group_data.loc["Yo", prop])
    
                # propagate error of summed groups: sqrt(a**2 + b**2 + ...)
                mol_err = round(math.sqrt(sum([float(err)**2 for err in prop_errs])), n_dec)
    
                # # format output as string (preserves trailing zeros)
                # mol_prop = format(mol_prop, '.'+str(n_dec)+'f')
                # mol_err = format(mol_err, '.'+str(n_dec)+'f')

                self.__setattr__(prop, mol_prop)
                self.__setattr__(err_str, mol_err)

        if len(error_groups) > 0:
            msg = self.name + " encountered errors with group(s): " +\
                str(error_groups) + ". Are these groups assigned "+\
                "ideal gas properties in the data file?"
            self.err_handler.raise_exception(msg)
                    
        # calculate Sig
        self.Sig = ((self.Gig - self.Hig)/-298.15)*1000 + self.Selements
        
    
    def __est_joback(self):
        
        # values to be added to final estimate of each property
        joback_props = {"Gig":53.88, "Hig":68.29, # kJ/mol
                        "Cpig_a":-37.93, "Cpig_b":0.210, # j/mol/K
                        "Cpig_c":-3.91*10**-4, "Cpig_d":2.06*10**-7} # j/mol/K
        
        for prop in joback_props.keys():
            mol_prop = 0
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
                    except:
                        error_groups.append(group)
                        
                if len(error_groups) == 0:
                    self.__setattr__(prop, mol_prop+joback_props[prop])

        if len(error_groups) > 0:
            self.err_handler.raise_exception("" + self.name + " encountered errors with group(s): "
                ""+str(error_groups) + ". Are these groups assigned "
                "ideal gas properties in the Joback data file?")
            
        # calculate Cpig
        T=298.15
        self.Cpig = self.Cpig_a + self.Cpig_b*T + self.Cpig_c*T**2 +\
                    self.Cpig_d*T**3
        
        # calculate Sig
        self.Sig = ((self.Gig - self.Hig)/-298.15)*1000 + self.Selements
    
    
    def __est_hydration(self, props=["Gh", "Hh", "Sh", "Cph", "V"]):

        for prop in props:
            if self.__getattribute__(prop) == None:
                err_str = prop + "_err"

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

                    # check whether Gh or Hh as the fewest sigfigs
                    sf = min([find_sigfigs(self.Gh), find_sigfigs(self.Hh)])

                    # round Sh to this number of sigfigs
                    mol_prop = sigfig.round(str(mol_prop), sigfigs=sf)

                    # check how many decimal places Sh has after sigfig rounding
                    if "." in mol_prop:
                        this_split = mol_prop.split(".")
                        n_dec = len(this_split[len(this_split)-1])
                    else:
                        n_dec = 0

                    # assign Sh and Sh_err
                    #self.__setattr__(prop, mol_prop) # for trailing zeros, but must store Sh as str.
                    #self.__setattr__(err_str, format(mol_err, '.'+str(n_dec)+'f')) # for trailing zeros, but must store Sh_err as str.
                    self.__setattr__(prop, float(mol_prop))
                    self.__setattr__(err_str, round(float(mol_err), n_dec))


                    continue

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
    
                            # round property to smallest number of decimal places
    
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

#                     # format output as string (preserves trailing zeros)
#                     mol_prop = format(mol_prop, '.'+str(n_dec)+'f')
#                     mol_err = format(mol_err, '.'+str(n_dec)+'f')

                    self.__setattr__(prop, mol_prop)
                    self.__setattr__(err_str, mol_err)

                else:
                    msg = self.name + " encountered errors with group(s): " +\
                        str(error_groups) + ". Are these groups assigned "+\
                        "hydration properties in the data file?"
                    self.err_handler.raise_exception(msg)
        
        if self.Gig != None and self.Hig != None and self.Sig != None and self.Cpig != None:
            # no ideal gas estimation needed.
            # TODO: Modify if statement to allow calculating remainder if
            # two out of three are provided for: Gig, Hig, Sig
            pass
        else:
            # Joback estimation of the Gibbs free energy of formation of the
            # ideal gas (Joule-based).
            # try:
            J_estimate = Joback(self.name, smiles=self.smiles)
            
            if self.Gig == None:
                self.Gig = J_estimate["Gig"]
            if self.Hig == None:
                self.Hig = J_estimate["Hig"]
            if self.Sig == None:
                self.Sig = ((float(self.Gig) - float(self.Hig))/-298.15)*1000 + self.Selements
            if self.Cpig == None:
                self.Cpig = J_estimate["Cpig"]
            # except:
            #     if isinstance(self.name, str):
            #         report_name = self.name
            #     elif isinstance(self.smiles, str):
            #         report_name = self.smiles
            #     else:
            #         report_name = "ERROR"
            #     self.err_handler.raise_exception("The properties of aqueous '"+report_name+"' could not be "
            #         "estimated because its ideal gas properties could not be "
            #         "estimated with the Joback method.")
                
        # estimate the Gibbs free energy of formation of the aqueous molecule by summing
        # its ideal gas and hydration properties.
        # TODO: if ideal gas properties are NaN, ensure aqueous properties are too.
        # TODO: determine estimation error of ideal gas, then propagate with hydration errors.
        # TODO: propagate errors into HKF parameter estimations.
        
        try:
            if self.Gaq == None:
                self.Gaq = float(self.Gig) + float(self.Gh)
        except:
            self.Gaq = float("NaN")

        try:
            if self.Haq == None:
                self.Haq = float(self.Hig) + float(self.Hh)
        except:
            self.Haq = float("NaN")

        try:
            if self.Saq == None:
                self.Saq = ((float(self.Gaq) - float(self.Haq))/-298.15)*1000 + self.Selements
        except:
            self.Saq = float("NaN")
        try:
            if self.Cpaq == None:
                self.Cpaq = self.Cpig + float(self.Cph)
        except:
            self.Cpaq = float("NaN")

        # calculate HKF parameters
        try:
            # find_HKF requires calories
            hkf_dict, eq = find_HKF(Gh=float(self.Gh)*1000/4.184,
                                    V=float(self.V),
                                    Cp=float(self.Cpaq)/4.184,
                                    Gf=float(self.Gaq)*1000/4.184,
                                    Hf=float(self.Haq)*1000/4.184,
                                    Saq=float(self.Saq)/4.184,
                                    Z=float(self.charge))

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

    def __estimate_hydration(self):
        self.__set_groups()
        self.__est_hydration()
        return self.__convert_to_OBIGT()

    def __estimate_joback(self):
        self.__set_groups()
        self.__est_joback()

    def __estimate_ig(self):
        self.__set_groups()
        self.__est_ig()


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
    if group_data is None:
        with import_package_file(__name__, 'data/joback_groups.csv', as_file=True) as path:
            group_data = pd.read_csv(path, dtype=str)
    
    ig_est = Estimate(name, smiles=smiles, state='gas', ig_method="Joback", show=False,
                      group_data=group_data, index_col="groups")
    
    return {'Gig':ig_est.Gig, 'Hig':ig_est.Hig,
            'Sig':ig_est.Sig, 'Cpig':ig_est.Cpig}