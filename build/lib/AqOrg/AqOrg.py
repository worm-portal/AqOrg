from IPython.display import SVG
from rdkit import Chem
from rdkit.Chem import rdDepictor, Draw
from rdkit.Chem.Draw import rdMolDraw2D
import pandas as pd
import math
import sigfig
import pubchempy as pcp
import os
import thermo
from chemparse import parse_formula
import pkg_resources
from datetime import datetime

# for benson group additivity
from pgradd.GroupAdd.Library import GroupLibrary
import pgradd.ThermoChem


# function to find number of significant digits. Requires a string.
# modified from a solution from https://stackoverflow.com/questions/8142676/
def find_sigfigs(x):
    '''Returns the number of significant digits in a number. This takes into account
    strings formatted in 1.23e+3 format and even strings such as 123.450

    Example:
    ```find_sigfigs("5.220")```

    '''
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


# a function to round to n sig figs
# solution from https://stackoverflow.com/questions/3410976
def round_to_n(x, n): return x if x == 0 else round(x, -int(math.floor(math.log10(abs(x)))) + (n - 1))
# round_to_n(55.2, 2)


class AqOrganicEstimator():
    """AqOrganicEstimator class calculates thermodynamic properties of aqueous organic molecules."""

    def __init__(self):

        # properties of the elements
        # Cox, J. D., Wagman, D. D., and Medvedev, V. A., CODATA Key Values for Thermodynamics, Hemisphere Publishing Corp., New York, 1989.
        # Compiled into a CSV by Jeffrey Dick for CHNOSZ
        element_data = pd.read_csv(pkg_resources.resource_stream(__name__, 'data/element.csv'), index_col="element")
        self.element_data = element_data.loc[element_data['source'] == "CWM89"]

        self.groups = list() # a list of all groups relevant to this dataset

        self.df_c = pd.DataFrame()

        # load 2nd order group contribution data
        self.load_group_data(pkg_resources.resource_stream(__name__, "data/group_contribution_data.csv"))

    def load_group_data(self, db_filename):
        self.df_gc = pd.read_csv(db_filename, dtype=str)
        self.df_gc['elem'] = self.df_gc['elem'].fillna('')
        self.pattern_dict = pd.Series(self.df_gc["elem"].values, index=self.df_gc["smarts"]).to_dict()
        self.df_gc = self.df_gc.set_index("smarts")

    def set_groups(self, input_name='', output_name=''):

        if ".csv" in input_name:
            df_inp = pd.read_csv(input_name)

        else:
            df_inp = pd.DataFrame({'compound':[input_name],	'test':[0]})

        ## get list of molecules to look up
        molecules = df_inp["compound"]

        # create a df of names and groups
        df = pd.DataFrame()
        vetted_mol = []
        for molecule in molecules:
            try:
                df = df.append(self.match_groups(molecule), ignore_index=True)
                vetted_mol.append(molecule) # matches online
            except:
                print("Error: Could not find", molecule, "in online pubchem database.")

        df.index = vetted_mol
        props = df_inp[[colname for colname in df_inp.columns.values if colname not in ["compound"]]]
        prop_names = df_inp.columns.values[df_inp.columns.values != 'compound']
        props.index = molecules
        self.df_c = pd.concat([props, df], axis=1, sort=False)

        if ".csv" in output_name:
            self.df_c.to_csv(output_name, index_label="compound")

        # remove columns with no matches
        self.df_c = self.df_c.loc[:, (self.df_c.sum(axis=0) != 0)]
        
        # get a list of relevent groups
        self.groups = [group for group in self.df_c.columns if group not in ["formula"]+list(prop_names)]
        

    def entropy(self, name, unit="J/mol/K"):
        """ Calculate the standard molal entropy of elements in a molecule.
        """
        this_compound = pcp.get_compounds(name, "name")
        formula = parse_formula(this_compound[0].molecular_formula)
        entropies = [(self.element_data.loc[elem, "s"]/self.element_data.loc[elem, "n"])*formula[elem] for elem in list(formula.keys())]
        if unit == "J/mol/K":
            unit_conv = 4.184
        elif unit == "cal/mol/K":
            unit_conv = 1
        else:
            print("Error in entropy: specified unit", unit, "is not recognized. Returning entropy in J/mol/K")
            unit_conv = 4.184
            
        return sum(entropies)*unit_conv

    def dict_to_formula(self, formula_dict):
        """
        Converts a formula dictionary into a formula string.
        Example:
        ```dict_to_formula(parse_formula("CO3-2"))```
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

    def match_groups(self, name, show=False, save=False):
        patterns = self.pattern_dict.keys()
        this_compound, this_smile, mol = self.mol_from_smiles(name)

        match_dict = dict(zip(patterns, [0]*len(patterns))) # initialize match_dict
        problem_keys = []
        for pattern in patterns:
            if pattern != "Yo": # never match material point
                try:
                    match_dict[pattern] = len(mol.GetSubstructMatches(Chem.MolFromSmarts(pattern)))
                except:
                        print("Warning in match_groups(): problem identifying SMARTS group", pattern, ". Skipping this group from now on...")
                        problem_keys.append(pattern)

        for key in problem_keys:
            self.pattern_dict.pop(key, None)
            match_dict.pop(key, None)

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
        atomic_info = this_compound[0].record["atoms"]
        chargedict = {}
        if "charge" in atomic_info.keys():
            all_charges = [chargedict.get("value", 0) for chargedict in atomic_info["charge"]]
            pos_charge = sum([charge for charge in all_charges if charge > 0])
            neg_charge = abs(sum([charge for charge in all_charges if charge < 0]))
            if pos_charge > 0:
                chargedict['+']=float(pos_charge)
            if neg_charge > 0:
                chargedict['-']=float(neg_charge)
        else:
            chargedict = {}

        # perform the comparison
        test_dict = parse_formula(this_compound[0].molecular_formula)
        test_dict.update(chargedict)
        if total_formula_dict != test_dict:
            print("Warning! The formula of", name, "does not equal the the elemental composition of the matched groups! This could be because the structure of", name, "is missing representative groups.")
            print("Formula of", name + ":")
            pcp_dict = parse_formula(this_compound[0].molecular_formula)
            pcp_dict.update(chargedict)
            print(pcp_dict)
            print("Total formula of group matches:")
            print(total_formula_dict)
        
        # add molecular formula to match dictionary
        match_dict["formula"] = self.dict_to_formula(total_formula_dict)
        
        ### create a png and svg of the molecule
        self.display_molecule(name, mol_smiles=mol, show=show, save=save)
        return match_dict

    def mol_from_smiles(self, name):
        this_compound = pcp.get_compounds(name, "name")
        this_smile = this_compound[0].canonical_smiles
        mol = Chem.MolFromSmiles(this_smile)
        return this_compound, this_smile, mol

    def display_molecule(self, name, mol_smiles=None, show=True, save=False):
        if mol_smiles == None:
            this_compound, this_smile, mol_smiles = self.mol_from_smiles(name)
        
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
            #Draw.MolToFile( mol, "mol_svg/"+name+".svg" )
            Draw.MolToFile(mol_smiles, "mol_png/"+name+".png")

    def get_smarts(self, name):
        this_compound = pcp.get_compounds(name, "name")
        this_smile = this_compound[0].canonical_smiles
        return Chem.MolToSmarts(Chem.MolFromSmiles(this_smile))

    def get_smiles(self, name):
        this_compound = pcp.get_compounds(name, "name")
        this_smile = this_compound[0].canonical_smiles
        return Chem.MolFromSmiles(this_smile)

        # functions for benson group additivity

    def BensonG(self, name, print_groups=False):
        this_compound = pcp.get_compounds(name, "name")
        this_smile = this_compound[0].canonical_smiles
        lib = GroupLibrary.Load('BensonGA')
        descriptors = lib.GetDescriptors(this_smile)
        if print_groups:
            print(descriptors)
        thermochem = lib.Estimate(descriptors,'thermochem')
        return thermochem.get_G(298.15, units="kJ/mol")

    def BensonHSCp(self, name, print_groups=False):
        this_compound = pcp.get_compounds(name, "name")
        this_smile = this_compound[0].canonical_smiles
        lib = GroupLibrary.Load('BensonGA')
        descriptors = lib.GetDescriptors(this_smile)
        if print_groups:
            print(descriptors)
        thermochem = lib.Estimate(descriptors,'thermochem')
        return thermochem.get_H(298.15, units="kJ/mol"), thermochem.get_S(298.15, units="J/mol/K"), thermochem.get_Cp(298.15, units="J/mol/K")

    def test_group_match(self, molecule, show=True, save=False):
        # test group matching and error messages
        try:
            match_dict = self.match_groups(molecule, show=show, save=save)
            return {key:value for key,value in zip(match_dict.keys(), match_dict.values()) if value !=0}
        except:
            print("Could not find group matches. Is the molecule name '{}' spelled correctly?".format(molecule))

    # create a dataframe to store estimated properties of molecules
    def create_df_est(self, csv_out_name='', ig_method="Joback"):
        """
        ig_method: String. Accepts "Joback" or "Benson". Group contribution method for ideal gas properties. 
        """
        self.df_est = pd.DataFrame(index=self.df_c.index)

        hkf_params = ["a1", "a2", "a3", "a4", "c1", "c2", "omega"]
        props = ["Gh", "Hh", "Sh", "Cph", "V"]

        for prop in props:
            # add column for property and error
            self.df_est[prop] = ""
            self.df_est[prop+"_err"] = ""   

        for param in hkf_params:
            # add column for parameter
            self.df_est[param] = ""

        for prop in ["Gig", "Hig", "Sig", "Cpig", "Gaq", "Haq", "Saq", "Cpaq"]:
            self.df_est[param] = ""
            
        self.df_est["note"] = ""
            
        for molecule in self.df_c.index:
            
            try:
                formula = self.df_c.loc[molecule, "formula"]
            except:
                print("Error:", molecule, "has no formula.")
                continue
            
            for prop in props:
                
                err_str = prop + "_err"
                
                # derive Sh, entropy of hydration, in J/mol K
                if prop == "Sh":
                    try:
                        # Entropy calculated from S = (G-H)/(-Tref)
                        mol_prop = (float(self.df_est.loc[molecule, "Gh"]) - float(self.df_est.loc[molecule, "Hh"]))/(-298.15)
                        mol_prop = mol_prop*1000 # convert kJ/molK to J/molK
                        
                        # propagate error from Gh and Hh to estimate Sh error.
                        # equation used: Sh_err = Sh*sqrt((Gh_err/Gh)^2 + (Hh_err/Hh)^2)
                        Gh_err = float(self.df_est.loc[molecule, "Gh_err"])/float(self.df_est.loc[molecule, "Gh"])
                        Hh_err = float(self.df_est.loc[molecule, "Hh_err"])/float(self.df_est.loc[molecule, "Hh"])
                        mol_err = abs(mol_prop)*math.sqrt(Gh_err**2 + Hh_err**2)
                        
                        # check whether Gh or Hh as the fewest sigfigs
                        sf = min([find_sigfigs(self.df_est.loc[molecule, p]) for p in ["Gh", "Hh"]])
                        
                        # round Sh to this number of sigfigs
                        mol_prop = sigfig.round(str(mol_prop), sigfigs=sf)
                        
                        # check how many decimal places Sh has after sigfig rounding
                        if "." in mol_prop:
                            this_split = mol_prop.split(".")
                            n_dec = len(this_split[len(this_split)-1])
                        else:
                            n_dec = 0
                        
                        # assign Sh and Sh_err to dataframe
                        self.df_est.at[molecule, prop] = mol_prop
                        self.df_est.at[molecule, err_str] = format(mol_err, '.'+str(n_dec)+'f')
                        
                    except:
                        pass
                    
                    continue
                    
                else:
                    pass
                
                # initialize variables and lists
                mol_prop = 0
                mol_err = 999
                prop_errs = []
                n_dec = 999
                error_groups = []

                for group in self.groups:
                    
                    try:
                        contains_group = self.df_c.loc[molecule, group][0] != 0
                    except:
                        contains_group = self.df_c.loc[molecule, group] != 0

                    # if this molecule contains this group...
                    if contains_group:

                        try:
                            # add number of groups multiplied by its contribution
                            mol_prop += self.df_c.loc[molecule, group] * float(self.df_gc.loc[group, prop])

                            # round property to smallest number of decimal places
                            if "." in self.df_gc.loc[group, prop]:
                                this_split = self.df_gc.loc[group, prop].split(".")
                                n_dec_group = len(this_split[len(this_split)-1])
                            else:
                                n_dec_group = 0
                                
                            if n_dec_group < n_dec:
                                n_dec = n_dec_group
                                
                            # handle group std errors
                            try:
                                float(self.df_gc.loc[group, err_str]) # assert that this group's error is numeric
                                prop_errs.append(self.df_gc.loc[group, err_str]) # append error
                            except:
                                # if group's error is non-numeric, pass
                                pass

                        except:
                            error_groups.append(group)

                if len(error_groups) == 0:

                    # add Y0
                    mol_prop += float(self.df_gc.loc["Yo", prop])

                    # propagate error of summed groups: sqrt(a**2 + b**2 + ...)
                    mol_err = round(math.sqrt(sum([float(err)**2 for err in prop_errs])), n_dec) # propagate error from groups
                    
                    # format output
                    mol_prop = format(mol_prop, '.'+str(n_dec)+'f')
                    mol_err = format(mol_err, '.'+str(n_dec)+'f')
                    
                    #print(molecule, "\t\t", mol_prop, u'\u00b1', mol_err)
                    self.df_est.at[molecule, prop] = mol_prop
                    self.df_est.at[molecule, err_str] = mol_err

                else:
                    message1 = molecule + " encountered errors with group(s): " + str(error_groups) + "."
                    message2 = "Are these groups assigned properties in the data file? ;"
                    self.df_est.at[molecule, "note"] = self.df_est.loc[molecule, "note"] + message1 + " " + message2
                    print(message1)
                    print(message2)
            
            try:
                Selements = self.entropy(molecule)
            except:
                print("Error! Could not calculate entropy from the elements of", molecule)
                Selements = float("NaN")
            
            if ig_method == "Joback":
                # Joback estimation of the Gibbs free energy of formation of the ideal gas (Joule-based)
                try:
                    mol = self.get_smiles(molecule)
                    J = thermo.Joback(mol) 
                    Gig = J.estimate()['Gf']/1000
                    Hig = J.estimate()['Hf']/1000
                    Sig = ((Gig - Hig)/-298.15)*1000 + Selements
                    Cpig = J.estimate()['Cpig'](T=298.15)
                except:
                    Gig = float("NaN")
                    Hig = float("NaN")
                    Sig = float("NaN")
                    Cpig = float("NaN")
            
            elif ig_method == "Benson":
                # Benson estimation of the Gibbs free energy of formation of the ideal gas (Joule-based)
                try:
                    Hig, Sig, Cpig = self.BensonHSCp(molecule)
                    delta_Sig = Sig - Selements
                    Gig = Hig - 298.15*delta_Sig/1000
                except:
                    Gig = float("NaN")
                    Hig = float("NaN")
                    Sig = float("NaN")
                    Cpig = float("NaN")
            else:
                print("Error! The ideal gas property estimation method", ig_method, "is not recognized. Try 'Joback' or 'Benson'.")
            
            # estimate the Gibbs free energy of formation of the aqueous molecule by summing
            # its ideal gas and hydration properties.
            try:
                # TODO: if ideal gas properties are NaN, ensure aqueous properties are too.
                Gaq = Gig + float(self.df_est.loc[molecule, "Gh"])
                Haq = Hig + float(self.df_est.loc[molecule, "Hh"])
                Saq = ((Gaq - Haq)/-298.15)*1000 + Selements
                Cpaq = Cpig + float(self.df_est.loc[molecule, "Cph"])
                
            except:
                Gaq = float("NaN")
                Haq = float("NaN")
                Saq = float("NaN")
                Cpaq = float("NaN")
                
            self.df_est.at[molecule, "Gig"] = Gig
            self.df_est.at[molecule, "Hig"] = Hig
            self.df_est.at[molecule, "Sig"] = Sig
            self.df_est.at[molecule, "Cpig"] = Cpig
            self.df_est.at[molecule, "Gaq"] = Gaq
            self.df_est.at[molecule, "Haq"] = Haq
            self.df_est.at[molecule, "Saq"] = Saq
            self.df_est.at[molecule, "Cpaq"] = Cpaq
            self.df_est.at[molecule, "formula"] = formula

            # calculate HKF parameters
            try:
                hkf_dict = self.find_HKF(Gh=float(self.df_est.loc[molecule, "Gh"]),
                                Vh=float(self.df_est.loc[molecule, "V"]),
                                Cp=Cpaq, Gf=Gaq, Hf=Haq,
                                Saq=Saq, charge=0, J_to_cal=False)
                for param in hkf_params:
                    self.df_est.at[molecule, param] = hkf_dict[param]
            except:
                print("Could not calculate HKF parameters for", molecule)
                pass
        
        if '.csv' in csv_out_name:
            self.df_est.to_csv(csv_out_name)


    # convert dataframe into an OBIGT table with an option to write to a csv file.
    def convert_to_OBIGT(self, filename=''):
        name = list(self.df_est.index)

        try:
            abbrv = list(self.df_est["formula"])
        except:
            print("Error: could not return an OBIGT entry because one or more chemical formulas are missing.")
            return

        formula = list(self.df_est["formula"])
        state = ["aq"]*len(self.df_est.index)
        ref1 = ["AqOrg"]*len(self.df_est.index)
        ref2 = ["GrpAdd"]*len(self.df_est.index)
        date = [datetime.now().strftime("%d/%m/%Y %H:%M:%S")]*len(self.df_est.index)
        E_units = ["J"]*len(self.df_est.index)
        G = [float(value)*1000 for value in list(self.df_est["Gaq"])]
        H = [float(value)*1000 for value in list(self.df_est["Haq"])]
        S = list(self.df_est["Saq"])
        Cp = list(self.df_est["Cpaq"])
        V = list(self.df_est["V"])
        a1 = list(self.df_est["a1"])
        a2 = list(self.df_est["a2"])
        a3 = list(self.df_est["a3"])
        a4 = list(self.df_est["a4"])
        c1 = list(self.df_est["c1"])
        c2 = list(self.df_est["c2"])
        omega = list(self.df_est["omega"])
        Z = ["0"]*len(self.df_est.index) # !

        obigt_out = pd.DataFrame(zip(name, abbrv, formula,
                                    state, ref1, ref2,
                                    date, E_units, G,
                                    H, S, Cp, V, a1, a2,
                                    a3, a4, c1, c2,
                                    omega, Z),
                                columns =['name', 'abbrv', 'formula',
                                        'state', 'ref1', 'ref2',
                                        'date', 'E_units', 'G',
                                        'H', 'S', 'Cp', 'V', 'a1.a', 'a2.b',
                                        'a3.c', 'a4.d', 'c1.e', 'c2.f',
                                        'omega.lambda', 'z.T'])


        obigt_out = obigt_out.dropna() # remove any rows with 'NaN'

        if '.csv' in filename:
            obigt_out.to_csv(filename, index=False)

        return obigt_out


    def find_HKF(self, Gh=float('NaN'), Vh=float('NaN'), Cp=float('NaN'),
                Gf=float('NaN'), Hf=float('NaN'), Saq=float('NaN'),
                charge=float('NaN'), J_to_cal=True):

        # define eta (angstroms*cal/mol)
        eta = (1.66027*10**5)

        # define YBorn (1/K)
        YBorn = -5.81*10**-5

        # define QBorn (1/bar)
        QBorn = 5.90*10**-7

        # define XBorn (1/K^2)
        XBorn = -3.09*10**-7

        # define abs_protonBorn (cal/mol), mentioned in text after Eq 47 in Shock and Helgeson 1988
        abs_protonBorn = (0.5387 * 10**5)

        if not pd.isnull(Gh) and charge == 0:

            # find omega*10^-5 (j/mol) if neutral and Gh available
            # Eq 8 in Plyasunov and Shock 2001
            HKFomega = 2.61+(324.1/(Gh-90.6))

        elif charge == 0:

            # find omega*10^-5 (j/mol) if neutral and Gh unavailable
            # Eq 61 in Shock and Helgeson 1990 for NONVOLATILE neutral organic species
            HKFomega = (10 ^ -5)*((-1514.4*(Saq/4.184)) + (0.34*10**5))*4.184

        elif charge != 0:

            # define alphaZ (described in text after Eq 59 in Shock and Helgeson 1990)
            if (abs(charge) == 1):
                alphaZ = 72
            elif (abs(charge) == 2):
                alphaZ = 141
            elif (abs(charge) == 3):
                alphaZ = 211
            elif (abs(charge) == 4):
                alphaZ = 286
            else:
                alphaZ = float('NaN')

            # define BZ
            BZ = ((-alphaZ*eta)/(YBorn*eta - 100)) - charge * \
                abs_protonBorn  # Eq 55 in Shock and Helgeson 1990

            # find ion omega*10^-5, (J/mol) if charged
            HKFomega = (10 ^ -5)*(-1514.4*(Saq/4.184) + BZ) * \
                4.184  # Eq 58 in Shock and Helgeson 1990

            ### METHOD FOR INORGANIC AQUEOUS ELECTROLYTES USING SHOCK AND HELGESON 1988:

            # find rej (angstroms), ions only
            #rej <- ((charge^2)*(eta*YBorn-100))/((Saq/4.184)-71.5*abs(charge)) # Eqs 46+56+57 in Shock and Helgeson 1988

            # find ion absolute omega*10^-5, (cal/mol)
            #HKFomega_abs_ion <- (eta*(charge^2))/rej # Eq 45 in Shock and Helgeson 1988

            # find ion omega*10^-5, (J/mol)
            #HKFomega2 <- (10^-5)*(HKFomega_abs_ion-(charge*abs_protonBorn))*4.184 # Eq 47 in Shock and Helgeson 1988

        else:
            HKFomega = float('NaN')

        # find delta V solvation (cm3/mol)
        # Eq 5 in Shock and Helgeson 1988, along with a conversion of 10 cm3 = 1 joule/bar
        V_solv = -(HKFomega/10**-5)*QBorn*10

        # find delta V nonsolvation (cm3/mol)
        V_nonsolv = Vh - V_solv  # Eq 4 in Shock and Helgeson 1988

        # find sigma (cm3/mol)
        HKFsigma = 1.11*V_nonsolv + 1.8  # Eq 87 in Shock and Helgeson

        # find delta cp solvation (J/mol*K)
        # Eq 35 in Shock and Helgeson 1988 dCpsolv = omega*T*X
        cp_solv = ((HKFomega/10**-5)*298.15*XBorn)

        # find delta cp nonsolvation (J/mol*K)
        cp_nonsolv = Cp - cp_solv  # Eq 29 in Shock and Helgeson 1988

        if not pd.isnull(Gh) and charge == 0:
            # find a1*10 (j/mol*bar)
            # Eq 10 in Plyasunov and Shock 2001
            HKFa1 = (0.820-((1.858*10**-3)*(Gh)))*Vh
            # why is this different than Eq 16 in Sverjensky et al 2014? Regardless, results seem to be very close using this eq vs. Eq 16.

            # find a2*10^-2 (j/mol)
            # Eq 11 in Plyasunov and Shock 2001
            HKFa2 = (0.648+((0.00481)*(Gh)))*Vh

            # find a4*10^-4 (j*K/mol)
            # Eq 12 in Plyasunov and Shock 2001
            HKFa4 = 8.10-(0.746*HKFa2)+(0.219*Gh)

        elif charge != 0:
            # find a1*10 (j/mol*bar)
            # Eq 16 in Sverjensky et al 2014, after Plyasunov and Shock 2001, converted to J/mol*bar. This equation is used in the DEW model since it works for charged and noncharged species up to 60kb
            HKFa1 = (0.1942*V_nonsolv + 1.52)*4.184

            # find a2*10^-2 (j/mol)
            # Eq 8 in Shock and Helgeson, rearranged to solve for a2*10^-2. Sigma is divided by 41.84 due to the conversion of 41.84 cm3 = cal/bar
            HKFa2 = (10**-2)*(((HKFsigma/41.84) -
                            ((HKFa1/10)/4.184))/(1/(2601)))*4.184

            # find a4*10^-4 (j*K/mol)
            # Eq 88 in Shock and Helgeson, solve for a4*10^-4
            HKFa4 = (10**-4)*(-4.134*(HKFa2/4.184)-27790)*4.184

        else:
            HKFa1 = float('NaN')
            HKFa2 = float('NaN')
            HKFa3 = float('NaN')

        # find c2*10^-4 (j*K/mol)
        if not pd.isnull(Gh) and charge == 0:
            HKFc2 = 21.4+(0.849*Gh)  # Eq 14 in Plyasunov and Shock 2001
        elif not pd.isnull(Cp) and charge != 0:
            # Eq 89 in Shock and Helgeson 1988
            HKFc2 = (0.2037*(Cp/4.184) - 3.0346)*4.184
        else:
            HKFc2 = float('NaN')

        # find c1 (j/mol*K)
        # Eq 31 in Shock and Helgeson 1988, rearranged to solve for c1
        HKFc1 = cp_nonsolv-(((HKFc2)/10**-4)*(1/(298.15-228))**2)

        # find a3 (j*K/mol*bar)
        # Eq 11 in Shock and Helgeson 1988, rearranged to solve for a3. Vh is divided by 10 due to the conversion of 10 cm3 = J/bar
        HKFa3 = (((Vh/10)-(HKFa1/10)-((HKFa2/10**-2)/2601) +
                ((HKFomega/10**-5)*QBorn))/(1/(298.15-228)))-((HKFa4/10**-4)/2601)

        if J_to_cal:
            conv = 4.184
        else:
            conv = 1

        # report results in calorie scale, ready to be pasted into OBIGT
        out = {
            "G": (Gf/conv)*1000,
            "H": (Hf/conv)*1000,
            "S": Saq/conv,
            "Cp": Cp/conv,
            "V": Vh,
            "a1": HKFa1/conv,
            "a2": HKFa2/conv,
            "a3": HKFa3/conv,
            "a4": HKFa4/conv,
            "c1": HKFc1/conv,
            "c2": HKFc2/conv,
            "omega": HKFomega/conv,
            "Z": charge,
            "Vsolv": V_solv,
            "Vnonsolv": V_nonsolv,
            "sigma": HKFsigma}

        return out

    def find_HKF_test(self):
        print("phenolate", self.find_HKF(Gh=-80.74, Vh=68.16, Cp=105, Gf=5.795, Hf=-129.0, Saq=76.6, charge=-1))
        print("phenolate, 1988 method", self.find_HKF(Gh=float('NaN'), Vh=68.16, Cp=105, Gf=5.795, Hf=-129.0, Saq=76.6, charge=-1))

        print("Be+2", self.find_HKF(Vh=-25.4, Cp=-1.3*4.184, Gf=(-83500*4.184) /
                                    1000, Hf=(-91500*4.184)/1000, Saq=-55.7*4.184, charge=2))
        print("NH4+", self.find_HKF(Vh=18.13, Cp=15.74*4.184, Gf=(-18990*4.184) /
                                    1000, Hf=(-31850*4.184)/1000, Saq=26.57*4.184, charge=1))
        print("Li+", self.find_HKF(Vh=-0.87, Cp=14.2*4.184, Gf=(-69933*4.184) /
                                   1000, Hf=(-66552*4.184)/1000, Saq=2.70*4.184, charge=1))

        # Compare to table 4 of Plyasunov and Shock 2001
        # (may be slightly different due to using Eq 16 in Sverjensky et al 2014 for calculating a1)
        print("SO2", self.find_HKF(Gh=-0.51, Vh=39.0,
                                   Cp=146, charge=0, J_to_cal=False))
        print("Pyridine", self.find_HKF(Gh=-11.7, Vh=77.1,
                                        Cp=306, charge=0, J_to_cal=False))
        print("1,4-Butanediol", self.find_HKF(Gh=-37.7, Vh=88.23,
                                              Cp=347, charge=0, J_to_cal=False))
        print("beta-alanine", self.find_HKF(Gh=-74, Vh=58.7,
                                            Cp=76, charge=0, J_to_cal=False))
    
    def estimate(self, input_name='', output_name='', csv_out_name='', ig_method="Joback", OBIGT_filename='', show=True):

        if show and '.csv' not in input_name:
            self.display_molecule(name=input_name)

        self.set_groups(input_name, output_name)
        self.create_df_est(csv_out_name, ig_method=ig_method)
        return self.convert_to_OBIGT(filename=OBIGT_filename)

