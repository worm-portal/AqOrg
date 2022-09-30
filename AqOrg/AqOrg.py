from IPython.display import SVG
from rdkit import Chem
from rdkit.Chem import rdDepictor, Draw
from rdkit.Chem.Draw import rdMolDraw2D
import pandas as pd
import math
import sigfig
import pubchempy as pcp
import os
from chemparse import parse_formula
import pkg_resources
from datetime import datetime

# for benson group additivity
from pgradd.GroupAdd.Library import GroupLibrary
import pgradd.ThermoChem

def find_HKF(Gh=float('NaN'), V=float('NaN'), Cp=float('NaN'),
             Gf=float('NaN'), Hf=float('NaN'), Saq=float('NaN'),
             charge=float('NaN'), J_to_cal=True, print_eq=False):
    
    """
    Estimate HKF parameters from standard state thermodynamic properties of an
    aqueous organic molecule.
    
    Parameters
    ----------
    Gh : numeric
        Standard state partial molal Gibbs free energy of hydration in kJ/mol.
    
    V : numeric
        Standard state partial molal volume in cm3/mol.
    
    Cp : numeric
        Standard state partial molal heat capacity in J/mol/K.
    
    Gf : numeric
        Standard state partial molal Gibbs free energy of formation in kJ/mol.
    
    Hf : numeric
        Standard state partial molal enthalpy of formation in kJ/mol.
    
    Saq : numeric
        Standard state partial molal third law entropy in J/mol/K.
    
    charge : numeric
        The charge of the molecule.
    
    J_to_cal : bool, default True
        Should the output be calorie-based? kJ/mol will be converted to cal/mol
        and J/mol/K will be converted to cal/mol/K.
    
    print_eq : bool, default False
        Print equations used in estimation? Equations are printed in the order
        they are calculated.
        
    Returns
    ----------
    out : dict
        A dictonary of properties and parameters. These will either be
        Joule-based or calorie-based depending on the parameter `J_to_cal`.
    """

    # define eta (angstroms*cal/mol)
    eta = (1.66027*10**5)

    # define YBorn (1/K)
    YBorn = -5.81*10**-5

    # define QBorn (1/bar)
    QBorn = 5.90*10**-7

    # define XBorn (1/K**2)
    XBorn = -3.09*10**-7
    
    if print_eq:
        print("eta = {} (angstroms*cal/mol), YBorn = {} (1/K), QBorn = {} (1/bar), XBorn = {} (1/K**2)\n".format(eta, YBorn, QBorn, XBorn))

    # define abs_protonBorn (cal/mol), mentioned in text after Eq 47 in Shock and Helgeson 1988
    abs_protonBorn = (0.5387 * 10**5)
    if print_eq:
        print("abs_protonBorn = (0.5387 * 10**5), mentioned in text after Eq 47 in Shock and Helgeson 1988\n")


    if not pd.isnull(Gh) and charge == 0:
        if print_eq:
            print("Gh is provided and charge equals zero so estimate omega from Plyasunov and Shock 2001...")

        # find omega*10**-5 (j/mol) if neutral and Gh available
        # Eq 8 in Plyasunov and Shock 2001
        HKFomega = 2.61+(324.1/(Gh-90.6))
        if print_eq:
            print("HKFomega = 2.61+(324.1/(Gh-90.6)), Eq 8 in Plyasunov and Shock 2001, omega*10**-5 (j/mol)\n")

    elif charge == 0:
        if print_eq:
            print("Gh is not provided and charge equals zero so estimate omega for neutral solutes from Shock and Helgeson 1990...")

        # find omega*10**-5 (j/mol) if neutral and Gh unavailable
        # Eq 61 in Shock and Helgeson 1990 for NONVOLATILE neutral organic species
        HKFomega = (10**-5)*((-1514.4*(Saq/4.184)) + (0.34*10**5))*4.184
        if print_eq:
            print("HKFomega = (10**-5)*((-1514.4*(Saq/4.184)) + (0.34*10**5))*4.184, Eq 61 in Shock and Helgeson 1990, omega*10**-5 (j/mol)\n")

    elif charge != 0:
        if print_eq:
            print("Gh is not provided and charge does not equal zero so estimate omega for ionic species from Shock and Helgeson 1990...")
            
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
        if print_eq and alphaZ != float('NaN'):
            print("alphaZ = {} because charge = {}, described in text after Eq 59 in Shock and Helgeson 1990\n".format(alphaZ, charge))
            
        # define BZ
        BZ = ((-alphaZ*eta)/(YBorn*eta - 100)) - charge * \
            abs_protonBorn  # Eq 55 in Shock and Helgeson 1990
        if print_eq:
            print("BZ = ((-alphaZ*eta)/(YBorn*eta - 100)) - charge * abs_protonBorn, Eq 55 in Shock and Helgeson 1990\n")
            
        # find ion omega*10**-5, (J/mol) if charged
        HKFomega = (10**-5)*(-1514.4*(Saq/4.184) + BZ) * \
            4.184  # Eq 58 in Shock and Helgeson 1990
        if print_eq:
            print("HKFomega = (10**-5)*(-1514.4*(Saq/4.184) + BZ) * 4.184, Eq 58 in Shock and Helgeson 1990, omega*10**-5, (J/mol)\n")

        ### METHOD FOR INORGANIC AQUEOUS ELECTROLYTES USING SHOCK AND HELGESON 1988:

        # find rej (angstroms), ions only
        #rej <- ((charge**2)*(eta*YBorn-100))/((Saq/4.184)-71.5*abs(charge)) # Eqs 46+56+57 in Shock and Helgeson 1988

        # find ion absolute omega*10**-5, (cal/mol)
        #HKFomega_abs_ion <- (eta*(charge**2))/rej # Eq 45 in Shock and Helgeson 1988

        # find ion omega*10**-5, (J/mol)
        #HKFomega2 <- (10**-5)*(HKFomega_abs_ion-(charge*abs_protonBorn))*4.184 # Eq 47 in Shock and Helgeson 1988

    else:
        HKFomega = float('NaN')

    # find delta V solvation (cm3/mol)
    # Eq 5 in Shock and Helgeson 1988, along with a conversion of 10 cm3 = 1 joule/bar
    V_solv = -(HKFomega/10**-5)*QBorn*10
    if print_eq:
        print("V_solv = -(HKFomega/10**-5)*QBorn*10, Eq 5 in Shock and Helgeson 1988, along with a conversion of 10 cm3 = 1 joule/bar, delta V solvation (cm3/mol)\n")

    # find delta V nonsolvation (cm3/mol)
    V_nonsolv = V - V_solv  # Eq 4 in Shock and Helgeson 1988
    if print_eq:
        print("V_nonsolv = V - V_solv, Eq 4 in Shock and Helgeson 1988, delta V nonsolvation (cm3/mol)\n")

    # find sigma (cm3/mol)
    HKFsigma = 1.11*V_nonsolv + 1.8  # Eq 87 in Shock and Helgeson
    if print_eq:
        print("HKFsigma = 1.11*V_nonsolv + 1.8, Eq 87 in Shock and Helgeson, sigma (cm3/mol)\n")
        
    # find delta Cp solvation (J/mol*K)
    # Eq 35 in Shock and Helgeson 1988 dCpsolv = omega*T*X
    cp_solv = ((HKFomega/10**-5)*298.15*XBorn)
    if print_eq:
        print("cp_solv = ((HKFomega/10**-5)*298.15*XBorn), Eq 35 in Shock and Helgeson 1988, dCpsolv = omega*T*X, delta Cp solvation (J/mol*K)\n")
        
    # find delta Cp nonsolvation (J/mol*K)
    cp_nonsolv = Cp - cp_solv  # Eq 29 in Shock and Helgeson 1988
    if print_eq:
        print("cp_nonsolv = Cp - cp_solv, Eq 29 in Shock and Helgeson 1988, delta Cp nonsolvation (J/mol*K)\n")
        
    if not pd.isnull(Gh) and charge == 0:
        if print_eq:
            print("Gh is provided and charge is neutral, so estimate a1, a2, and a4 from Plysunov and Shock 2001")
        # find a1*10 (j/mol*bar)
        # Eq 10 in Plyasunov and Shock 2001
        HKFa1 = (0.820-((1.858*10**-3)*(Gh)))*V
        # why is this different than Eq 16 in Sverjensky et al 2014? Regardless, results seem to be very close using this eq vs. Eq 16.
        if print_eq:
            print("HKFa1 = (0.820-((1.858*10**-3)*(Gh)))*V, Eq 10 in Plyasunov and Shock 2001, a1*10 (j/mol*bar)\n")
            
        # find a2*10**-2 (j/mol)
        # Eq 11 in Plyasunov and Shock 2001
        HKFa2 = (0.648+((0.00481)*(Gh)))*V
        if print_eq:
            print("HKFa2 = (0.648+((0.00481)*(Gh)))*V, Eq 11 in Plyasunov and Shock 2001, a2*10**-2 (j/mol)\n")
            
        # find a4*10**-4 (j*K/mol)
        # Eq 12 in Plyasunov and Shock 2001
        HKFa4 = 8.10-(0.746*HKFa2)+(0.219*Gh)
        if print_eq:
            print("HKFa4 = 8.10-(0.746*HKFa2)+(0.219*Gh), Eq 12 in Plyasunov and Shock 2001, a4*10**-4 (j*K/mol)\n")

    else:
        if print_eq:
            print("Gh is unavailable and/or charge is not 0, so estimate a2, a4 from Shock and Helgeson 1988, and a1 from Sverjensky et al 2014")
        
        # find a1*10 (j/mol*bar)
        # Eq 16 in Sverjensky et al 2014, after Plyasunov and Shock 2001, converted to J/mol*bar. This equation is used in the DEW model since it works for charged and noncharged species up to 60kb
        HKFa1 = (0.1942*V_nonsolv + 1.52)*4.184
        if print_eq:
            print("HKFa1 = (0.1942*V_nonsolv + 1.52)*4.184, Eq 16 in Sverjensky et al 2014, after Plyasunov and Shock 2001, converted to J/mol*bar, a1*10 (j/mol*bar)\n")
            
            
        # find a2*10**-2 (j/mol)
        # Eq 8 in Shock and Helgeson 1988, rearranged to solve for a2*10**-2. Sigma is divided by 41.84 due to the conversion of 41.84 cm3 = cal/bar
        HKFa2 = (10**-2)*(((HKFsigma/41.84) -
                        ((HKFa1/10)/4.184))/(1/(2601)))*4.184
        if print_eq:
            print("HKFa2 = (10**-2)*(((HKFsigma/41.84) - ((HKFa1/10)/4.184))/(1/(2601)))*4.184, Eq 8 in Shock and Helgeson 1988, rearranged to solve for a2*10**-2 (j/mol). Sigma is divided by 41.84 due to the conversion of 41.84 cm3 = cal/bar\n")
            
        # find a4*10**-4 (j*K/mol)
        # Eq 88 in Shock and Helgeson 1988, solve for a4*10**-4
        HKFa4 = (10**-4)*(-4.134*(HKFa2/4.184)-27790)*4.184
        if print_eq:
            print("HKFa4 = (10**-4)*(-4.134*(HKFa2/4.184)-27790)*4.184, Eq 88 in Shock and Helgeson 1988, a4*10**-4 (j*K/mol)\n")
            
#     else:
#         HKFa1 = float('NaN')
#         HKFa2 = float('NaN')
#         HKFa3 = float('NaN')
#         HKFa4 = float('NaN')
#         print("HKF parameters a1, a2, a3, and a4 could not be estimated.\n")
           
    # find c2*10**-4 (j*K/mol)
    if not pd.isnull(Gh) and charge == 0:
        HKFc2 = 21.4+(0.849*Gh)  # Eq 14 in Plyasunov and Shock 2001
        if print_eq:
            print("HKFc2 = 21.4+(0.849*Gh), Eq 14 in Plyasunov and Shock 2001, c2*10**-4 (j*K/mol)\n")
#     elif not pd.isnull(Cp) and charge != 0:
    else:
        # Eq 89 in Shock and Helgeson 1988
        HKFc2 = (0.2037*(Cp/4.184) - 3.0346)*4.184
        if print_eq:
            print("HKFc2 = (0.2037*(Cp/4.184) - 3.0346)*4.184, Eq 89 in Shock and Helgeson 1988, c2*10**-4 (j*K/mol)\n")
#     else:
#         HKFc2 = float('NaN')
#         if print_eq:
#             print("HKF parameter c2 could not be estimated.")
            
    # find c1 (j/mol*K)
    # Eq 31 in Shock and Helgeson 1988, rearranged to solve for c1
    HKFc1 = cp_nonsolv-(((HKFc2)/10**-4)*(1/(298.15-228))**2)
    if print_eq:
        print("HKFc1 = cp_nonsolv-(((HKFc2)/10**-4)*(1/(298.15-228))**2), Eq 31 in Shock and Helgeson 1988, rearranged to solve for c1 (j/mol*K)\n")
            
    # find a3 (j*K/mol*bar)
    # Eq 11 in Shock and Helgeson 1988, rearranged to solve for a3. V is divided by 10 due to the conversion of 10 cm3 = J/bar
    HKFa3 = (((V/10)-(HKFa1/10)-((HKFa2/10**-2)/2601) +
            ((HKFomega/10**-5)*QBorn))/(1/(298.15-228)))-((HKFa4/10**-4)/2601)
    if print_eq:
        print("HKFa3 = (((V/10)-(HKFa1/10)-((HKFa2/10**-2)/2601) + ((HKFomega/10**-5)*QBorn))/(1/(298.15-228)))-((HKFa4/10**-4)/2601), Eq 11 in Shock and Helgeson 1988, rearranged to solve for a3 (j*K/mol*bar). V is divided by 10 due to the conversion of 10 cm3 = J/bar\n")
            
    if J_to_cal:
        conv = 4.184
    else:
        conv = 1

    out = {
        "G": (Gf/conv)*1000,
        "H": (Hf/conv)*1000,
        "S": Saq/conv,
        "Cp": Cp/conv,
        "V": V,
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


def find_HKF_test(print_eq=False):
    
    """
    Test the HKF estimation function by regenerating published values.
    
    Parameters
    ----------
    print_eq : bool, default False
        Print equations used in estimation?
    """
    
    #print("\n---------------------------------------------")
    
#     print("phenolate\n---------")
#     print("Input parameters:")
#     print("Gh=-80.74, V=68.16, Cp=105, Gf=5.795, Hf=-129.0, Saq=76.6, charge=-1\n")
#     out = find_HKF(Gh=-80.74, V=68.16, Cp=105,
#                    Gf=5.795, Hf=-129.0, Saq=76.6, charge=-1,
#                    print_eq=print_eq)
    
#     print("phenolate, 1988 method\n---------")
#     print("Input parameters:")
#     print("Gh=float('NaN'), V=68.16, Cp=105, Gf=5.795, Hf=-129.0, Saq=76.6, charge=-1\n")
#     out = find_HKF(Gh=float('NaN'), V=68.16, Cp=105,
#                    Gf=5.795, Hf=-129.0, Saq=76.6, charge=-1,
#                    print_eq=print_eq)

#     print("Be+2\n---------")
#     print("Input parameters:")
#     print("V=-25.4, Cp=-1.3*4.184, Gf=(-83500*4.184)/1000, Hf=(-91500*4.184)/1000, Saq=-55.7*4.184, charge=2\n")
#     out = find_HKF(V=-25.4, Cp=-1.3*4.184, Gf=(-83500*4.184)/1000,
#                    Hf=(-91500*4.184)/1000, Saq=-55.7*4.184, charge=2,
#                    print_eq=print_eq)
    
#     print("NH4+\n---------")
#     print("Input parameters:")
#     print("V=18.13, Cp=15.74*4.184, Gf=(-18990*4.184)/1000, Hf=(-31850*4.184)/1000, Saq=26.57*4.184, charge=1\n")
#     out = find_HKF(V=18.13, Cp=15.74*4.184, Gf=(-18990*4.184)/1000,
#                    Hf=(-31850*4.184)/1000, Saq=26.57*4.184, charge=1,
#                    print_eq=print_eq)
    
#     print("Li+\n---------")
#     print("Input parameters:")
#     print("V=-0.87, Cp=14.2*4.184, Gf=(-69933*4.184)/1000, Hf=(-66552*4.184)/1000, Saq=2.70*4.184, charge=1\n")
#     out = find_HKF(V=-0.87, Cp=14.2*4.184, Gf=(-69933*4.184)/1000,
#                    Hf=(-66552*4.184)/1000, Saq=2.70*4.184, charge=1,
#                    print_eq=print_eq)

    # Compare to table 4 of Plyasunov and Shock 2001
    # (may be slightly different due to using Eq 16 in Sverjensky et al 2014 for calculating a1)
    print("PLYASUNOV AND SHOCK 2001, TABLE 4\n---------------------------------------------")
    
    print("SO2\n---------")
    print("Input parameters:")
    print("Gh=-0.51, V=39.0, Cp=146, charge=0, J_to_cal=False\n")
    out = find_HKF(Gh=-0.51, V=39.0, Cp=146, charge=0, J_to_cal=False, print_eq=print_eq)
    pub = {"omega":"-0.95", "a1":"32.02", "a2":"25.17",
           "a3":"18.71", "a4":"-10.79", "c1":"93.2", "c2":"20.97"}
    print("Published: {}, \tCalculated: {}, \tomega*10**-5".format(pub["omega"], round(out["omega"], 2)))
    print("Published: {}, \tCalculated: {}, \ta1*10".format(pub["a1"], round(out["a1"], 2)))
    print("Published: {}, \tCalculated: {}, \ta2*10**-2".format(pub["a2"], round(out["a2"], 2)))
    print("Published: {}, \tCalculated: {}, \ta3".format(pub["a3"], round(out["a3"], 2)))
    print("Published: {}, \tCalculated: {}, \ta4*10**-4".format(pub["a4"], round(out["a4"], 2)))
    print("Published: {}, \tCalculated: {}, \tc1".format(pub["c1"], round(out["c1"], 1)))
    print("Published: {}, \tCalculated: {}, \tc2*10**-4".format(pub["c2"], round(out["c2"], 2)))
    print("")
    
    print("Pyridine\n---------")
    print("Input parameters:")
    print("Gh=-11.7, V=77.1, Cp=306, charge=0, J_to_cal=False\n")
    out = find_HKF(Gh=-11.7, V=77.1, Cp=306, charge=0, J_to_cal=False, print_eq=print_eq)
    pub = {"omega":"-0.56", "a1":"64.89", "a2":"45.62",
           "a3":"69.94", "a4":"-28.50", "c1":"278.1", "c2":"11.47"}
    print("Published: {}, \tCalculated: {}, \tomega*10**-5".format(pub["omega"], round(out["omega"], 2)))
    print("Published: {}, \tCalculated: {}, \ta1*10".format(pub["a1"], round(out["a1"], 2)))
    print("Published: {}, \tCalculated: {}, \ta2*10**-2".format(pub["a2"], round(out["a2"], 2)))
    print("Published: {}, \tCalculated: {}, \ta3".format(pub["a3"], round(out["a3"], 2)))
    print("Published: {}, \tCalculated: {}, \ta4*10**-4".format(pub["a4"], round(out["a4"], 2)))
    print("Published: {}, \tCalculated: {}, \tc1".format(pub["c1"], round(out["c1"], 1)))
    print("Published: {}, \tCalculated: {}, \tc2*10**-4".format(pub["c2"], round(out["c2"], 2)))
    print("")
    
    print("1,4-Butanediol\n---------")
    print("Input parameters:")
    print("Gh=-37.7, V=88.23, Cp=347, charge=0, J_to_cal=False\n")
    out = find_HKF(Gh=-37.7, V=88.23, Cp=347, charge=0, J_to_cal=False, print_eq=print_eq)
    pub = {"omega":"0.08", "a1":"78.50", "a2":"41.17",
           "a3":"76.32", "a4":"-30.87", "c1":"369.2", "c2":"-10.61"}
    print("Published: {}, \tCalculated: {}, \tomega*10**-5".format(pub["omega"], round(out["omega"], 2)))
    print("Published: {}, \tCalculated: {}, \ta1*10".format(pub["a1"], round(out["a1"], 2)))
    print("Published: {}, \tCalculated: {}, \ta2*10**-2".format(pub["a2"], round(out["a2"], 2)))
    print("Published: {}, \tCalculated: {}, \ta3".format(pub["a3"], round(out["a3"], 2)))
    print("Published: {}, \tCalculated: {}, \ta4*10**-4".format(pub["a4"], round(out["a4"], 2)))
    print("Published: {}, \tCalculated: {}, \tc1".format(pub["c1"], round(out["c1"], 1)))
    print("Published: {}, \tCalculated: {}, \tc2*10**-4".format(pub["c2"], round(out["c2"], 2)))
    print("")
    
    print("beta-alanine\n---------")
    print("Input parameters:")
    print("Gh=-74, V=58.7, Cp=76, charge=0, J_to_cal=False\n")
    out = find_HKF(Gh=-74, V=58.7, Cp=76, charge=0, J_to_cal=False, print_eq=print_eq)
    pub = {"omega":"0.64", "a1":"56.17", "a2":"17.14",
           "a3":"54.55", "a4":"-20.90", "c1":"165.5", "c2":"-41.43"}
    print("Published: {}, \tCalculated: {}, \tomega*10**-5".format(pub["omega"], round(out["omega"], 2)))
    print("Published: {}, \tCalculated: {}, \ta1*10".format(pub["a1"], round(out["a1"], 2)))
    print("Published: {}, \tCalculated: {}, \ta2*10**-2".format(pub["a2"], round(out["a2"], 2)))
    print("Published: {}, \tCalculated: {}, \ta3".format(pub["a3"], round(out["a3"], 2)))
    print("Published: {}, \tCalculated: {}, \ta4*10**-4".format(pub["a4"], round(out["a4"], 2)))
    print("Published: {}, \tCalculated: {}, \tc1".format(pub["c1"], round(out["c1"], 1)))
    print("Published: {}, \tCalculated: {}, \tc2*10**-4".format(pub["c2"], round(out["c2"], 2)))
    print("")

def find_sigfigs(x):
    
    '''
    Get the number of significant digits in a string representing a number up to
    eight digits long.

    Parameters
    ----------
    x : str
        A string denoting a number. This can include scientific notation.
    
    Parameters
    ----------
    int
        The number of significant digits.
    
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
    name : str
        Name of the aqueous organic molecule that will have its thermodynamic
        properties estimated.
    
    ig_method : str, default "Joback"
        Group contribution method for estimating ideal gas properties. Accepts
        "Joback" or "Benson".
                       
    show : bool, default True
        Show a diagram of the molecule?
    
    group_data : str, optional
        Name of a CSV containing custom group contribution data.
    
    test : bool, default False
        Perform a simple group matching test instead of estimating properties?
    
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
    
    """
    
    def __init__(self, name, ig_method="Joback", show=True, group_data=None,
                       test=False, state='aq', **kwargs):
                       # E_units="J" # not implemented... tricky because groups
                                     # are in both kJ and J units.

        self.name = name
        self.ig_method = ig_method
        
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
        
        # load group contribution data
        if group_data == None:
            group_data = pkg_resources.resource_stream(__name__, "data/group_contribution_data.csv")
#         elif '.csv' in group_data:
#             pass
#         else:
#             raise Exception("group_data must be a CSV file.")
        self.__load_group_data(group_data)
        
        # look up compound on PubChem
        self.pcp_compound = pcp.get_compounds(self.name, "name")
        if len(self.pcp_compound) == 0:
            raise Exception("Could not find '" + self.name + "' in PubChem's online database.")
        self.smiles = self.pcp_compound[0].canonical_smiles
        self.formula = self.pcp_compound[0].molecular_formula
        self.formula_dict = parse_formula(self.formula)
        
        if "-" in self.formula_dict.keys() or "+" in self.formula_dict.keys():
            mssg = self.name + " cannot be estimated because it has a net charge."
            raise Exception(mssg)

        if show:
            self.__display_molecule()

        if test:
            print(self.__test_group_match())            
        else:
            # load properties of the elements
            # Cox, J. D., Wagman, D. D., and Medvedev, V. A., CODATA Key Values
            # for Thermodynamics, Hemisphere Publishing Corp., New York, 1989.
            # Compiled into a CSV by Jeffrey Dick for CHNOSZ
            element_data = pd.read_csv(pkg_resources.resource_stream(__name__, 'data/element.csv'), index_col="element")
            self.element_data = element_data.loc[element_data['source'] == "CWM89"]
            
            self.Selements = self.__entropy()
            self.note = ""
            self.charge = 0 # !
            
            if state == 'aq':
                self.OBIGT = self.__estimate()
            elif state == 'gas':
                self.__estimate_joback()

    def __load_group_data(self, db_filename):
        self.group_data = pd.read_csv(db_filename, dtype=str)
        self.group_data['elem'] = self.group_data['elem'].fillna('')
        self.pattern_dict = pd.Series(self.group_data["elem"].values,
                                      index=self.group_data["smarts"]).to_dict()
        self.group_data = self.group_data.set_index("smarts")

        
    def __set_groups(self):
        
        self.group_matches = pd.DataFrame(self.__match_groups(), index=[self.name])

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

    
    def __dict_to_formula(self, formula_dict):
        
        """
        Convert a formula dictionary into a formula string.
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

    
    def __match_groups(self, show=False, save=False):
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
        atomic_info = self.pcp_compound[0].record["atoms"]
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
        test_dict = parse_formula(self.pcp_compound[0].molecular_formula)
        test_dict.update(chargedict)
        if total_formula_dict != test_dict:
            mssg = "The formula of " + self.name + \
                " does not equal the the elemental composition of the " + \
                "matched groups. This could be because the database " + \
                "is missing representative groups.\nFormula of " + \
                self.name + ":\n"
            pcp_dict = parse_formula(self.pcp_compound[0].molecular_formula)
            pcp_dict.update(chargedict)
            mssg = mssg + str(pcp_dict) + "\nTotal formula of group matches:\n" + \
                str(total_formula_dict)
            mssg = mssg + "\nIncomplete group matches:\n" + \
                str({k:v for k,v in zip(match_dict.keys(), match_dict.values()) if v!= 0})
            raise Exception(mssg)
        
        # add molecular formula to match dictionary
        match_dict["formula"] = self.__dict_to_formula(total_formula_dict)
        
        return match_dict

    
    def __display_molecule(self, show=True, save=False):
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
            #Draw.MolToFile( mol, "mol_svg/"+self.name+".svg" )
            Draw.MolToFile(mol_smiles, "mol_png/"+self.name+".png")
    
    def __BensonHSCp(self, print_groups=False):
        this_smile = self.pcp_compound[0].canonical_smiles
        lib = GroupLibrary.Load('BensonGA')
        descriptors = lib.GetDescriptors(this_smile)
        if print_groups:
            print(descriptors)
        thermochem = lib.Estimate(descriptors,'thermochem')
        H = thermochem.get_H(298.15, units="kJ/mol")
        S = thermochem.get_S(298.15, units="J/mol/K")
        Cp = thermochem.get_Cp(298.15, units="J/mol/K")
        return H, S, Cp

    
    def __test_group_match(self):
        match_dict = self.__match_groups()
        return {key:value for key,value in zip(match_dict.keys(), match_dict.values()) if value !=0}

    
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
                else:
                    msg = self.name + " encountered errors with group(s): " +\
                        str(error_groups) + ". Are these groups assigned "+\
                        "ideal gas properties in the Joback data file?"
                    raise Exception(msg)
            
        # calculate Cpig
        T=298.15
        self.Cpig = self.Cpig_a + self.Cpig_b*T + self.Cpig_c*T**2 +\
                    self.Cpig_d*T**3
        
        # calculate Sig
        self.Sig = ((self.Gig - self.Hig)/-298.15)*1000 + self.Selements
    
    
    def __est_calcs(self):

        props = ["Gh", "Hh", "Sh", "Cph", "V"]
        
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
                    raise Exception(msg)
        
        ig_gas_error = False
        if self.Gig != None and self.Hig != None and self.Sig != None and self.Cpig != None:
            # no ideal gas estimation needed.
            # TODO: Modify if statement to allow calculating remainder if
            # two out of three are provided for: Gig, Hig, Sig
            pass
        elif self.ig_method == "Joback":
            # Joback estimation of the Gibbs free energy of formation of the
            # ideal gas (Joule-based).
            try:
                J_estimate = Joback(self.name)
                
                if self.Gig == None:
                    self.Gig = J_estimate["Gig"]
                if self.Hig == None:
                    self.Hig = J_estimate["Hig"]
                if self.Sig == None:
                    self.Sig = ((float(self.Gig) - float(self.Hig))/-298.15)*1000 + self.Selements
                if self.Cpig == None:
                    self.Cpig = J_estimate["Cpig"]
            except:
                ig_gas_error = True

        elif self.ig_method == "Benson":
            # Benson estimation of the Gibbs free energy of formation of
            # the ideal gas (Joule-based).
            try:
                Hig_ben, Sig_ben, Cpig_ben = self.__BensonHSCp()
                if self.Hig == None:
                    self.Hig = Hig_ben
                if self.Sig == None:
                    self.Sig = Sig_ben
                if self.Cpig == None:
                    self.Cpig == Cpig_ben
                delta_Sig = self.Sig - self.Selements
                if self.Gig == None:
                    self.Gig = self.Hig - 298.15*delta_Sig/1000
            except:
                ig_gas_error = True
        else:
            print("Error! The ideal gas property estimation method", self.ig_method, "is not recognized. Try 'Joback' or 'Benson'.")

        if ig_gas_error:
            msg = "The properties of aqueous "+self.name+" could not be " + \
                "estimated because its ideal gas properties could not be " + \
                "estimated with the "+self.ig_method+" method."
            raise Exception(msg)
                
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
            hkf_dict = find_HKF(Gh=float(self.Gh),
                                V=float(self.V),
                                Cp=float(self.Cpaq),
                                Gf=float(self.Gaq),
                                Hf=float(self.Haq),
                                Saq=float(self.Saq),
                                charge=float(self.charge),
                                J_to_cal=False)
            for param in hkf_dict.keys():
                self.__setattr__(param, hkf_dict[param])

        except:
            print("Could not calculate HKF parameters for", self.name)
            pass

    # convert dataframe into an OBIGT table with an option to write to a csv file.
    def __convert_to_OBIGT(self):
        
        df = pd.DataFrame({'name':self.name,
                           'abbrv':self.formula,
                           'formula':self.formula,
                           'state':'aq',
                           'ref1':'AqOrg',
                           'ref2':'GrpAdd',
                           'date':datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                           'E_units':'J',
                           'G':float(self.Gaq)*1000,
                           'H':float(self.Haq)*1000,
                           'S':float(self.Saq),
                           'Cp':float(self.Cpaq),
                           'V':float(self.V),
                           'a1.a':float(self.a1),
                           'a2.b':float(self.a2),
                           'a3.c':float(self.a3),
                           'a4.d':float(self.a4),
                           'c1.e':float(self.c1),
                           'c2.f':float(self.c2),
                           'omega.lambda':float(self.omega),
                           'z.T':self.charge}, index=[0])

        return df

    def __estimate(self):
        self.__set_groups()
        self.__est_calcs()
        return self.__convert_to_OBIGT()

    def __estimate_joback(self):
        self.__set_groups()
        self.__est_joback()

def Joback(name):
    
    """
    Estimate standard state ideal gas properties of a molecule using the Joback
    method. (Joback K. G., Reid R. C., "Estimation of Pure-Component Properties
    from Group-Contributions", Chem. Eng. Commun., 57, 233–243, 1987.)
    
    Parameters
    ----------
    name : str
        Name of the molecule for which to estimate ideal gas properties.
        
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
    
    ig_est = Estimate(name, state='gas', show=False,
                      group_data=pkg_resources.resource_stream(__name__, 'data/joback_groups.csv'), index_col="groups")
    
    return {'Gig':ig_est.Gig, 'Hig':ig_est.Hig,
            'Sig':ig_est.Sig, 'Cpig':ig_est.Cpig}