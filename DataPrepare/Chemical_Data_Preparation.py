import numpy as np
import os
import sys
import shutil
import math
from pandas import pandas as pd
import pickle
from enum import Enum

#For metal atoms, we may need to make its atomic number effect smaller than regular using a fixed number or natural log or square root
MetalAtomicNumber = Enum('MetalAtomicNumber', 'Regular Fixed Log Sqrt')
#Out of the many runs for each species on each metal surface, whether to take all of them, take the minimum for each or
#take the values close to a reference value (supplied in a csv file).
EnergyPicker = Enum('EnergyPicker','All Minimum CloseToReference')

#This class encapsulates the calculation of various features/descriptors for adsorption energy prediction
#Features/descriptors include Coulomb matrix (CM), Bag-of-bonds (BoB), atom-wise Fingerprints for subnet etc
#These features are calculated from the files (such as POSCAR or CONTCAR) that contain the coordinates for each atom
#of a species on a metal surface.
class ChemDataPrepare():
	#maxAtoms: maximum number of each type of atom in the species
	#E_H, E_O, E_C: gas phase energies for hydrogen, oxygen and carbon, respectively
	#CleanEnergiesForMetals: surface energies for each of the metal surface
	#numberOfMetalAtomsInEachLayer: number of atoms in each metal surface layer
	#excludeMetal: boolean. if true, metal atoms are not included in CM/BoB calculation. Otherwise they are included.
	#metalAtomicNumType: enumeration MetalAtomicNumber.
	#fixedMetalAtomicNumbers: works if metalAtomicNumType is 'Fixed'. Dictionary with keys as metals and values as corresponding fixed atomic number (Z)
	#isVerbose: if true, extra details are printed
	#usePoscars: if true, use POSCARs as the geometric coordinate files. Otherwise use CONTCARs.
    def __init__(self, maxAtoms, E_H, E_O, E_C, 
                 CleanEnergiesForMetals, numberOfMetalAtomsInEachLayer, excludeMetal,
                 metalAtomicNumType, fixedMetalAtomicNumbers=None, isVerbose = False, usePoscars = False):
        if type(maxAtoms) is not dict:
            raise ValueError('Passed argument maxAtoms is not a dictionary but it should be!')
        self.atomMaxDict = maxAtoms
        self.maxMoleculeSize = maxAtoms['C'] + maxAtoms['H'] + maxAtoms['O']
        self.metalAtomicNumType = metalAtomicNumType
        self.fixedMetalAtomicNumbers = fixedMetalAtomicNumbers
        self.E_H = E_H
        self.E_O = E_O
        self.E_C = E_C
        self.cleanEnergies = CleanEnergiesForMetals
        self.numberOfMetalAtomsInEachLayer = numberOfMetalAtomsInEachLayer
        self.excludeMetal = excludeMetal
        self.isVerbose = isVerbose
        self.usePoscars = usePoscars
        
        
    def getAtomicNumberForMetal(self, atom, regularAtomicNumber):
        if self.metalAtomicNumType == MetalAtomicNumber.Regular:
            return regularAtomicNumber
        elif self.metalAtomicNumType == MetalAtomicNumber.Fixed:
            return fixedMetalAtomicNumbers[atom]
        elif self.metalAtomicNumType == MetalAtomicNumber.Log:
            return np.log(regularAtomicNumber)
        elif self.metalAtomicNumType == MetalAtomicNumber.Sqrt:
            return np.sqrt(regularAtomicNumber)
        else:
            raise ValueError('MetalAtomicNumberType Enum was passed wrong value!')
            
        
    def NuclearChargeForAtom(self, atom):
        latom = atom.lower()
        ret = 0
        if latom == 'c':
            ret = 6
        elif latom == 'h':
            ret = 1
        elif latom == 'o':
            ret = 8
        elif latom == 'x':
            ret = 0   
        elif latom == 'pd':
            ret = self.getAtomicNumberForMetal(atom, 46)
        elif latom == 'pt':
            ret = self.getAtomicNumberForMetal(atom, 78)
        elif latom == 're':
            ret = self.getAtomicNumberForMetal(atom, 75)
        elif latom == 'rh':
            ret = self.getAtomicNumberForMetal(atom, 45)
        elif latom == 'ru':
            ret = self.getAtomicNumberForMetal(atom, 44)
        elif latom == 'ag':
            ret = self.getAtomicNumberForMetal(atom, 47)
        elif latom == 'ni':
            ret = self.getAtomicNumberForMetal(atom, 28)
        elif latom == 'cu':
            ret = self.getAtomicNumberForMetal(atom, 29)
        else:
            print(atom)
            raise ValueError('Wrong atom type passed!')
        return ret
    
    
    def AtomicDistance(self, atom1, atom2):
        return np.sqrt((atom1[1] - atom2[1])**2 + (atom1[2] - atom2[2])**2 + (atom1[3] - atom2[3])**2)
    

    def calculateReferencedEnergy(self, energy, metal, currentDirectory):
        c, h, o = 0, 0, 0
        f = open(currentDirectory + '\\' + 'CONTCAR', "r")
        lines = f.readlines()
        atomdict = dict(zip(lines[5].split(), list(map(int, lines[6].split()))))
        if 'C' in atomdict:
            c = atomdict['C']
        if 'H' in atomdict:
            h = atomdict['H']
        if 'O' in atomdict:
            o = atomdict['O']
        return energy - (c * self.E_C) - (h * self.E_H) - (o * self.E_O) - self.cleanEnergies[metal]
    
    
    def BuildCoulombMatrixFromStruct(self, struct, r_power = 1, useCutoff = False, cutoff_dist_angstrom = 4.0):
        dim = struct.shape[0]
        cmat = np.empty([dim,dim])
        for i in range(dim):
            for j in range(dim):
                if i==j:
                    cmat[i,j] = 0.5*(self.NuclearChargeForAtom(struct.iloc[i,0]))**(2.4)
                else:
                    r = self.AtomicDistance(struct.iloc[i,:], struct.iloc[j,:])
                    if useCutoff == False:
                        cmat[i,j] = ((self.NuclearChargeForAtom(struct.iloc[i,0]) * \
                                      self.NuclearChargeForAtom(struct.iloc[j,0]))/(np.power(r, r_power)))
                    else:
                        if r > cutoff_dist_angstrom:# and (i_metal == True or j_metal == True):
                            cmat[i,j] = 0.0
                        else:
                            cmat[i,j] = ((self.NuclearChargeForAtom(struct.iloc[i,0]) * \
                                          self.NuclearChargeForAtom(struct.iloc[j,0]))/(np.power(r, r_power))) - \
                                        ((self.NuclearChargeForAtom(struct.iloc[i,0]) * \
                                          self.NuclearChargeForAtom(struct.iloc[j,0]))/(np.power(cutoff_dist_angstrom,r_power)))
                if math.isnan(cmat[i,j]) == True:
                    cmat[i,j] = 0.0
        eigvals, v = np.linalg.eig(cmat)
        eigvals = np.sort(eigvals) 
        return eigvals
    
    
    
    def BuildBagOfBondsFromStruct(self, struct, boblen, maxCountInAGroup = -1, 
                                  r_power = 1, useCutoff = False, cutoff_dist_angstrom = 4.0):
        dim = struct.shape[0]
        bobdf = pd.DataFrame(index=range(0,boblen), columns=['bondtype','value'])
        i = 1
        k = 0
        while i<dim:
            j = 0
            while j<i:
                a = struct.iloc[i,0] + struct.iloc[j,0]
                b = ''.join(sorted(a))
                if b!='WW' and b!='HH' and b!='HW':
                    bobdf.iloc[k,0] = b
                    if struct.iloc[i, 4] == 'pad' or struct.iloc[j, 4] == 'pad':
                        bobdf.iloc[k, 1] = 0.0
                    else:
                        r = self.AtomicDistance(struct.iloc[i,:], struct.iloc[j,:])
                        if useCutoff == False:
                            bobdf.iloc[k,1] = ((self.NuclearChargeForAtom(struct.iloc[i,5]) * \
                                                self.NuclearChargeForAtom(struct.iloc[j,5]))/(np.power(r, r_power)))
                        else:
                            if r > cutoff_dist_angstrom:# and (struct.iloc[i,0] == 'W' or struct.iloc[j,0]) == 'W':
                                bobdf.iloc[k,1] = 0.0
                            else:
                                bobdf.iloc[k,1] = ((self.NuclearChargeForAtom(struct.iloc[i,5]) * \
                                                    self.NuclearChargeForAtom(struct.iloc[j,5]))/(np.power(r, r_power))) - \
                                                  ((self.NuclearChargeForAtom(struct.iloc[i,5]) * \
                                                    self.NuclearChargeForAtom(struct.iloc[j,5]))/ \
                                                   (np.power(cutoff_dist_angstrom, r_power)))
                    k = k + 1
                j = j + 1

            i = i + 1
        bobdfsorted = bobdf.sort_values(by=['bondtype','value'], ascending=[True,False])
        if maxCountInAGroup > 0:
            bobdfsorted = bobdfsorted.groupby('bondtype').head(maxCountInAGroup).reset_index(drop=True)
        return bobdfsorted

        
    def GetBoBStruct(self, struct, cho_order, atomdict):
        structbob = pd.DataFrame(index=range(0, struct.shape[0]), columns=range(0,6))
        if 'C' not in atomdict:
            cho_order.append('C')
            atomdict['C'] = 0
        if 'H' not in atomdict:
            cho_order.append('H')
            atomdict['H'] = 0
        if 'O' not in atomdict:
            cho_order.append('O')
            atomdict['O'] = 0
        row = 0
        j = 0
        for cho in cho_order:
            if cho == 'C' or cho == 'H' or cho == 'O':
                for i in range(atomdict[cho]):
                    structbob.iloc[row, :] = struct.iloc[j, 0], struct.iloc[j, 1], struct.iloc[j, 2], struct.iloc[j, 3], \
                                             'reg', struct.iloc[j, 0]
                    row = row + 1
                    j = j + 1
                for i in range(self.atomMaxDict[cho] - atomdict[cho]):
                    structbob.iloc[row, :] = cho, 0.0, 0.0, 0.0, 'pad', cho
                    row = row + 1
            elif self.excludeMetal == False:
                for i in range(self.numberOfMetalAtomsInEachLayer*2):
                    structbob.iloc[row, :] = 'W', struct.iloc[j, 1], struct.iloc[j, 2], struct.iloc[j, 3], 'reg', \
                                             struct.iloc[j, 0]
                    row = row + 1
                    j = j + 1
        structbob_sorted = structbob.sort_values(by=[0,4], ascending=[True,False])        
        return structbob_sorted
    
    
    def f_c_of_Rij(self, Rij, cutoff_dist_angstrom):
        if Rij > cutoff_dist_angstrom:
            return 0
        return 0.5 * (np.cos(np.pi * Rij / cutoff_dist_angstrom) + 1)
    
    
    def getFP_PairwiseDistance(self, struct, i, dim, cutoff_dist_angstrom):
        fp = 0.0
        for j in range(dim):
            if j == i or struct.iloc[j,4] == 'pad':
                continue
            Rij = self.AtomicDistance(struct.iloc[i,:], struct.iloc[j,:])
            fp += np.exp(-Rij**2) * self.f_c_of_Rij(Rij, cutoff_dist_angstrom)
        return fp
    
    
    def getFP_TripleDistance(self, struct, i, dim, cutoff_dist_angstrom):
        fp = 0.0
        for j in range(dim):
            if j == i or struct.iloc[j,4] == 'pad':
                continue
            for k in range(dim):
                if k == i or k == j or struct.iloc[k,4] == 'pad':
                    continue
                Rij = self.AtomicDistance(struct.iloc[i,:], struct.iloc[j,:])
                Rik = self.AtomicDistance(struct.iloc[i,:], struct.iloc[k,:])
                Rjk = self.AtomicDistance(struct.iloc[j,:], struct.iloc[k,:])
                RijDotRik = (struct.iloc[i,1]-struct.iloc[j,1])*(struct.iloc[i,1]-struct.iloc[k,1]) + \
                            (struct.iloc[i,2]-struct.iloc[j,2])*(struct.iloc[i,2]-struct.iloc[k,2]) + \
                            (struct.iloc[i,3]-struct.iloc[j,3])*(struct.iloc[i,3]-struct.iloc[k,3])
                theta_ijk = RijDotRik/(Rij * Rik)
                fp += (1+theta_ijk) * np.exp(-((Rij**2)+(Rik**2)+(Rjk**2))) * self.f_c_of_Rij(Rij, cutoff_dist_angstrom) * \
                                    self.f_c_of_Rij(Rik, cutoff_dist_angstrom) * self.f_c_of_Rij(Rjk, cutoff_dist_angstrom)
        return fp
    
    
    def getFP_FourSeparatePairwiseDistance(self, struct, i, dim, cutoff_dist_angstrom):
        fpc = fph = fpo = fpm = 0.0
        for j in range(dim):
            if j == i or struct.iloc[j,4] == 'pad':
                continue
            otheratom = struct.iloc[j,0].upper()
            Rij = self.AtomicDistance(struct.iloc[i,:], struct.iloc[j,:])
            val = np.exp(-Rij**2) * self.f_c_of_Rij(Rij, cutoff_dist_angstrom)
            if otheratom == 'C':
                fpc += val
            elif otheratom == 'H':
                fph += val
            elif otheratom == 'O':
                fpo += val
            else:
                fpm += val
        return fpc, fph, fpo, fpm
    
    
    def BuildTwoFingerprintsForSubnet(self, struct, cutoff_dist_angstrom):
        dim = struct.shape[0]
        retvec = np.zeros([self.maxMoleculeSize * 2])
        
        for i in range(dim):
            curratom = struct.iloc[i,0].upper()
            if curratom != 'C' and curratom != 'H' and curratom != 'O':
                continue
                
            fp1 = fp2 = 0.0
            if struct.iloc[i,4] == 'reg':
                fp1 = self.getFP_PairwiseDistance(struct, i, dim, cutoff_dist_angstrom)            
                fp2 = self.getFP_TripleDistance(struct, i, dim, cutoff_dist_angstrom)
                
            retvec[i*2] = fp1
            retvec[i*2 + 1] = fp2
            
        return retvec
    
    
    def BuildFourFingerprintsForSubnet(self, struct, cutoff_dist_angstrom):
        dim = struct.shape[0]
        retvec = np.zeros([self.maxMoleculeSize * 4])
        
        for i in range(dim):
            curratom = struct.iloc[i,0].upper()
            if curratom != 'C' and curratom != 'H' and curratom != 'O':
                continue
                
            fpc = fph = fpo = fpm = 0.0
            if struct.iloc[i,4] == 'reg':
                fpc, fph, fpo, fpm = self.getFP_FourSeparatePairwiseDistance(struct, i, dim, cutoff_dist_angstrom)
                
            retvec[i*4] = fpc
            retvec[i*4 + 1] = fph
            retvec[i*4 + 2] = fpo
            retvec[i*4 + 3] = fpm
            
        return retvec
    
    
    def BuildFiveFingerprintsForSubnet(self, struct, cutoff_dist_angstrom):
        dim = struct.shape[0]
        retvec = np.zeros([self.maxMoleculeSize * 5])
        
        for i in range(dim):
            curratom = struct.iloc[i,0].upper()
            if curratom != 'C' and curratom != 'H' and curratom != 'O':
                continue
                
            fpc = fph = fpo = fpm = fptriple = 0.0
            if struct.iloc[i,4] == 'reg':
                fpc, fph, fpo, fpm = self.getFP_FourSeparatePairwiseDistance(struct, i, dim, cutoff_dist_angstrom)
                fptriple = self.getFP_TripleDistance(struct, i, dim, cutoff_dist_angstrom)                
                                    
            retvec[i*4] = fpc
            retvec[i*4 + 1] = fph
            retvec[i*4 + 2] = fpo
            retvec[i*4 + 3] = fpm
            retvec[i*4 + 4] = fptriple
            
        return retvec
    
    #This function takes structural info for all species on all surfaces, calculate the speciefied number of fingerprints
	#for each and stores the fingerprints in a csv file
    def EncodeUsingSubnetFingerprints(self, structFile, outFileName, num_of_fingerprints = 2, cutoff_dist_angstrom = 4.0):
        f = open(structFile, 'rb')
        allStructs = pickle.load(f)
        f.close()
        structs = allStructs['structs']
        cho_orders = allStructs['cho_orders']
        atomdicts = allStructs['atomdicts']
        df = pd.DataFrame(index=range(0, len(structs)), columns=range(0, self.maxMoleculeSize * num_of_fingerprints))
        df_row = 0
        for struct in structs:
            bobStruct = self.GetBoBStruct(struct, cho_orders[df_row], atomdicts[df_row])
			if num_of_fingerprints == 2:
				df.iloc[df_row,:] = self.BuildTwoFingerprintsForSubnet(bobStruct, cutoff_dist_angstrom)
			elif num_of_fingerprints == 4:
				df.iloc[df_row,:] = self.BuildFourFingerprintsForSubnet(bobStruct, cutoff_dist_angstrom)
			else:
				df.iloc[df_row,:] = self.BuildFiveFingerprintsForSubnet(bobStruct, cutoff_dist_angstrom)
            df_row += 1
            if self.isVerbose == True:
                print(df_row)
        df.to_csv(outFileName)

    #This function takes structural info for all species on all surfaces, calculate the Coulomb matrix for each
	#and store them in a csv file
    def EncodeUsingCM(self, structFile, outFileName, r_power = 1, useCutoff = False, cutoff_dist_angstrom = 4.0,
                     setValForMetalAtomicNumType = None):
        f = open(structFile, 'rb')
        allStructs = pickle.load(f)
        f.close()
        if setValForMetalAtomicNumType is not None:
            self.metalAtomicNumType = setValForMetalAtomicNumType
        structs = allStructs['structs']
        metals = allStructs['metal']
        species = allStructs['species']
        metalDescs = allStructs['metalDescs']
        dim = self.maxMoleculeSize + (self.numberOfMetalAtomsInEachLayer*2)
        df = pd.DataFrame(index=range(0, len(structs)), columns=range(0, dim + 2 + metalDescs.shape[1]))
        df_row = 0
        for struct in structs:
            cmvec = self.BuildCoulombMatrixFromStruct(struct, r_power, useCutoff, cutoff_dist_angstrom)
            df.iloc[df_row, 0] = metals[df_row]
            df.iloc[df_row, 1] = species[df_row]
            for i in range(metalDescs.shape[1]):
                df.iloc[df_row, i+2] = metalDescs[df_row, i]
            l = 2 + metalDescs.shape[1]
            for k in range(dim-1, -1, -1):
                df.iloc[df_row, l] = cmvec[k]
                l += 1
            df_row += 1
            if df_row % 100 == 0 and self.isVerbose == True:
                print(df_row)
        df.to_csv(outFileName)
    
    
    def lenForBond(self, bndAtom1, bndAtom2, maxAtomsInAGroup):
        count = bndAtom1*bndAtom2
        if count > maxAtomsInAGroup:
            return maxAtomsInAGroup
        return count
        
    def getBoBLenForTopAtomsInGroup(self, maxAtomsInAGroup):
        cc = int((self.atomMaxDict['C']-1)*self.atomMaxDict['C']/2)
        if cc > maxAtomsInAGroup:
            cc = maxAtomsInAGroup
        oo = int((self.atomMaxDict['O']-1)*self.atomMaxDict['O']/2)
        if oo > maxAtomsInAGroup:
            oo = maxAtomsInAGroup
        ch = self.lenForBond(self.atomMaxDict['C'], self.atomMaxDict['H'], maxAtomsInAGroup)
        co = self.lenForBond(self.atomMaxDict['C'], self.atomMaxDict['O'], maxAtomsInAGroup)
        oh = self.lenForBond(self.atomMaxDict['O'], self.atomMaxDict['H'], maxAtomsInAGroup)
        cm = self.lenForBond(self.atomMaxDict['C'], self.atomMaxDict['M'], maxAtomsInAGroup)
        om = self.lenForBond(self.atomMaxDict['O'], self.atomMaxDict['M'], maxAtomsInAGroup)
        return cc + oo + co + ch + oh + cm + om
        
    #This function takes structural info for all species on all surfaces, calculate the Bag-of-bonds for each
	#and store them in a csv file    
    def EncodeUsingBoB(self, structFile, outFileName, maxCountInAGroup = -1,
                       r_power = 1, useCutoff = False, cutoff_dist_angstrom = 4.0, setValForMetalAtomicNumType = None):
        f = open(structFile, 'rb')
        allStructs = pickle.load(f)
        f.close()
        if setValForMetalAtomicNumType is not None:
            self.metalAtomicNumType = setValForMetalAtomicNumType
        structs = allStructs['structs']
        cho_orders = allStructs['cho_orders']
        atomdicts = allStructs['atomdicts']
        metals = allStructs['metal']
        species = allStructs['species']
        metalDescs = allStructs['metalDescs']
        
        numOfMetalAtomsInStruct = self.numberOfMetalAtomsInEachLayer * 2
        numOfHydrogenInStruct = self.atomMaxDict['H']
        dim = self.maxMoleculeSize + numOfMetalAtomsInStruct
        
        boblen = boblen_reduced = 0
        if maxCountInAGroup > 0:
            boblen_reduced = self.getBoBLenForTopAtomsInGroup(maxCountInAGroup)
            
        if self.excludeMetal == True: # since there is no metal atom, nC2 combinations and deduct H-H
            boblen = int((dim-1)*dim/2) - int((numOfHydrogenInStruct-1)*numOfHydrogenInStruct/2)
        else: # first take all nC2 combinations then deduct the metal-metal, H-H and H-M combinations
            boblen = int((dim-1)*dim/2) - int((numOfMetalAtomsInStruct-1)*numOfMetalAtomsInStruct/2) \
                     - int((numOfHydrogenInStruct-1)*numOfHydrogenInStruct/2) - numOfMetalAtomsInStruct*numOfHydrogenInStruct
        
        if maxCountInAGroup <= 0:
            boblen_reduced = boblen
        df = pd.DataFrame(index=range(0, len(structs)), columns=range(0, boblen_reduced + 2 + metalDescs.shape[1]))
        df_row = 0
        for struct in structs:
            bobStruct = self.GetBoBStruct(struct, cho_orders[df_row], atomdicts[df_row])
            bobvec = self.BuildBagOfBondsFromStruct(bobStruct, boblen, maxCountInAGroup, 
                                                    r_power, useCutoff, cutoff_dist_angstrom)
            df.iloc[df_row, 0] = metals[df_row]
            df.iloc[df_row, 1] = species[df_row]
            for i in range(metalDescs.shape[1]):
                df.iloc[df_row, i+2] = metalDescs[df_row, i]
            for i in range(boblen_reduced):
                df.iloc[df_row, i+2+metalDescs.shape[1]] = bobvec.iloc[i,1]
            df_row += 1
            if df_row % 100 == 0 and self.isVerbose == True:
                print(df_row)
        df.to_csv(outFileName)
            
        
    def ReadStructureFile(self, fpath):
        fname = None
        if self.usePoscars == True:
            fname = 'POSCAR'
        else:
            fname = 'CONTCAR'
        f = open(fpath + '\\' + fname, "r")
        lines = f.readlines()
        atomdict = dict(zip(lines[5].split(), list(map(int, lines[6].split()))))
        atomorder = lines[5].split()
        dim = self.maxMoleculeSize + (self.numberOfMetalAtomsInEachLayer*2)
        struct = pd.DataFrame(index=range(0, dim), columns=['atom','x','y','z'])
        cho_order = []
        currline = 2
        mat = np.zeros((3,3))
        for i in range(3):
            mat[i,:] = list(map(float, lines[currline].split()))
            currline = currline + 1
        currline = 9
        i = 0
        for atom in atomorder:
            linesToRead = atomdict[atom]            
            if atom == 'C' or atom == 'H' or atom == 'O':   
                cho_order.append(atom)
                for line in range(linesToRead):
                    words = lines[currline].split()
                    v = np.array([float(k) for k in words[:3]])
                    w = np.dot(mat,v)
                    struct.iloc[i, 0:4] = atom, w[0], w[1], w[2] 
                    i = i + 1
                    currline = currline + 1
            elif self.excludeMetal == True:
                currline = currline + linesToRead
            elif self.excludeMetal == False:
                cho_order.append(atom)
                metalCoords = pd.DataFrame(index=range(0, self.atomMaxDict['M']), columns=range(0,4))
                for line in range(linesToRead):
                    words = lines[currline].split()
                    v = np.array([float(k) for k in words[:3]])
                    w = np.dot(mat,v)
                    metalCoords.iloc[line, 0:4] = atom, w[0], w[1], w[2]
                    currline = currline + 1
                p = metalCoords.sort_values(by=[3], ascending=False)
                for j in range(self.numberOfMetalAtomsInEachLayer*2):
                    struct.iloc[i, 0:4] = atom, p.iloc[j,1], p.iloc[j,2], p.iloc[j,3]
                    i = i + 1
                
        while i < dim:
            struct.iloc[i, :] = 'X', 0.0, 0.0, 0.0
            i = i + 1
        return struct, cho_order, atomdict

        
    #This function reads all input directories, find appropriate folders containing calculation for each species on each metal surface,
	#read energies and coordinates from files in that folder, saves energies and structural info in output files.
	#param list:
	#rootdirectories: dictionary with metals as keys and corresponding input directories as values
	#metals: list of metal names
	#metalDescriptors: dictionary with metals as keys and list metal descriptors as values
	#datasize: int. Total number of data points
	#typeOfRun: enumeration EnergyPicker. Defines how data will be read for each species
	#closestEnergyFileName: works if typeOfRun = EnergyPicker.CloseToReference. The reference energies are contained in this file.
	#structFileName: name of the output pickle file that should hold the serialized structural info for each data points
	#energyFileName: name of the output csv file containing the energies for each species on each surface
	#speciesList: list of species names.
    def PrepareStructs(self, rootdirectories, metals, metalDescriptors, datasize, typeOfRun, 
                       closestEnergyFileName, structFileName, energyFileName, speciesList=None):
        structs = []
        cho_orders = []
        atomdicts = []
        metalInfos = []
        speciesInfos = []
        metalDescs = np.empty([datasize, len(metalDescriptors[list(metalDescriptors.keys())[0]])])
        energy_csv = pd.DataFrame(index = range(datasize), columns=['path','metal','species','energy'])
        closestEnergyDF = None
        if isinstance(closestEnergyFileName, str) and len(closestEnergyFileName) > 0:
            closestEnergyDF = pd.read_csv(closestEnergyFileName, skiprows=0, index_col=0)
        rownum = 0
        for metal in metals:
            rootdirectory = rootdirectories[metal]
            for subname in os.listdir(rootdirectory):
                currDir = os.path.join(rootdirectory, subname)
                if not os.path.isdir(currDir) or (speciesList is not None and 
                                                  (subname not in speciesList and subname.upper() not in speciesList)):
                    continue
                min_energy = 1.1
                min_energy_path = ''     
                closest_energy_gap = 500.1
                closest_energy = 1.1
                closest_energy_path = ''
                energyFromExcel = -1
                if closestEnergyDF is not None:
                    energyFromExcel = closestEnergyDF.loc[(closestEnergyDF['metal']==metal) & 
                                                      ((closestEnergyDF['species']==subname) | 
                                                       (closestEnergyDF['species']==subname.upper()))].iloc[0,:]['energy']
                for root, subdirs, files in os.walk(currDir):
                    if 'OSZICAR' not in files or '331' in subdirs:
                        continue
                    if self.isVerbose == True:
                        print(root)
                    f = open(root + '\\' + 'OSZICAR', "r")
                    lines = f.readlines()
                    currline_no = len(lines) - 1
                    while len(lines[currline_no].split()) == 0:
                        currline_no = currline_no - 1
                    words = lines[currline_no].split()
                    energy_file = float(words[words.index('E0=')+1])
                    if typeOfRun == EnergyPicker.All:
                        energy = self.calculateReferencedEnergy(energy_file, metal, root)
                        struct, cho_order, atomdict = self.ReadStructureFile(root)                    
                        energy_csv.iloc[rownum,:] = root, metal, subname, energy
                        structs.append(struct) 
                        cho_orders.append(cho_order)
                        atomdicts.append(atomdict)
                        metalInfos.append(metal)
                        speciesInfos.append(subname)
                        metalDescs[rownum,:] = metalDescriptors[metal]
                        rownum += 1
                    elif typeOfRun == EnergyPicker.Minimum:
                        if energy_file < min_energy: 
                            min_energy = energy_file
                            min_energy_path = root
                    elif typeOfRun == EnergyPicker.CloseToReference:
                        if abs(energyFromExcel - energy_file) < closest_energy_gap:
                            closest_energy_gap = abs(energyFromExcel - energy_file)
                            closest_energy = energy_file
                            closest_energy_path = root
                if typeOfRun != EnergyPicker.All:
                    energy = None
                    energyPath = None
                    if typeOfRun == EnergyPicker.Minimum:
                        energy = min_energy
                        energyPath = min_energy_path
                    elif typeOfRun == EnergyPicker.CloseToReference:
                        energy = closest_energy
                        energyPath = closest_energy_path
                    energy = self.calculateReferencedEnergy(energy, metal, energyPath)
                    struct, cho_order, atomdict = self.ReadStructureFile(energyPath)                    
                    energy_csv.iloc[rownum,:] = energyPath, metal, subname, energy
                    structs.append(struct) 
                    cho_orders.append(cho_order)
                    atomdicts.append(atomdict)
                    metalInfos.append(metal)
                    speciesInfos.append(subname)
                    metalDescs[rownum,:] = metalDescriptors[metal]
                    rownum += 1
        print(rownum)    
        allStructs = {'structs':structs, 'cho_orders':cho_orders, 'atomdicts':atomdicts, 
                      'metal':metalInfos, 'species':speciesInfos, 'metalDescs':metalDescs}
        f = open(structFileName, 'wb')
        pickle.dump(allStructs, f, pickle.HIGHEST_PROTOCOL)
        f.close()
        energy_csv.to_csv(energyFileName)