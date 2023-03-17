#!/usr/bin/env python
# coding: utf-8

##########################################################################################
#
# Author: Michal Bajczyk michal.bajczyk@gmail.com
#
##########################################################################################


import os
import sys
import pickle
from rdkit import Chem
from rdkit.Chem import RDConfig
import pandas as pd
import numpy as np
import pprint
import glob
import seaborn as sns
import matplotlib.pyplot as plt
from rdkit.Chem import AllChem, QED, Descriptors, Descriptors3D
from rdkit.DataStructs.cDataStructs import BulkTanimotoSimilarity
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
import openbabel
from openbabel import pybel

""""
run_analysis() = format_data() -> process_data() -> getDiversity()

format_data() = filter_by_validity() -> count_duplicates() -> frame()

process_data() = find_in_X(), getX()

getDiversity() = getMorganFingerprint() -> get_pairwise_similarities()

getOccurrenceFrequency()

shape_plot() -> plot_shape()

required files:
    'merged_db_MA.csv'
    'dataset_smiles_qm9.pkl'
    'train_smiles.p'
    'test_smiles.p'
""""


def run_analysis(directory, titles, file_format='.p'):
    
""""
this is the main method, which can be run from terminal

directory = string
titles = list of strings
file_format = string: '.p' or '.csv'

the list of strings could be written manually, e.g:

    ED_ESP_titles=['0.5_40000_2022-01-21',
        '1.0_40000_2022-01-21',
        '1.5_40000_2022-01-21',
        '2.0_40000_2022-01-21',
        '2.5_40000_2022-01-21',
        '5.0_40000_2022-01-21',
        '10.0_40000_2022-01-21',
        '25.0_40000_2022-01-21',
        '50.0_40000_2022-01-21']

or by using these lines:

    files=os.listdir()
    titles=[file.split(".")[0] for file in files]
    
example:
    run_analysis('ED_ESP/', ED_ESP_titles)
""""
    
    if file_format.lower()=='csv' or file_format.lower()=='.csv':
        in_extension='.csv'        
    elif file_format.lower()=='p' or file_format.lower()=='.p':
        in_extension='.p'
    
    if type(titles)==str:
        print(directory, titles, in_extension, sep='')
        with open(directory+titles+in_extension, 'rb') as handle:
            data = pickle.load(handle)
        title=str(titles)
        df_valid, df_invalid=format_data(data)
        df_valid.to_csv(directory+title+'.csv', index=False)
        df_invalid.to_csv(directory+title+'_invalid.csv', index=False)
        process_data(directory, title)
        diversityInt, tanimotosInt = getDiversity(directory+title+'.csv', external=False)
        diversityExt, tanimotosExt = getDiversity(directory+title+'.csv', external=True)
        print(title, diversityInt, diversityExt)
        df_tan_int=pd.DataFrame.from_dict(tanimotosInt, orient='index', columns=['internal', 'internal/total'])
        df_tan_ext=pd.DataFrame.from_dict(tanimotosExt, orient='index', columns=['external', 'external/total'])
        df_tan=pd.concat([df_tan_int, df_tan_ext], axis=1)
        df_tan_int.to_csv(directory+title+'_InternalTanimotos.csv', index=True)
        df_tan_ext.to_csv(directory+title+'_ExternalTanimotos.csv', index=True)
        df_tan.to_csv(directory+title+'_Tanimotos.csv', index=True)
            
    elif type(titles)==list:
        int_divs=[]
        ext_divs=[]
        for title in titles:
            print(directory, title, in_extension, sep='')
            with open(directory+title+in_extension, 'rb') as handle:
                data = pickle.load(handle)
            df_valid, df_invalid=format_data(data)
            df_valid.to_csv(directory+title+'.csv', index=False)
            df_invalid.to_csv(directory+title+'_invalid.csv', index=False)
            process_data(directory, title)
            diversityInt, tanimotosInt = getDiversity(directory+title+'.csv', external=False)
            diversityExt, tanimotosExt = getDiversity(directory+title+'.csv', external=True)
            print(title, diversityInt, diversityExt)
            df_tan_int=pd.DataFrame.from_dict(tanimotosInt, orient='index', columns=['internal', 'internal/total'])
            df_tan_ext=pd.DataFrame.from_dict(tanimotosExt, orient='index', columns=['external', 'external/total'])
            df_tan=pd.concat([df_tan_int, df_tan_ext], axis=1)
            df_tan_int.to_csv(directory+title+'_InternalTanimotos.csv', index=True)
            df_tan_ext.to_csv(directory+title+'_ExternalTanimotos.csv', index=True)
            df_tan.to_csv(directory+title+'_Tanimotos.csv', index=True)
            int_divs.append(diversityInt)
            ext_divs.append(diversityExt)
            name=directory.replace('/','')
        stats=consolidate(directory, titles, name)
        for i in range(0, len(stats)):
            stats.at[i, 'internal diversity']=int_divs[i]
            stats.at[i, 'external diversity']=ext_divs[i]
        stats.to_csv(directory+name+'.csv', index=False)
        

def format_data(data):
""""
organizes molecule sets into Pandas dataframes
this method is used by: run_analysis()
""""  
    valid, invalid = filter_by_validity(data)
    set_valid=count_duplicates(valid)
    set_invalid=count_duplicates(invalid)
    df_valid=frame(set_valid)
    df_invalid=frame(set_invalid)
    df_valid=df_valid.sort_values(by='repeats', ascending=False)
    df_invalid=df_invalid.sort_values(by='repeats', ascending=False)
    return df_valid, df_invalid
    
def filter_by_validity(smiles_list):
""""
checks the validity of smiles representations and segregates them
this method is used by: format_data()
""""  
    valid=[]
    invalid=[]

    for i, smiles in enumerate(smiles_list):
        if len(smiles)>0:
            try:
                mol=Chem.MolFromSmiles(smiles)
                smiles=Chem.MolToSmiles(mol)
                mol=Chem.SanitizeMol(mol)
                #print('good')
                valid+=[(i, smiles)]
            except:
                invalid+=[(i, smiles)]
                #print('bad')
        else:
            invalid+=[(i, smiles)]
    return valid, invalid

def count_duplicates(smiles_list):
""""
collects data on duplicity and organizes the molecule sets
uses canonical forms of smiles to find matches
this method is used by: format_data()
"""" 
    new_list={}
    
    for j, entry in enumerate(smiles_list):
        index, smiles = entry
        if smiles not in new_list.keys():
            new_list[str(smiles)]=[index]
        elif smiles in new_list.keys():
            new_list[str(smiles)]+=[index]
    return new_list

def frame(smiles_dict):
""""
transforms dataframe containing duplicity info into a dictionary
this method is used by: format_data()
""""
    dataFrame=pd.DataFrame({'smiles': [], 'repeats': []})
    for key, value in smiles_dict.items():
        line=pd.DataFrame({'smiles': [key], 
                           'repeats': [len(value)]})
        dataFrame=dataFrame.append(line)
    return dataFrame

def process_data(directory, title):
""""
benchmarks the molecules against various datasets
evaluates structural correctness, size, shape and other parameters
this method is used by: run_analysis()
""""  
    filter_by_MolGen(directory+title+'.csv')
    find_in_commercial(directory+title+'.csv')
    find_in_QM9(directory+title+'.csv')
    find_in_TrainingSet(directory+title+'.csv')
    #check_correctness(directory+title+'.csv')
    getSize(directory+title+'.csv')
    getShape(directory+title+'.csv')
    getQEDs(directory+title+'.csv')
    getSyntheticAvailability(directory+title+'.csv')
    getLogP(directory+title+'.csv')
    print(title, 'done')

def sort_data(dataFrame, value):
    dataFrame=dataFrame.sort_values(by=value)
    dataFrame.to_csv(filename, index=False)
    
def filter_by_MolGen(filename):
""""
these are forbidden motifs from MolGen, translated into SMARTS and generalized,
    i.e., here carbons are any atoms and single bonds are any bonds
this method is used by: process_data()
""""
    dataFrame=pd.read_csv(filename)
    
    evilSMARTS=51*[None]
    evil_motifs=[]
    evilSMARTS[0]='[*]1~[*]#[*]~1'
    evilSMARTS[1]='[*]1~[*]~[*]#[*]~1'
    evilSMARTS[2]='[*]1~[*]~[*]#[*]~[*]~1'
    evilSMARTS[3]='[*]1~[*]~[*]~[*]#[*]~[*]~1'
    evilSMARTS[4]='[*]1~[*]~[*]~[*]#[*]~[*]~[*]~1'
    evilSMARTS[5]='[*]1=[*]=[*]~1'
    evilSMARTS[6]='[*]1~[*]=[*]=[*]~1'
    evilSMARTS[7]='[*]1~[*]~[*]=[*]=[*]~1'
    evilSMARTS[8]='[*]1~[*]~[*]=[*]=[*]~[*]~1'
    evilSMARTS[9]='[*]1~[*]~[*]~[*]=[*]=[*]~[*]~1'
    evilSMARTS[10]='[*]1~[*]~[*]~[*]=[*]=[*]~[*]~[*]~1'
    evilSMARTS[11]='[*]1~[*]2~[*]=[*]~1~2'
    evilSMARTS[12]='[*]1~[*]2=[*]~1~[*]~2'
    evilSMARTS[13]='[*]1~[*]~[*]2=[*]~[*]~1~2'
    evilSMARTS[14]='[*]1~[*]=[*]2~[*]~[*]~1~2'
    evilSMARTS[15]='[*]1~[*]~[*]2=[*]~1~[*]~2'
    evilSMARTS[16]='[*]1~[*]~[*]2~[*]=[*]~2~[*]~1'
    evilSMARTS[17]='[*]1~[*]~[*]2~[*]~[*]~2=[*]~1'
    evilSMARTS[18]='[*]1~[*]~[*]2=[*](~[*]~1)~[*]~2'
    evilSMARTS[19]='[*]1~[*]~[*]~[*]2=[*]~[*]~2~[*]~1'
    evilSMARTS[20]='[*]1~[*]~[*]=[*]2~[*]~[*]~2~[*]~1'
    evilSMARTS[21]='[*]1~[*]~[*]2=[*]~[*]~[*]~1~2'
    evilSMARTS[22]='[*]1~[*]~[*]2~[*]~[*]=[*]~2~[*]~1'
    evilSMARTS[23]='[*]1~[*]~[*]2~[*]~[*]~[*]~2=[*]~1'
    evilSMARTS[24]='[*]1~[*]2~[*]~[*]~1=[*]~2'
    evilSMARTS[25]='[*]1~[*]~[*]2=[*]~[*]~1~[*]~2'
    evilSMARTS[26]='[*]1~[*]=[*]2~[*]~[*]~1~[*]~2'
    evilSMARTS[27]='[*]1~[*]~[*]2~[*]~[*](=[*]~2)~[*]~1'
    evilSMARTS[28]='[*]1~[*]~[*]2~[*]~[*](=[*]~1)~[*]~2'
    evilSMARTS[29]='[*]1~[*]~[*]2=[*]~[*]~1~[*]~[*]~2'
    evilSMARTS[30]='[*]1~[*]~[*]2=[*]~[*]~[*]~1~[*]~2'
    evilSMARTS[31]='[*]1~[*]~[*]2~[*]~[*]~[*](=[*]~2)~[*]~1'
    evilSMARTS[32]='[*]1~[*]~[*]2~[*]~[*]=[*](~[*]~1)~[*]~2'
    evilSMARTS[33]='[*]1~[*]~[*]2~[*]~[*]~[*](=[*]~1)~[*]~2'
    evilSMARTS[34]='[*]1~[*]~[*]2=[*]~[*]~[*]~1~[*]~[*]~2'
    evilSMARTS[35]='[*]1~[*]2~[*]3~[*]~[*]~1~2~3'
    evilSMARTS[36]='[*]1~[*]~[*]23~[*]4~[*]~2~[*]~1~3~4'
    evilSMARTS[37]='[*]1~[*]23~[*]~[*]~1(~[*]~2)~[*]~3'
    evilSMARTS[38]='[*]1~[*]2:[*]:[*]:[*]:[*]:[*]:2~1'
    evilSMARTS[39]='[*]1~[*]~[*]2:[*]:[*]:[*]:[*]:[*]:2~1'
    evilSMARTS[40]='[*]1~[*]2:[*]:[*]:[*]:[*]~1:[*]:2'
    evilSMARTS[41]='[*]1~[*]~[*]2:[*]:[*]:[*]:[*]~1:[*]:2'
    evilSMARTS[42]='[*]1~[*]~[*]2:[*]:[*]:[*]:[*](:[*]:2)~[*]~1'
    evilSMARTS[43]='[*]1~[*]2:[*]:[*]:[*]~1:[*]:[*]:2'
    evilSMARTS[44]='[*]1~[*]~[*]2:[*]:[*]:[*]~1:[*]:[*]:2'
    evilSMARTS[45]='[*]1~[*]~[*]2:[*]:[*]:[*](:[*]:[*]:2)~[*]~1'
    evilSMARTS[46]='[*]1:[*]:[*]2:[*]:[*]~2:[*]:1'
    evilSMARTS[47]='[*]1:[*]:[*]2:[*]:[*]:[*]:1~2'
    evilSMARTS[48]='[*]1~[*]~[*]~[*]2:[*]:[*]:[*]:[*](:[*]:2)~[*]~1'
    evilSMARTS[49]='[*]1~[*]~[*]~[*]2:[*]:[*]:[*]:[*](:[*]:2)~[*]~[*]~1'
    evilSMARTS[50]='[*]1~[*]~[*]~[*]2:[*]:[*]:[*](:[*]:[*]:2)~[*]~1'
    
    for i, item in enumerate(evilSMARTS):
        subMol = Chem.MolFromSmarts(item)
        evil_motifs+=[subMol]
    
    for i in range(0, len(dataFrame)):
        smiles=dataFrame.at[i, 'smiles']
        #print(smiles)
        mol=Chem.MolFromSmiles(smiles)
        dataFrame.at[i, 'forbidden substructures']=0
        for j, motif in enumerate(evil_motifs):
                if mol.HasSubstructMatch(motif) == True:
                    dataFrame.at[i, 'forbidden substructures']+=1
    dataFrame.to_csv(filename, index=False)

    
def find_in_commercial(filename):
""""
this method is used by: process_data()
checks which smiles are of commercially available compounds
""""
    dataFrame=pd.read_csv(filename)
    db=getCommercial()
    for i in range(0, len(dataFrame)):
        smiles=dataFrame.at[i, 'smiles']
        if smiles in db:
            dataFrame.at[i, 'commercial']=1
        else:
            dataFrame.at[i, 'commercial']=0
    dataFrame.to_csv(filename, index=False)
    
def getCommercial(removeStereo=True):
""""
this method is used by: find_in_commercial()
requires a file containing a list of commercially available compounds,
e.g., 'merged_db_MA.csv'
""""
    commercialDB = pd.read_csv('merged_db_MA.csv', usecols=['smiles'])
    commercial_smiles = [ s for s in commercialDB.smiles ]
    if removeStereo==True:
        commercial_smiles = [ smiles.replace('@','') for smiles in commercial_smiles]
        commercial_smiles = [ smiles.replace('/','') for smiles in commercial_smiles]
        commercial_smiles = [ smiles.replace('\\','') for smiles in commercial_smiles]
    commercial_mol=[Chem.MolFromSmiles(smiles) for smiles in commercial_smiles]
    commercial_smiles=[Chem.MolToSmiles(mol) for mol in commercial_mol]
    return commercial_smiles
    
def find_in_QM9(filename):
""""
this method is used by: process_data()
checks which smiles belong to QM9 set
""""
    dataFrame=pd.read_csv(filename)
    db=getQM9()
    for i in range(0, len(dataFrame)):
        smiles=dataFrame.at[i, 'smiles']
        if smiles in db:
            dataFrame.at[i, 'QM9']=1
        else:
            dataFrame.at[i, 'QM9']=0
    dataFrame.to_csv(filename, index=False)
    
def getQM9(removeStereo=True):
""""
this method is used by: find_in_QM9()
requires a file containing molecules from QM9 dataser,
e.g., 'dataset_smiles_qm9.pkl'
""""
    with open('dataset_smiles_qm9.pkl', 'rb') as handle:
        full_smiles_list = pickle.load(handle)
    if removeStereo==True:
        full_smiles_list = [ smiles.replace('@','') for smiles in full_smiles_list]
        full_smiles_list = [ smiles.replace('/','') for smiles in full_smiles_list]
        full_smiles_list = [ smiles.replace('\\','') for smiles in full_smiles_list]
    full_mol_list=[Chem.MolFromSmiles(smiles) for smiles in full_smiles_list]
    full_smiles_list=[Chem.MolToSmiles(mol) for mol in full_mol_list]
    return full_smiles_list

def find_in_TrainingSet(filename):
""""
this method is used by: process_data()
checks which smiles belong to the training set
""""
    dataFrame=pd.read_csv(filename)
    db=getTrainingSet()
    for i in range(0, len(dataFrame)):
        smiles=dataFrame.at[i, 'smiles']
        if smiles in db:
            dataFrame.at[i, 'training']=1
        else:
            dataFrame.at[i, 'training']=0
    dataFrame.to_csv(filename, index=False)

def getTrainingSet(removeStereo=True):
""""
this method is used by: find_in_TrainingSet()
requires a file of the training set,
e.g., 'train_smiles.p'
""""
    with open('train_smiles.p', 'rb') as handle:
        full_smiles_list = pickle.load(handle)
    if removeStereo==True:
        full_smiles_list = [ smiles.replace('@','') for smiles in full_smiles_list]
        full_smiles_list = [ smiles.replace('/','') for smiles in full_smiles_list]
        full_smiles_list = [ smiles.replace('\\','') for smiles in full_smiles_list]
    full_mol_list=[Chem.MolFromSmiles(smiles) for smiles in full_smiles_list]
    full_smiles_list=[Chem.MolToSmiles(mol) for mol in full_mol_list]
    return full_smiles_list

def find_in_TestSet(filename):
""""
this method is used by: process_data()
checks which smiles belong to the test set
""""
    dataFrame=pd.read_csv(filename)
    db=getTestSet()
    for i in range(0, len(dataFrame)):
        smiles=dataFrame.at[i, 'smiles']
        if smiles in db:
            dataFrame.at[i, 'test']=1
        else:
            dataFrame.at[i, 'test']=0
    dataFrame.to_csv(filename, index=False)

def getTestSet(removeStereo=True):
""""
this method is used by: find_in_TestSet()
requires a file of the test set,
e.g., 'test_smiles.p'
""""
    with open('test_smiles.p', 'rb') as handle:
        full_smiles_list = pickle.load(handle)
    if removeStereo==True:
        full_smiles_list = [ smiles.replace('@','') for smiles in full_smiles_list]
        full_smiles_list = [ smiles.replace('/','') for smiles in full_smiles_list]
        full_smiles_list = [ smiles.replace('\\','') for smiles in full_smiles_list]
    full_mol_list=[Chem.MolFromSmiles(smiles) for smiles in full_smiles_list]
    full_smiles_list=[Chem.MolToSmiles(mol) for mol in full_mol_list]
    return full_smiles_list
    
def check_correctness(filename):
""""
this method is optional: process_data()
currently hashed out, removes smiles that can not yield sanitizable mol objects
""""
    dataFrame=pd.read_csv(filename)
    for i in range(0, len(dataFrame)):
        smiles=dataFrame.at[i, 'smiles']
        mol=Chem.MolFromSmiles(smiles)
        try:
            mol=Chem.SanitizeMol(mol)
            dataFrame.at[i, 'chemically valid']=1
        except:
            dataFrame.at[i, 'chemically valid']=0
    dataFrame.to_csv(filename, index=False)

def getSize(filename):
""""
this method is used by: process_data()
counts atoms and bonds, calculates molecular mass
""""
    dataFrame=pd.read_csv(filename)
    for i in range(0, len(dataFrame)):  
        smiles=dataFrame.at[i,'smiles']
        mol=Chem.MolFromSmiles(smiles)
        mass=Descriptors.MolWt(mol)
        dataFrame.at[i, 'weight']=mass
        
        atoms=mol.GetNumAtoms()
        dataFrame.at[i, 'atoms']=atoms
        
        atoms=mol.GetAtoms()
        dataFrame.at[i, 'carbons']=0
        dataFrame.at[i, 'nitrogens']=0
        dataFrame.at[i, 'oxygens']=0
        for atom in atoms:
            element=atom.GetAtomicNum()
            if element==6:
                dataFrame.at[i, 'carbons']+=1
            elif element==7:
                dataFrame.at[i, 'nitrogens']+=1
            elif element==8:
                dataFrame.at[i, 'oxygens']+=1
        
        bonds=(mol.GetBonds())
        dataFrame.at[i, 'bonds']=len(bonds)
        dataFrame.at[i, 'single']=0
        dataFrame.at[i, 'double']=0
        dataFrame.at[i, 'triple']=0
        for bond in bonds:
            if str(bond.GetBondType())=="SINGLE":
                dataFrame.at[i, 'single']+=1
            elif str(bond.GetBondType())=='DOUBLE':
                dataFrame.at[i, 'double']+=1
            elif str(bond.GetBondType())=='TRIPLE':
                dataFrame.at[i, 'triple']+=1
    dataFrame.to_csv(filename, index=False)

def getShape(filename):
""""
this method is used by: process_data()
uses RDKit descriptor package to calculate the molecule shapes
""""
    dataFrame=pd.read_csv(filename)
    for i in range(0, len(dataFrame)):
        smiles=dataFrame.at[i,'smiles']
        mol=Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        if AllChem.EmbedMolecule(mol)==-1:
            dataFrame.at[i, 'embed_pass']=0
            obConversion = openbabel.OBConversion()
            obConversion.SetInAndOutFormats("smi", "mdl")
            mol = openbabel.OBMol()
            obConversion.ReadString(mol, smiles)
            mol.AddHydrogens()
            mol=pybel.Molecule(mol)
            mol.addh()
            mol.make3D()
            mol=Chem.MolFromMolBlock(mol.write("mol"))
            mol=Chem.AddHs(mol)
        else:
            dataFrame.at[i, 'embed_pass']=1
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except:
            dataFrame.at[i, 'embed_pass']=0
            continue
        dataFrame.at[i, 'sphericity']=Chem.Descriptors3D.SpherocityIndex(mol)
        dataFrame.at[i, 'asphericity']=Chem.Descriptors3D.Asphericity(mol)
        dataFrame.at[i, 'eccentricity']=Chem.Descriptors3D.Eccentricity(mol)
        dataFrame.at[i, 'gyration_radius']=Chem.Descriptors3D.RadiusOfGyration(mol)
        dataFrame.at[i, 'NPR1']=Chem.Descriptors3D.NPR1(mol)
        dataFrame.at[i, 'NPR2']=Chem.Descriptors3D.NPR2(mol)
        dataFrame.at[i, 'ISF']=Chem.Descriptors3D.InertialShapeFactor(mol)
    dataFrame.to_csv(filename, index=False)
    
def getQEDs(filename):
""""
this method is used by: process_data()
gives Quantitative Estimation of Drug-likeness by RDKit
""""
    dataFrame=pd.read_csv(filename)
    for i in range(0, len(dataFrame)):
        smiles=dataFrame.at[i,'smiles']
        mol=Chem.MolFromSmiles(smiles)
        qed=Chem.QED.qed(mol)
        dataFrame.at[i,'QED']=qed
    dataFrame.to_csv(filename, index=False)

def getSyntheticAvailability(filename):
    dataFrame=pd.read_csv(filename)
    for i in range(0, len(dataFrame)):
        smiles=dataFrame.at[i,'smiles']
        mol=Chem.MolFromSmiles(smiles)
        s = sascorer.calculateScore(mol)
        dataFrame.at[i,'synthetic availability']=s
    dataFrame.to_csv(filename, index=False)
    
def getLogP(filename):
    dataFrame=pd.read_csv(filename)
    for i in range(0, len(dataFrame)):
            smiles=dataFrame.at[i,'smiles']
            mol=Chem.MolFromSmiles(smiles)
            LogP = Descriptors.MolLogP(mol)
            dataFrame.at[i,'logP']=LogP       
    dataFrame.to_csv(filename, index=False)
    
def getDiversity(filename, external=False):
""""
this method is used by: run_analysis()
uses tanimoto to assess dataset diversity
internal = pairwise comparisons within the same dataset
external = pairwise comparisons between the given set and the training set
""""    
    dataFrame=pd.read_csv(filename)
    molTotal=int(len(dataFrame))
    fingerprints=[None]*molTotal
    #fingerprints=np.array(fingerprints)
    for i in range(0, molTotal):
        smiles=dataFrame.at[i,'smiles']
        mol=Chem.MolFromSmiles(smiles)
        fingerprint=getMorganFingerprint(mol)
        fingerprints[i]=fingerprint    
    if external==True:
        trainingSet=getTestSet(removeStereo=True)
        refTotal=len(trainingSet)
        exFingerprints=[None]*refTotal
        for i in range(0, refTotal):
            smiles=trainingSet[i]
            mol=Chem.MolFromSmiles(smiles)
            fingerprint=getMorganFingerprint(mol)
            exFingerprints[i]=fingerprint
        total=molTotal*refTotal
        histogram=get_pairwise_similarities(fingerprints, exFingerprints)
    else:
        refTotal=len(fingerprints)
        total=molTotal*refTotal
        histogram=get_pairwise_similarities(fingerprints, fingerprints)
    tanimotos=0
    count=0
    for value, key in histogram.items():
        tanimotos+=(value*key)
        count+=key
    print(count==total)
    diversity=tanimotos/total
    
    for key, value in histogram.items():
        histogram[key]=[value, (value/total)]
    
    return diversity, histogram

def getMorganFingerprint(mol):
""""
this method is used by: get_pairwise_similarities()
""""    
    return AllChem.GetHashedMorganFingerprint(mol, 2)

def get_pairwise_similarities(fps1, fps2):
""""
this method is used by: getDiversity()
""""    
    values=np.around(np.linspace(0, stop=1, num=101), decimals=2)
    histogram={i:0 for i in values}
    for fp in fps1:
        tanimotos=BulkTanimotoSimilarity(fp, fps2)
        for tanimoto in tanimotos:
            anti_tanimoto=np.around((1-tanimoto), decimals=2)
            histogram[anti_tanimoto]+=1
    return histogram

def plot_pws(pws_random, pws_smiles, output):
"""plot similarity data """

    dat = pd.DataFrame({'sims_random': pws_random, 'sims_smiles': pws_smiles})
    # print(dat['sims_smiles'][0]) # --- the answer --- !!!
    # dat.to_csv(f'similarity_data_{output}.csv')
    return dat['sims_smiles'][0] # --- the answer --- !!!!!!!!!!!!!!!!!!
    # plt.savefig(f'figures/{output}-pws.jpg')
    
def consolidate(directory, titles, name):
""""
this method is used by: process_data()
puts together and organizes all the analysis&benchmarking data
""""    
    new_dataFrame=pd.DataFrame()
    for title in titles:
        dataFrame=pd.read_csv(directory+title+'.csv')
        dataFrame_invalid=pd.read_csv(directory+title+'_invalid.csv')
        total=len(dataFrame)+len(dataFrame_invalid)
        new_dataFrame.at[title, 'unique']=total/40000
        new_dataFrame.at[title, 'unique_c']=total
        new_dataFrame.at[title, 'valid']=len(dataFrame)/total
        new_dataFrame.at[title, 'valid_c']=len(dataFrame)
        new_dataFrame.at[title, 'MolGen_pass']=0
        new_dataFrame.at[title, 'MolGen_pass_c']=0
        new_dataFrame.at[title, 'geom_pass']=0
        new_dataFrame.at[title, 'geom_pass_c']=0
        new_dataFrame.at[title, 'legit']=0
        new_dataFrame.at[title, 'legit_c']=0        
        new_dataFrame.at[title, 'novelty']=0
        new_dataFrame.at[title, 'novel']=0
        new_dataFrame.at[title, 'novelty_weighted']=0
        new_dataFrame.at[title, 'novel_weighted']=0
        new_dataFrame.at[title, 'commercial']=0
        new_dataFrame.at[title, 'commercial_c']=0
        
        new_dataFrame.at[title, 'invalid']=len(dataFrame_invalid)/40000
        new_dataFrame.at[title, 'invalid_c']=len(dataFrame_invalid)        
        new_dataFrame.at[title, 'QM9']=0
        new_dataFrame.at[title, 'QM9_c']=0
        new_dataFrame.at[title, 'Reaxys']=0
        new_dataFrame.at[title, 'Reaxys_c']=0
        
        new_dataFrame.at[title, 'avg_weight']=np.average(dataFrame['weight'])
        new_dataFrame.at[title, 'error_weight']=np.std(dataFrame['weight'])
        new_dataFrame.at[title, 'avg_atoms']=np.average(dataFrame['atoms'])
        new_dataFrame.at[title, 'error_atoms']=np.std(dataFrame['atoms'])
        new_dataFrame.at[title, 'avg_carbons']=np.average(dataFrame['carbons'])
        new_dataFrame.at[title, 'error_carbons']=np.std(dataFrame['carbons'])
        new_dataFrame.at[title, 'avg_nitrogens']=np.average(dataFrame['nitrogens'])
        new_dataFrame.at[title, 'error_nitrogens']=np.std(dataFrame['nitrogens'])
        new_dataFrame.at[title, 'avg_oxygens']=np.average(dataFrame['oxygens'])
        new_dataFrame.at[title, 'error_oxygens']=np.std(dataFrame['oxygens'])
        
        new_dataFrame.at[title, 'avg_bonds']=np.average(dataFrame['bonds'])
        new_dataFrame.at[title, 'error_bonds']=np.std(dataFrame['bonds'])
        new_dataFrame.at[title, 'avg_single']=np.average(dataFrame['single'])
        new_dataFrame.at[title, 'error_single']=np.std(dataFrame['single'])
        new_dataFrame.at[title, 'avg_double']=np.average(dataFrame['double'])
        new_dataFrame.at[title, 'error_double']=np.std(dataFrame['double'])
        new_dataFrame.at[title, 'avg_triple']=np.average(dataFrame['triple'])
        new_dataFrame.at[title, 'error_triple']=np.std(dataFrame['triple'])
        
        new_dataFrame.at[title, 'avg_sphericity']=np.mean(dataFrame['sphericity'])
        new_dataFrame.at[title, 'error_sphericity']=np.std(dataFrame['sphericity'])
        new_dataFrame.at[title, 'avg_asphericity']=np.mean(dataFrame['asphericity'])
        new_dataFrame.at[title, 'error_asphericity']=np.std(dataFrame['asphericity'])
        new_dataFrame.at[title, 'avg_eccentricity']=np.mean(dataFrame['eccentricity'])
        new_dataFrame.at[title, 'error_eccentricity']=np.std(dataFrame['eccentricity'])        
        new_dataFrame.at[title, 'avg_gyration_radius']=np.mean(dataFrame['gyration_radius'])
        new_dataFrame.at[title, 'error_gyration_radius']=np.std(dataFrame['gyration_radius'])    
        new_dataFrame.at[title, 'avg_NPR1']=np.mean(dataFrame['NPR1'])
        new_dataFrame.at[title, 'error_NPR1']=np.std(dataFrame['NPR1'])   
        new_dataFrame.at[title, 'avg_NPR2']=np.mean(dataFrame['NPR2'])
        new_dataFrame.at[title, 'error_NPR2']=np.std(dataFrame['NPR2'])   
        new_dataFrame.at[title, 'avg_ISF']=np.mean(dataFrame['ISF'])
        new_dataFrame.at[title, 'error_ISF']=np.std(dataFrame['ISF'])   
        
        for i in range(0, len(dataFrame)):
            if dataFrame.at[i, 'forbidden substructures']==0:
                new_dataFrame.at[title, 'MolGen_pass_c']+=1
            if dataFrame.at[i, 'commercial']==1:
                new_dataFrame.at[title, 'commercial_c']+=1
            if dataFrame.at[i, 'training']==0:
                new_dataFrame.at[title, 'novel']+=1
                new_dataFrame.at[title, 'novel_weighted']+=dataFrame.at[i, 'repeats']
            if dataFrame.at[i, 'QM9']==1:
                new_dataFrame.at[title, 'QM9_c']+=1
            #if dataFrame.at[i, 'Reaxys']==1:
                #new_dataFrame.at[title, 'Reaxys_c']+=1   
            if dataFrame.at[i, 'embed_pass']==1:
                new_dataFrame.at[title, 'geom_pass_c']+=1  
            if dataFrame.at[i, 'forbidden substructures']==0 and dataFrame.at[i, 'embed_pass']==1:
                new_dataFrame.at[title, 'legit_c']+=1  
                
        new_dataFrame.at[title, 'MolGen_pass']=new_dataFrame.at[title, 'MolGen_pass_c']/len(dataFrame)
        new_dataFrame.at[title, 'novelty']=new_dataFrame.at[title, 'novel']/len(dataFrame)
        new_dataFrame.at[title, 'novelty_weighted']=new_dataFrame.at[title, 'novel_weighted']/len(dataFrame)
        new_dataFrame.at[title, 'commercial']=new_dataFrame.at[title, 'commercial_c']/len(dataFrame)
        new_dataFrame.at[title, 'QM9']=new_dataFrame.at[title, 'QM9_c']/len(dataFrame)
        #new_dataFrame.at[title, 'Reaxys']=new_dataFrame.at[title, 'Reaxys_c']/len(dataFrame)
        new_dataFrame.at[title, 'geom_pass']=new_dataFrame.at[title, 'geom_pass_c']/len(dataFrame)
        new_dataFrame.at[title, 'legit']=new_dataFrame.at[title, 'legit_c']/len(dataFrame)
        
    new_dataFrame.to_csv(directory+name+'.csv')
    return new_dataFrame

def getOccurrenceFrequency(directory, titles, name):
        
""""
this is the separate method, which can be run from terminal

directory = string
titles = list of strings
name = string

ranks and sorts the molecules based on their number of occurences
""""
    new_dataFrame=pd.DataFrame()
    for title in titles:
        dataFrame=pd.read_csv(directory+title+'.csv')
        dataFrame=dataFrame.sort_values(by='repeats', ascending=False, ignore_index=True)
        column=dataFrame['repeats']
        new_dataFrame=pd.concat([new_dataFrame, column], axis=1)
        new_dataFrame=new_dataFrame.rename(columns={'repeats': title.split('_')[0]})
    new_dataFrame.index+=1
    new_dataFrame.to_csv(directory+name+'_occurences.csv')
    
def shape_plot(directory, titles):
""""
this is the separate method, which can be run from terminal

directory = string
titles = list of strings

calculates shapes for the triangular plot, i.e.,  sphere vs disc vs rod
""""
    new_dataFrame=pd.DataFrame()
    for title in titles:
        temp_dataFrame=pd.DataFrame()
        dataFrame=pd.read_csv(directory+title+'.csv')
        title=title.split('_')[0]
        temp_dataFrame['smiles']=dataFrame['smiles']
        temp_dataFrame['type']=title
        new_dataFrame=pd.concat([temp_dataFrame,new_dataFrame], ignore_index=True)
        
    for i in range(0, len(new_dataFrame)):
        smiles=new_dataFrame.at[i, 'smiles']
        mol=Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        if AllChem.EmbedMolecule(mol)==-1:
            dataFrame.at[i, 'embed_pass']=0
            obConversion = openbabel.OBConversion()
            obConversion.SetInAndOutFormats("smi", "mdl")
            mol = openbabel.OBMol()
            obConversion.ReadString(mol, smiles)
            mol.AddHydrogens()
            mol=pybel.Molecule(mol)
            mol.addh()
            mol.make3D()
            mol=Chem.MolFromMolBlock(mol.write("mol"))
            mol=Chem.AddHs(mol)
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except:
            continue
        NPR1=Chem.Descriptors3D.NPR1(mol)
        NPR2=Chem.Descriptors3D.NPR2(mol)
        new_dataFrame.at[i, 'NPR1']=NPR1
        new_dataFrame.at[i, 'NPR2']=NPR2
        new_dataFrame.at[i, 'NPR1']=0
        new_dataFrame.at[i, 'NPR2']=0
    return new_dataFrame

def plot_shape(df):
""""
this is the separate method, which can be run from terminal

directory = dataframe obtained from shape_plot()

produces the triangular plot, i.e.,  sphere vs disc vs rod
""""

    # remove those where shape could not be calculated
    df = df[(df.NPR1 != 0.0) & (df.NPR2 != 0.0)]

    #pal = ['firebrick', 'dodgerblue', ]
    h = sns.jointplot(data=df, x="NPR1", y="NPR2", height=10, hue="type", kind="kde", palette='hls', alpha=0.5)
    
    h.ax_joint.plot([0.5, 1.0], [0.5, 1.0], c='grey')
    h.ax_joint.plot([0.5, 0.0], [0.5, 1.0], c='grey')
    h.ax_joint.plot([0.0, 1.0], [1.0, 1.0], c='grey')
    
    h.ax_joint.set_ylim([0.45, 1.05])
    h.ax_joint.set_xlim([-0.05, 1.05])
    h.ax_joint.tick_params(axis="both", labelsize=14)
    h.set_axis_labels("NPR1", "NPR2", fontsize=14)
    
    h.ax_joint.legend(fontsize=14)

    h.ax_joint.text(0.0, 1.02, 'Rod', fontweight='bold', fontsize=14)
    h.ax_joint.text(0.46, 0.47, 'Disk', fontweight='bold', fontsize=14)
    h.ax_joint.text(0.87, 1.02, 'Sphere', fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    #plt.savefig(f'figures/{output}-shape.jpg')

