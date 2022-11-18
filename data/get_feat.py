import pandas as pd
from tqdm import tqdm
import torch
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import numpy as np
breakpoint()

def mol_features(smiles_string, is_prot=False):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """
    if not is_prot:
        mol = Chem.MolFromSmiles(smiles_string)
    else:
        mol = Chem.MolFromFASTA(smiles_string)

    descriptor = rdMolDescriptors.MQNs_(mol)
    return descriptor

def num_atoms(smiles_data, is_prot=False):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """
    lens = []    
    breakpoint()
    for index, row in smiles_data.iterrows():
        #breakpoint()
        smiles_string, prot_name, label = row
        mol = Chem.MolFromSmiles(smiles_string)
        atoms = rdMolDescriptors.CalcNumAtoms(mol)
        lens.append(atoms)
        
    #print(f"Total proteins: {len(smiles_data.Smiles)} -- Total lens: {len(lens)}"

    return lens
    
'''def transform_molecule_pg(smiles, label, args=None, advs=False,
                          received_mol=False, saliency=False, is_prot=False,):

    if is_prot:
        descrip_list = smiles_to_graph(smiles, is_prot)
        x_p = torch.tensor(x_p)
        edge_index_p = torch.tensor(edge_index_p)
        edge_attr_p = torch.tensor(edge_attr_p)

        return DataFrameMol

    else:
        if args.advs or received_mol:
            if advs or received_mol:
                edge_attr, edge_index, x = smiles_to_graph_advs(
                    smiles,
                    args,
                    advs=True,
                    received_mol=received_mol,
                    saliency=saliency,
                )
            else:
                edge_attr, edge_index, x = smiles_to_graph_advs(
                    smiles, args, received_mol=received_mol, saliency=saliency
                )
        else:
            edge_attr, edge_index, x = smiles_to_graph(smiles)

        if not saliency:
            x = torch.tensor(x)
        y = torch.tensor([label])
        edge_index = torch.tensor(edge_index)
        if not args.advs and not received_mol:
            edge_attr = torch.tensor(edge_attr)

        if received_mol:
            mol = smiles
        else:
            mol = Chem.MolFromSmiles(smiles)

        return Data(
            edge_attr=edge_attr, edge_index=edge_index, x=x, y=y, mol=mol, smiles=smiles
        )'''

def get_features(dataset, data_fasta):

    feature_df = pd.DataFrame()
    for index, row in dataset.iterrows():
        #if index == 20: break
        #breakpoint()
        smiles, prot_name, label = row
        desc_smiles = mol_features(smiles)
        desc_protein = mol_features(data_fasta[data_fasta.Target == prot_name].Fasta.item(), True)
        feat = desc_smiles + desc_protein
        df = pd.DataFrame(feat).T
        feature_df = pd.concat([feature_df, df], ignore_index = True, axis = 0)

    return feature_df


def load_dataset(binary_task):
    """
    Load data and return data in dataframes format for each split and the loader of each split.
    Args:
        binary_tast (boolean): Whether to perform binary classification or multiclass classification.
        args (parser): Complete arguments (configuration) of the model.
        use_prot (boolean): Whether to use the PM module.
        advs (boolean): Whether to train the LM module with adversarial augmentations.
        test (boolean): Whether the model is being tested or trained.
    Return:
        train (loader): Training loader
        valid (loader): Validation loader
        test (loader): Test loader
        data_train (dataframe): Training data dataframe
        data_valid (dataframe): Validation data dataframe
        data_test (dataframe): Test data dataframe

    """
    breakpoint()
    # Read all data files
    if binary_task:
        path = "datasets/AD/"
        add_val = '_AD'
    else:
        path = "datasets/DUDE/"
        add_val = ''

    print("Loading data...")

    A = pd.read_csv(path + f"Smiles{add_val}_1.csv", names=["Smiles", "Target", "Label"])
    B = pd.read_csv(path + f"Smiles{add_val}_2.csv", names=["Smiles", "Target", "Label"])
    C = pd.read_csv(path + f"Smiles{add_val}_3.csv", names=["Smiles", "Target", "Label"])
    D = pd.read_csv(path + f"Smiles{add_val}_4.csv", names=["Smiles", "Target", "Label"])

    # Get dataset for each split
    total_data = pd.concat([A, B, C, D], ignore_index=True)
    #total_data = A
    data_target = pd.read_csv(path + "Targets_Fasta.csv", names=["Fasta", "Target", "Label"])
    num_ats = num_atoms(total_data)
    #df_features = get_features(total_data, data_target)

    breakpoint()
    #joint_df = pd.concat([total_data, df_features], axis=1)


    print("Done.")
    return num_ats, np.max(num_ats), np.min(num_ats)

breakpoint()
load = load_dataset(binary_task=True)
#load.to_csv('planet_feat.csv')

def remove_target():
    df_OG = pd.read_csv("planet_feat.csv", names=["Smiles", "Target", "Label",
                                                  0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,
                                                  21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,
                                                  39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,
                                                  57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,
                                                  75,76,77,78,79,80,81,82,83])

    df_OG = df_OG.drop(columns=['Target'])
    breakpoint()
    df_OG.to_csv('planet_sinTarget.csv',index=False)

#remove_target()
breakpoint()
print('fin')