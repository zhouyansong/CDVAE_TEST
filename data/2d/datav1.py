import pandas as pd
from sklearn.model_selection import train_test_split
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter
import os
import pathlib
from pymatgen.analysis import local_env
from pymatgen.analysis.graphs import StructureGraph
import numpy as np

def validate_structure_connectivity(structure, search_cutoff=8.0, max_bond=4.0, min_bond=1.3, min_neighbors=1):
    crystal_nn = local_env.CrystalNN(
        distance_cutoffs=None, 
        x_diff_weight=-1, 
        porous_adjustment=False, 
        search_cutoff=search_cutoff  
    )
    structure_graph = StructureGraph.with_local_env_strategy(structure, crystal_nn)
    isolated_atoms = []
    neighbor_counts = []
    closest_neighbor_distances = []
    for i in range(len(structure)):
        neighbors = structure_graph.get_connected_sites(i)
        valid_distances = []
        valid_indices = []
        for neighbor in neighbors:
            distance = structure.get_distance(i, neighbor.index)
            if distance >= min_bond and neighbor.index != i:
                valid_distances.append(distance)
                valid_indices.append(neighbor.index)
        neighbor_count = len(valid_distances)
        neighbor_counts.append(neighbor_count)
        #print(f"Atom {i}: valid neighbors {valid_indices}, distances {valid_distances}")

        if neighbor_count >= min_neighbors:
            closest_distance = min(valid_distances)  # 找到最近邻的距离
            closest_neighbor_distances.append(closest_distance)

            if closest_distance > max_bond:
                isolated_atoms.append(i)
                print(f"Atom {i}: closest neighbor at {closest_distance:.3f} Å (> {max_bond} Å)")
        else:
            # No neighbors found even within search_cutoff
            isolated_atoms.append(i)
            closest_neighbor_distances.append(float('inf'))
            print(f"Atom {i}: no neighbors found within {search_cutoff} Å")
    
    # Determine if structure is valid
    is_valid = len(isolated_atoms) == 0
    
    validation_info = {
        'total_atoms': len(structure),
        'isolated_atoms': isolated_atoms,
        'isolated_count': len(isolated_atoms),
        'neighbor_counts': neighbor_counts,
        'closest_neighbor_distances': closest_neighbor_distances,}
    
    return is_valid, validation_info


def split_data(df, train_size=0.6, test_size=0.2, val_size=0.2, random_state=None):
    train_df, temp_df = train_test_split(df, test_size=(1 - train_size), random_state=random_state)
    val_ratio_adjusted = val_size / (test_size + val_size)
    val_df, test_df = train_test_split(temp_df, test_size=(1 - val_ratio_adjusted), random_state=random_state)
    return train_df, test_df, val_df

if __name__ == "__main__":
    # Example usage
    path = os.getcwd()
    rootdir = pathlib.Path(path)/'vasp_structure'
    df = pd.read_csv('formation.csv')
    cif_data = []
    invalid_data = []

    print(f"Processing {len(df)} structures...")
    for idx,row in df.iterrows():
        uid = row['cif']
        # if uid != 'Ag2As4S12_10_164.vasp':
        #     continue
        try:
            file = rootdir / uid
            struct = Structure.from_file(file)
            total_atoms = len(struct)
            if total_atoms > 50:
                print(f"Skipping {uid} due to too many atoms: {total_atoms}")
                invalid_data.append({'material_id': uid, 'formation_energy_per_atom': row['formation_energy_per_atom'], 'error': 'Too many atoms'})
                continue
            #crystal_graph = StructureGraph.with_local_env_strategy(struct, CrystalNN)
            # Validate structure connectivity
            is_valid, validation_info = validate_structure_connectivity(struct, max_bond=4.0)
            if not is_valid:
                print(f"Skipping {uid} due to isolated atoms: {validation_info['isolated_count']} isolated atoms")
                invalid_data.append({
                    'material_id': uid,
                    'formation_energy_per_atom': row['formation_energy_per_atom'],
                    'error': f"Isolated atoms: {validation_info['isolated_count']} atoms",
                    'validation_info': validation_info
                })
                continue
             # If structure is valid, process it
            print(f'Processing structure {idx}: {uid} - Valid structure with {total_atoms} atoms')

            cif_writer = CifWriter(struct)
            cif_string = str(cif_writer)
            cif_data.append({'material_id': uid, 'cif': cif_string, 
                             'formation_energy_per_atom': row['formation_energy_per_atom'],'numbers': total_atoms, 
                             'closest_neighbor_distances': validation_info['closest_neighbor_distances']})
        
        except Exception as e:
            print(f"Error processing {uid}: {e}")
            invalid_data.append({'material_id': uid, 'formation_energy_per_atom': row['formation_energy_per_atom'], 
                                 'error': str(e)})
    
    print(f"Valid structures: {len(cif_data)}")
    print(f"Invalid structures: {len(invalid_data)}")
    valid_df = pd.DataFrame(cif_data)
    valid_df.to_csv('c2db_with_cif.csv', index=False)

    train_df, test_df, val_df = split_data(valid_df, train_size=0.8, test_size=0.1, val_size=0.1, random_state=42)
    train_df.to_csv('train.csv', index=False)  # index=False
    test_df.to_csv('test.csv', index=False)
    val_df.to_csv('val.csv', index=False)

    # Save invalid structures for analysis
    if invalid_data:
        invalid_df = pd.DataFrame(invalid_data)
        invalid_df.to_csv('invalid_structures.csv', index=False)
        print(f"Invalid structures saved to 'invalid_structures.csv'")
