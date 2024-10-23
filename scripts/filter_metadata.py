import os
import pickle

import pandas as pd
from tqdm import tqdm

from matplotlib import pyplot as plt
import seaborn as sns


def postprocess_data(path_to_csv, output_dir):
    data = pd.read_csv(path_to_csv)
    os.makedirs(output_dir, exist_ok=True)
    name_to_data = {}
    visited_pdb_id = []

    lengths = data['modeled_seq_len']
    # plot to histogram and save to file
    save_to = os.path.join(output_dir, 'modeled_seq_len_hist.png')
    if not os.path.exists(save_to):
        plt.figure(figsize=(18, 6))
        sns.histplot(lengths, bins=100)
        plt.title('Distribution of modeled sequence lengths')
        plt.xlabel('Sequence length')
        plt.ylabel('Count')
        plt.savefig(save_to)
        plt.close()

    print(f">>> Total number of chains: {len(data)} initialily")

    data = data[data['modeled_seq_len'] <= 1000]

    data = data[data['modeled_seq_len'] >= 10]

    print(f">>> Total number of chains: {len(data)} after length selection [10, 1000]")

    data = data[data['resolution'] <= 5.0]
    data = data[data['resolution'] >= 0.01]

    print(f">>> Total number of chains: {len(data)} after resolution selection [0.01, 5.0]")
    
    # for idx, row in tqdm(data.iterrows(), total=len(data)):
    #     pkl_path = row['processed_path']
    #     pdb_id, chain_id = row['pdb_chain_name'].split('_')
    #     # if homomer, add any chain (only once)
    #     # if heteromer, add all chains
    #     qc = row['quaternary_category']
    #     if qc == 'homomer':
    #         if pdb_id in visited_pdb_id:
    #             continue
    #     elif qc == 'heteromer':
    #         pass
    #     else:
    #         raise ValueError(f"Unknown quaternary category: {qc}")  # actually no else

    #     # with open(pkl_path, 'rb') as pf:
    #     name_to_data[row['pdb_chain_name']] = 1 #pickle.load(pf)
    #     visited_pdb_id.append(pdb_id)

    homomer = data[data['quaternary_category'] == 'homomer']
    heteromer = data[data['quaternary_category'] == 'heteromer']
    homomer_gt1 = homomer[homomer['num_chains'] > 1]
    homomer_1 = homomer[homomer['num_chains'] == 1]

    def custom_agg(group):
        # Take all rows where 
        part1 = group[group['quaternary_category'] == 'heteromer']
        # Take the first row where 
        part2 = group[group['quaternary_category'] == 'homomer'].head(1)
        return pd.concat([part1, part2])

    # Group by column 'A' and apply custom aggregation function
    data = pd.concat([homomer_1, homomer_gt1.groupby('pdb_name').head(1), heteromer]).reset_index(drop=True)

    print(f">>> Total number of chains: {len(data)} after homomer/heteromer selection ")

    allowable_oligo = [
        ','.join(['monomeric'] * i) for i in range(1, 100)
    ]
    data = data[data['oligomeric_detail'].isin(allowable_oligo)]
    print(f">>> Total number of chains: {len(data)} after oligomeric_detail selection")

    return data
    
    
if __name__ == '__main__':
    path_to_dataset = ".../metadata.csv"
    output_dir = os.path.join(os.path.dirname(os.path.dirname(path_to_dataset)), 'processed_chains_asset')
    data = postprocess_data(path_to_dataset, output_dir=output_dir)
    save_to = os.path.join(output_dir, f'filtered_{len(data)}.csv')
    data.to_csv(save_to, index=False)

