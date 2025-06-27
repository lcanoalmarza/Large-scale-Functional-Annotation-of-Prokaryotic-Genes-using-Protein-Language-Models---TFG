#!/usr/bin/env python

# Load dependencies
import numpy as np
import pandas as pd
import argparse
import h5py
import torch
import torch.nn.functional as F
import glob
import time
from datetime import datetime
import resource
import gc
from tqdm import tqdm
import joblib
import csv
import pickle
import sys
import os


###############################################################################
#                                 TRACK RESOURCES                             #
###############################################################################
t0 = datetime.now()

def log(*args):
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3
    dt = datetime.now() - t0
    print(f'[{dt}] [{round(mem,2)} GB]', *args, flush=True)


##################################################
#                PARSE ARGUMENTS                 #
##################################################
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run ko_ProstT5_NN model on sample embeddings with optional dimensionality reduction.")
    parser.add_argument("--sample", type=str, required=True,
                        help="Path to the .h5 file containing embeddings generated with the ProstT5 model.")
    parser.add_argument("--model", type=str, default = "./model/",
         help="Path to directory containing ko_nn.py, its dataloader, trained weights and UMAP reducer")
    parser.add_argument("--output", required=True,
                       help="File name to store annotations")
    args = parser.parse_args()
    return args


##################################################
#              LOAD & READ SAMPLES               #
##################################################

def load_embeddings_for_umap(embeddings_path):
    """
    Loads all embeddings contained in embeddings_path

    Args:
        embeddings_path: path to directory containing .h5 files with protein embeddings

    Returns:
        Tuple with (embeddings_array, ids)
    """
    # Create external links
    table_link = 'table_links.h5'

    h5_files = sorted(glob.glob(f'{embeddings_path}/*.h5'))
    log(f'Found {len(h5_files)} H5 files in {embeddings_path}')

    with h5py.File(table_link, mode='w') as h5fw:
        for i, h5name in enumerate(h5_files, 1):
            h5fw[f'link{i}'] = h5py.ExternalLink(h5name, '/')

    log(f'Linked h5 files: {len(h5_files)}')

    # First round: get embedding dimension and number to estimate needed memory
    embedding_dim = None
    total_embeddings = 0

    log('Counting embeddings and getting dimensions...')
    with h5py.File('table_links.h5', 'r') as myfile:
        for group_name in tqdm(myfile.keys(), desc="Scanning files"):
            if isinstance(myfile[group_name], h5py.Group):
                group = myfile[group_name]

                for dataset_name in group.keys():
                    # Read embedding's shape
                    if embedding_dim is None:
                        embedding_shape = group[dataset_name].shape
                        embedding_dim = embedding_shape[0]

                    total_embeddings += 1

    log(f'Embeddings dimension: {embedding_dim}')
    log(f'Number of embeddings to load: {total_embeddings}')

        # Preallocate array to save embeddings
    # Using float32 instead of float64 to save memory if precision is sufficient
    all_embeddings = np.zeros((total_embeddings, embedding_dim), dtype=np.float32)
    all_ids = []

    # Second round: load embeddings
    log('Loading embeddings...')
    current_index = 0

    with h5py.File('table_links.h5', 'r') as myfile:
        # Process groups in sorted order for reproducibility
        for group_name in tqdm(sorted(myfile.keys()), desc="Loading embeddings"):
            if isinstance(myfile[group_name], h5py.Group):
                group = myfile[group_name]

                # Prepare dataset list first to minimize IO operations
                datasets_to_load = list(group.keys())

                # Load embeddings of current group
                for dataset_name in datasets_to_load:
                    dataset_id = dataset_name.replace(".", "_").rstrip().split(" ")[0]
                    all_embeddings[current_index] = group[dataset_name][()]
                    all_ids.append(dataset_id)
                    current_index += 1

                # Release memory after processing each group
                gc.collect()

    log(f'Loaded embeddings: {current_index}/{total_embeddings}')

    return all_embeddings, all_ids

##################################################
#                 MAIN WORKFLOW                 #
##################################################
if __name__ == "__main__":

    # Parse arguments
    args = parse_arguments()
    sample=args.sample
    model_path=args.model
    reducer = glob.glob(os.path.join(model_path, '*umap_model.pkl'))
    output=args.output

    # Print execution parameters
    print('\n#### EXECUTION PARAMETERS ####')
    print("Input embeddings:", sample)
    print("Using reducer: ",reducer)
    print("Output files: ", output, "\n\n")

    # Load sample embeddings
    sample_emb, sample_ids = load_embeddings_for_umap(sample)

    # Perform umap
    log('Loading reducer...')
    reducer_model = joblib.load(reducer)
    log('Done!')
    log('Performing UMAP')
    sample_emb_reduced = reducer_model.transform(sample_emb)
    log('Done!')
    
    # Save reduced embeddings to .npy file
    log('Saving reduced embeddings...')
    output_npy = f"{output}_reduced_embeddings.npy"
    np.save(output_npy, sample_emb_reduced)
    log(f'Reduced embeddings saved to: {output_npy}')
    
    # Optionally, also save the IDs for reference
    output_ids = f"{output}_sample_ids.txt"
    with open(output_ids, 'w') as f:
        for sample_id in sample_ids:
            f.write(f"{sample_id}\n")
    log(f'Sample IDs saved to: {output_ids}')
    
    log('Process completed successfully!')
