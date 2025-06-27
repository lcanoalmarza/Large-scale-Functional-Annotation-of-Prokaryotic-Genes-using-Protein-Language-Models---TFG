#!/usr/bin/env python

# Compute one UMAP from information stored in different embeddings file

# Import dependencies
import h5py
import glob
import pandas as pd
import umap
import numpy as np
import sys
import argparse
import time
from datetime import datetime
import resource
import gc
from tqdm import tqdm
import joblib

###############################################################################
#                                 TRACK RESOURCES                             #
###############################################################################
t0 = datetime.now()

def log(*args):
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3
    dt = datetime.now() - t0
    print(f'[{dt}] [{round(mem,2)} GB]', *args, flush=True)


###############################################################################
#                                 HANDLE ARGUMENTS                            #
###############################################################################

def parse_arguments():
    """Parse command line arguments with proper flags and help information."""
    parser = argparse.ArgumentParser(
        description="Process metadata and embeddings files for UMAP dimensional reduction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Required arguments
    parser.add_argument("-m", "--metadata", 
                        required=True,
                        help="Path to the embedding's ko annotations (.csv)")
    
    parser.add_argument("-e", "--embeddings", 
                        required=True,
                        help="Path to directory storing embeddings (.h5)")
    
    parser.add_argument("-o", "--output", 
                        required=True,
                        help="Name of output files")
    
    parser.add_argument("-n", "--ncomponents", 
                        type=int, 
                        default=2,
                        help="Number of components for dimensionality reduction")
    
    # Optional arguments example
    parser.add_argument("-v", "--verbose", 
                        action="store_true",
                        help="Enable verbose output")
    
    return parser.parse_args()


###############################################################################
#                    READ METADATA OF ANNOTATED SEQUENCES                     #
###############################################################################
def read_metadata(metadata_file):
    """
    Reads file containing metadata from KEGG Orthology db. 

    Format:
        - One line per protein
        - Tab-separated files: 'sequence_ID', 'ko', 'length', 'description'

    Returns:
        - metadata (df): contains metadata file information
        - selected_IDs (list): IDs in metadata
    """
    log('Reading metadata file...')
    metadata = pd.read_csv(metadata_file,
                    sep='\t', 
                    names=['sequence_ID', 'ko', 'length', 'description'],
                    header=None)
    
    # Optimize by converting to a set for faster lookups
    metadata['sequence_ID'] = metadata['sequence_ID'].str.replace('.', '_', regex=False)
    selected_IDs = set(metadata['sequence_ID'])  # Using set instead of list for O(1) lookups
    
    log(f'Metadata loaded: {len(selected_IDs)} unique sequence IDs')
    return metadata, selected_IDs


###############################################################################
#                 GET EMBEDDINGS FROM DIFFERENT H5 FILES                      #
###############################################################################

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


###############################################################################
#                                   UMAP                                      #
###############################################################################
def build_embeddings_dict(ids, embeddings, selected_ids):
    '''
    Optimized version: Builds a dictionary matching keys: embedding IDs and values: embeddings.
    Considers only proteins with embeddings and metadata annotation
    
    Args:
        ids: List of embedding IDs
        embeddings: NumPy array of embeddings
        selected_ids: Set of IDs to include
        
    Returns:
        Dictionary with filtered embeddings and list of corresponding IDs
    '''
    log('Building dictionary with only annotated embeddings...')
    
    if len(ids) != len(embeddings):
        raise ValueError("Length of ids and embeddings must match.")
    
    # Create mask for faster filtering
    mask = np.zeros(len(ids), dtype=bool)
    
    # Build mask of IDs to keep
    for i, emb_id in enumerate(ids):
        if emb_id in selected_ids:  # Using set membership test (O(1))
            mask[i] = True
    
    # Get the indices where mask is True
    indices = np.where(mask)[0]
    
    # Create dictionary and filtered arrays in one go
    filtered_ids = [ids[i] for i in indices]
    filtered_embeddings = embeddings[indices]  # Fast NumPy indexing
    
    # Build dictionary (if needed)
    emb_dict = {filtered_ids[i]: filtered_embeddings[i] for i in range(len(filtered_ids))}
    
    log(f'Built dictionary with {len(emb_dict)} annotated embeddings')
    
    return emb_dict, filtered_ids, filtered_embeddings


def save_results(metadata, annotated_ids, emb_umap, output_file):
    """
    Save UMAP results and correspondig KO annotation of each sequence

    Input:
        - metadata (df)
        - annotated_ids (list): IDs of the filtered embeddings
        - emb_umap (numpy array): umap coordinates
        - output_file (str): base name for output files
    
    Return: two files whose name is specified by -output argument
        - output.csv: 1-col file storing ko annotation of each embedding
        - ouput.npy: numpy file storing umap coordinates
    """
    log('Saving results...')
    
    # Convert IDs to list if it's a set
    if isinstance(annotated_ids, set):
        annotated_ids = list(annotated_ids)
    
    # Filter metadata efficiently
    filtered_metadata = metadata[metadata['sequence_ID'].isin(annotated_ids)]
    
    # Save KO annotations
    with open(f'{output_file}.csv', "w") as f:
        f.write("\n".join(filtered_metadata.iloc[:, 1].astype(str)) + "\n")

    # Save UMAP result to .npy file
    np.save(f'{output_file}.npy', emb_umap)
    
    # Also save a mapping file for future reference
    mapping_data = {
        'ids': annotated_ids,
        'coordinates': emb_umap
    }
    np.savez(f'{output_file}_mapping.npz', **mapping_data)
    
    log('\n=== Results ===')
    log(f'Embeddings KO annotation saved to {output_file}.csv')
    log(f'UMAP results saved to {output_file}.npy')
    log(f'ID-to-coordinate mapping saved to {output_file}_mapping.npz')


#####################################################################
#                           MAIN WORKFLOW                           # 
#####################################################################

if __name__ == "__main__":
    try:
        # Start time is already initialized at the top of the script
        
        # Access arguments
        args = parse_arguments()  
        metadata_file = args.metadata
        embeddings_path = args.embeddings
        output_file = args.output
        ncomp = args.ncomponents
        verbose = args.verbose
        
        # Read metadata
        metadata, selected_IDs = read_metadata(metadata_file)
        
        # Load embeddings
        log('Loading embeddings...')
        embeddings, embeddings_ids = load_embeddings_for_umap(embeddings_path)
        
        # Build embeddings dict for UMAP input
        log('Building dictionary with only annotated embeddings...')
        emb_dict, filtered_ids, filtered_embeddings = build_embeddings_dict(embeddings_ids, embeddings, selected_IDs)
        
        # Clear original embeddings to free memory
        log('Clearing original embeddings from memory...')
        del embeddings
        gc.collect()
        
        # Perform UMAP
        log('Performing UMAP dimension reduction...')
        reducer = umap.UMAP(
            n_components=ncomp,
            metric='euclidean',
            verbose=verbose   
        )
        
        emb_umap = reducer.fit_transform(filtered_embeddings)
        joblib.dump(reducer, f'{output_file}_umap_model.pkl')
	#log(f'UMAP completed: reduced to {ncomp} dimensions')
        
        # Save results
        save_results(metadata, filtered_ids, emb_umap, output_file)
        
        # Show total execution time
        total_time = datetime.now() - t0
        log(f'Total execution time: {total_time}')
        log('End of script')
        
    except Exception as e:
        log(f'ERROR: {str(e)}')
        import traceback
        log(traceback.format_exc())
        sys.exit(1)
