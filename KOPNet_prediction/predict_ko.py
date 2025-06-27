#!/usr/bin/env python
# Load dependencies
import numpy as np
import pandas as pd
import argparse
import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
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
def parse_arguments():
    parser = argparse.ArgumentParser(description="Run ko_ProstT5_NN model on reduced sample embeddings.")
    parser.add_argument("--reduced_sample", type=str, required=True,
                        help="Path to the .npy file containing reduced embeddings.")
    parser.add_argument("--sample_ids", type=str, required=True,
                        help="Path to the .txt file containing sample IDs in the same order as reduced embeddings.")
    parser.add_argument("--model", type=str, default="./model/",
                        help="Path to directory containing ko_nn.py, its dataloader, trained weights and UMAP reducer")
    parser.add_argument("--threshold", default=-1, type=float,
                        help="Threshold above which a KO term is shown as annotation of a sequence")
    parser.add_argument("--k", default=1, type=int,
                         help="Select k most probable KO terms")
    parser.add_argument("--output", required=True,
                        help="File name to store annotations")
    args = parser.parse_args()
    return args

##################################################
#             GET KO-TERMS METADATA              #
##################################################
def get_ko_name(ko):
    if "/" in ko:
        part1, part2 = ko.split("/")
        return f"{ko_names.get(part1, 'NA')}/{ko_names.get(part2, 'NA')}"
    else:
        return ko_names.get(ko, 'NA')

def get_ko_metric(ko, metric):
    if "/" in ko:
        part1, part2 = ko.split("/")
        return f"{ko_metrics.get(part1, 'NA')}/{ko_metrics.get(part2, 'NA')}"
    else:
        return str(ko_metrics.get(ko, {}).get(metric, 'NA'))

##################################################
#             OPTIMIZED PROCESSING               #
##################################################
def process_annotations_vectorized(probabilities, ko_terms, sample_ids, t, k):
    """Vectorized processing of annotations for maximum efficiency"""
    filtered_results = []
    
    if t == -1 and k == 1:
        # Most efficient case: just get argmax using vectorized operations
        log('Using vectorized argmax for single best annotation...')
        max_indices = torch.argmax(probabilities, dim=1)
        max_probs = torch.gather(probabilities, 1, max_indices.unsqueeze(1)).squeeze(1)
        
        for sample_id, max_idx, max_prob in zip(sample_ids, max_indices, max_probs):
            ko_id = ko_terms[max_idx.item()]
            filtered_results.append({
                "sequence_id": sample_id,
                "annotations": [(ko_id, max_prob.item())]
            })
    
    elif k > 1:
        # Get top-k efficiently using torch.topk
        log(f'Using vectorized topk for {k} best annotations...')
        k_actual = min(k, len(ko_terms))
        top_probs, top_indices = torch.topk(probabilities, k=k_actual, dim=1, sorted=True)
        
        for i, sample_id in enumerate(sample_ids):
            selected_annotations = [
                (ko_terms[idx.item()], prob.item()) 
                for idx, prob in zip(top_indices[i], top_probs[i])
            ]
            filtered_results.append({
                "sequence_id": sample_id,
                "annotations": selected_annotations
            })
    
    else:
        # Threshold-based: optimized with vectorized sorting and cumsum
        log(f'Using vectorized threshold processing (t={t})...')
        sorted_probs, sorted_indices = torch.sort(probabilities, dim=1, descending=True)
        cumsum_probs = torch.cumsum(sorted_probs, dim=1)
        
        for i, sample_id in enumerate(sample_ids):
            # Find where cumulative sum exceeds threshold
            exceed_mask = cumsum_probs[i] > t
            if exceed_mask.any():
                stop_idx = exceed_mask.nonzero(as_tuple=True)[0][0].item() + 1
            else:
                stop_idx = len(sorted_probs[i])
            
            selected_annotations = [
                (ko_terms[sorted_indices[i][j].item()], sorted_probs[i][j].item())
                for j in range(stop_idx)
            ]
            
            filtered_results.append({
                "sequence_id": sample_id,
                "annotations": selected_annotations
            })
    
    return filtered_results

def create_ko_caches(filtered_results):
    """Pre-compute all KO metadata for efficient lookup"""
    log('Creating KO metadata caches...')
    
    # Collect all unique KOs
    unique_kos = set()
    for entry in filtered_results:
        for ko, _ in entry["annotations"]:
            unique_kos.add(ko)
    
    log(f'Caching metadata for {len(unique_kos)} unique KO terms...')
    
    # Pre-compute all metadata
    ko_recall_cache = {}
    ko_precision_cache = {}
    ko_occurrence_cache = {}
    ko_name_cache = {}
    
    for ko in unique_kos:
        ko_recall_cache[ko] = get_ko_metric(ko, "recall")
        ko_precision_cache[ko] = get_ko_metric(ko, "precision")
        ko_occurrence_cache[ko] = get_ko_metric(ko, 'model ocurrences')
        ko_name_cache[ko] = get_ko_name(ko)
    
    return ko_recall_cache, ko_precision_cache, ko_occurrence_cache, ko_name_cache

def write_results_optimized(filtered_results, output):
    """Optimized batch writing with pre-computed lookups"""
    log('Pre-computing KO metadata...')
    ko_recall_cache, ko_precision_cache, ko_occurrence_cache, ko_name_cache = create_ko_caches(filtered_results)
    
    log('Writing results to output file...')
    with open(output, "w", newline="") as tsvfile:
        writer = csv.writer(tsvfile, delimiter="\t")
        writer.writerow([
            "sequence_id",
            "KO:probability",
            "recall",
            "precision",
            "occurrences in training dataset",
            "KO name"
        ])
        
        # Process in batches for better I/O performance
        batch_rows = []
        batch_size = 1000
        
        for entry in filtered_results:
            ko_probs = entry["annotations"]
            annotations_str = ";".join(f"{ko}:{prob:.4f}" for ko, prob in ko_probs)
            rec_str = ";".join(ko_recall_cache[ko] for ko, _ in ko_probs)
            prec_str = ";".join(ko_precision_cache[ko] for ko, _ in ko_probs)
            occ_str = ";".join(ko_occurrence_cache[ko] for ko, _ in ko_probs)
            names_str = ";".join(ko_name_cache[ko] for ko, _ in ko_probs)
            
            batch_rows.append([
                entry["sequence_id"],
                annotations_str,
                rec_str,
                prec_str,
                occ_str,
                names_str
            ])
            
            # Write in batches
            if len(batch_rows) >= batch_size:
                writer.writerows(batch_rows)
                batch_rows = []
        
        # Write remaining rows
        if batch_rows:
            writer.writerows(batch_rows)

def optimize_batch_size(total_samples, device):
    """Dynamically determine optimal batch size based on available memory"""
    if device.type == 'cuda':
        # Try to use larger batches for GPU
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        if gpu_memory > 8:
            return min(4096, total_samples)
        elif gpu_memory > 4:
            return min(2048, total_samples)
        else:
            return min(1024, total_samples)
    else:
        # CPU processing
        return min(512, total_samples)

##################################################
#                 MAIN WORKFLOW                 #
##################################################
if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()
    reduced_sample = args.reduced_sample
    sample_ids_file = args.sample_ids
    model_path = args.model
    m_weights = f"{model_path}13M.pt"
    ko_terms_file = f"{model_path}13M_ko.csv"
    t = args.threshold
    k = args.k
    output = args.output

    # Print execution parameters
    print('\n#### EXECUTION PARAMETERS ####')
    print("Input reduced embeddings:", reduced_sample)
    print("Sample IDs file:", sample_ids_file)
    print("Using model in", model_path)
    if t == -1 and k==1:
        print("Annotation criteria: KO with highest probability")
    elif k>1:
        print(f"Annotation criteria: annotate {k} most probable KO terms")
    else:
        print(f"Annotation criteria: include KOs until cumulative probability exceeds threshold t={t}")
    print("Output files:", output, "\n\n")

    # Import the pretrained neural network ko_ProstT5
    sys.path.append(model_path)
    from kos_nn_w import ko_ProstT5_NN
    from kos_nn_w import ResidualLinearBlock

    # Load model files
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f'Using computing device: {device}')
    
    # Enable optimizations
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    model = torch.load(m_weights, weights_only=False, map_location=device)
    model.eval()  # Critical for inference mode
    model.to(device)

    # Load reduced sample embeddings
    log('Loading reduced sample embeddings...')
    sample_emb_reduced = np.load(reduced_sample)
    log(f'Loaded reduced embeddings with shape: {sample_emb_reduced.shape}')

    # Load sample IDs with optimized reading
    log('Loading sample IDs...')
    with open(sample_ids_file, 'r') as f:
        sample_ids = f.read().strip().split('\n')
    log(f'Loaded {len(sample_ids)} sample IDs')

    # Verify that the number of embeddings matches the number of IDs
    if len(sample_ids) != sample_emb_reduced.shape[0]:
        raise ValueError(f"Mismatch: {len(sample_ids)} sample IDs but {sample_emb_reduced.shape[0]} embeddings")

    # Transform to tensor with memory optimization
    log('Converting to tensor...')
    x = torch.tensor(sample_emb_reduced, dtype=torch.float32)
    
    # Clear the numpy array to free memory
    del sample_emb_reduced
    gc.collect()

    # Optimize batch size based on available resources
    batch_size = optimize_batch_size(len(sample_ids), device)
    log(f'Using optimized batch size: {batch_size}')
    
    dataset = TensorDataset(x)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        pin_memory=(device.type == 'cuda'),
        num_workers=0  # Avoid multiprocessing overhead for this use case
    )

    # Predict and normalize results
    log('Making predictions with optimized batching...')
    all_probs = []
    total_batches = len(dataloader)

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch_x = batch[0]
            if device.type == 'cuda':
                batch_x = batch_x.to(device, non_blocking=True)
            else:
                batch_x = batch_x.to(device)
            
            pred = model(batch_x)
            probs = F.softmax(pred, dim=1)
            
            # Move to CPU immediately to free GPU memory
            all_probs.append(probs.cpu())
            
            # Progress reporting
            if batch_idx % max(1, total_batches // 10) == 0:
                progress = (batch_idx + 1) * batch_size
                log(f'Processed {min(progress, len(sample_ids))}/{len(sample_ids)} samples ({batch_idx+1}/{total_batches} batches)')
            
            # Force garbage collection periodically
            if batch_idx % 50 == 0:
                gc.collect()

    # Concatenate all probabilities
    log('Concatenating prediction results...')
    probabilities = torch.cat(all_probs, dim=0)
    
    # Clear intermediate results to free memory
    del all_probs, x
    gc.collect()
    
    log('Predictions completed!')

    # Load KO terms with optimized reading
    log('Loading KO terms...')
    ko_terms_df = pd.read_csv(ko_terms_file, header=None, dtype=str)[0]
    ko_terms = sorted(set(ko_terms_df))
    del ko_terms_df  # Free memory
    
    log('Loading KO metadata...')
    # Load KO terms names
    with open(f'{model_path}ko_names.pkl', 'rb') as f:
        ko_names = pickle.load(f)

    # Load metrics per KO
    with open(f'{model_path}13M_metrics_per_ko.pkl', 'rb') as f:
        ko_metrics = pickle.load(f)

    log(f"Number of KO terms: {len(ko_terms)}")
    log(f"Probability matrix shape: {probabilities.shape}")

    # Process annotations using optimized vectorized approach
    log('Processing annotations with vectorized operations...')
    filtered_results = process_annotations_vectorized(probabilities, ko_terms, sample_ids, t, k)
    
    # Clear probabilities tensor to free memory
    del probabilities
    gc.collect()

    # Write results with optimized batch writing
    write_results_optimized(filtered_results, output)

    log('Process completed successfully!')
    log(f'Results saved to: {output}')
    
    # Final memory cleanup
    gc.collect()
