import numpy as np
import pandas as pd
import argparse
import sys
import os

print('Dependencies loaded')

def normalize_id(seq_id):
    """Normalize sequence ID by replacing dots with underscores and cleaning whitespace"""
    return seq_id.replace(".", "_").rstrip().split(" ")[0]

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Process KEGG prokaryote data and create KO annotations')
    parser.add_argument('-input', required=True, 
                       help='Input .npz file path (e.g., 13M_mapping.npz)')
    parser.add_argument('--reference', required=True,
                       help='Reference file path (prokaryotes.clean.dat)')
    
    args = parser.parse_args()
    
    # Check if input files exist
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found")
        sys.exit(1)
        
    if not os.path.exists(args.reference):
        print(f"Error: Reference file '{args.reference}' not found")
        sys.exit(1)
    
    # 1. Load the prokaryotes.clean.dat file and create dictionary {ID: KO}
    ko_dict = {}
    with open(args.reference, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                norm_id = normalize_id(parts[0])
                ko_dict[norm_id] = parts[1]
    
    print(f"Loaded {len(ko_dict)} IDs with KO from {args.reference}")
    
    # 2. Load UMAP data and IDs from the input .npz file
    try:
        data = np.load(args.input, allow_pickle=True)
    except FileNotFoundError:
        print(f"Error: {args.input} file not found")
        sys.exit(1)
    
    # data should have 'coordinates' and 'ids'
    coordinates = data['coordinates']
    ids = data['ids']
    print(f"Loaded {coordinates.shape[0]} UMAP coordinates and {len(ids)} IDs")
    
    # 3. Align KO with IDs (handling if IDs come as bytes)
    aligned_ko = [ko_dict.get(id_.decode() if isinstance(id_, bytes) else id_, "NA") for id_ in ids]
    
    # 4. Create DataFrame with everything together
    df = pd.DataFrame({
        "id": [id_.decode() if isinstance(id_, bytes) else id_ for id_ in ids],
        "ko": aligned_ko,
    })
    
    # Add coordinates as separate columns (e.g., if they are 2D or 3D)
    for dim in range(coordinates.shape[1]):
        df[f"UMAP_{dim+1}"] = coordinates[:, dim]
    
    print(df.head())
    
    # 5. Save only KO annotations (in the same order as UMAP)
    ko_array = np.array(aligned_ko)
    np.savetxt("labels_ko.csv", ko_array, fmt="%s")  # Text CSV option
    print("KO labels saved in labels_ko.csv")

if __name__ == "__main__":
    main()