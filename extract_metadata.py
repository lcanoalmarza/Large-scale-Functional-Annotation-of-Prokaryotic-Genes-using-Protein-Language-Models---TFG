from Bio import SeqIO
import sys

def extract_metadata(fasta_file, metadata_file, output_file):
    """
    Extract metadata lines where the first column matches a protein ID in the FASTA file.
    Optimized for performance on large files.
    """
    # Build a set of protein IDs from the FASTA file
    protein_ids = set()
    with open(fasta_file, "r") as fasta_handle:
        for protein in SeqIO.parse(fasta_handle, "fasta"):
            protein_ids.add(protein.description.split()[0])

    # Stream through metadata file and write matches directly to output
    match_count = 0
    with open(metadata_file, 'r') as meta_handle, open(output_file, 'w') as out_handle:
        for line in meta_handle:
            line = line.strip()
            if line.split('\t')[0] in protein_ids:
                out_handle.write(line + '\n')
                match_count += 1

    # Reporting
    print('\n-------------------- Results ----------------------')
    if match_count > 0:
        print(f'\nMetadata extracted for {match_count} sequence(s)')
        print(f'Metadata written to: {output_file}\n')
    else:
        print(f'No matching sequences found between {fasta_file} and {metadata_file}\n')

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <input_fasta> <metadata_file> <output_file>")
        sys.exit(1)

    input_fasta = sys.argv[1]
    metadata_file = sys.argv[2]
    output_file = sys.argv[3]

    print('\n---------------- Script parameters ----------------')
    print(f'\nInput FASTA: {input_fasta}')
    print(f'Input metadata: {metadata_file}')
    print(f'Output file: {output_file}\n')

    extract_metadata(input_fasta, metadata_file, output_file)

