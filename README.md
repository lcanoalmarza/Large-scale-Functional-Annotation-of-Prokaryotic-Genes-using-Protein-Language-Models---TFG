# Large-scale-Functional-Annotation-of-Prokaryotic-Genes-using-Protein-Language-Models---TFG

### Objectives of this project

The goal of this project is to develop an AI-based bioinformatics framework for efficiently and 
scalably predicting the function of microbial sequences that are too divergent to be annotated with 
traditional, homology-based methods. To achieve this objective, we provide here the scripts used to
develop the KOPNet-pipeline, a new bioinformatics workflow to predict KEGG KO functional terms based 
on the comparison of protein embeddings derived from ProsT5 model available at https://github.com/mheinzinger/ProstT5. 

This project was developed as Final Bachelor Thesis (Trabajo Fin de Grado, TFG) for the Biotechnology
Degree at Universidad Politécnica de Madrid (UPM).

### Important information about heavy files

Some of the files generated during the training of KOPNet—the neural network underlying KO prediction in this pipeline—are too large to be published in this GitHub repository. 

If you would like to access these files, please send an email to: **laura.cano@alumnos.upm.es**

### KOPNet training

(1) Remove proteins with 100% identity using CD-HIT

<pre> cd-hit -i input_sequences.fa -o non_red_sequences -c 1 -d 0 -n 5 -M 160000 -T 8 </pre>

(2) Extract metadata of non-redundant sequences
<pre> python extract_metadata.py non_red_sequences.fa input_sequences.dat non_red_input_sequences.dat </pre>

(3) Per-protein embedding computing
All per-protein embeddings were derived using the emebdder tool provided by ProtTrans in: https://github.com/agemagician/ProtTrans/blob/master/Embedding/prott5_embedder.py
Note that, to use ProstT5 model instead of ProtT5, transformer name should be appropriatelly updated.

(4) Compute UMAP for training dataset derived from KEGG database
<pre> python efficient_umap.py -m non_red_input_sequences.dat -e embeddings/ -o UMAP_reduction -n 40 -v </pre>

Where ```embeddings/``` is the folder storing per-protein embeddings computed by ProstT5

(5) Propper formatting of labels 


(4) Train KOPNet
<pre> python kos_nn.py --samples --labels -s -e 3 -l 0.01 -n 100 400 1600 4000 6000 </pre>

### Predict KO terms using KOPNet-pipeline


