# Large-scale-Functional-Annotation-of-Prokaryotic-Genes-using-Protein-Language-Models---TFG

### Objectives of this project

The goal of this project is to develop an AI-based bioinformatics framework for efficiently and 
scalably predicting the function of microbial sequences that are too divergent to be annotated with 
traditional, homology-based methods. To achieve this objective, we provide here the scripts used to
develop the KOPNet-pipeline, a new bioinformatics workflow to predict KEGG KO functional terms based 
on the comparison of protein embeddings derived from ProsT5 model available at https://github.com/mheinzinger/ProstT5. 

![KOPNet-pipeline](KOPNet_pipeline.drawio.svg)

This project was developed as Final Bachelor Thesis (Trabajo Fin de Grado, TFG) for the Biotechnology
Degree at Universidad Politécnica de Madrid (UPM).

### Important information about heavy files and hardware resources

Some of the files generated during the training of KOPNet—the neural network underlying KO prediction in this pipeline—are too large to be published in this GitHub repository. 
If you would like to access these files, please send an email to: **laura.cano@alumnos.upm.es**

To run this pipeline you must have access to GPU facilities as well as high memory nodes. In our case we used:
    - Quad CPU Xeon G6230, 80 CPU cores, 1,5 T RAM for UMAP reduction
    - NVIDIA® A100, 32 CPU cores, 125G RAM, 41G vRAM, 2 x SSD 447.1G for KOPNet training
    - NVIDIA® TESLA V100, 40 CPU cores, 188G RAM, 16G vRAM, SSD 222.6G for KOPNet KO predictions

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
<pre> python prepare_labels.py -input UMAP_reduction.npz --reference non_red_input_sequences.dat</pre>

This will generate ```labels_ko.csv``` file suitable for training.

(4) Train KOPNet
<pre> python kos_nn.py --samples UMAP_reduction.npy --labels labels_ko.csv -s -e 3 -l 0.01 -n 100 400 1600 4000 6000 </pre>

Where: 
    ```-s```: shuffles data instances to make neural network learning independent of data order
    ```-e```: number of epochs for training
    ```-l```: learning rate
    ```-n```: each value defines number of neurons in the fully-connected layer of that layer's block

### Predict KO terms using KOPNet-pipeline

(1) Per-protein embedding computing for target proteins (analogously to embedding computing in training)

(2) Target protein reduction within UMAP model
<pre> sample_UMAP_reduction.py --sample sample_embeddings/ --output reduced_sample </pre>

(3) Run KOPNet KO prediction 
<pre> python predict_ko.py --reduced_sample reduced_sample.npy --sample_ids reduced_sample_sample_ids.txt --output KOPNet_annotation.tsv </pre>

