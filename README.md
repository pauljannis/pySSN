# pySSN
Calculation of sequence similarity networks

## Requirements
Python 3.7 with
biopython, scikit-bio, python-Levenshtein, umap, scikit-learn, seaborn, pandas, numpy, matplotlib, tqdm

## Usage
```
pySSN.py [-h] [-T THREADS] [-v] [-i INPUT]
                [--metric {Levenshtein,Alignment}] [--grouping GROUPING]
                [--reducer {UMAP,tSNE}]

Calculation of SSNs. Author: Paul Zurek (pjz26@cam.ac.uk). Version 1.0

optional arguments:
  -h, --help            show this help message and exit
  -T THREADS, --threads THREADS
                        Number of threads to execute in parallel. Defaults to
                        CPU count.
  -v, --version         show program's version number and exit
  -i INPUT, --input INPUT
                        Please provide one of the following input files:
                        FASTA: List of records to calculate distance matrix
                        from. CSV: Distance matrix checkpoint. NPY: Reducer
                        embeddings checkpoint.
  --metric {Levenshtein,Alignment}
                        Metic used for distance calculation: Levenshtein or
                        Alignment. Use Levenshtein for close sequences and
                        Alignment for less homologous sequences.
  --grouping GROUPING   TXT file for group information. Can be used to color
                        the SSN.
  --reducer {UMAP,tSNE}
                        Choice of dimensionality reduction method: UMAP or
                        tSNE. Defaults to UMAP.
```

## Example 1
We have a set of sequence variants with point mutations, such as those generated by three rounds of directed evolution in Zurek _et al_, [2020](https://www.nature.com/articles/s41467-020-19687-9). 

As these sequences (see _evolved_variants.fasta_) are pretty close, sequence wise, we'll calculate the distance matrix via Levenshtein distances, as follows:

`python pySSN.py -i evolved_variants.fasta --metric Levenshtein --reducer UMAP`

Our output are checkpoint files as well as the UMAP representation of the sequence space. We might want to try the tSNE representation to find a prettier view, calculated on the automatically generated distance matric checkpoint:

`python pySSN.py -i evolved_variants-LevenshteinDM-checkpoint.csv --reducer tSNE`

This one nicely shows the clusters emerging over the course of directed evolution. But why not add some color? We can assign color values to each point by passing a grouping file. We know from which of the three rounds of directed evolution each sequence originates and can assign a color accordingly in a simple txt file (see _evolved_variants_groupings.txt_). We'll use the tSNE checkpoint to not have to calculate everything again:

`python pySSN.py -i evolved_variants-tSNEcheckpoint.npy --grouping evolved_variants_groupings.txt`


<img src="https://raw.githubusercontent.com/pauljannis/pySSN/main/Example1/evolved_variants-tSNE.png" align="middle" alt="pySSN tSNE of evolved variants" width="485" height="482"/>


## Example 2
We have a protein sequence of interest (in this case a phenylalanine dehydrogenase ID Q59771) and want to generate a sequence similarity network. First, we look for homologous sequences by blasting the UniRef90 database and taking a total of 250 hits. Those are our input as _PheDH_250_UniRef90.fasta_. We want to align those sequences, because they could be quite dissimilar. Also, a UMAP might look pretty. Thus:

`python pySSN.py -i PheDH_250_UniRef90.fasta --metric Alignment --reducer UMAP`

We get a nice UMAP representation of the sequence space and might want to look further into the clusters by coloring interesting clades, as exemplified in Example 1.


<img src="https://raw.githubusercontent.com/pauljannis/pySSN/main/Example2/PheDH_250_UniRef90-UMAP.png" align="middle" alt="pySSN UMAP of PheDH sequence space" width="485" height="482"/>
