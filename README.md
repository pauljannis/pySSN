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


