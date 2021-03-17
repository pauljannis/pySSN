#Calculation of SSNs
#Author: Paul Jannis Zurek, pjz26@cam.ac.uk
#17/03/2021 v 1.0
# Calculates SSW alignments or Levensthein distance matrices
# Calculates tSNEs or UMAPs

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from skbio.alignment import StripedSmithWaterman
import multiprocessing
import itertools
import os
import numpy as np
from matplotlib import pyplot as plt
import argparse
from tqdm import tqdm
from Levenshtein import distance as LevenDist
import pandas as pd
import seaborn as sns
import umap
sns.set(style='white', context='notebook', rc={'figure.figsize':(6,6)})


parser = argparse.ArgumentParser(description="""Calculation of SSNs.
                                 Author: Paul Zurek (pjz26@cam.ac.uk).
                                 Version 1.0""")
parser.add_argument('-T', '--threads', type=int, default=0, help='Number of threads to execute in parallel. Defaults to CPU count.')
parser.add_argument('-v', '--version', action='version', version='1.0')
parser.add_argument('-i', '--input', type=str, help="""Please provide one of the following input files:\n 
                                                      FASTA: List of records to calculate distance matrix from.\n
                                                      CSV: Distance matrix checkpoint.
                                                      NPY: UMAP embeddings checkpoint.
                                                      """)
parser.add_argument('--metric', type=str, choices=['Levenshtein','Alignment'], help='Metic used for distance calculation: Levenshtein or Alignment. Use Levenshtein for close sequences and Alignment for less homologous sequences.')
parser.add_argument('--grouping', type=str, help='TXT file as a list group information. Can be used for coloring the SSN.')


#Parse arguments
args = parser.parse_args()
threads = args.threads
metric = args.metric
grouping = args.grouping

if threads == 0:
    threads = multiprocessing.cpu_count()

input_file = args.input
name = ".".join(input_file.split(".")[:-1])
ftype = input_file.split(".")[-1].lower()


##############
# DEFINITIONS

def all_scores(query_nr, record):   #Calculates all scores for distance_matrix
    global query_lst
    aln = query_lst[query_nr](record)
    #Alignment score
    score = aln.optimal_alignment_score
    #Calculate query coverage
    query_length = len(aln.query_sequence)
    aligned_query_length = aln.query_end - aln.query_begin + 1
    coverage = aligned_query_length / query_length
    #Calculate %identity
    aln_query = aln.aligned_query_sequence
    aln_target = aln.aligned_target_sequence
    aln_length = len(aln_query)
    same_aa = sum(e1 == e2 for e1, e2 in zip(aln_query, aln_target))
    ident = same_aa / aln_length
    return [score, ident, coverage]


def distance_matrix(records):   #Calculates sparse distance matrix, based on %mismatches after SSW alignment
    rec_lst = records.copy()
    global query_lst
    query_lst = [StripedSmithWaterman(rec) for rec in rec_lst]
    pool = multiprocessing.Pool(processes=threads)
    score_lst_lst = []
    N_rec = len(rec_lst)
    for i in tqdm(range(N_rec)):
        rec_lst.pop(0)
        score_lst = pool.starmap(all_scores, zip(itertools.repeat(i), rec_lst))
        identlst = [1-elem[1] for elem in score_lst] #1-identity = dissimilarity
        score_lst_lst.append(identlst)
    pool.close()
    print('finished generating the alignment score matrix')
    return score_lst_lst

def convert_DM_tofull(sparse_matrix):   #Converts sparse distance matrix to full matrix
    elem = len(sparse_matrix)
    full_matrix = [[0 for i in range(elem)] for j in range(elem)]
    for i in range(len(sparse_matrix)):
        for j in range(len(sparse_matrix[i])):
            full_matrix[i][j+1+i] = sparse_matrix[i][j]
            full_matrix[j+1+i][i] = sparse_matrix[i][j]
    return full_matrix

def LevDistMat(records):
    rec_lst = records.copy()
    score_lst_lst = []
    N_rec = len(rec_lst)
    pool = multiprocessing.Pool(processes=threads)
    max_dist = 0
    for i in tqdm(range(N_rec)):
        ref = rec_lst.pop(0)
        score_lst = pool.starmap(LevenDist, zip(itertools.repeat(ref), rec_lst))
        if max(score_lst) > max_dist:
            max_dist = max(score_lst)
        score_lst_lst.append(score_lst)
    pool.close()
    #Scale 0-1
    for i in range(len(score_lst_lst)):
        for j in range(len(score_lst_lst[i])):
            score_lst_lst[i][j] = score_lst_lst[i][j] / max_dist
    print('finished generating the alignment score matrix')
    return score_lst_lst


def calc_UMAP(DM):
    reducer = umap.UMAP(metric="precomputed")
    embedding = reducer.fit_transform(DM)
    np.save(f"{name}-UMAPcheckpoint.npy", embedding, allow_pickle=True)
    return embedding

def plot_scatter_colored(embedding):
    colors = ["k" for _ in range(len(embedding[:,1]))]
    if grouping is not None:
        with open(grouping, 'r') as f:
            lines = f.readlines()
        if len(lines) != len(colors):
            raise ValueError("Number of groupings does not match number of embedded sequences.")
        colors = [l.strip("\n") for l in lines]
        try:
            colors = [int(c) for c in colors]
        except Exception:
            pass
    print(colors)
    plt.figure()
    plt.scatter(embedding[:,0], embedding[:,1], c=colors, s=1)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f"{name}-UMAP.png", bbox_inches="tight", dpi=600)


if ftype == "fasta":
    if metric is None:
        raise NameError("Please specify a distance metric")
    records = [str(rec.seq) for rec in SeqIO.parse(input_file, "fasta")]
    print(f"{len(records)} records loaded.")
    print(f"Calculating distance matrix via {metric}:")
    if metric == 'Alignment':
        DM = distance_matrix(records)
        fullDM = convert_DM_tofull(DM)
        fullDM = pd.DataFrame(fullDM)
        fullDM.to_csv(f"{name}-DMcheckpoint.csv") 
    elif metric == 'Levenshtein':
        DM = distance_matrix(records)
        fullDM = convert_DM_tofull(DM)
        fullDM = pd.DataFrame(fullDM)
        fullDM.to_csv(f"{name}-DMcheckpoint.csv") 
    #Get and plot UMAP
    embeddings = calc_UMAP(fullDM)
    plot_scatter_colored(embeddings)
elif ftype == "csv":
    print("Loading precomputed distance matrix")
    fullDM = pd.read_csv(input_file, index_col=0)
    print("calculating UMAP embeddings")
    embeddings = calc_UMAP(fullDM)
    plot_scatter_colored(embeddings)
elif ftype == "npy":
    embeddings = np.load(input_file, allow_pickle=True)
    plot_scatter_colored(embeddings)


