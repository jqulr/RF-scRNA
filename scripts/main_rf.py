#!/usr/bin/env python3
"""
main_rf.py

Random Forest classifier trained on scRNA‑seq data to classify cell types.

Takes in gene expression matrix from scRNA-seq as training data, runs RF on it and evaluates performance 
using test dataset based on f1-score. 
"""
import os
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report
import scanpy as sc
from anndata import AnnData
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Random Forest on scRNA-seq with GridSearchCV")
    parser.add_argument("--series_mtx", required=True, help="Path to series matrix with cell type labels")
    parser.add_argument("--exp_mtx", required=True, help="Path to gene expression matrix")

    return parser.parse_args()

# runs basic RF 
def run_RF(exp_mtx, series_matrix):
    adata = make_adata(exp_mtx, series_matrix)
    X_tr, x_ts, Y_tr, y_ts = split_and_normalize(adata)

    print("starting training RF")
    train_RF(X_tr, x_ts, Y_tr, y_ts)


    

# Extracts a label vector of length n_samples from a series matrix, and the length of the label vector
def extract_labels(file_path):
    
    labels = []
    with open(file_path, "r") as fh:
        for line in fh:
            if not line.startswith("!Sample_characteristics_ch1"):
                continue
            
            parts = line.rstrip("\n").split("\t")[1:]
            first_key = parts[0].strip('"').split(": ", 1)[0].lower()

            if not first_key.startswith("cell"):
                # not the cell type metadata
                continue
            
            for entries in parts:
                cells = line.rstrip('\n').split('\t')
                cells = cells[1:]
                for entries in cells:
                    # splitting on first encounter of ": ", then take the cell type only
                    entry = entries.strip('"').split(": ", 1)[1]
                    labels.append(entry)
                break

    length = len(labels)

    return labels, length

# make an annData object to store the gene expression matrix and cell labels
# takes in the exp-mtx (genes x samples)and file path to series matrix, check that the length matches, add cell types into annData object
def make_adata(exp_mtx, file_path):
    labels, n_labels = extract_labels(file_path)
    
    # add a check for file, if .txt, convert to .csv first 
    root, ext = os.path.splitext(exp_mtx)
    if ext.lower() == ".txt":
        new_path = root + ".csv"
        df = pd.read_csv(exp_mtx, sep="\t", header=0)
        df.to_csv(new_path, index=False)
        exp_mtx = new_path

    adata = sc.read_csv(exp_mtx, first_column_names=True).T


    assert n_labels == adata.n_obs, f"{adata.n_obs=} cells vs. {n_labels=} labels"
    
    adata.obs["cell_type"] = labels

    # outpath = "/Users/junequ/RF-scRNA/expr.h5ad"
    # adata.write_h5ad(outpath)

    return adata

# split by 20/80
def split_and_normalize(adata, label_key="cell_type", test_size=0.2, random_state=42):
    
    ad = adata.copy()

    # filter out data without less than 3 cells
    sc.pp.filter_genes(ad, min_cells=3)

    # normalize
    sc.pp.normalize_total(ad)
    sc.pp.log1p(ad)

    # extract feature matrix and label vector
    X = ad.X
    y = ad.obs[label_key].values

    # split 20/80
    X_train, X_test, y_train, y_test = train_test_split(
      X, y,
      test_size=test_size,
      stratify=y,
      random_state=random_state
    )

    return X_train, X_test, y_train, y_test


# basic training of RF
def train_RF(X_train, x_test, y_train, y_test, n_jobs=5, n_estimators=100):
    sel = RandomForestClassifier(n_jobs=5, n_estimators=100)
    sel.fit(X_train, y_train)

    # evaluation on test set
    y_pred = sel.predict(x_test)
    print(classification_report(y_test, y_pred, digits=4))
    
      
    
def main():
    args = parse_args()
    run_RF(args.exp_mtx, args.series_mtx)

    
if __name__ == "__main__":
    main()