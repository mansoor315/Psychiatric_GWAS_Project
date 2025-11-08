#!/usr/bin/env Rscript

# -------------------------------------------------------------------
# RNA-seq Read Count Normalization using DESeq2
#
# Purpose: This script loads raw gene counts from a specified file,
#          performs sample metadata recreation, filtering, and
#          DESeq2 normalization.
#
# It exports two primary files:
#   1. normalized_counts_for_ml.csv: A matrix of normalized counts
#      with gene IDs, ready for use as input in the Python ML pipeline.
#      (Saved to data/)
#   2. DESeq2_annotated_results.csv: The full DESeq2 differential
#      expression results, annotated with gene names via biomaRt.
#      (Saved to results_R/)
# -------------------------------------------------------------------


# --- 1. Package Management ---
# Checks for, installs (if necessary), and loads all required packages.

message("Loading required R packages...")

# List of required packages
packages <- c("DESeq2", "ggplot2", "pheatmap", "readr", "biomaRt", "here")

# Check, install, and load packages
for (pkg in packages) {
  if (!require(pkg, character.only = TRUE)) {
    message(paste("Installing", pkg, "..."))
    if (pkg %in% c("DESeq2", "biomaRt")) {
      if (!requireNamespace("BiocManager", quietly = TRUE))
        install.packages("BiocManager")
      BiocManager::install(pkg)
    } else {
      install.packages(pkg)
    }
    library(pkg, character.only = TRUE)
  }
}

# Set global options for R
options(stringsAsFactors = FALSE)


# --- 2. Configuration ---
# Define all inputs, outputs, and parameters in one place.

message("Setting up configuration...")

# --- Input File ---
# Assumes the raw counts file is in the 'data/' subfolder
COUNTS_FILE <- "GSE243841_raw_counts_all.csv"

# --- Output Files ---
# Main output: Normalized counts for Python/ML (will be saved in 'data/')
NORM_COUNTS_OUT <- "normalized_counts_for_ml.csv"
# Secondary output: Full DESeq2 results (will be saved in 'results_R/')
ANNOT_RESULTS_OUT <- "DESeq2_annotated_results.csv"

# --- Parameters ---
# Number of leading columns that are annotations (not samples)
N_ANNOT_COLS <- 6
# Column name in the counts file that holds Ensembl IDs
GENE_ID_COL <- "Geneid"

# --- Output Directory ---
# Create the R-specific results directory using 'here'
# 'here::here()' builds a path from the project's root folder
R_RESULTS_DIR <- here::here("results_R")
dir.create(R_RESULTS_DIR, showWarnings = FALSE)


# --- 3. Data Loading & Cleaning ---

message("Loading and cleaning count data...")

# Build the full path to the counts file
counts_file_path <- here::here("data", COUNTS_FILE)

# Check if file exists
if (!file.exists(counts_file_path)) {
  stop(paste("Error: Cannot find counts file at:", counts_file_path,
             "\nPlease check the 'data' folder and 'COUNTS_FILE' variable."))
}
counts_data <- read.csv(counts_file_path)

# Store Gene IDs *before* any manipulation
if (!GENE_ID_COL %in% colnames(counts_data)) {
  stop(paste("Error: Cannot find Gene ID column:", GENE_ID_COL,
             "\nPlease check the 'GENE_ID_COL' variable."))
}
gene_ids <- counts_data[[GENE_ID_COL]]

# Clean sample names (logic specific to this dataset)
# This finds and replaces patterns in the column names
original_colnames <- colnames(counts_data)
cleaned_names <- gsub(".*BAM[./]*", "", original_colnames)
cleaned_names <- gsub("_sorted.bam", "", cleaned_names)
colnames(counts_data) <- cleaned_names

# Separate count matrix from annotations
# Assumes first N_ANNOT_COLS are annotations
count_matrix <- counts_data[, -(1:N_ANNOT_COLS)]

# Get the cleaned sample names (which are the columns of count_matrix)
cleaned_sample_names <- colnames(count_matrix)


# --- 4. Recreate Metadata ---

message("Recreating sample metadata...")

# Recreate metadata based on sample name prefixes
condition <- ifelse(grepl("^YD", cleaned_sample_names), "control",
                    ifelse(grepl("^ow", cleaned_sample_names), "schizophrenia", NA))

if (any(is.na(condition))) {
  warning("Some samples were not matched to a condition! Check prefixes.")
}

metadata <- data.frame(row.names = cleaned_sample_names,
                       condition = factor(condition, levels = c("control", "schizophrenia")))


# --- 5. Filtering & Pre-processing ---

message("Filtering low-count genes...")

# Sanity check: ensure data and metadata align perfectly
stopifnot(all(colnames(count_matrix) == rownames(metadata)))

# Filter genes: keep genes with at least 10 counts in 2 or more samples
keep_genes <- rowSums(count_matrix >= 10) >= 2
filtered_counts <- count_matrix[keep_genes, ]

# Get the corresponding Gene IDs for the filtered rows
gene_ids_filtered <- gene_ids[keep_genes]

# Set rownames of the filtered matrix to be the Gene IDs
rownames(filtered_counts) <- gene_ids_filtered


# --- 6. DESeq2 Normalization ---

message("Running DESeq2 normalization...")

# Create DESeqDataSet object
# DESeq2 requires integers, so we round the counts
dds <- DESeqDataSetFromMatrix(countData = round(filtered_counts),
                              colData = metadata,
                              design = ~ condition)

# Run DESeq analysis (this performs normalization and other steps)
dds <- DESeq(dds)

# Get the normalized counts (these are the values for plotting/ML)
norm_counts <- counts(dds, normalized = TRUE)


# --- 7. Export Normalized Counts (for ML) ---

# This file is the key INPUT for your Python script
norm_counts_path <- here::here("data", NORM_COUNTS_OUT)
message(paste("Exporting normalized counts to:", norm_counts_path))

# Convert matrix to a data frame
norm_counts_df <- as.data.frame(norm_counts)

# Move the rownames (Ensembl IDs) into a proper column
norm_counts_df$ensembl_gene_id <- rownames(norm_counts_df)

# Reorder columns to put the Gene ID first for easy reference
norm_counts_df <- norm_counts_df[, c("ensembl_gene_id", setdiff(names(norm_counts_df), "ensembl_gene_id"))]

# Export the file to the 'data/' folder
write.csv(norm_counts_df,
          file = norm_counts_path,
          row.names = FALSE)


# --- 8. Annotation & Full Results (for Analysis) ---

message("Fetching annotations from biomaRt...")

# Get full DESeq2 results (log-fold change, p-vals, etc.)
res <- results(dds)
ensembl_ids_in_results <- rownames(res)

# Use biomaRt to get gene names, biotypes, etc.
# This requires an internet connection and Ensembl servers to be online.
# We wrap this in a 'tryCatch' in case the server is down.
tryCatch({
  
  # Note: If you are in China, the Asia mirror may be faster
  # ensembl_mart <- useMart("ensembl",
  #                         dataset = "hsapiens_gene_ensembl",
  #                         host = "https://asia.ensembl.org")
  
  ensembl_mart <- useMart("ensembl", dataset = "hsapiens_gene_ensembl")
  
  annotations <- getBM(
    attributes = c("ensembl_gene_id", "hgnc_symbol", "gene_biotype", "description"),
    filters = "ensembl_gene_id",
    values = ensembl_ids_in_results,
    mart = ensembl_mart
  )
  
  # Merge DESeq2 results with annotations
  res_df <- as.data.frame(res)
  res_df$ensembl_gene_id <- rownames(res_df)
  res_annotated <- merge(res_df, annotations, by = "ensembl_gene_id", all.x = TRUE)
  
  # Export to the 'results_R/' directory
  output_path_annot <- here::here(R_RESULTS_DIR, ANNOT_RESULTS_OUT)
  write.csv(res_annotated, output_path_annot, row.names = FALSE)
  
  message(paste("Successfully generated annotated results:", output_path_annot))
  
}, error = function(e) {
  # If biomaRt fails, just print a warning and finish
  warning(paste("biomaRt annotation failed. Skipping annotation.",
                "This is often due to an internet/server issue.",
                "Error was:", e$message))
})

message("R script finished successfully.")