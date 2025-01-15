import pandas as pd
import numpy as np

def create_taxonomy_similarity_matrix(csv_file, output_csv_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    accessions = df['Accession'].values
    N = len(accessions)

    # Initialize the similarity matrix with zeros
    similarity_matrix = np.zeros((N, N), dtype=int)

    # Extract taxonomy columns and fill missing values with placeholders
    families = df['Family'].fillna('MISSING_FAMILY').values
    genera = df['Genus'].fillna('MISSING_GENUS').values
    species = df['Species'].fillna('MISSING_SPECIES').values

    # Create boolean arrays indicating known values
    family_known = df['Family'].notnull().values
    genus_known = df['Genus'].notnull().values
    species_known = df['Species'].notnull().values

    # Create outer comparison arrays
    family_match = np.equal.outer(families, families)
    genus_match = np.equal.outer(genera, genera)
    species_match = np.equal.outer(species, species)

    # Known arrays for both elements in the pair
    both_family_known = np.logical_and.outer(family_known, family_known)
    both_genus_known = np.logical_and.outer(genus_known, genus_known)
    both_species_known = np.logical_and.outer(species_known, species_known)

    # Apply scoring:

    # 4: species known and match
    condition_4 = both_species_known & species_match
    similarity_matrix[condition_4] = 4

    # 3: genus and species known, genus match, species mismatch
    condition_3 = (both_species_known & both_genus_known & genus_match & ~species_match & (similarity_matrix == 0))
    similarity_matrix[condition_3] = 3

    # 2: family, genus, species known; family match, genus mismatch, species mismatch
    condition_2 = (both_species_known & both_genus_known & both_family_known &
                   family_match & ~genus_match & ~species_match & (similarity_matrix == 0))
    similarity_matrix[condition_2] = 2

    # 1: family known and mismatch
    condition_1 = (both_family_known & ~family_match & (similarity_matrix == 0))
    similarity_matrix[condition_1] = 1

    # All others remain 0 by default

    # Create a DataFrame for the similarity matrix with accessions as labels
    similarity_df = pd.DataFrame(similarity_matrix, index=accessions, columns=accessions)

    # Save the similarity matrix to a CSV file
    similarity_df.to_csv(output_csv_path)
