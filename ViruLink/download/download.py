import os
import gdown
import logging
import sys
import pandas as pd
from glob import glob
from tqdm import tqdm
import numpy as np
from Bio.Seq import Seq
from ViruLink.download.vogdb import vogdb_download
from ViruLink.utils import read_fasta, write_fasta

def fix_family(row):
    if pd.isna(row["Family_y"]) and not pd.isna(row["Family_x"]):
        return row["Family_x"]
    
    else:
        return row["Family_y"]

def EnrichCsv(ncbi_df, ictv_csv):
    ictv_df = pd.read_csv(ictv_csv).drop(columns=["Species"])
    
    # Deduplicate ICTV DataFrame
    ictv_df = ictv_df.drop_duplicates(subset=["Genus", "Family"])
    
    ncbi_df.loc[:, "Genus"] = ncbi_df["Genus"].map(lambda x: x if not pd.isnull(x) else "Unknown")
    ncbi_df.loc[:, "Family"] = ncbi_df["Family"].map(lambda x: x if not pd.isnull(x) else "Unknown")

    
    ncbi_mask = (~(ncbi_df["Family"] == "Unknown")) & (ncbi_df["Genus"] == "Unknown")
    ncbi_only_fam = ncbi_df[ncbi_mask]
    ncbi_rest = ncbi_df[~ncbi_mask]
    
    merged_rest = ncbi_rest.merge(ictv_df, how="left", on="Genus")
    
    if "Family_x" in merged_rest.columns:
        merged_rest["Family"] = merged_rest.apply(fix_family, axis=1)
        merged_rest = merged_rest.drop(columns=["Family_x", "Family_y"])
    
    ictv_df = ictv_df.drop(columns=["Genus"])
    merged_only_fam = ncbi_only_fam.merge(ictv_df, how="left", on="Family")
    
    result = pd.concat([merged_rest, merged_only_fam], ignore_index=True)
    
    result.drop_duplicates(subset=["Accession"], keep="first", inplace=True)
    assert len(result) == len(ncbi_df), "Row count mismatch after enrichment"
    return result

_NAN_LIKE_STRINGS_LOWERCASE = ["", "unknown", "unclassified", "none", "<na>", "na", "null", "nan"]

def _clean_series(series_to_clean: pd.Series, series_name_for_log: str = "Unnamed Series") -> pd.Series:
    """Cleans a pandas Series for taxonomic comparisons."""
    if not isinstance(series_to_clean, pd.Series):
        logging.warning(f"Input to _clean_series '{series_name_for_log}' is not a pandas Series (type: {type(series_to_clean)}). Returning as is.")
        return series_to_clean

    s_cleaned = series_to_clean.copy() # Work on a copy

    if s_cleaned.hasnans: # Optimization: only replace if NaNs are present
        s_cleaned = s_cleaned.replace({pd.NA: np.nan}) # Standardize pd.NA to np.nan
    
    # Convert to string, strip whitespace, convert to lowercase. np.nan becomes "nan".
    s_cleaned = s_cleaned.astype(str).str.strip().str.lower()
    # Replace list of nan-like strings (and "nan" from np.nan conversion) with actual np.nan
    s_cleaned = s_cleaned.replace(_NAN_LIKE_STRINGS_LOWERCASE, np.nan)
    return s_cleaned

def Add_ICTVData(ncbi_df: pd.DataFrame, ictv_csv: str, 
                 taxa_name: str = None, taxa_name_level: str = None) -> pd.DataFrame:
    logging.info(f"Starting Add_ICTVData. Args: taxa_name='{taxa_name}', taxa_name_level='{taxa_name_level}'")

    try:
        ictv_df_original = pd.read_csv(ictv_csv)
    except FileNotFoundError:
        logging.error(f"ICTV CSV file not found: {ictv_csv}. Returning original ncbi_df.")
        return ncbi_df.copy() # Return a copy to signify no operation and prevent modifying input on error
    except Exception as e:
        logging.error(f"Error reading ICTV CSV {ictv_csv}: {e}. Returning original ncbi_df.")
        return ncbi_df.copy()
    
    # Work on copies to avoid modifying original DataFrames passed to the function,
    # unless ncbi_df is explicitly intended to be modified in place by the caller.
    # For this function signature, it's safer to assume ncbi_df should be copied if modified.
    # However, the prompt's original complex code modified ncbi_df in place.
    # To match that and for performance on large dfs, we will modify ncbi_df.
    # Caller should pass ncbi_df.copy() if they want to preserve their original.
    ictv_df = ictv_df_original.copy()

    # --- Determine ICTV Taxa Levels ---
    ictv_column_names_from_file = ictv_df.columns.to_list()
    genome_col_name = None
    for col_name in ictv_column_names_from_file: # Case-insensitive check for "Genome"
        if col_name.lower() == "genome":
            genome_col_name = col_name
            break
    
    # This list will contain ICTV taxonomic rank column names, excluding "Genome"
    # Assumes the ICTV CSV is generally ordered from Realm down to Species.
    ictv_taxa_cols_r2s = list(ictv_column_names_from_file) # Realm to Species order (original if so)
    if genome_col_name:
        ictv_taxa_cols_r2s.remove(genome_col_name)
    else:
        logging.warning("Column 'Genome' (case-insensitive) not found in ICTV data columns for removal. Proceeding with all columns as potential taxa levels.")

    ictv_taxa_cols_s2r = list(ictv_taxa_cols_r2s[::-1]) # Species to Realm order (for hierarchy indexing)
    
    logging.info(f"ICTV taxonomic columns (assumed R2S-like order from file): {ictv_taxa_cols_r2s}")
    logging.info(f"ICTV taxonomic columns (S2R-like order for hierarchy): {ictv_taxa_cols_s2r}")
    
    if not ictv_taxa_cols_s2r:
        logging.warning("No ICTV taxa levels determined. Enrichment will be limited or ineffective.")
        # Consider returning ncbi_df early if this is critical

    # --- NCBI Taxa Levels for Lookup ---
    # Ordered from most specific to least, as this order is preferred for attempting matches.
    ncbi_base_key_levels = ["Species", "Genus", "Family"]
    ncbi_taxa_levels_for_lookup = []
    for lvl in ncbi_base_key_levels:
        if lvl not in ncbi_df.columns:
            logging.info(f"NCBI lookup key '{lvl}' not found in ncbi_df columns. Will not be used.")
            continue
        if lvl not in ictv_taxa_cols_s2r : # Check against S2R list for hierarchy consistency
             logging.info(f"NCBI lookup key '{lvl}' is not among the recognized ICTV taxonomic levels ({ictv_taxa_cols_s2r}). It cannot be used for hierarchical mapping.")
             continue
        ncbi_taxa_levels_for_lookup.append(lvl)

    if not ncbi_taxa_levels_for_lookup:
        logging.warning("No valid NCBI taxa levels (Species, Genus, Family) found in ncbi_df that are also recognized ICTV levels. Primary mapping might be skipped or limited.")

    # --- Clean NCBI DataFrame Columns ---
    # Clean NCBI lookup keys and any columns in ncbi_df that happen to have names matching ICTV taxa levels.
    cols_to_clean_in_ncbi = list(set(ncbi_taxa_levels_for_lookup + [lvl for lvl in ictv_taxa_cols_r2s if lvl in ncbi_df.columns]))
    for col in cols_to_clean_in_ncbi:
        # Already checked if lvl in ncbi_df.columns for the second part of list comprehension
        ncbi_df[col] = _clean_series(ncbi_df[col], series_name_for_log=f"ncbi_df['{col}']")
    
    # --- Initialize/Prepare ICTV Columns in NCBI DataFrame ---
    for lvl in ictv_taxa_cols_r2s: # Use R2S order for creating/accessing columns by convention
        if lvl not in ncbi_df.columns:
            ncbi_df[lvl] = np.nan # Initialize with np.nan
        
        # Ensure column is object type if it's float (e.g., all np.nan initially)
        # This prepares it for string values from mapping or "Unknown" fill.
        if pd.api.types.is_float_dtype(ncbi_df[lvl]):
            if ncbi_df[lvl].isna().all(): # Only convert if currently all NaN
                 ncbi_df[lvl] = ncbi_df[lvl].astype(object)
            # If not all NaN but float, pandas fillna will handle upcasting if strings are filled.

    # --- Clean ICTV DataFrame Columns ---
    # Clean all identified ICTV taxa levels and any NCBI key levels if present in ICTV table (e.g. ICTV also has a 'Species' column)
    cols_to_clean_in_ictv = list(set(ictv_taxa_cols_r2s + [lvl for lvl in ncbi_taxa_levels_for_lookup if lvl in ictv_df.columns]))
    for col in cols_to_clean_in_ictv:
        if col in ictv_df.columns: # Should always be true due to list comprehension logic
            ictv_df[col] = _clean_series(ictv_df[col], series_name_for_log=f"ictv_df['{col}']")

    # --- Main Cross-Referencing Logic ---
    logging.info(f"\nStarting main cross-referencing. NCBI lookup keys (most specific first): {ncbi_taxa_levels_for_lookup}")
    for ncbi_key_col_name in ncbi_taxa_levels_for_lookup:
        # ncbi_key_col_name is confirmed to be in ncbi_df.columns and ictv_taxa_cols_s2r
        idx_ncbi_key_in_s2r = ictv_taxa_cols_s2r.index(ncbi_key_col_name)
        logging.info(f"Processing with NCBI lookup key: '{ncbi_key_col_name}' (S2R index: {idx_ncbi_key_in_s2r})")
        
        ncbi_key_series = ncbi_df[ncbi_key_col_name] # This series contains cleaned keys from ncbi_df
        if ncbi_key_series.isna().all():
            logging.info(f"  NCBI key column '{ncbi_key_col_name}' in ncbi_df is all NaN. No mapping possible with this key.")
            continue

        # Iterate over target ICTV columns to fill (using S2R order for consistent hierarchy checks)
        for ictv_target_col_name in ictv_taxa_cols_s2r:
            if ictv_target_col_name not in ictv_df.columns: # Target column must exist in ICTV data to create a map
                logging.debug(f"  Skipping ICTV target column '{ictv_target_col_name}': not found as a column in (cleaned) ICTV dataframe.")
                continue
            
            idx_ictv_target_in_s2r = ictv_taxa_cols_s2r.index(ictv_target_col_name)

            # HIERARCHICAL CONSTRAINT:
            # NCBI key must be MORE specific or EQUAL to target ICTV column.
            # In S2R list (Species=idx 0, Genus=idx 1,...), more specific means lower index.
            # So, idx_ncbi_key_in_s2r <= idx_ictv_target_in_s2r.
            if idx_ncbi_key_in_s2r > idx_ictv_target_in_s2r:
                # logging.debug(f"  Hierarchical Skip: Cannot map from '{ncbi_key_col_name}' (idx {idx_ncbi_key_in_s2r}) to more specific target '{ictv_target_col_name}' (idx {idx_ictv_target_in_s2r}).")
                continue
            
            # Create mapping dictionary: ictv_df[key_column] -> ictv_df[value_column]
            # .dropna() removes rows where EITHER key OR value is NaN in ICTV data, ensuring only valid, known relationships are mapped.
            map_source_df = ictv_df[[ncbi_key_col_name, ictv_target_col_name]].dropna()
            if map_source_df.empty:
                continue
            
            # If multiple target values exist for the same key (e.g. Genus G -> Family F1, Genus G -> Family F2),
            # drop_duplicates picks the first one. This is acceptable given the hierarchical constraint;
            # for valid ranks, a more specific key (e.g. Genus) maps to a unique less specific rank (e.g. Family).
            conversion_map_df = map_source_df.drop_duplicates(subset=[ncbi_key_col_name], keep='first')
            conversion_map = dict(zip(conversion_map_df[ncbi_key_col_name], conversion_map_df[ictv_target_col_name]))
            
            if not conversion_map:
                continue

            mapped_values_series = ncbi_key_series.map(conversion_map)
            
            if mapped_values_series.notna().any():
                target_col_in_ncbi_df = ncbi_df[ictv_target_col_name]
                # Ensure target column can accept string values if it's currently float (e.g. all np.nan)
                # and mapped_values actually contain some non-numeric strings
                if pd.api.types.is_float_dtype(target_col_in_ncbi_df) and not pd.api.types.is_numeric_dtype(mapped_values_series.dropna()):
                    if target_col_in_ncbi_df.isna().all(): 
                         ncbi_df[ictv_target_col_name] = target_col_in_ncbi_df.astype(object)
                # If not all NaN and float, pandas .fillna will upcast to object if string values are introduced.

                original_na_count = target_col_in_ncbi_df.isna().sum()
                ncbi_df[ictv_target_col_name].fillna(mapped_values_series, inplace=True) # Use .fillna to only fill NaNs
                
                filled_count = original_na_count - ncbi_df[ictv_target_col_name].isna().sum()
                if filled_count > 0:
                     logging.info(f"  Filled {filled_count} NA values in ncbi_df['{ictv_target_col_name}'] using map: '{ncbi_key_col_name}' (idx {idx_ncbi_key_in_s2r}) -> '{ictv_target_col_name}' (idx {idx_ictv_target_in_s2r}).")

    # --- Fallback using taxa_name and taxa_name_level (if provided) ---
    logging.info(f"\nApplying fallback using taxa_name='{taxa_name}', taxa_name_level='{taxa_name_level}'")
    
    cleaned_fallback_taxa_name, effective_taxa_name_level_col = None, None
    is_valid_fallback_taxa_name, is_valid_fallback_taxa_level = False, False

    if taxa_name is not None and pd.notna(taxa_name):
        temp_name_str = str(taxa_name).strip().lower()
        if temp_name_str and (temp_name_str not in _NAN_LIKE_STRINGS_LOWERCASE):
            cleaned_fallback_taxa_name = temp_name_str
            is_valid_fallback_taxa_name = True
    
    if taxa_name_level is not None and pd.notna(taxa_name_level):
        temp_level_str = str(taxa_name_level).strip() 
        if temp_level_str and (temp_level_str.lower() not in _NAN_LIKE_STRINGS_LOWERCASE):
            # Check if this column name (original case) exists in ICTV DataFrame and S2R hierarchy
            if temp_level_str in ictv_df.columns and temp_level_str in ictv_taxa_cols_s2r:
                effective_taxa_name_level_col = temp_level_str 
                is_valid_fallback_taxa_level = True
            else:
                logging.warning(f"  Fallback taxa_name_level '{temp_level_str}' not found as a column in ICTV data or not in S2R hierarchy. Fallback for this level skipped.")

    if is_valid_fallback_taxa_name and is_valid_fallback_taxa_level:
        # Values in ictv_df[effective_taxa_name_level_col] are already cleaned (lowercase string or np.nan)
        ictv_ref_rows_df = ictv_df[ictv_df[effective_taxa_name_level_col] == cleaned_fallback_taxa_name]

        if not ictv_ref_rows_df.empty:
            ictv_ref_data_series = ictv_ref_rows_df.iloc[0] # Use the first match from ICTV as the reference lineage
            logging.info(f"  Found reference ICTV lineage for {effective_taxa_name_level_col}='{cleaned_fallback_taxa_name}'.")

            idx_fallback_level_in_s2r = ictv_taxa_cols_s2r.index(effective_taxa_name_level_col)

            # Iterate through ICTV levels (S2R order) to fill ncbi_df based on the reference lineage
            for current_col_idx_s2r, ictv_col_to_fill_from_ref in enumerate(ictv_taxa_cols_s2r):
                # Hierarchical restriction for fallback:
                # Only fill if ictv_col_to_fill_from_ref is the effective_taxa_name_level_col itself
                # or one of its PARENT ranks (more general).
                # In S2R list (Species=0, ..., Realm=N), parents have index >= current rank's index.
                # So, fill if current_col_idx_s2r >= idx_fallback_level_in_s2r.
                if current_col_idx_s2r < idx_fallback_level_in_s2r:
                    # logging.debug(f"    Fallback Skip: Target '{ictv_col_to_fill_from_ref}' (idx {current_col_idx_s2r}) is more specific than fallback level '{effective_taxa_name_level_col}' (idx {idx_fallback_level_in_s2r}).")
                    continue
                
                if ictv_col_to_fill_from_ref in ictv_ref_data_series.index:
                    value_from_ictv_ref = ictv_ref_data_series[ictv_col_to_fill_from_ref] # This value is already cleaned
                    
                    if pd.notna(value_from_ictv_ref): # Only fill if the reference value from ICTV is not NaN
                        target_col_in_ncbi_df = ncbi_df[ictv_col_to_fill_from_ref]
                        # Ensure target column in ncbi_df can accept the string value
                        if pd.api.types.is_float_dtype(target_col_in_ncbi_df) and \
                           not isinstance(value_from_ictv_ref, (int, float)): 
                            if target_col_in_ncbi_df.isna().all(): 
                                ncbi_df[ictv_col_to_fill_from_ref] = target_col_in_ncbi_df.astype(object)
                        
                        original_na_count = target_col_in_ncbi_df.isna().sum()
                        ncbi_df[ictv_col_to_fill_from_ref].fillna(value_from_ictv_ref, inplace=True)
                        
                        filled_count = original_na_count - ncbi_df[ictv_col_to_fill_from_ref].isna().sum()
                        if filled_count > 0:
                            logging.info(f"    Fallback: Filled {filled_count} NA values in ncbi_df['{ictv_col_to_fill_from_ref}'] with '{value_from_ictv_ref}'.")
        else:
            logging.info(f"  No ICTV reference data found for {effective_taxa_name_level_col}='{cleaned_fallback_taxa_name}'. Fallback using these parameters cannot proceed.")
    else:
        details = []
        if not is_valid_fallback_taxa_name: details.append(f"taxa_name ('{taxa_name}') invalid/missing")
        if not is_valid_fallback_taxa_level: details.append(f"taxa_name_level ('{taxa_name_level}') invalid/missing or not a recognized ICTV column/hierarchy member")
        logging.info(f"  Skipping taxa_name/taxa_name_level fallback: {', '.join(details) if details else 'insufficient parameters'}.")

    # --- Final fill with "Unknown" for any remaining NaNs in ICTV-derived columns ---
    final_unknown_value = "Unknown" # Use title case for the final fill.
    logging.info(f"\nFinalizing: Filling remaining NaNs in ICTV-derived columns with '{final_unknown_value}'.")
    for lvl_col_name in ictv_taxa_cols_r2s: # Iterate using one of the definitive lists of ICTV columns (e.g., R2S order)
        if lvl_col_name in ncbi_df.columns:
            # Column should ideally be object type by now if strings were involved.
            # If it's still float (all NaNs and no non-numeric strings mapped), ensure it becomes object for "Unknown".
            if pd.api.types.is_float_dtype(ncbi_df[lvl_col_name]) and ncbi_df[lvl_col_name].isna().any():
                 ncbi_df[lvl_col_name] = ncbi_df[lvl_col_name].astype(object)
            
            na_count_before_final_fill = ncbi_df[lvl_col_name].isna().sum()
            if na_count_before_final_fill > 0:
                ncbi_df[lvl_col_name].fillna(final_unknown_value, inplace=True)
                logging.info(f"  Finalized: Filled {na_count_before_final_fill} NA values in ncbi_df['{lvl_col_name}'] with '{final_unknown_value}'.")
            
    logging.info(f"\nAdd_ICTVData complete. Example ncbi_df columns: {ncbi_df.columns.tolist()[:5]} ...")
    return ncbi_df

def ClassDownloadProcessor(class_outpath, ictv_path, class_name, class_level):
    csv_paths = glob(f"{class_outpath}/{class_name}.csv")
    
    initial_csv = [csv for csv in csv_paths if not "complete" in csv]
    
    if len(initial_csv) == 0:
        logging.info(f"Class {class_outpath} missing initial CSV files.")
        logging.info(f"Use --database flag to redownload.")
        sys.exit(1)
    
    fasta_paths = glob(f"{class_outpath}/*.fasta")
    
    if len(fasta_paths) == 0:
        logging.info(f"Class {class_outpath} missing FASTA files.")
        logging.info(f"Use --database flag to redownload.")
        sys.exit(1)
    
    seq_list = read_fasta(fasta_paths[0])
    seq_dict = {seq.id.split('.')[0]: seq for seq in seq_list}
    seq_df = pd.read_csv(initial_csv[0])
    
    
    if "Assembly" not in seq_df.columns or "Accession" not in seq_df.columns:
        logging.error(f"CSV at {initial_csv[0]} missing required columns 'Assembly' or 'Accession'.")
        sys.exit(1)
    
    representative_records = []
    for assembly, group in seq_df.groupby("Assembly"):
        representative_row = group.iloc[0]
        representative_accession = representative_row["Accession"]
        representative_accession = representative_row["Accession"].split('.')[0]
        representative_seq = seq_dict[representative_accession] 
        
        concatenated_sequence = ''.join([
            str(seq_dict[a.split('.')[0]].seq)
            for a in group["Accession"]
            if a.split('.')[0] in seq_dict
        ])
        representative_seq.seq = Seq(concatenated_sequence)
        
        representative_records.append(representative_seq)
    
    write_fasta(representative_records, fasta_paths[0])
    
    fixed_seq_df = seq_df.drop_duplicates(subset="Assembly", keep="first")
    fixed_seq_df.to_csv(initial_csv[0], index=False)
    logging.info(f"Updated DataFrame saved to {initial_csv[0]}.")
    logging.info(f"Updated FASTA file saved to {fasta_paths[0]}.")
    
    ictv_csv = glob(f"{ictv_path}/*.csv")
    
    new_df = Add_ICTVData(fixed_seq_df, ictv_csv[0], class_name, class_level)
    
    #new_df = EnrichCsv(fixed_seq_df, ictv_csv[0])
    new_df.to_csv(initial_csv[0], index=False)
    
    

def GoogleDownload(url, output_dir):
    logging.info(f"Downloading {url} to {output_dir}")
    gdown.download_folder(url,output=output_dir,quiet=False)
    logging.info(f"Downloaded {url} to {output_dir}")

def PreparingDownload(row, output_dir, ictv_path, force):
    to_download = {"unprocessed": row["Unprocessed_url"]}
    class_name = row["Class"]
    class_path = os.path.join(output_dir, row["Class"])
    
    
    for folder, url in to_download.items():
        logging.info(f"Downloading {row['Class']} {folder} data.")
        class_outpath = os.path.join(class_path) # no seperate for unprocessed and processed data.
        if not os.path.exists(class_outpath) or os.path.getsize(class_outpath) == 0 or force:
            os.makedirs(class_path, exist_ok=True)
            GoogleDownload(url, class_outpath) 
        logging.info(f"Downloaded {row['Class']} {folder} data to {class_outpath}")
        ClassDownloadProcessor(class_outpath, ictv_path, class_name, row["Level"])

def DownloadHandler(arguments, classes_df):
    vogdb_url = "https://fileshare.csb.univie.ac.at/vog/vog227/vog.faa.tar.gz"
    ictv_MSL_url = "https://drive.google.com/drive/folders/1okNtAJfBwng1FoRvT5y45PwpyFGv16r3?usp=drive_link"
    
    ictv_path = os.path.join(arguments.output, "ictv")
    if not os.path.exists(ictv_path):
        logging.info(f"Downloading ICTV MSL data to {ictv_path}, since none available.")
        gdown.download_folder(ictv_MSL_url, output=ictv_path, quiet=False)
    
    # Download the VOG database
    vogdb_dir = os.path.join(arguments.output, "VOGDB")
    vogdb_unproc_dir = vogdb_dir # keeping same directory for now.

    # Download if vogdb command invoked.
    if not os.path.exists(vogdb_unproc_dir) or arguments.vogdb:
        os.makedirs(vogdb_dir, exist_ok=True)
        os.makedirs(vogdb_unproc_dir, exist_ok=True)
        vogdb_download(vogdb_url, vogdb_unproc_dir)
    
    if arguments.all:
        classes_df.apply(PreparingDownload, axis=1, args=(arguments.output, ictv_path), force=arguments.force)
    
    if not arguments.database==None:
        row = classes_df[classes_df["Class"]==arguments.database].iloc[0]
        PreparingDownload(row, arguments.output, ictv_path, arguments.force)