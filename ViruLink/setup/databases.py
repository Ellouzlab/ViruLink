import pandas as pd
def database_info():
    classes_data = [
        ["Caudoviricetes", "https://drive.google.com/drive/folders/1n8c7zDcGxfq8HjueIt2rt-GfDrLCbtya?usp=drive_link", "unknown_processed", "--medium", "--medium"]
    ]
    classes_df = pd.DataFrame(classes_data, columns=["Class", "Unprocessed_url", "Processed_url", "skani_sketch_mode", "skani_dist_mode"])
    return classes_df