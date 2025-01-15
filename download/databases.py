import pandas as pd
def database_info():
    classes_data = [
        ["Caudoviricetes", "https://drive.google.com/drive/folders/1n8c7zDcGxfq8HjueIt2rt-GfDrLCbtya?usp=drive_link", "unknown_processed"]
    ]
    classes_df = pd.DataFrame(classes_data, columns=["Class", "Unprocessed_url", "Processed_url"])
    return classes_df