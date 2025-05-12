import pandas as pd
def database_info():
    classes_data = [
        ["Caudoviricetes", "https://drive.google.com/drive/folders/1n8c7zDcGxfq8HjueIt2rt-GfDrLCbtya?usp=drive_link", "unknown_processed", "--medium", "--medium"],
        ["Herviviricetes", "https://drive.google.com/drive/folders/1TI5qJ7Gsdp0ksqAUlD_kAsbdCe4LtsWk?usp=drive_link", "unknown_processed", "--medium", "--medium"],
        ["Monjiviricetes", "https://drive.google.com/drive/folders/1EwIf_UGogrN9zAKJbJbaYEJK4RFTP_Rf?usp=drive_link", "unknown_processed", "--medium", "--small-genomes"],
        ["Leviviricetes", "https://drive.google.com/drive/folders/1jcPRTOx_DLt1Osu4woMNZG6Os-4PCvMp?usp=drive_link", "unknown_processed", "--medium", "--small-genomes"],
        ["Repensiviricetes", "https://drive.google.com/drive/folders/1LUz6YPQ0raHY0tUuIOPVTHqVsiDRSlm7?usp=drive_link", "unknown_processed", "--medium", "--small-genomes"],
        ["Arfiviricetes", "https://drive.google.com/drive/folders/1Le4_VnnQyAbLyksHjbgBRsR5ssc4mhb3?usp=drive_link", "unknown_processed", "--medium", "--small-genomes"],
        ["Megaviricetes", "https://drive.google.com/drive/folders/1FSPCiJcVO4XE2PaxwqNTU9w-pqqmObvQ?usp=drive_link", "unknown_processed", "--medium", "--medium"]
        
    ]
    classes_df = pd.DataFrame(classes_data, columns=["Class", "Unprocessed_url", "Processed_url", "skani_sketch_mode", "skani_dist_mode"])
    return classes_df