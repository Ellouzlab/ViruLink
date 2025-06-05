import pandas as pd
def database_info():
    classes_data = [
        ["Caudoviricetes", "https://drive.google.com/drive/folders/1n8c7zDcGxfq8HjueIt2rt-GfDrLCbtya?usp=drive_link", "unknown_processed", "--medium", "--medium", "Class"],
        ["Herviviricetes", "https://drive.google.com/drive/folders/1TI5qJ7Gsdp0ksqAUlD_kAsbdCe4LtsWk?usp=drive_link", "unknown_processed", "--medium", "--medium", "Class"],
        ["Monjiviricetes", "https://drive.google.com/drive/folders/1EwIf_UGogrN9zAKJbJbaYEJK4RFTP_Rf?usp=drive_link", "unknown_processed", "--medium", "--small-genomes", "Class"],
        ["Leviviricetes", "https://drive.google.com/drive/folders/1jcPRTOx_DLt1Osu4woMNZG6Os-4PCvMp?usp=drive_link", "unknown_processed", "--medium", "--small-genomes", "Class"],
        ["Repensiviricetes", "https://drive.google.com/drive/folders/1LUz6YPQ0raHY0tUuIOPVTHqVsiDRSlm7?usp=drive_link", "unknown_processed", "--medium", "--small-genomes", "Class"],
        ["Arfiviricetes", "https://drive.google.com/drive/folders/1Le4_VnnQyAbLyksHjbgBRsR5ssc4mhb3?usp=drive_link", "unknown_processed", "--medium", "--small-genomes", "Class"],
        ["Revtraviricetes", "https://drive.google.com/drive/folders/1VQMbr_gSzLdPKP5rIwG8hlpvQCNdhBOp?usp=drive_link", "unknown_processed", "--medium", "--small-genomes", "Class"],
        ["Faserviricetes", "https://drive.google.com/drive/folders/1Docwdob4ah_8uFcKEjHd6WLHtVeWnC0F?usp=drive_link", "unknown_processed", "--medium", "--small-genomes", "Class"],
        ["Malgrandaviricetes", "https://drive.google.com/drive/folders/1rk-EgdVJwqpDhYVEPDwWL6Rx5h30kceS?usp=drive_link", "unknown_processed", "--medium", "--small-genomes", "Class"],
        ["Orthornavirae", "https://drive.google.com/drive/folders/1UdDf--dgQ7-ahz2yij5r2w45hblBaKc5?usp=drive_link", "unknown_processed", "--medium", "--small-genomes", "Kingdom"],
        ["Lenarviricota", "https://drive.google.com/drive/folders/1Vr-fgQBeszhGssJG-65S5wAT0DILD8DS?usp=drive_link", "unknown_processed", "--medium", "--small-genomes", "Phylum"],

    ]
    classes_df = pd.DataFrame(classes_data, columns=["Class", "Unprocessed_url", "Processed_url", "skani_sketch_mode", "skani_dist_mode", "Level"])
    return classes_df