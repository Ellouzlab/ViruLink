from ViruLink.relations.relationship_edges import build_relationship_edges
from ViruLink.score_profile import score_config
from ViruLink.utils import get_file_path
from ViruLink.download.databases import database_info
import pandas as pd
import logging, os

def DatabaseTesting(arguments, database_name):
    database_location = f"{arguments.databases_loc}/{database_name}"
    if not os.path.exists(database_location) or os.path.getsize(database_location) == 0:
        logging.error(f"Database {database_name} not found at {database_location}")
        return
    
    
    meta_path = f"{database_location}/{database_name}.csv"
    meta_df = pd.read_csv(meta_path)
    rel_scores = score_config[database_name]
    
    '''
    # Test case
    
    #   Dataframe:
    #   Accession	Family	Genus	Species
    #   A	 F1	 G1	 S1
    #   B	 F1	 G1	 S2
    #   C	 F1	 –	 S1
    #   D	 F2	 G2	 S3
    
    # Result:
    # source target    lower    upper
    #      A      B    Genus    Genus
    #      A      C  Species  Species
    #      A      D       NR       NR
    #      C      D       NR       NR
    #      B      C   Family    Genus

    
    meta_df = pd.DataFrame({
        "Accession": ["A",  "B",  "C",  "D"],
        "Family"   : ["F1", "F1", "F1", "F2"],
        "Genus"    : ["G1", "G1", None, "G2"],
        "Species"  : ["S1", "S2", "S1", "S3"],
    })
    rel_scores = {"NR": 0, "Family": 1, "Genus": 2, "Species": 3}
    '''
    
    
    edges_df = build_relationship_edges(meta_df, rel_scores)
    print(edges_df.head())
    
def TestHandler(arguments):
    all_databases = database_info()
    if arguments.all:
        for database in all_databases["Class"]:
            if not database == "VOGDB":
                logging.info(f"Testing {database}")
                DatabaseTesting(arguments, database)