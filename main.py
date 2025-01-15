import os
import logging
import pandas as pd
from ViraLink.arguments import argparser
from ViraLink.utils import init_logging
from ViraLink.download.databases import database_info


def main():
    classes_df = database_info()
    arguments = argparser(classes_df["Class"].to_list())

    if arguments.command == "download":
        from ViraLink.download.download import DownloadHandler
        logging_folder = f"{arguments.output}/logs"
        os.makedirs(logging_folder, exist_ok=True)
        os.makedirs(arguments.output, exist_ok=True)
        init_logging(f"{logging_folder}/download.log")
        DownloadHandler(arguments, classes_df)

    if arguments.command == "process":
        from ViraLink.process.process import ProcessHandler
        log_path = f"{arguments.databases_loc}/logs"
        
        if not os.path.exists(log_path):
            print(f"Logs not found at {log_path}")
            os.makedirs(log_path, exist_ok=True)
            print(f"Logs created at {log_path}")
            
        init_logging(f"{log_path}/process.log")
        ProcessHandler(arguments, classes_df)
        
    if arguments.command == "single_use":
        if arguments.function == "prep_class_db":
            from ViraLink.single_use_scripts.prep_class_db import prepare_class_db
            prepare_class_db(arguments)

if __name__ == "__main__":
    main()