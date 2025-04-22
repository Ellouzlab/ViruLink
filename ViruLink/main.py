import os
import logging
import pandas as pd
from ViruLink.arguments import argparser
from ViruLink.utils import init_logging
from ViruLink.download.databases import database_info


def main():
    classes_df = database_info()
    arguments = argparser(classes_df["Class"].to_list())

    if arguments.command == "download":
        from ViruLink.download.download import DownloadHandler
        logging_folder = f"ViruLink_logs"
        os.makedirs(logging_folder, exist_ok=True)
        os.makedirs(arguments.output, exist_ok=True)
        init_logging(f"{logging_folder}/download.log")
        DownloadHandler(arguments, classes_df)

    if arguments.command == "process":
        log_path = f"ViruLink_logs"
        
        if not os.path.exists(log_path):
            print(f"Logs not found at {log_path}")
            os.makedirs(log_path, exist_ok=True)
            print(f"Logs created at {log_path}")
            
        init_logging(f"{log_path}/process.log")
        if arguments.mmseqs:
            from ViruLink.process.mmseqs_process import ProcessHandler
        else:
            from ViruLink.process.process import ProcessHandler
        
        ProcessHandler(arguments, classes_df)
    
    if arguments.command == "test":
        log_path = f"ViruLink_logs"
        
        if not os.path.exists(log_path):
            print(f"Logs not found at {log_path}")
            os.makedirs(log_path, exist_ok=True)
            print(f"Logs created at {log_path}")
        
        init_logging(f"{log_path}/test.log")
        from ViruLink.test.test import TestHandler
        TestHandler(arguments)
        
        
    if arguments.command == "single_use":
        if arguments.function == "prep_class_db":
            logging_folder = f"ViruLink_logs"
            os.makedirs(arguments.output, exist_ok=True)
            os.makedirs(logging_folder, exist_ok=True)
            init_logging(f"{logging_folder}/prep_class_db.log")
            from ViruLink.single_use_scripts.make_class_db import make_class_db
            make_class_db(arguments)

if __name__ == "__main__":
    main()