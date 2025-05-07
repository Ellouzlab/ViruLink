import argparse, os, sys
from multiprocessing import cpu_count
from pathlib import Path

def verify(arguments):
    '''
    Verify the arguments passed to the program.
    '''
    subcommands = ["download", "process","test" ,"single_use", "classify"]
    if arguments.command not in subcommands:
        print("Please specify a valid command.")
        print("Available commands:", subcommands)
        sys.exit(1)
    
    if arguments.command == "download":
        if arguments.all and not arguments.database==None:
            print(arguments.database, arguments.all)
            print("Please specify either --all or use the --database flag. Not both.")
            sys.exit(1)
        if not arguments.all and arguments.database==None:
            print("Please specify either --all or use the --database flag.")
            sys.exit(1)
    
    if arguments.command == "process":
        print("Please note, processing the databases is not required if you have downloaded the processed databases.")
        if not arguments.all and arguments.database==None:
            print("Please specify either --all or use the --database flag.")
            sys.exit(1)
        
        databases_exist = os.path.exists(arguments.databases_loc)
        if not databases_exist:
            print("Please download the databases first, or specify the correct location.")
            sys.exit(1)
            
    if arguments.command == "test":
        print("Please note, processing the databases is not required if you have downloaded the processed databases.")
        if not arguments.all and arguments.database==None:
            print("Please specify either --all or use the --database flag.")
            sys.exit(1)
        
        databases_exist = os.path.exists(arguments.databases_loc)
        if not databases_exist:
            print("Please download the databases first, or specify the correct location.")
            sys.exit(1)
            
    if arguments.command == "single_use":
        if arguments.function not in ["prep_class_db", "split_train_test"]:
            print("Please specify a valid function.")
            print("Available functions:", ["prep_class_db", "split_train_test"])
            sys.exit(1)
    


def argparser(classes):
    args = argparse.ArgumentParser(description="Classify viruses")
    args.add_argument('-v', '--version', action='version', version="0.0.0")
    subparsers = args.add_subparsers(dest='command', help='sub-command help')

    download_parser = subparsers.add_parser('download', help='Download Reference Data')
    download_parser.add_argument("--all", help="download all reference data", action="store_true")
    download_parser.add_argument("--vogdb", help="download VOGDB", action="store_true")
    download_parser.add_argument("--database", help="Specify which class database to download.", choices=classes, default=None)
    download_parser.add_argument("--unprocessed", help="download only unprocessed data", action="store_true")
    download_parser.add_argument("--output", help="output directory", default=f"{Path.home()}/.cache/ViruLink")
    
    process_parser = subparsers.add_parser('process', help='Process database data for ViraLink (NOT REQUIRED IF YOU DOWNLOADED PROCESSED)')
    process_parser.add_argument("--databases_loc", help=f"location of database to process default: {Path.home()}/.cache/ViruLink", choices=classes, default=f"{Path.home()}/.cache/ViruLink")
    process_parser.add_argument("--mmseqs", help="use mmseqs for processing", action="store_true")
    process_parser.add_argument("--database", help="database to process", choices=classes, default=None)
    process_parser.add_argument("--all", help="output directory", action="store_true")
    process_parser.add_argument("--threads", help="number of threads to use", default=cpu_count(), type=int)
    process_parser.add_argument("--memory", help="memory to use (default is 16G)", default="16G")
    process_parser.add_argument("--bitscore", help="bitscore to use", default=50, type=int)
    process_parser.add_argument("--eval", help="evalue to use", default=0.00001, type=float)
    process_parser.add_argument("--force", help="force reprocessing of database", action="store_true")
    process_parser.add_argument("--ANI_FRAC_weights", help="weights for calculated with a product of ANI + ANI Frac", action="store_true")
    
    test_parser = subparsers.add_parser('test', help='Run Tests')
    test_parser.add_argument("--databases_loc", help=f"location of database to process default: {Path.home()}/.cache/ViruLink", choices=classes, default=f"{Path.home()}/.cache/ViruLink")
    test_parser.add_argument("--database", help="database to process", choices=classes, default=None)
    test_parser.add_argument("--all", help="output directory", action="store_true")
    test_parser.add_argument("--threads", help="number of threads to use", default=cpu_count(), type=int)
    
    
    classify_parser = subparsers.add_parser('classify', help='Classify a sequence')
    classify_parser.add_argument("--database", help="database to use as reference", choices=classes, default=None)
    classify_parser.add_argument("--database_loc", help=f"location of database to process default: {Path.home()}/.cache/ViruLink", choices=classes, default=f"{Path.home()}/.cache/ViruLink")
    classify_parser.add_argument("--threads", help="number of threads to use", default=cpu_count(), type=int)
    classify_parser.add_argument("--query", help="query file to classify", required=True)
    classify_parser.add_argument("--output", help="output directory", default=f"{Path.home()}/.cache/ViruLink")
    classify_parser.add_argument("--eval", help="evalue to use", default=0.00001, type=float)
    classify_parser.add_argument("--bitscore", help="bitscore to use", default=50, type=int)
    classify_parser.add_argument("--temp_dir", help="temporary directory to use", default=f"tmp")
    classify_parser.add_argument("--keep_temp", help="keep temporary files", action="store_true")
    classify_parser.add_argument(
        "--ANI_FRAC_weights",
        help="weights for calculated with a product of ANI + ANI Frac ONLY USE THIS MODE IF YOU USED IT DURING THE PROCESS COMMAND",
        action="store_true"
    )
    
    '''Not meant for general use, but for preparations of databases'''
    single_use_parser = subparsers.add_parser('single_use', help='Run a single use script, Not meant for general users')
    single_use_subparsers = single_use_parser.add_subparsers(dest='function', help='Single-use scripts')
    prep_class_db_parser = single_use_subparsers.add_parser('prep_class_db', help='Prepare class database')
    prep_class_db_parser.add_argument("--fasta", help="fasta file to process", required=True)
    prep_class_db_parser.add_argument("--ncbi_csv", help="NCBI CSV file to process", required=True)
    prep_class_db_parser.add_argument("--Acc2Assem", help="Accession to assembly csv", required=True)
    prep_class_db_parser.add_argument("--ictv_csv", help="ICTV CSV file for reference", required=True)
    prep_class_db_parser.add_argument("--output", help="output directory", default=f"{Path.home()}/.cache/ViruLink/class_db")
    prep_class_db_parser.add_argument("--num_class", help="number of max assemblies per class", default=200)
    
    arguments = args.parse_args()
    verify(arguments)
    return arguments

