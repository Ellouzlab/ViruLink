import argparse, os, sys
from multiprocessing import cpu_count

def verify(arguments):
    '''
    Verify the arguments passed to the program.
    '''
    subcommands = ["download", "process", "single_use"]
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
    


def argparser(classes):
    args = argparse.ArgumentParser(description="Classify viruses")
    args.add_argument('-v', '--version', action='version', version="0.0.0")
    subparsers = args.add_subparsers(dest='command', help='sub-command help')

    download_parser = subparsers.add_parser('download', help='Download Reference Data')
    download_parser.add_argument("--all", help="download all reference data", action="store_true")
    download_parser.add_argument("--vogdb", help="download VOGDB", action="store_true")
    download_parser.add_argument("--database", help="Specify which class database to download.", choices=classes, default=None)
    download_parser.add_argument("--unprocessed", help="download only unprocessed data", action="store_true")
    download_parser.add_argument("--output", help="output directory", default="Databases")
    
    process_parser = subparsers.add_parser('process', help='Process database data for ViraLink (NOT REQUIRED IF YOU DOWNLOADED PROCESSED)')
    process_parser.add_argument("--databases_loc", help="database to process", choices=classes, default="Databases")
    process_parser.add_argument("--database", help="database to process", choices=classes, default=None)
    process_parser.add_argument("--all", help="output directory", action="store_true")
    process_parser.add_argument("--threads", help="number of threads to use", default=cpu_count(), type=int)
    process_parser.add_argument("--memory", help="memory to use (default is 16G)", default="16G")
    process_parser.add_argument("--bitscore", help="bitscore to use", default=50, type=int)
    process_parser.add_argument("--eval", help="evalue to use", default=0.00001, type=float)
    
    
    
    '''Not meant for general use, but for preparations of databases'''
    single_use_parser = subparsers.add_parser('single_use', help='Run a single use script, Not meant for general users')
    single_use_parser.add_argument("--function", help="function to run", choices=["prep_class_db"], required=True)
    single_use_parser.add_argument("--fasta", help="fasta file to process")
    single_use_parser.add_argument("--ncbi_csv", help="NCBI CSV file to process")
    single_use_parser.add_argument("--Acc2Assem", help="Accession to assembly csv")
    single_use_parser.add_argument("--ictv_csv", help="ICTV CSV file for reference")
    single_use_parser.add_argument("--output", help="output directory", default="Databases")
    
    arguments = args.parse_args()
    verify(arguments)
    return arguments

