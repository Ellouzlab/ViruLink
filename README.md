# ViruLink
A tool to classify Viruses (or at the moment, only Caudoviricetes)

### Download/Installation
1. Clone this repository:
   
	`git clone https://github.com/Ellouzlab/ViruLink.git`
3. Create a conda env with python>=3.10, Diamond (later than version 2), MMseqs2 (later than version 14) and skani (0.2.2):

	`mamba create -n ViruLink -c conda-forge -c bioconda python>=3.10 diamond>=2 mmseqs2>=14 skani=0.2.2`
4. Activate environment:
   
	`mamba activate ViruLink`
6. change directories into ViruLink:
   
	`cd ViruLink`
8. Install program via pip:
   
	`python -m pip install -e .`


### Usage
1. Get help:
   
	`ViruLink -h`
3. Download all databases to .cache:
   
	`ViruLink download --all`
5. Process databases and build ANI and hypergeometric graphs:
   
	`ViruLink process --all`
7. Run performance tests (Optional):
   
	`ViruLink test --all`
8. Classify a query virus:

   	`ViruLink classify --query 'PATH_TO_YOUR_QUERY' --database 'OPTION_FROM_AMONG_DATABASES' --output 'PATH_TO_OUT_CSV'`

NOTE: If you have a viral genome with multiple contigs, you can concatenate them end-to-end all the contigs and use ViruLink - it won't make a difference in the results. Here is an example:

	>virus_contig_1
 	AAAAAAAAAAAAAAA
  	>virus_contig_2
   	TTTTTTTTTTTTTTT

		||
  		\/
    >virus
	AAAAAAAAAAAAAA
 	TTTTTTTTTTTTTT



Please check ViruLink/ViruLink/setup/score_profiles to change what ranks can be predicted. Currently, the code in this repository will not lead to performance tests using the Evo2 encoder. Code only works on linux systems.

### Contacts
Creator: Muhammad Sulman (sulmanmu40@gmail.com)
