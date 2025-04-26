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
Currently, only performance can be tested. I have not yet set up an easy method to classify viruses
1. Get help:
	`ViruLink -h`
2. Download all databases:
	`ViruLink download --all`
3. Process databases and build ANI and hypergeometric graphs:
	`ViruLink process --all`
4. Run performance tests:
	`ViruLink test --all`

Please check ViruLink/ViruLink/setup/score_profiles to change what ranks can be predicted. Currently, the code in this repository will not lead to performance tests using the Evo2 encoder. 

### Contact
Muhammad Sulman: sulmanmu40@gmail.com
