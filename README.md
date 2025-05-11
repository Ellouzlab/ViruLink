# ViruLink
**ViruLink** is a tool for virus classification  
Currently supports **Caudoviricetes**, **Monjiviricetes**, and **Herviviricetes**. Soon, we will support all well-described viral classes.

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/Ellouzlab/ViruLink.git
cd ViruLink

# 2. Create a Conda/Mamba environment
mamba create -n ViruLink python=3.12 "bioconda::diamond>=2" "bioconda::skani=0.2.2"

# 3. Activate the environment
mamba activate ViruLink

# 4. Install ViruLink in editable mode
python -m pip install -e .
```

---

## Usage

### Get help
```bash
# Get a list of supported options
ViruLink -h

# Learn what databases (viral classes) are currently supported by ViruLink
ViruLink download -h
```

### One-time setup

#### Handle all viruses:
```bash
# Download all databases (run once)
ViruLink download --all

# Build ANI + hypergeometric graphs (run once)
ViruLink process --all
```
#### Or alternatively, handle one specific class of virus
```bash
# Download a database (run once)
ViruLink download --database NAME_OF_DATABASE

# Build ANI + hypergeometric graphs (run once)
ViruLink process --database NAME_OF_DATABASE
```

### (Optional) run built-in tests
```bash
ViruLink test --all
```

### Classify a query genome
```bash
ViruLink classify \
  --query    PATH_TO_QUERY.fasta \
  --database DATABASE_NAME \
  --output   results.csv
```

#### Input requirements
* Provide each genome as **one** FASTA record with a unique ID.
* If your genome spans several contigs, simply concatenate them; classification results are unchanged.  Multiple queries can be in one fasta file - each record will be predicted seperately.
  Example:

  ```fasta
  >virus_contig_1
  AAAAAAAAAAAAAAA
  >virus_contig_2
  TTTTTTTTTTTTTTT
  ```

  becomes

  ```fasta
  >virus
  AAAAAAAAAAAAAAA
  TTTTTTTTTTTTTTT
  ```


#### Computational Requirements
*Currently Linux ONLY* 

Windows and Mac OS support will arrive within the next few weeks!

RNA virus databases require ~2Gb of RAM. DNA viruses, ~8Gb.

---


## Configuration

To change which taxonomic ranks are predicted, edit  
`ViruLink/setup/score_profiles/*`.



---

## Contact

Muhammad Sulman · <sulmanmu40@gmail.com>
