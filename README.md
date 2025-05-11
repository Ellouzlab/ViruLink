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
ViruLink -h
```

### One-time setup
```bash
# Download all databases (run once)
ViruLink download --all

# Build ANI + hypergeometric graphs (run once)
ViruLink process --all
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
* If your genome spans several contigs, simply concatenate them; classification results are unchanged.  
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

---

## Configuration

To change which taxonomic ranks are predicted, edit  
`ViruLink/setup/score_profiles/*`.

*Evo2 encoder performance tests are disabled in this repo.  
Linux only.*

---

## Contact

Muhammad Sulman · <sulmanmu40@gmail.com>
