# ViruLink

**ViruLink** is a tool for taxonomic classification of viruses.  
It currently supports:

- **Caudoviricetes**  
- **Monjiviricetes**  
- **Herviviricetes**
- **Leviviricetes** (struggles to differentiate between Family and Genus)
- **Repensiviricetes**
- **Arfiviricetes**
- **Megaviricetes** (struggles to differentiate between Genus and Species, as such Species toggled off)
- **Revtraviricetes**
- **Faserviricetes**
- **Malgrandaviricetes**

In the near future any-contig-to-genus-level classification will become available for all viruses! 

---

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/Ellouzlab/ViruLink.git
   cd ViruLink
   ```

2. **Create a Conda/Mamba environment**

   ```bash
   mamba create -n ViruLink python=3.12 "bioconda::diamond>=2" "bioconda::skani=0.2.2" "mmseqs2>16"
   ```

3. **Activate the environment**

   ```bash
   mamba activate ViruLink
   ```

4. **Install ViruLink in editable mode**

   ```bash
   python -m pip install -e .
   ```

---

## Usage

### Help

```bash
# List global options
ViruLink -h

# Discover which databases are available
ViruLink download -h
```

### One‑time setup

#### All supported virus classes

```bash
# Download every database (run once)
ViruLink download --all

# Build ANI + hypergeometric graphs (run once)
ViruLink process --all
```

#### A single virus class

```bash
# Download one database (run once)
ViruLink download --database NAME_OF_DATABASE

# Build ANI + hypergeometric graphs (run once)
ViruLink process --database NAME_OF_DATABASE
```

### (Optional) run built‑in tests

```bash
ViruLink test --all
```

### Classify a query genome

```bash
ViruLink classify \
  --query    PATH_TO_QUERY.fasta \
  --database NAME_OF_DATABASE \
  --output   results.csv
```

#### Input requirements

- Provide **one** FASTA record per genome. Each record must have a unique identifier.  
- If your genome spans multiple contigs, concatenate them into a single record; results are unchanged.  
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

## Computational requirements

Classification times hover around 30s (except Caudoviricetes, which is closer to 1min)

| Database type | RAM needed | GPU benefit |
|---------------|-----------:|-------------|
| RNA viruses   | ~2 GB      | Minor speed‑up (major if --swiglu used) |
| DNA viruses   | ~8 GB      | Minor speed‑up (major if --swiglu used) |

> **Note:** ViruLink is currently **Linux‑only**. Windows and macOS support will arrive in the coming weeks.

---

## Configuration

To change which taxonomic ranks are predicted, edit the files in  
`ViruLink/setup/score_profiles.py`.

---

## Contact

Muhammad Sulman • <sulmanmu40@gmail.com>

If you have any suggestions for the tool, or need any help, let me know!
