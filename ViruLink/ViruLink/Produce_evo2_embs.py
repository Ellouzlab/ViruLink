#!/usr/bin/env python3
"""
Evo2 FASTA embedder

• Splits each record into ≤8 192-nt windows.
• Runs Evo2 in batches of 8 windows.
• Mean-pools tokens → one vector per layer → averages across chunks.
• Writes <seq_id>.npz (25 separate layer vectors) as soon as each record finishes.
"""

from pathlib import Path
from typing import List

import numpy as np
import torch
from Bio import SeqIO                      # pip install biopython
from evo2 import Evo2                      # pip install evo2
from tqdm import tqdm


def _clean(layer: str) -> str:
    return layer.replace(".", "_")


@torch.inference_mode()
def fasta_to_evo2(
    fasta_path: str | Path,
    out_dir: str | Path,
    *,
    model_name: str = "evo2_1b_base",
    chunk_size: int = 8_192,
    batch_size: int = 8,
) -> None:
    """
    Stream a FASTA file, encode in 8-chunk mini-batches, and write one .npz
    per sequence with 25 layer vectors (blocks.*.mlp.l3).
    """
    fasta_path, out_dir = Path(fasta_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    evo2 = Evo2(model_name)               # auto-loads on GPU
    layers = [f"blocks.{i}.mlp.l3" for i in range(25)]
    safe   = [_clean(l) for l in layers]

    for rec in tqdm(SeqIO.parse(fasta_path, "fasta"), desc="FASTA records"):
        seq_id = rec.id.split()[0]
        seq    = str(rec.seq).upper()
        sums: List[torch.Tensor | None] = [None] * len(layers)
        n_chunks = 0

        batch_tok: List[List[int]] = []

        for pos in range(0, len(seq), chunk_size):
            batch_tok.append(evo2.tokenizer.tokenize(seq[pos: pos + chunk_size]))

            is_last = pos + chunk_size >= len(seq)
            if len(batch_tok) == batch_size or is_last:
                ids = torch.nn.utils.rnn.pad_sequence(
                    [torch.tensor(t, dtype=torch.int) for t in batch_tok],
                    batch_first=True,
                    padding_value=evo2.tokenizer.pad_id,
                ).to("cuda:0")                                    # [B, Lmax]

                _, emb = evo2(ids, return_embeddings=True, layer_names=layers)

                for i, lyr in enumerate(layers):
                    vecs = emb[lyr].mean(1).to(torch.float32)     # [B, H]  ← cast
                    tot  = vecs.sum(0)                            # [H]
                    sums[i] = tot if sums[i] is None else sums[i] + tot

                n_chunks += len(batch_tok)
                batch_tok.clear()
                del emb, ids
                torch.cuda.empty_cache()

        if n_chunks == 0:
            continue

        np.savez_compressed(
            out_dir / f"{seq_id}.npz",
            **{safe[i]: (sums[i] / n_chunks).cpu().numpy()
               for i in range(len(layers))}
        )
        del sums
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main() -> None:
    fasta_to_evo2(
        fasta_path="Caudoviricetes_partial.fasta",
        out_dir="testing_evo2",
        model_name="evo2_7b_base",
        chunk_size=8_192,
        batch_size=1,
    )


if __name__ == "__main__":
    main()
