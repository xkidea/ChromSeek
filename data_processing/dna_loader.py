import re
import os
import numpy as np

def preprocess_fasta(fa_path, out_dir):
    """
    Preprocess chr1-chr22 sequences and save each chromosome as a separate .npy file.
    Each sequence is stored as np.uint8 ASCII codes.
    """
    os.makedirs(out_dir, exist_ok=True)
    target_chrs = {f"chr{i}" for i in range(1, 23)}
    current_chr = None
    buf = []

    with open(fa_path, "r") as f:
        for line in f:
            if line.startswith(">"):
                # Save the previous chromosome.
                if current_chr in target_chrs and buf:
                    seq_np = np.frombuffer("".join(buf).encode("ascii"), dtype=np.uint8)
                    out_path = os.path.join(out_dir, f"{current_chr}.npy")
                    np.save(out_path, seq_np)
                    print(f"[SAVE] {current_chr}: {len(seq_np):,} bp -> {out_path}")
                    buf.clear()

                # Extract the chromosome ID.
                current_chr = re.match(r"^>(\S+)", line).group(1)
            else:
                if current_chr in target_chrs:
                    buf.append(line.strip().upper())

        # Save the final chromosome.
        if current_chr in target_chrs and buf:
            seq_np = np.frombuffer("".join(buf).encode("ascii"), dtype=np.uint8)
            out_path = os.path.join(out_dir, f"{current_chr}.npy")
            np.save(out_path, seq_np)
            print(f"[SAVE] {current_chr}: {len(seq_np):,} bp -> {out_path}")

    print(f"[INFO] All chromosomes saved to {out_dir}")


def build_one_hot_table():
    """
    Build the ASCII to one-hot lookup table with shape (128, 4).
    """
    table = np.zeros((128, 4), dtype=np.float32)
    table[ord('A')] = [1, 0, 0, 0]
    table[ord('C')] = [0, 1, 0, 0]
    table[ord('G')] = [0, 0, 1, 0]
    table[ord('T')] = [0, 0, 0, 1]
    table[ord('N')] = [0, 0, 0, 0]
    return table


def load_chr(chr_name, cache_dir, mmap=True):
    """
    Load a single chromosome on demand.
    When mmap=True, use a read-only memory map with minimal RAM usage.
    """
    path = os.path.join(cache_dir, f"{chr_name}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found, please run preprocess_fasta first!")

    if mmap:
        seq_np = np.load(path, mmap_mode="r")  # Return a memmap object.
    else:
        seq_np = np.load(path)
    return seq_np


def get_encoded_segment(chr_arr, start, end, one_hot_table):
    """
    Slice a segment from one chromosome array and return one-hot encoding.
    chr_arr: np.ndarray or np.memmap
    """
    seq = chr_arr[start:end]
    return one_hot_table[seq]
