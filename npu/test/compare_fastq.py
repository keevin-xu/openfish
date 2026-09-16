#!/usr/bin/env python3
"""Informational: exact-sequence match rate between two slorado FASTQ files (matched by read id)."""

import sys


def read_fastq(path):
    seqs = {}
    with open(path) as f:
        while True:
            h = f.readline()
            if not h:
                break
            s = f.readline().rstrip("\n")
            f.readline()
            f.readline()
            seqs[h[1:].split()[0]] = s
    return seqs


a, b = read_fastq(sys.argv[1]), read_fastq(sys.argv[2])
common = a.keys() & b.keys()
same = sum(a[k] == b[k] for k in common)
lens_a = sum(len(a[k]) for k in common)
lens_b = sum(len(b[k]) for k in common)
print(f"reads: {len(a)} vs {len(b)}, common {len(common)}")
print(f"exact sequence match: {same}/{len(common)} = {100.0 * same / max(len(common), 1):.2f}%")
print(f"total bases: {lens_a} vs {lens_b} ({100.0 * (lens_b - lens_a) / max(lens_a, 1):+.4f}%)")
