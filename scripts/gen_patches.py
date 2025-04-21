#!/usr/bin/env python3
import argparse, json, os, itertools
import torch

def compute_next_bit(patch, birth, survival):
    # patch: list of 9 bits (row-major 3x3), center at index 4
    total = sum(patch)
    center = patch[4]
    neigh = total - center
    if center == 1:
        return 1 if neigh in survival else 0
    else:
        return 1 if neigh in birth else 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate 3x3 patch datasets for CA rules')
    parser.add_argument('--rules', type=str, default='data/rules.json', help='path to rules JSON')
    parser.add_argument('--out', type=str, default='data/patches', help='output directory for .pt files')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    with open(args.rules, 'r') as f:
        rules = json.load(f)

    for idx, rule in enumerate(rules):
        birth = set(rule['birth'])
        survival = set(rule['survival'])
        patches = []
        labels = []
        for bits in itertools.product([0,1], repeat=9):
            patch = list(bits)
            patches.append(patch)
            labels.append(compute_next_bit(patch, birth, survival))
        patches_tensor = torch.tensor(patches, dtype=torch.uint8)
        labels_tensor = torch.tensor(labels, dtype=torch.uint8)
        filename = os.path.join(args.out, f'rule_{idx}.pt')
        torch.save({'patches': patches_tensor, 'labels': labels_tensor}, filename)
        print(f"Saved patches for rule {idx} -> {filename}")
