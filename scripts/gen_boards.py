#!/usr/bin/env python3
import argparse, json, os, torch
import numpy as np

def step_board(board, birth, survival):
    # board: np.array HxW of 0/1
    H,W = board.shape
    new = np.zeros_like(board)
    for i in range(H):
        for j in range(W):
            neigh = board[max(i-1,0):i+2, max(j-1,0):j+2].sum() - board[i,j]
            if board[i,j] == 1:
                new[i,j] = 1 if neigh in survival else 0
            else:
                new[i,j] = 1 if neigh in birth else 0
    return new

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate full-board datasets for CA rules')
    parser.add_argument('--rules', type=str, default='data/rules.json')
    parser.add_argument('--out', type=str, default='data/boards')
    parser.add_argument('--size', type=int, default=50, help='board side length')
    parser.add_argument('--boards', type=int, default=100, help='num boards to sample per rule')
    parser.add_argument('--steps', type=int, default=1, help='steps per board-state pair')
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.seed is not None:
        np.random.seed(args.seed)

    with open(args.rules, 'r') as f:
        rules = json.load(f)

    for idx, rule in enumerate(rules):
        birth = set(rule['birth'])
        survival = set(rule['survival'])
        boards = []
        nexts = []
        for b in range(args.boards):
            board = (np.random.rand(args.size, args.size) < 0.3).astype(int)
            next_b = board.copy()
            for _ in range(args.steps):
                next_b = step_board(next_b, birth, survival)
            boards.append(board)
            nexts.append(next_b)
        boards = torch.tensor(boards, dtype=torch.uint8)
        nexts = torch.tensor(nexts, dtype=torch.uint8)
        filename = os.path.join(args.out, f'rule_{idx}.pt')
        torch.save({'boards': boards, 'nexts': nexts}, filename)
        print(f"Saved boards for rule {idx} -> {filename}")
