#!/usr/bin/env python3
import argparse, json, os
import torch
from torch.utils.data import DataLoader, TensorDataset
from models.maml_patch import MAMLPatch

def load_patch_data(patches_dir, rule_idx):
    path = os.path.join(patches_dir, f'rule_{rule_idx}.pt')
    data = torch.load(path)
    patches = data['patches'].float()  # [512,9]
    labels = data['labels'].float()    # [512]
    return patches, labels

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MAML on CA patch tasks')
    parser.add_argument('--rules', type=str, default='data/rules.json', help='path to rules JSON')
    parser.add_argument('--patches', type=str, default='data/patches', help='directory of patch .pt files')
    parser.add_argument('--shots', type=int, default=5, help='num support samples')
    parser.add_argument('--query', type=int, default=20, help='num query samples')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--out', type=str, default='checkpoints')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load rules list to count tasks
    with open(args.rules, 'r') as f:
        rules = json.load(f)
    num_rules = len(rules)

    # Model and optimizer
    model = MAMLPatch().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for rule_idx in range(num_rules):
            patches, labels = load_patch_data(args.patches, rule_idx)
            # sample support and query indices
            idx = torch.randperm(len(patches))
            support_idx = idx[:args.shots]
            query_idx = idx[args.shots:args.shots+args.query]
            support_x = patches[support_idx].to(device)
            support_y = labels[support_idx].to(device)
            query_x   = patches[query_idx].to(device)
            query_y   = labels[query_idx].to(device)
            # compute meta-loss
            loss = model(support_x, support_y, query_x, query_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / num_rules
        print(f"Epoch {epoch}/{args.epochs} - Meta Loss: {avg_loss:.4f}")
        # save checkpoint
        torch.save(model.state_dict(), os.path.join(args.out, f'meta_patch_ep{epoch}.pt'))
