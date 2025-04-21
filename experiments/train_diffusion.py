#!/usr/bin/env python3
import argparse, json, os, torch
from torch.utils.data import DataLoader, TensorDataset
from models.diffusion_unet import DiffusionUNet

def load_board_data(boards_dir, rule_idx):
    data = torch.load(os.path.join(boards_dir, f'rule_{rule_idx}.pt'))
    boards = data['boards'].float().unsqueeze(1)  # [N,1,H,W]
    nexts  = data['nexts'].float().unsqueeze(1)
    return boards, nexts

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train conditional diffusion model on CA boards')
    parser.add_argument('--rules', type=str, default='data/rules.json')
    parser.add_argument('--boards', type=str, default='data/boards')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--bs', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--out', type=str, default='checkpoints')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    with open(args.rules, 'r') as f:
        rules = json.load(f)
    num_rules = len(rules)

    model = DiffusionUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    for epoch in range(1, args.epochs+1):
        total_loss = 0.0
        count = 0
        for rule_idx in range(num_rules):
            boards, nexts = load_board_data(args.boards, rule_idx)
            dataset = TensorDataset(boards, nexts)
            loader = DataLoader(dataset, batch_size=args.bs, shuffle=True)
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                # dummy timestep tensor
                t = torch.zeros(x.size(0), dtype=torch.long, device=device)
                pred = model(x, t, x)
                loss = criterion(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                count += 1
        avg_loss = total_loss / count
        print(f"Epoch {epoch}/{args.epochs} - Loss: {avg_loss:.4f}")
        checkpoint = os.path.join(args.out, f'diffusion_ep{epoch}.pt')
        torch.save(model.state_dict(), checkpoint)
        print(f"Saved checkpoint: {checkpoint}")
