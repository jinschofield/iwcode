#!/usr/bin/env python3
import argparse, json, random, os

def sample_rule():
    # sample non-empty birth and survival subsets of neighbor counts 0..8
    counts = list(range(9))
    # choose birth subset size 1..9
    k_b = random.randint(1, 9)
    k_s = random.randint(1, 9)
    birth = sorted(random.sample(counts, k_b))
    survival = sorted(random.sample(counts, k_s))
    return {"birth": birth, "survival": survival}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate random Life-like CA rules')
    parser.add_argument('--n', type=int, default=10, help='number of rules to generate')
    parser.add_argument('--out', type=str, default='data/rules.json', help='output JSON file path')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
    rules = [sample_rule() for _ in range(args.n)]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(rules, f, indent=2)
    print(f"Generated {len(rules)} rules -> {args.out}")
