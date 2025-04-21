# DNS-MetaCA

Differentiable Neuro-Symbolic Meta-Learner for cellular automaton rules (e.g., Conway's Game of Life).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Project Structure

```
├── .gitignore
├── README.md
├── requirements.txt
├── setup.cfg
├── data/
│   ├── rules.json      # generated CA rules
│   ├── patches/        # 3x3 patch datasets
│   └── boards/         # full-board datasets
├── scripts/
│   ├── gen_rules.py
│   ├── gen_patches.py
│   └── gen_boards.py
├── models/
│   ├── logic_bottleneck.py
│   ├── maml_patch.py
│   ├── diffusion_unet.py
│   └── adversarial_sampler.py
└── experiments/
    ├── train_meta_patch.py
    └── train_diffusion.py
``` 

## Data Generation

```bash
scripts/gen_rules.py --n 50 --out data/rules.json
scripts/gen_patches.py --rules data/rules.json --out data/patches
scripts/gen_boards.py --rules data/rules.json --out data/boards --size 50 --boards 100 --steps 1
```

## Experiments

- **Meta-patch (MAML)**
  ```bash
  python experiments/train_meta_patch.py --shots 5 --query 20 --epochs 30 --device cuda
  ```

- **Conditional Diffusion**
  ```bash
  python experiments/train_diffusion.py --epochs 10 --bs 8 --device cuda
  ```

## Utilities

- `scripts/eval_utils.py` contains metrics (patch_accuracy, threshold extraction).

## License

MIT License
