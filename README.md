# BalancedSoftmax
Balanced Softmax for classification.

This repository only considers Balanced Softmax.

- **paper**: [Balanced Meta-Softmax for Long-Tailed Visual Recognition](https://proceedings.neurips.cc/paper/2020/file/2ba61cc3a8f44143e1f2f13b2b729ab3-Paper.pdf) (NeurIPS 2020)
- **official github**: https://github.com/jiawei-ren/BalancedMetaSoftmax-Classification/tree/main

# Environment

I use a docker image. `nvcr.io/nvidia/pytorch:22.12-py3`

```bash
pip install -r requirements.txt
```

# Datasets

[datasets/build.py](https://github.com/TooTouch/BalancedSoftmax/blob/main/datasets/build.py)

- CIFAR10-LT
- CIFAR100-LT

```python
from datasets import CIFAR10LT

trainset = CIFAR10LT(
    root             = '/datasets/CIFAR10',
    train            = True,
    download         = True,
    imb_type         = 'exp',
    imbalance_factor = 200,
)

print(trainset.num_per_cls)
>> {0: 5000,
    1: 2775,
    2: 1540,
    3: 854,
    4: 474,
    5: 263,
    6: 146,
    7: 81,
    8: 45,
    9: 25}
```

# Balanced Softmax

[losses.py](https://github.com/TooTouch/BalancedSoftmax/blob/main/losses.py)

```python
from losses import BalancedSoftmax

num_per_cls = list(trainset.num_per_cls.values())
criterion = BalancedSoftmax(num_per_cls=num_per_cls)
```

# Experiments

## 1. Experiment setting

[configs.yaml](https://github.com/TooTouch/BalancedSoftmax/blob/main/configs.yaml)

```yaml
DEFAULT:
  seed: 0
  savedir: ./results
  exp_name: CE-IF_1
DATASET:
  datadir: /datasets
  batch_size: 32
  test_batch_size: 2048
  num_workers: 12
  imbalance_type: null
  imbalance_factor: 1
  aug_info:
    - RandomCrop
    - RandomHorizontalFlip
LOSS:
  name: CrossEntropyLoss
OPTIMIZER:
  name: SGD
  lr: 0.1
SCHEDULER:
  sched_name: cosine_annealing
  params:
    t_mult: 1
    eta_min: 0.00001
TRAIN:
  epochs: 50
  grad_accum_steps: 1
  mixed_precision: fp16
  log_interval: 10
  ckp_metric: bcr
  wandb:
    use: true
    entity: tootouch
    project_name: Balanced Softmax
MODEL:
  name: resnet18
  pretrained: false
```

## 2. Run

[run.sh](https://github.com/TooTouch/BalancedSoftmax/blob/main/run.sh)

```bash
dataname='CIFAR10LT CIFAR100LT'
IF='1 10 50 100 200'
losses='CrossEntropyLoss BalancedSoftmax'

for d in $dataname
do
    for f in $IF
    do
        for l in $losses
        do
            if [ $f == '1' ] && [ $l == 'BalancedSoftmax' ]; then
                continue
            else
                echo "dataset: $d, loss: $l, IF: $f"
                python main.py --config configs.yaml \
                            DEFAULT.exp_name $l-IF_$f \
                            DATASET.name $d \
                            DATASET.imbalance_type exp \
                            DATASET.imbalance_factor $f \
                            LOSS.name $l
            fi
        done
    done
done

```

## 3. Results

TBD