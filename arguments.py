from omegaconf import OmegaConf
import argparse
from datasets import stats

def convert_type(value):
    # None
    if value == 'None':
        return None
    
    # bool
    for t in [bool, float, int, list, dict]:
        check, value = str_to_type(value=value, type=t)
        if check:
            return value
    
    return value

def str_to_type(value, type):
    try:
        check = isinstance(eval(value), type)
        out = [True, eval(value)] if check else [False, value]
        return out
    except NameError:
        return False, value


def parser():
    parser = argparse.ArgumentParser(description='BalancedSoftmax')
    parser.add_argument('--config', type=str, default=None, help='config file')
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # load default config
    cfg = OmegaConf.load(args.config)
    
    # assert experiment name
    assert cfg.DEFAULT.get('exp_name', False) != False, 'exp_name is not defined.'
    
    # update cfg
    for k, v in zip(args.opts[0::2], args.opts[1::2]):
        OmegaConf.update(cfg, k, convert_type(v), merge=True)
       
    # load dataset statistics
    cfg.DATASET.update(stats.datasets[cfg.DATASET.name])
    
    print(OmegaConf.to_yaml(cfg))
    
    return cfg