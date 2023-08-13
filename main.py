import numpy as np
import json
import os
import random
import wandb
import torch
import logging
from torch.utils.data import DataLoader
from accelerate import Accelerator
from omegaconf import OmegaConf

from train import fit, test
from utils import MyEncoder
from losses import create_criterion
from log import setup_default_logging
from arguments import parser


_logger = logging.getLogger('train')

def torch_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU 
    # CUDA randomness
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)


def run(cfg):

    # set accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps = cfg.TRAIN.grad_accum_steps,
        mixed_precision             = cfg.TRAIN.mixed_precision
    )

    setup_default_logging()
    torch_seed(cfg.DEFAULT.seed)

    # set device
    _logger.info('Device: {}'.format(accelerator.device))

    # load dataset
    trainset, testset = __import__('datasets').__dict__[f"load_{cfg.DATASET.name.lower()}"](
        datadir          = cfg.DATASET.datadir, 
        img_size         = cfg.DATASET.img_size,
        mean             = cfg.DATASET.mean, 
        std              = cfg.DATASET.std,
        aug_info         = cfg.DATASET.aug_info,
        imbalance_type   = cfg.DATASET.imbalance_type,
        imbalance_factor = cfg.DATASET.imbalance_factor
    )
    
    # make save directory
    savedir = os.path.join(cfg.DEFAULT.savedir, cfg.DATASET.name, cfg.MODEL.name, cfg.DEFAULT.exp_name)
    
    assert not os.path.isdir(savedir), f'{savedir} already exists'
    os.makedirs(savedir)
    
    # save configs
    OmegaConf.save(cfg, os.path.join(savedir, 'configs.yaml'))
    
    # initialize wandb
    if cfg.TRAIN.wandb.use:
        wandb.init(name=cfg.DEFAULT.exp_name, project=cfg.TRAIN.wandb.project_name, entity=cfg.TRAIN.wandb.entity, config=OmegaConf.to_container(cfg))
    
     # logging
    _logger.info('Total samples: {}'.format(len(trainset)))

    # define train dataloader
    trainloader = DataLoader(
        dataset     = trainset,
        batch_size  = cfg.DATASET.batch_size,
        shuffle     = True,
        num_workers = cfg.DATASET.num_workers
    )
    
    # define test dataloader
    testloader = DataLoader(
        dataset     = testset,
        batch_size  = cfg.DATASET.test_batch_size,
        shuffle     = False,
        num_workers = cfg.DATASET.num_workers
    )

    # load model
    model = __import__('models').__dict__[cfg.MODEL.name](
        num_classes = cfg.DATASET.num_classes, 
        img_size    = cfg.DATASET.img_size
    )
    
    # optimizer
    optimizer = __import__('torch.optim', fromlist='optim').__dict__[cfg.OPTIMIZER.name](
        model.parameters(), 
        lr = cfg.OPTIMIZER.lr, 
        **cfg.OPTIMIZER.get('params',{})
    )

    # scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cfg.TRAIN.epochs, T_mult=1, eta_min=0.00001)

    # criterion 
    criterion = create_criterion(
        name        = cfg.LOSS.name, 
        num_per_cls = list(trainset.num_per_cls.values()), 
        params      = cfg.LOSS.get('params', {})
    )
    
    # prepraring accelerator
    model, optimizer, trainloader, testloader, scheduler = accelerator.prepare(
        model, optimizer, trainloader, testloader, scheduler
    )

    # fitting model
    fit(
        model        = model, 
        trainloader  = trainloader, 
        testloader   = testloader, 
        criterion    = criterion, 
        optimizer    = optimizer, 
        scheduler    = scheduler,
        accelerator  = accelerator,
        epochs       = cfg.TRAIN.epochs, 
        use_wandb    = cfg.TRAIN.wandb.use,
        log_interval = cfg.TRAIN.log_interval,
        savedir      = savedir,
        seed         = cfg.DEFAULT.seed,
        ckp_metric   = cfg.TRAIN.ckp_metric
    )
    
    # save model
    torch.save(model.state_dict(), os.path.join(savedir, f"model_seed{cfg.DEFAULT.seed}.pt"))
    
    # load best checkpoint 
    model.load_state_dict(torch.load(os.path.join(savedir, f'model_seed{cfg.DEFAULT.seed}_best.pt')))

    # test results
    test_results = test(
        model            = model, 
        dataloader       = testloader, 
        criterion        = criterion, 
        log_interval     = cfg.TRAIN.log_interval,
        return_per_class = True
    )

    # save results per class
    json.dump(
        obj    = test_results['per_class'], 
        fp     = open(os.path.join(savedir, f"results-seed{cfg.DEFAULT.seed}-per_class.json"), 'w'), 
        cls    = MyEncoder,
        indent = '\t'
    )
    del test_results['per_class']

    # save results
    json.dump(test_results, open(os.path.join(savedir, f'results-seed{cfg.DEFAULT.seed}.json'), 'w'), indent='\t')
    

if __name__=='__main__':

    # config
    cfg = parser()
    
    # run
    run(cfg)