from omegaconf import OmegaConf
from libs.dist_utils import setup_ddp, setup_logging, cleanup_ddp
import os
import logging
from pathlib import Path
from libs.evaluator import Evaluator

def main():
    # setup config
    cli_args = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli_args.config)
    del cli_args.config
    default_cfg = OmegaConf.load('configs/default.yaml')
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg._root = os.path.join(cfg._root, cfg.name)
    Path(cfg._root).mkdir(parents=True, exist_ok=True)

    # setup DDP
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    is_distributed = world_size > 1
    if is_distributed:
        setup_ddp(rank, world_size)
        print(f"Rank {rank} is initialized.")
    cfg._world_size = world_size
    cfg._distributed = is_distributed

    setup_logging(log_dir=os.path.join(cfg._root, 'logs'), log_file_name='evaluate.log')
    logger = logging.getLogger(__name__)
    logger.info(f'Evaluate on {world_size} GPUs.')
    
    logger.info(cfg)
    evaluator = Evaluator(OmegaConf.to_object(cfg))
    evaluator.load(cfg.eval.ckpt)
    evaluator.run()

    if is_distributed:
        cleanup_ddp()

if __name__ == '__main__':
    main()