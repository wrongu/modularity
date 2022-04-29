import torch
from models import LitWrapper
from pathlib import Path
import argparse


def create_dummy_checkpoint(dataset, task, uid, save_dir=Path(), extra_model_args={}):
    mdl = LitWrapper(dataset=dataset, task=task, l2=0., l1=0., drop=0., run=uid)

    the_path = save_dir / mdl.get_uid()
    the_path.mkdir(exist_ok=True)
    the_file = the_path / 'dummy.ckpt'
    if not the_file.exists():
        mdl.init_model(set_seed=True, **extra_model_args)
        dummy_data = {
            LitWrapper.CHECKPOINT_HYPER_PARAMS_KEY: {
                'dataset': dataset,
                'task': task,
                'l2': 0.0,
                'l1': 0.0,
                'drop': 0.0,
                'run': uid,
                'model_args': extra_model_args
            },
            'state_dict': mdl.state_dict()
        }
        torch.save(dummy_data, the_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--num', default=500, type=int)
    parser.add_argument('--save-dir', default=Path(), type=Path)
    parser.add_argument('--model-args', default=None, type=str)
    args = parser.parse_args()

    if args.model_args is not None:
        try:
            args.model_args = eval(args.model_args)
        except:
            raise ValueError(f"Failed to parse extra args for the model: {args.model_args}")
    else:
        args.model_args = {}

    for i in range(args.num):
        create_dummy_checkpoint(dataset=args.dataset,
                                task=args.task,
                                save_dir=args.save_dir,
                                uid=1000+i,
                                extra_model_args=args.model_args)
