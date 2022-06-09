# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Calculate quality metrics for previous training run or pretrained network pickle."""

import os
import click
import json
import tempfile
import copy
import torch
import dnnlib

from metrics import metric_main
from metrics import metric_utils
from torch_utils import training_stats
from torch_utils import custom_ops
from torch_utils import misc

#----------------------------------------------------------------------------

def subprocess_fn(rank, args, temp_dir):
    dnnlib.util.Logger(should_flush=True)

    # Init torch.distributed.
    if args.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=args.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=args.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if args.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0 or not args.verbose:
        custom_ops.verbosity = 'none'
    device = torch.device('cuda', rank)

    # Calculate each metric.
    for metric in args.metrics:
        if rank == 0 and args.verbose:
            print(f'Calculating {metric}...')
        progress = metric_utils.ProgressMonitor(verbose=args.verbose)
        result_dict = metric_main.calc_metric(metric=metric, dataset_kwargs=args.dataset_kwargs,
            testset_kwargs=args.testset_kwargs, num_gpus=args.num_gpus, rank=rank, device=device, progress=progress)
        if rank == 0:
            metric_main.report_metric(result_dict, run_dir=args.run_dir)
        if rank == 0 and args.verbose:
            print()

    # Done.
    if rank == 0 and args.verbose:
        print('Exiting...')

#----------------------------------------------------------------------------

class CommaSeparatedList(click.ParamType):
    name = 'list'

    def convert(self, value, param, ctx):
        _ = param, ctx
        if value is None or value.lower() == 'none' or value == '':
            return []
        return value.split(',')

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--metrics', help='Comma-separated list or "none"', type=CommaSeparatedList(), default='fid50k_full', show_default=True)
@click.option('--data', help='Dataset to evaluate metrics against (directory or zip) [default: same as training data]', metavar='PATH')
@click.option('--testdata', help='Generated testset to evaluate metrics against', metavar='PATH')
@click.option('--mirror', help='Whether the dataset was augmented with x-flips during training [default: look up]', type=bool, metavar='BOOL')
@click.option('--gpus', help='Number of GPUs to use', type=int, default=1, metavar='INT', show_default=True)
@click.option('--resolution', help='Resolution', type=int, default=256, metavar='INT', show_default=True)
@click.option('--cond', help='Whether generation is conditional', type=bool, default=False, metavar='BOOL', show_default=True)
@click.option('--verbose', help='Print optional information', type=bool, default=True, metavar='BOOL', show_default=True)

def calc_metrics(ctx, metrics, data, testdata, mirror, gpus, resolution, cond, verbose):
    """Calculate quality metrics for previous training run or pretrained network pickle.

    Examples:

    \b
    # Previous training run: look up options automatically, save result to JSONL file.
    python calc_metrics.py --metrics=pr50k3_full

    \b
    # Pre-trained network pickle: specify dataset explicitly, print result to stdout.
    python calc_metrics.py --metrics=fid50k_full --data=~/datasets/ffhq.zip --mirror=1

    Available metrics:

    \b
      ADA paper:
        fid50k_full  Frechet inception distance against the full dataset.
        kid50k_full  Kernel inception distance against the full dataset.
        pr50k3_full  Precision and recall againt the full dataset.
        is50k        Inception score for CIFAR-10.

    \b
      StyleGAN and StyleGAN2 papers:
        fid50k       Frechet inception distance against 50k real images.
        kid50k       Kernel inception distance against 50k real images.
        pr50k3       Precision and recall against 50k real images.

    """
    dnnlib.util.Logger(should_flush=True)

    # Validate arguments.
    args = dnnlib.EasyDict(metrics=metrics, num_gpus=gpus, verbose=verbose)
    if not all(metric_main.is_valid_metric(metric) for metric in args.metrics):
        ctx.fail('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))
    if not args.num_gpus >= 1:
        ctx.fail('--gpus must be at least 1')

    # Initialize dataset options.
    if data is not None:
        args.dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=data)
    else:
        ctx.fail('Could not look up dataset options; please specify --data')

    # Finalize dataset options.
    args.dataset_kwargs.resolution = resolution
    args.dataset_kwargs.use_labels = cond
    if mirror is not None:
        args.dataset_kwargs.xflip = mirror

    # Initialize testset options.
    if testdata is not None:
        args.testset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=testdata)
    else:
        ctx.fail('Could not look up dataset options; please specify --testdata')

    # Finalize testset options.
    args.testset_kwargs.resolution = resolution
    args.testset_kwargs.use_labels = cond

    # Print dataset options.
    if args.verbose:
        print('Dataset options:')
        print(json.dumps(args.dataset_kwargs, indent=2))
        print('Testset options:')
        print(json.dumps(args.testset_kwargs, indent=2))

    # Locate run dir.
    args.run_dir = None

    # Launch processes.
    if args.verbose:
        print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if args.num_gpus == 1:
            subprocess_fn(rank=0, args=args, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(args, temp_dir), nprocs=args.num_gpus)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    calc_metrics() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
