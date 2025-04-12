"""CryoGEM: Physics-Informed Generative Cryo-Electron Microscopy"""

import sys
import argparse
import importlib.metadata
import logging
import os

from cryogem.commands import gen_data, esti_ice, train, test, analysis_fspace, train_ddpm, eval_fid, video, gallery, ddpm_pipeline
logger = logging.getLogger(__name__)

def main():
    # configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(name)s - %(levelname)s - %(message)s'
    )
    
    # define commands
    command_list = {
        'gen_data': gen_data,
        'esti_ice': esti_ice,
        'train': train,
        'test': test, 
        'analysis_fspace': analysis_fspace,
        'video': video,
        'gallery': gallery,
        'train_ddpm': train_ddpm,  # Add new DDPM training command
        'eval_fid': eval_fid,      # Add new FID evaluation command
        'ddpm_pipeline': ddpm_pipeline,  # Add new DDPM pipeline command
    }
    
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument('--version', '-V', action='version',
                      version=importlib.metadata.version("cryogem"))
    subparsers = parser.add_subparsers(dest='command', help='cryogem command')
    subparsers.required = True
    
    # register commands
    for name, module in command_list.items():
        subparser = subparsers.add_parser(name, help=getattr(module, '__doc__', None))
        module.add_args(subparser)
    
    # parse the command line
    args = parser.parse_args()
    
    # Call appropriate module's main() with args
    command_list[args.command].main(args)

if __name__ == "__main__":
    main()