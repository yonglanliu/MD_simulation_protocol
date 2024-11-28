"""
Creating path for simulation results
Created by: Yonglan Liu
Date: 04/26/2024
"""

import os
import argparse

# Folders
def create_folders(args):
    try:
        os.makedirs("./" + args.project)
    except OSError:
        pass

    try:
        os.makedirs("./" + args.project + "/" + args.system)
    except OSError:
        pass

    dirs = ["chk", "traj", "state", "output"]
    for d in dirs:
        try:
            os.makedirs("./" + args.project + "/" + args.system + "/" + d)
        except OSError:
            pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, help="Project path")
    parser.add_argument("--system", type=str, help="System Name")
    args = parser.parse_args()
    create_folders(args)


