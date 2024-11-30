#! /bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate openpoints
cd /root/autodl-tmp/Pointnext/examples/shapenetpart

nohup python pointconv.py > /root/autodl-tmp/Pointnext/output/pointconv.log 2>&1 &