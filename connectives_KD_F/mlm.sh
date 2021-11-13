#!/bin/bash

#SBATCH --account=qiantieyun
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00

#module load google/tensorflow/python3.6-gpu
#nvidia-smi
#

CUDA_VISIBLE_DEVICES=1 python train_GCNbert.py --seed 1 --batch_size 16 --lr 0.00002 --alpha 0.08 --parse_tool l/ --class_num 4 --epochs 15
CUDA_VISIBLE_DEVICES=1 python train_GCNbert.py --seed 1337 --batch_size 16 --lr 0.00002 --alpha 0.08 --parse_tool l/ --class_num 4 --epochs 15
CUDA_VISIBLE_DEVICES=1 python train_GCNbert.py --seed 2021 --batch_size 16 --lr 0.00002 --alpha 0.08 --parse_tool l/ --class_num 4 --epochs 15
CUDA_VISIBLE_DEVICES=1 python train_GCNbert.py --seed 0 --batch_size 16 --lr 0.00002 --alpha 0.08 --parse_tool l/ --class_num 4 --epochs 15
CUDA_VISIBLE_DEVICES=1 python train_GCNbert.py --seed 2 --batch_size 16 --lr 0.00002 --alpha 0.08 --parse_tool l/ --class_num 4 --epochs 15

#CUDA_VISIBLE_DEVICES=0 python train_GCNbert.py --seed 1 --batch_size 16 --lr 0.00002 --alpha 0.4 --parse_tool l/ --class_num 4 --epochs 15
#CUDA_VISIBLE_DEVICES=0 python train_GCNbert.py --seed 1337 --batch_size 16 --lr 0.00002 --alpha 0.4 --parse_tool l/ --class_num 4 --epochs 15
#CUDA_VISIBLE_DEVICES=0 python train_GCNbert.py --seed 2021 --batch_size 16 --lr 0.00002 --alpha 0.4 --parse_tool l/ --class_num 4 --epochs 15
#CUDA_VISIBLE_DEVICES=0 python train_GCNbert.py --seed 0 --batch_size 16 --lr 0.00002 --alpha 0.4 --parse_tool l/ --class_num 4 --epochs 15
#CUDA_VISIBLE_DEVICES=0 python train_GCNbert.py --seed 2 --batch_size 16 --lr 0.00002 --alpha 0.4 --parse_tool l/ --class_num 4 --epochs 15
#
#CUDA_VISIBLE_DEVICES=0 python train_GCNbert.py --seed 1 --batch_size 16 --lr 0.00002 --alpha 0.6 --parse_tool l/ --class_num 4 --epochs 15
#CUDA_VISIBLE_DEVICES=0 python train_GCNbert.py --seed 1337 --batch_size 16 --lr 0.00002 --alpha 0.6 --parse_tool l/ --class_num 4 --epochs 15
#CUDA_VISIBLE_DEVICES=0 python train_GCNbert.py --seed 2021 --batch_size 16 --lr 0.00002 --alpha 0.6 --parse_tool l/ --class_num 4 --epochs 15
#CUDA_VISIBLE_DEVICES=0 python train_GCNbert.py --seed 0 --batch_size 16 --lr 0.00002 --alpha 0.6 --parse_tool l/ --class_num 4 --epochs 15
#CUDA_VISIBLE_DEVICES=0 python train_GCNbert.py --seed 2 --batch_size 16 --lr 0.00002 --alpha 0.6 --parse_tool l/ --class_num 4 --epochs 15
#
#CUDA_VISIBLE_DEVICES=0 python train_GCNbert.py --seed 1 --batch_size 16 --lr 0.00002 --alpha 0.8 --parse_tool l/ --class_num 4 --epochs 15
#CUDA_VISIBLE_DEVICES=0 python train_GCNbert.py --seed 1337 --batch_size 16 --lr 0.00002 --alpha 0.8 --parse_tool l/ --class_num 4 --epochs 15
#CUDA_VISIBLE_DEVICES=0 python train_GCNbert.py --seed 2021 --batch_size 16 --lr 0.00002 --alpha 0.8 --parse_tool l/ --class_num 4 --epochs 15
#CUDA_VISIBLE_DEVICES=0 python train_GCNbert.py --seed 0 --batch_size 16 --lr 0.00002 --alpha 0.8 --parse_tool l/ --class_num 4 --epochs 15
#CUDA_VISIBLE_DEVICES=0 python train_GCNbert.py --seed 2 --batch_size 16 --lr 0.00002 --alpha 0.8 --parse_tool l/ --class_num 4 --epochs 15
#
#CUDA_VISIBLE_DEVICES=0 python train_GCNbert.py --seed 1 --batch_size 16 --lr 0.00002 --alpha 1.0 --parse_tool l/ --class_num 4 --epochs 15
#CUDA_VISIBLE_DEVICES=0 python train_GCNbert.py --seed 1337 --batch_size 16 --lr 0.00002 --alpha 1.0 --parse_tool l/ --class_num 4 --epochs 15
#CUDA_VISIBLE_DEVICES=0 python train_GCNbert.py --seed 2021 --batch_size 16 --lr 0.00002 --alpha 1.0 --parse_tool l/ --class_num 4 --epochs 15
#CUDA_VISIBLE_DEVICES=0 python train_GCNbert.py --seed 0 --batch_size 16 --lr 0.00002 --alpha 1.0 --parse_tool l/ --class_num 4 --epochs 15
#CUDA_VISIBLE_DEVICES=0 python train_GCNbert.py --seed 2 --batch_size 16 --lr 0.00002 --alpha 1.0 --parse_tool l/ --class_num 4 --epochs 15
##
