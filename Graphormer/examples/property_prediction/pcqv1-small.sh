#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

CUDA_VISIBLE_DEVICES=0 fairseq-train \
--user-dir ../../graphormer \
--num-workers 4 \
--ddp-backend=legacy_ddp \
--dataset-name pcqm4m \
--dataset-source ogb \
--task graph_prediction \
--criterion l1_loss \
--arch graphormer_base \
--num-classes 1 \
--attention-dropout 0.1 --act-dropout 0.1 --dropout 0.0 \
--optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm 5.0 --weight-decay 0.0 \
--lr-scheduler polynomial_decay --power 1 --warmup-updates 60000 --total-num-update 1000000 \
--lr 3e-4 --end-learning-rate 1e-9 \
--batch-size 128 \
--update-freq 8 \
--fp16 \
--data-buffer-size 20 \
--encoder-layers 6 \
--encoder-embed-dim 512 \
--encoder-ffn-embed-dim 512 \
--encoder-attention-heads 32 \
--max-epoch 300 \
--save-dir ./ckpts \
