#!/bin/bash
# AutoSF for wikikg2 dataset

# embedding dimension = 100
for i in $(seq 1 10)  
do   
CUDA_VISIBLE_DEVICES=1 python3 run.py --do_train --cuda --do_valid --do_test --evaluate_train \
  --valid_steps 2000 -randomSeed $i --print_on_screen \
  --model AutoSF -n 512 -b 1024 -d 100 -g 50 -a 3.0 -adv -r 1e-7 \
  -lr 3e-3 --max_steps 200000 --cpu_num 8 --test_batch_size 128
done  

# embedding dimension = 200
for i in $(seq 1 10)  
do   
CUDA_VISIBLE_DEVICES=1 python3 run.py --do_train --cuda --do_valid --do_test --evaluate_train \
  --valid_steps 2000 -randomSeed $i --print_on_screen \
  --model AutoSF -n 512 -b 1024 -d 200 -g 50 -a 3.0 -adv -r 1e-7 \
  -lr 3e-3 --max_steps 200000 --cpu_num 8 --test_batch_size 128
done  
