#!/bin/bash
# AutoSF for biokg dataset

# embedding dimension = 1000
for i in $(seq 1 10)  
do   
CUDA_VISIBLE_DEVICES=0 python3 run.py --do_train --cuda --do_valid --do_test --evaluate_train \
  --valid_steps 1000 -randomSeed $i --print_on_screen \
  --model AutoSF -n 256 -b 2048 -d 1000 -g 200 -a 1.0 -adv -r 1e-7 \
  -lr 1e-3 --max_steps 100000 --cpu_num 8 --test_batch_size 128
done  

# embedding dimension = 2000
for i in $(seq 1 10)  
do   
CUDA_VISIBLE_DEVICES=0 python3 run.py --do_train --cuda --do_valid --do_test --evaluate_train \
  --valid_steps 1000 -randomSeed $i --print_on_screen \
  --model AutoSF -n 256 -b 2048 -d 2000 -g 200 -a 1.0 -adv -r 1e-7 \
  -lr 1e-3 --max_steps 100000 --cpu_num 8 --test_batch_size 128
done  
