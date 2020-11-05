
# searching
python train.py --task_dir=KG_Data/WN18 --optim=adagrad --lamb=0.000282 --lr=0.37775 --n_dim=64 --n_epoch=250 --n_batch=1024 --epoch_per_test=50 --test_batch_size=50 --thres=0.0 --parrel=5 --decay_rate=0.99456;
python train.py --task_dir=KG_Data/FB15K --optim=adagrad --lamb=0.00158 --lr=0.417 --n_dim=64 --n_epoch=300 --n_batch=512 --epoch_per_test=60 --test_batch_size=80 --thres=0.0 --parrel=5 --decay_rate=0.992815;
python train.py --task_dir=KG_Data/WN18RR --optim=adagrad --lamb=0. --lr=0.7 --n_epoch=200 --n_dim=64 --n_batch=512 --epoch_per_test=40 --test_batch_size=50 --thres=0.0 --decay_rate=0.9915589 --parrel=4;
python train.py --task_dir=KG_Data/FB15K237 --optim=adagrad --lamb=0.059 --lr=0.299 --n_dim=64 --n_epoch=200 --n_batch=512 --epoch_per_test=40 --test_batch_size=80 --thres=0. --decay_rate=0.992813 --parrel=4;
python train.py --task_dir=KG_Data/YAGO --optim=adagrad --lamb=0.0005 --lr=0.5774 --n_epoch=280 --n_dim=64 --n_batch=1024 --epoch_per_test=40 --test_batch_size=50 --thres=0.0 --parrel=5 --decay_rate=0.9931


# evaluation
python evaluate.py --task_dir=KG_Data/WN18 --mode=evaluate;
python evaluate.py --task_dir=KG_Data/FB15K --mode=evaluate;
python evaluate.py --task_dir=KG_Data/WN18RR --mode=evaluate;
python evaluate.py --task_dir=KG_Data/FB15K237 --mode=evaluate;
python evaluate.py --task_dir=KG_Data/YAGO --mode=evaluate;

# fine-tuning
python evaluate.py --task_dir=KG_data/WN18 --mode=tune;
