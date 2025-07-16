python train.py --dataset clothing --epochs 100 --n_dim 32
python eval.py --dataset clothing --topk 1 --alpha 0.6
python eval.py --dataset clothing --topk 2 --alpha 0.6
python eval.py --dataset clothing --topk 3 --alpha 0.6
python eval.py --dataset clothing --topk 5 --alpha 0.6