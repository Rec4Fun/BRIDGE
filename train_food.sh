python train.py --dataset food --epochs 100 --n_dim 32
python eval.py --dataset food --topk 1 --alpha 0.8
python eval.py --dataset food --topk 2 --alpha 0.8
python eval.py --dataset food --topk 3 --alpha 0.8
python eval.py --dataset food --topk 5 --alpha 0.8