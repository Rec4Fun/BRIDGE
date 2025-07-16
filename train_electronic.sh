python train.py --dataset electronic --epochs 100 --n_dim 32
python eval.py --dataset electronic --topk 1 --alpha 0.2
python eval.py --dataset electronic --topk 2 --alpha 0.2
python eval.py --dataset electronic --topk 3 --alpha 0.2
python eval.py --dataset electronic --topk 5 --alpha 0.2