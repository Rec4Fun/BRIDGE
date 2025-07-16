python train.py --dataset Steam --epochs 100 --n_dim 16
python eval.py --dataset Steam --topk 1 --alpha 0.8
python eval.py --dataset Steam --topk 2 --alpha 0.8
python eval.py --dataset Steam --topk 3 --alpha 0.8
python eval.py --dataset Steam --topk 5 --alpha 0.8