CUDA_VISIBLE_DEVICES=0 python main.py --Data ICR --model R50 --dim 512 --lr 3e-2 --comb 0 --semi 0 &
CUDA_VISIBLE_DEVICES=1 python main.py --Data ICR --model R50 --dim 512 --lr 1e-2 --comb 0 --semi 0 &
CUDA_VISIBLE_DEVICES=2 python main.py --Data ICR --model R50 --dim 512 --lr 3e-3 --comb 0 --semi 0 &
CUDA_VISIBLE_DEVICES=3 python main.py --Data ICR --model R50 --dim 512 --lr 1e-3 --comb 0 --semi 0 