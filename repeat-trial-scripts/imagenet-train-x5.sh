CUDA_VISIBLE_DEVICES=0 python train_coreset_model.py --prune_rate 0.7 --dataset imagenet --batch_size 256 --epochs 60 --architecture resnet34 --data_dir ./data --num_workers 10 --device cuda --score_file ./results/imagenet/zcore-imagenet-clip-resnet18-1000Ks-2sd-ri-1000nn-4ex-0/score.npy
CUDA_VISIBLE_DEVICES=0 python train_coreset_model.py --prune_rate 0.8 --dataset imagenet --batch_size 256 --epochs 60 --architecture resnet34 --data_dir ./data --num_workers 10 --device cuda --score_file ./results/imagenet/zcore-imagenet-clip-resnet18-1000Ks-2sd-ri-1000nn-4ex-0/score.npy
CUDA_VISIBLE_DEVICES=0 python train_coreset_model.py --prune_rate 0.9 --dataset imagenet --batch_size 256 --epochs 60 --architecture resnet34 --data_dir ./data --num_workers 10 --device cuda --score_file ./results/imagenet/zcore-imagenet-clip-resnet18-1000Ks-2sd-ri-1000nn-4ex-0/score.npy