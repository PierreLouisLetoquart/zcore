# Zero-Shot Coreset Selection Simply Explained

```bash
uv run zeroshot_coreset_selection.py \
    --dataset eurosat10 \
    --data_dir ./data \
    --results_dir ./results \
    --embedding clip resnet18 \
    --num_workers 10
```

What's Happening:
1. ZCore is a tool that helps you select the most important images from a large dataset WITHOUT needing labels
2. It uses two powerful models (CLIP and ResNet18) to understand your images
3. It then scores each image based on how unique and representative it is

In this example (the basic/default one in the README):
- It's processing the EuroSAT10 dataset (satellite images)
- It creates embeddings (numerical representations) of each image using CLIP and ResNet18
- It then scores all images and saves the results in a score file

How to Use the Results:
1. The score file (`score.npy`) contains importance scores for each image
2. You can use these scores to keep only the most important images (e.g., keep only 30% of your data)
3. Train your models on this smaller, more efficient dataset

Simple Use Cases:

1. **Cleaning Product Image Dataset**:
```python
# Select most important product images
python zeroshot_coreset_selection.py --dataset product_images --data_dir ./my_products

# Train with 50% of most important images
python train_coreset_model.py --prune_rate 0.5 --dataset product_images
```

2. **Medical Image Analysis**:
```python
# Select key medical images
python zeroshot_coreset_selection.py --dataset xray_images --data_dir ./medical_data

# Train with 70% reduction
python train_coreset_model.py --prune_rate 0.7 --dataset xray_images
```

Key Benefits:
- Save storage space
- Reduce training time
- No need for labels
- Works with any image dataset
