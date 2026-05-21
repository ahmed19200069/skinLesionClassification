# Skin Lesion Classification Project

This is my course project for classifying skin lesions using the HAM10000 dataset. The main idea is to compare transfer learning (MobileNetV2) with a simple CNN I built from scratch and see which approach works better.

## What the project does

I trained 4 different models on HAM10000 which has 7 types of skin lesions. Three of them use MobileNetV2 with different training strategies, and one is a baseline CNN I wrote myself without any pretrained weights.

The 4 models:
- **Baseline CNN** - built from scratch, no pretrained weights
- **Feature Extraction** - MobileNetV2 with the backbone frozen, only training the last layers
- **Progressive Unfreezing** - start with frozen backbone then slowly unfreeze it during training
- **Full Fine-Tune** - train all layers of MobileNetV2 end to end

## Dataset

HAM10000 from Harvard Dataverse. It has 10,000+ dermoscopy images across 7 classes:
- akiec, bcc, bkl, df, mel, nv, vasc

The dataset is heavily imbalanced (nv has way more samples than the others) so I used weighted sampling and class weights in the loss function to deal with that.

One thing I focused on is the confusion between **mel** (melanoma) and **bkl** (benign keratosis) because they look similar and misclassifying melanoma as benign is a serious mistake.

## Files

```
src/
  dataset.py          - loading HAM10000, transforms, sampler
  baseline_cnn.py     - the scratch CNN architecture
  train_baseline.py   - training script for the baseline
  model.py            - MobileNetV2 with custom head
  train.py            - training script for all 3 MobileNetV2 strategies
  evaluate.py         - metrics, confusion matrix, bkl/mel analysis
  compare.py          - comparison charts across all 4 models
  ablation.py         - ablation experiments (lr, dropout, augmentation, imbalance)
```

## How to run

First install dependencies:
```bash
pip install -r requirements.txt
```

Download HAM10000 and put the files in:
```
data/HAM10000_metadata.csv
data/HAM10000_images_part1/
data/HAM10000_images_part2/
```

Then train each model:
```bash
python src/train_baseline.py --epochs 40

python src/train.py --strategy feature_extraction --epochs 20

python src/train.py --strategy progressive --epochs 30 --unfreeze5_epoch 10 --unfreeze10_epoch 20

python src/train.py --strategy full_finetune --epochs 30 --lr 1e-4
```

Then evaluate:
```bash
python src/evaluate.py --strategy baseline_cnn
python src/evaluate.py --strategy feature_extraction
python src/evaluate.py --strategy progressive
python src/evaluate.py --strategy full_finetune

python src/compare.py
```

Or just run everything at once:
```bash
python run_all.py
```

## Results output

Everything saves to `results/`. The main files are:
- `comparison_table.txt` - numbers for all 4 models side by side
- `comparison_overall.png` - bar chart of accuracy/F1
- `learning_curves.png` - training curves
- `bkl_mel_confusion.png` - the mel/bkl error rates specifically
- `{strategy}_confusion_matrix.png` and `{strategy}_eval.json` for each model

## Requirements

- Python 3.10+
- PyTorch 2.1+
- 8GB RAM minimum (no GPU needed but it's faster with one)

## References

- Tschandl et al., "The HAM10000 dataset" (2018)
- Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks" (2018)
