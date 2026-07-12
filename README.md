# SijiGokuldas_CS898BA_Project

# Plant Disease Detection from Leaf Images

Computer vision project for CS898BA — detects plant disease from leaf images using a CNN with a custom image processing pipeline.

## Setup
1. Create a virtual environment: `python -m venv venv`
2. Activate it: `.\venv\Scripts\Activate` (Windows)
3. Install dependencies: `pip install torch torchvision opencv-python matplotlib numpy pandas scikit-learn kagglehub`

## How to Run
1. `python download_data.py` — downloads the PlantVillage dataset
2. `python preprocessing.py` — runs the image processing pipeline (HSV conversion, leaf segmentation, noise reduction, contrast enhancement) on a sample image, saves result to `pipeline_demo.png`
3. `python train_baseline.py` — trains a baseline ResNet18 model on a subset of the data

## Results
- Baseline model: ResNet18 (transfer learning), trained on 2,000 images across 16 disease classes, 3 epochs on CPU
- Baseline test accuracy: **45.25%**
- Random guessing across 16 classes would be ~6%, so this shows the pipeline and model are learning real patterns

## Discussion
Training was done on a smaller subset due to no local GPU access. Next steps are training on more data with more epochs, and exploring GPU access before the final submission.