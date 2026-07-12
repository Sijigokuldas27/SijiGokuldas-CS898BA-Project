# Downloading the PlantVillage dataset (the one from your pitch slides)
import kagglehub

path = kagglehub.dataset_download("emmarex/plantdisease")
print("Dataset downloaded to:", path)