# Aircraft-Defect-Detection

# Aircraft defect classification: Hybrid Model + KAN classifier

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR-USER/YOUR-REPO/blob/main/hybrid_model_run.ipynb)

One notebook, [`hybrid_model_run.ipynb`](hybrid_model_run.ipynb), that runs the whole study on a **free Google Colab GPU**.

## What it does

The task is six-class image classification of aircraft surface defects (Crack, Dent, Paint Off, Missing Head, Scratch, Corrosion).
The notebook compares individual models with fused models under a leakage-free five-fold cross-validation:

- **Frozen feature extractors:** ResNet50, DenseNet121 and TinyViT-5M (ImageNet-pretrained).
- **15 configurations:** single streams, three dual-stream hybrids, one triple-stream hybrid, each with a linear head or a KAN head, plus a gated-versus-concatenation ablation.
- **Fine-tuned baselines:** the three individual models trained end to end (10 epochs).
- **Trivial baseline:** a majority-class classifier.
- **Analysis:** per-fold and pooled results, group-bootstrap intervals, per-class results, confusion matrices, cost (parameters, FLOPs, latency), learning curves, calibration and model agreement.

Main result: with frozen backbones and about 650 independent source photos, fusion did **not** give a measurable improvement over the best individual model
(macro-F1 0.488 for the hybrid with a KAN head against 0.502 for DenseNet121; a majority-class baseline gets 0.105).

## How to run (Google Colab)

1. **Get the dataset.** Download the Roboflow Universe project **Aircraft_Defect** (version 1, folder format, licence CC BY 4.0):
   <https://universe.roboflow.com/hediatma/aircraft_defect-vdd9u>
   Use the same export as the one in `splits/data_splits.csv`, whose paths look like `train/Corrosion/...jpg`, otherwise the file names will not match.
2. **Prepare Google Drive.** Create the folder `MyDrive/AircraftDefectDetection/` and put in it:
   ```
   AircraftDefectDetection/
   ├── Aircraft_Defect_Image_Dataset.zip   (download from the link above, then rename the zip to exactly this name)
   └── splits/
       ├── data_splits.csv                  (from this repository)
       └── split_config.json                (from this repository)
   ```
3. **Open the notebook in Colab** (badge above, or upload `hybrid_model_run.ipynb`) and choose *Runtime > Change runtime type > T4 GPU*.
4. **Quick check first (optional).** Set `DEBUG = True` in the first code cell and run all cells: a few minutes, uses one fold and two epochs.
5. **Full run.** Set `DEBUG = False` and run all cells. Results are saved to `MyDrive/AircraftDefectDetection/outputs/` after every run, and finished runs are skipped, so after a disconnect just reconnect and run all again.

Rough time on a free T4: feature extraction a few minutes (cached in `features/`), about 15 minutes for the 75 head-training runs, about 25 minutes for the 15 fine-tuning runs, then a few minutes for the analysis cells.

## What you get

- `outputs/hybrids/` and `outputs/baselines/`: one JSON file (metrics, per-class results, confusion matrix, history, cost) and one prediction CSV per model and fold.
- Tables in the notebook, and figures saved next to the results.

## Results and figures

All figures produced by the notebook (confusion matrices, per-fold scores, bootstrap intervals, cost and learning curves) are in the [`images/`](images/) folder.

## Data and credit

The images come from the Roboflow Universe project *Aircraft_Defect* by hediatma, licence **CC BY 4.0**. The dataset is not included in this repository.
The fusion head follows the HybridNet-S design of Ahmed et al. (2025), Knowledge-Based Systems 330, 114546; KAN layers come from [efficient-kan](https://github.com/Blealtan/efficient-kan).

## Notes

- The provider's train/validation/test split leaks variants of the same photo, so `splits/data_splits.csv` defines a group-aware, class-stratified five-fold split in which all variants of a source photo stay together.
- Results are for one public dataset and are not suitable for safety-critical decisions.
