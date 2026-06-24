# Image Generation Using Diffusion Models

**Name:** Numan Hussan
**Roll Number:** MSDS25067
**Assignment:** DL Spring 2025 - Assignment 5 (Bonus)

---

## Folder Contents

| File | Description |
|---|---|
| `model.py` | Diffusion process, UNet model, custom loss function |
| `MSDS25067_05.py` | Main script: dataset loading + training (**run this**) |
| `test_single_sample.ipynb` | Loads the trained model and generates one sample image (**open this during evaluation**) |
| `MSDS25067_05_allCode.py` | All code from the project combined into one file (for the grader — do not run directly) |
| `Saved_Models/` | Trained model checkpoint is saved here (`diffusion_model.pt`) |
| `outputs/` | Created automatically when training runs — contains `forward_process.png`, `loss_curve.png`, `samples_epoch_*.png`, `final_generated_samples.png` |
| `Report.pdf` | Written report with results, comments, and analysis |
| `requirements.txt` | Python dependencies |

---

## 1. Dataset Format

Organize your dataset like this before training:

```
dataset_root/
    class_1/
        img001.jpg
        img002.jpg
        ...
    class_2/
        ...
    ...   (15 animal classes total, provided by instructor)
```

You only need 5 classes and ~20 images per class — the script picks this subset for you automatically (or you can specify exactly which 5 classes to use).

---

## 2. How to Run — Training

Install dependencies (only needed once):

```bash
pip install torch torchvision matplotlib pillow
```

Basic run (5 random classes, 20 images each, default settings):

```bash
python MSDS25067_05.py --data_dir /path/to/dataset_root
```

Specify exact classes and tweak hyperparameters:

```bash
python MSDS25067_05.py \
    --data_dir /path/to/dataset_root \
    --classes "cat,dog,lion,tiger,elephant" \
    --images_per_class 20 \
    --epochs 150 \
    --batch_size 8 \
    --lr 0.0002 \
    --timesteps 300 \
    --img_size 64
```

### All available command-line arguments

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | *(required)* | Path to dataset root folder |
| `--classes` | random 5 | Comma-separated list of 5 class names |
| `--images_per_class` | 20 | How many images to use per class |
| `--img_size` | 64 | Image resolution |
| `--batch_size` | 8 | Training batch size |
| `--epochs` | 100 | Number of training epochs |
| `--lr` | 0.0002 | Learning rate |
| `--timesteps` | 300 | Number of diffusion steps T (assignment mentions T=1000; raise this if you want, but it trains more slowly) |
| `--base_channels` | 32 | UNet base channel width |
| `--n_test_samples` | 4 | How many images to generate after training |
| `--sample_every` | 20 | Save a sample grid every N epochs |
| `--output_dir` | `outputs` | Where to save plots/results |
| `--model_dir` | `Saved_Models` | Where to save the model checkpoint |

---

## 3. How to Run — Testing / Single Sample Generation

After training finishes, `Saved_Models/diffusion_model.pt` will exist. Open `test_single_sample.ipynb` (in Jupyter or Google Colab) and run all cells. It will load the saved model and generate one image purely from random noise using the reverse diffusion process.

---

## 4. Notes

- The forward (noising) process uses the proper closed-form diffusion equation:
  `x_t = sqrt(alpha_hat_t) * x0 + sqrt(1 - alpha_hat_t) * noise` — noise is **not** applied directly to raw pixel values.
- The loss function (`custom_mse_loss` in `model.py`) is written manually, not called from `nn.MSELoss()`.
- Only PyTorch (plus torchvision/matplotlib/PIL for data loading and plotting) is used, per assignment rules.
