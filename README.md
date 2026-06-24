================================================================
Readme.txt
================================================================
Name        : Numan Hussan
Roll Number : MSDS25067
Assignment  : DL Spring 2025 - Assignment 5 (Bonus)
              Image Generation Using Diffusion Models

----------------------------------------------------------------
FOLDER CONTENTS
----------------------------------------------------------------
model.py                   -> Diffusion process, UNet model, custom loss
MSDS25067_05.py             -> Main script: dataset loading + training (run this)
test_single_sample.ipynb   -> Loads the trained model and generates one
                               sample image (open this during evaluation)
MSDS25067_05_allCode.py     -> All code from the project combined into one
                               file (for the grader, do not run directly)
Saved_Models/               -> Trained model checkpoint is saved here
                               (diffusion_model.pt)
outputs/                    -> Created automatically when training runs.
                               Contains:
                                 forward_process.png   (noising visualization)
                                 loss_curve.png         (training loss curve)
                                 samples_epoch_*.png    (samples during training)
                                 final_generated_samples.png
Report.pdf                  -> Written report with results, comments,
                               and analysis (see project guide for outline)

----------------------------------------------------------------
1. DATASET FORMAT
----------------------------------------------------------------
Organize your dataset like this before training:

    dataset_root/
        class_1/
            img001.jpg
            img002.jpg
            ...
        class_2/
            ...
        ...   (15 animal classes total, provided by instructor)

You only need 5 classes and ~20 images per class - the script picks
this subset for you automatically (or you can specify exactly which
5 classes to use).

----------------------------------------------------------------
2. HOW TO RUN - TRAINING
----------------------------------------------------------------
Install dependencies (only needed once):
    pip install torch torchvision matplotlib pillow

Basic run (5 random classes, 20 images each, default settings):
    python MSDS25067_05.py --data_dir /path/to/dataset_root

Specify exact classes and tweak hyperparameters:
    python MSDS25067_05.py \
        --data_dir /path/to/dataset_root \
        --classes "cat,dog,lion,tiger,elephant" \
        --images_per_class 20 \
        --epochs 150 \
        --batch_size 8 \
        --lr 0.0002 \
        --timesteps 300 \
        --img_size 64

All available command-line arguments:
    --data_dir          (required) path to dataset root folder
    --classes            comma-separated list of 5 class names
                         (default: random 5 classes)
    --images_per_class  how many images to use per class (default 20)
    --img_size           image resolution, e.g. 64 (default 64)
    --batch_size         training batch size (default 8)
    --epochs             number of training epochs (default 100)
    --lr                 learning rate (default 0.0002)
    --timesteps          number of diffusion steps T (default 300;
                         assignment mentions T=1000, you may raise this,
                         but it will train more slowly)
    --base_channels      UNet base channel width (default 32)
    --n_test_samples     how many images to generate after training (default 4)
    --sample_every       save a sample grid every N epochs (default 20)
    --output_dir         where to save plots/results (default "outputs")
    --model_dir          where to save the model checkpoint (default "Saved_Models")

----------------------------------------------------------------
3. HOW TO RUN - TESTING / SINGLE SAMPLE GENERATION
----------------------------------------------------------------
After training finishes, "Saved_Models/diffusion_model.pt" will exist.
Open "test_single_sample.ipynb" (in Jupyter or Google Colab) and run all
cells. It will load the saved model and generate one image purely from
random noise using the reverse diffusion process.

----------------------------------------------------------------
4. NOTES
----------------------------------------------------------------
- The forward (noising) process uses the proper closed-form diffusion
  equation (sqrt(alpha_hat_t) * x0 + sqrt(1 - alpha_hat_t) * noise) -
  noise is NOT applied directly to raw pixel values.
- The loss function (custom_mse_loss in model.py) is written manually,
  not called from nn.MSELoss().
- Only PyTorch (plus torchvision/matplotlib/PIL for data loading and
  plotting) is used, per assignment rules.
