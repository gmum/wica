# Weighted Non-linear Independent Component Analysis.

This is an implementation of the work described in:
  
**Andrzej Bedychaj, Przemysław Spurek, Aleksandra Nowak, Jacek Tabor,** 
***[WICA: nonlinear weighted ICA](https://arxiv.org/abs/2001.04147)***.

## Dependencies.

All dependencies are listed in the requirements.txt.

Dataset of images for mixing is available [here](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300/html/dataset/images.html).

## Training the Model.
Run command
`python3 src/main.py`

You can easily manipulate parameters using optional arguments:
```bash
optional arguments:
  -h, --help            show this help message and exit
  --save_raw SAVE_RAW   save the raw or results in png
  --data-path DATA_PATH
                        path to the data
  --num-epochs NUM_EPOCHS
                        number of epochs
  --lr LR               the learning rate
  --batch-size BATCH_SIZE
                        size of one batch
  --beta BETA           independence scaling
  --cuda                whether to use cuda
  --rec-loss {mse,bce}  type of the reconstruction error function
  --folder FOLDER       output folder
  --save-every SAVE_EVERY
                        how often to save the images and model
  --latent-dim LATENT_DIM
                        latent dimension
  --normalize-img       whether to apply normlization to input images
  --power POWER         power argument in scaling
  --number_of_gausses NUMBER_OF_GAUSSES
                        how many gausses to use for the weighting
```
## License
MIT
