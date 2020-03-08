# Weighted non-linear independent component analysis

This is an implementation of the work described in [https://arxiv.org/abs/2001.04147]

## Dependencies

All dependencies in file requirements.

## Training a model
Run command
`python3 src/main.py`

You can easily manipulate parameters using optional arguments:
```bash
usage: main.py [-h] [--save_raw SAVE_RAW] [--data-path DATA_PATH] [--model {beta-vae,ae,gan,wica}] [--num-epochs NUM_EPOCHS] [--lr LR] [--batch-size BATCH_SIZE] [--beta BETA] [--cuda] [--rec-loss {mse,bce}] [--folder FOLDER]
               [--save-every SAVE_EVERY] [--latent-dim LATENT_DIM] [--normalize-img] [--power POWER] [--number_of_gausses NUMBER_OF_GAUSSES]

optional arguments:
  -h, --help            show this help message and exit
  --save_raw SAVE_RAW   save the raw or results in png
  --data-path DATA_PATH
                        path to the data
  --model {beta-vae,ae,gan,wica}
                        the model to use
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
