# MSA-VAE : Simple refactor to TF2

Since the original repository only supports TF1.0 which is outdated and not compatible with other libraries in TF ecosystem, this repo minimally refactors the code of MSA-VAE to TF2.0

Note that since AR-VAE is reported to have inferior performance, we have not refactor it. Also, the `n_conditions` functionality used in MSA-VAE has not been used in the training and generating scripts, this refactor simply ignore it.

See `msavae_example.ipynb` for an example of how to run.

Below is the original readme.md by alexhh, the original author.

-----
## Generating novel protein variants with variational autoencoders

This code provides implementations of variational autoencoder models designed to work with aligned and unaligned protein sequence data as described in the manuscript *Generating novel protein variants with variational autoencoders*.

### Dependencies

The code requires Python 3. Variational autoencoder models were implemented in keras (2.1.2) using the tensorflow backend (tensorflow 1.0.0). Full python dependencies are listed in requirements.txt.

Individual models were trained on a single Tesla K80 GPU with cuda 8.0.0, cudnn v5 and Python 3.6.0.

### Installation

To run code locally, first clone the repository, then install all dependencies (pip install -r requirements.txt)

### Training models

To train models run the corresponding script (training logs will be written to output/logs, and weights saved to output/weights at the end of training.)

``` bash
python scripts/train_msa.py
```

or 

``` bash
python scripts/train_raw.py
```

For the latter we recommend the use of a GPU, the former can run in a few hours on a standard CPU.

### Generating sequences (demo)

To generate sequences by sampling from the prior run scripts/generate_from_prior.py, passing the name of the weights file, and specifying the --unaligned flag if using an ARVAE model. Generated sequences will be written to a new fasta file in output/generated_sequences/

``` bash
python scripts/generate_from_prior.py data/weights/msavae.h5
```