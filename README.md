# SOM-VAE: Extended with High-Order Dynamics & Hexagonal Topologies

This repository is an extended implementation of the **SOM-VAE** ([Fortuin et al., 2019](https://arxiv.org/abs/1806.02199)). This framework combines Variational Autoencoders (VAEs) with Self-Organizing Maps (SOMs) to learn interpretable, discrete, and topologically ordered representations of time-series data.

This specific version introduces research extensions for **higher-order temporal dependencies** and **non-standard grid structures**, providing deeper insights into latent manifold organization.

---

## 🚀 Research Extensions

Compared to the original ETH Zurich implementation, this version includes:

- **2nd-Order Markov Dynamics:** Support for memory depth beyond a single step ($P\left(k_t | k_{t-1}, k_{t-2}\right)$). This captures momentum and directional intent in time-series trajectories, resolving ambiguities in state transitions.
- **Hexagonal (Triangular) Topology:** Implementation of 6-neighbor connectivity. Hexagonal tiling provides more uniform distances between nodes compared to rectangular grids, leading to smoother manifold mappings.
- **Restoration & Visualization Tool (`restore_model.py`):** A dedicated utility to reconstruct the SOM codebook from saved checkpoints. It allows for the extraction of individual node "prototypes" as images for critical assessment.

# SOM-VAE model

This repository contains a TensorFlow implementation of the self-organizing map variational autoencoder as described in the paper [SOM-VAE: Interpretable Discrete Representation Learning on Time Series](https://arxiv.org/abs/1806.02199).

If you like the SOM-VAE, you should also check out the DPSOM ([paper](https://arxiv.org/abs/1910.01590), [code](https://github.com/ratschlab/dpsom)), which yields better performance on many tasks.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

In order to install and run the model, you will need a working Python 3 distribution as well as a NVIDIA GPU with CUDA and cuDNN installed.

### Installing

In order to install the model and run it, you have to follow these steps:

- Clone the repository, i.e. run `git clone https://github.com/ratschlab/SOM-VAE`
- Change into the directory, i.e. run `cd SOM-VAE`
- Install the requirements, i.e. run `pip install -r requirements.txt`
- Install the package itself, i.e. run `pip install .`
- Change into the code directory, i.e. `cd som_vae`

Now you should be able to run the code, e.g. do `python somvae_train.py`.

### Training the model

The SOM-VAE model is defined in [somvae_model.py](som_vae/somvae_model.py).
The training script is [somvae_train.py](som_vae/somvae_train.py).

If you just want to train the model with default parameter settings, you can run

```bash
python somvae_train.py
```

This will download the MNIST data set into `data/MNIST_data/` and train on it. Afterwards, it will evaluate the trained model in terms of different clustering performance measures.

The parameters are handled using [sacred](https://github.com/IDSIA/sacred).
That means that if you want to run the model with a different parameter setting, e.g. a latent space dimensionality of 32, you can just call the training script like

```bash
python somvae_train.py with latent_dim=32
```

Per default, the script will generate time courses of linearly interpolated MNIST digits.
To train on normal MNIST instead, run

```bash
python somvae_train.py with time_series=False
```

Note that for non-time-series training, you should also set the loss parameters `gamma` and `tau` to 0.
If you want to save the model for later use, run

```bash
python somvae_train.py with save_model=True
```

If you want to train on [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) istead of normal MNIST, download the data set into `data/fashion/` and run

```bash
python somvae_train.py with data_set="fashion"
```

For more details regarding the different model parameters and how to set them, please look at the documentation in the [code](som_vae/somvae_train.py) and at the sacred documentation.

### Customizing Topology and Temporal Dynamics

In addition to standard VAE parameters, you can customize the structure of the Self-Organizing Map and the depth of the temporal transitions using the following options:

1. **SOM Topology** (`topology`) This parameter determines the connectivity between nodes in the 2D latent grid. The choice of topology affects how gradients are distributed during the SOM update and how the manifold is organized.
   - `rectangular` (Default): A standard grid where each node is connected to 4 immediate neighbors (up, down, left, right).

   - `triangular`: Implements a 6-neighbor connectivity (hexagonal tiling). This allows for a more uniform distance between a node and its neighbors, often resulting in a smoother latent manifold and lower topological error.

   To train with a hexagonal-style topology, run:

   ```bash
   python somvae_train.py with topology="triangular"
   ```

2. **Markov Order** (`markov_order`) This parameter defines the memory depth of the transition probability model. It determines how many previous steps the model considers when predicting the current node assignment $k_t$
   - 1 (Default): A first-order Markov model where the current state depends only on the previous step: $P\left(k_t | k_{t-1}\right)$.
   - 2: A second-order Markov model where the current state depends on the previous two steps: $P\left(k_t | k_{t-1}, k_{t-2}\right)$. This is particularly useful for sequences with "momentum," allowing the model to distinguish between different directions of travel through the latent space.

   To train with a second-order memory, run:

   ```bash
   python somvae_train.py with markov_order=2
   ```

### Hyperparameter optimization

If you want to optimize the models hyperparameters, you have to additionally install [labwatch](https://github.com/automl/labwatch) and [SMAC](https://github.com/automl/SMAC3) and comment the commented out lines in [somvae_train.py](som_vae/somvae_train.py) in.
Note that you also have to run a local distribution of the [MongoDB](ihttps://www.mongodb.com/).

### Train on other kinds of data

If you want to train on other types of data, you have to run the training with

```
python somvae_train.py with mnist=False
```

Moreover, you have to define the correct dimensionality in the respective `input_length` and `input_channels` parameters of the model, provide a suitable data generator in [somvae_train.py](som_vae/somvae_train.py) and potentially change the dimensionality of the layers in [somvae_model.py](som_vae/somvae_model.py).

To reproduce the experiments on eICU data, please use the preprocessing pipeline from this repository: https://github.com/ratschlab/variational-psom

## Authors

- **Vincent Fortuin** - [ETH website](https://bmi.inf.ethz.ch/people/person/vincent-fortuin/)

See also the list of [contributors](https://github.com/ratschlab/SOM-VAE/contributors) who participated in this project.

- Work extended by: Aral Dörtoğul

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details
