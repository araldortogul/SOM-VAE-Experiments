import tensorflow.compat.v1 as tf
import numpy as np
from somvae_model import SOMVAE
import matplotlib.pyplot as plt
import argparse
import os

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Visualize SOM-VAE model results.")

# Checkpoint path
parser.add_argument("--checkpoint", type=str, required=True, 
                    help="Path to the saved .ckpt file")

# SOM Dimensions (accepts two integers)
parser.add_argument("--som_dim", type=int, nargs=2, default=[5, 5],
                    help="Dimensions of the SOM grid (e.g., --som_dim (5 5)")

# Topology
parser.add_argument("--topology", type=str, default="rectangular",
                    choices=["rectangular", "triangular"],
                    help="SOM topology: 'rectangular' or 'triangular'")

# Latent Dimension (added for completeness)
parser.add_argument("--latent_dim", type=int, default=64,
                    help="Size of the latent space")

parser.add_argument("--save", action="store_true", 
                    help="If set, saves individual node images to images/ folder")


args = parser.parse_args()

# 1. Setup parameters from arguments
latent_dim = args.latent_dim
som_dim = args.som_dim
topology = args.topology
checkpoint_path = args.checkpoint 

# 2. Define the graph
tf.reset_default_graph()
x_placeholder = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
model = SOMVAE(inputs=x_placeholder, latent_dim=latent_dim, som_dim=som_dim, topology=topology)

# 3. Create the Saver
saver = tf.train.Saver()

# Load MNIST using Keras
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize and reshape
data_train = x_train.astype('float32') / 255.0
data_train = np.expand_dims(data_train, -1)
labels_train = y_train

# --- MODIFIED PLOTTING FUNCTION ---
def plot_seamless_grid(grid_data, dims):
    rows, cols = dims
    # Adjust figsize depending on how big you want the resulting image
    fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))
    
    # CRITICAL: Set spacing to zero
    plt.subplots_adjust(wspace=0, hspace=0)
    
    # Handle edge case if dims is [1,1]
    if rows > 1 or cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    for i in range(len(grid_data)):
        if i < len(axes):
            img = grid_data[i].reshape(28, 28)
            # Ensure aspect is equal so images don't stretch
            axes[i].imshow(img, cmap='gray', aspect='equal')
            # Turn off axis lines and numbers
            axes[i].axis('off')
            # REMOVED: Title is gone
            # axes[i].set_title(f"Node {i}")
            
    plt.show()

def save_individual_images(grid_data, folder="images"):
    """Saves each image in grid_data to the specified folder with its index."""
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Created directory: {folder}")
    
    for i, img_data in enumerate(grid_data):
        img = img_data.reshape(28, 28)
        file_path = os.path.join(folder, f"node_{i:03d}.png")
        # imsave saves the raw pixels without figure borders/axes
        plt.imsave(file_path, img, cmap='gray')
    
    print(f"Successfully saved {len(grid_data)} images to '{folder}/'")

with tf.Session() as sess:
    # 4. Restore weights
    try:
        saver.restore(sess, checkpoint_path)
        print(f"Model restored from: {checkpoint_path}")
    except Exception as e:
        print(f"Error reading checkpoint: {e}")
        exit(1)

    # 5. Extract the 5x5 Grid (Codebook)
    embeddings_np = sess.run(model.embeddings)
    flat_embeddings = embeddings_np.reshape(-1, latent_dim)
    manual_grid_images = sess.run(model.reconstruction_q, 
                                  feed_dict={model.z_q: flat_embeddings})

    # Show the seamless plot in the UI
    print(f"Displaying seamless {som_dim[0]}x{som_dim[1]} grid...")
    plot_seamless_grid(manual_grid_images, som_dim)

    # 6. Save individual files if requested
    if args.save:
        save_individual_images(manual_grid_images)