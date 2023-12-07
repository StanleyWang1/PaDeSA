import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

# data_dir = 'np_voxelgrids'
data_dir = './DATA/sphere_samples/'

def load_and_preprocess_data(directory):
    all_voxels = []
    for filename in os.listdir(directory):
        if filename.endswith('.npy'):
            file_path = os.path.join(directory, filename)
            voxel_grid = np.load(file_path, allow_pickle=True)
            normalized_voxel_grid = normalize_voxel_grid(voxel_grid)
            all_voxels.append(normalized_voxel_grid)
    return np.array(all_voxels)


def build_generator(noise_dim=100):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(9*9*9*128, activation='relu', input_shape=(noise_dim,)),
        tf.keras.layers.Reshape((9, 9, 9, 128)),
        tf.keras.layers.Conv3DTranspose(128, (4, 4, 4), strides=(2, 2, 2), padding='same', activation='relu'),
        tf.keras.layers.Conv3DTranspose(64, (4, 4, 4), strides=(2, 2, 2), padding='same', activation='relu'),
        tf.keras.layers.Conv3DTranspose(1, (4, 4, 4), strides=(1, 1, 1), padding='valid', activation='tanh')
    ])
    return model



def build_discriminator():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv3D(64, (4, 4, 4), strides=(2, 2, 2), padding='same', input_shape=[None, None, None, 1]),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Conv3D(128, (4, 4, 4), strides=(2, 2, 2), padding='same'),
        tf.keras.layers.LeakyReLU(),
        # Add a global average pooling layer
        tf.keras.layers.GlobalAveragePooling3D(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model



# Loss functions
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

# Function to normalize the voxel grids
def normalize_voxel_grid(voxel_grid):
   return voxel_grid * 2 - 1
    # return voxel_grid

def plot_3d_voxel(voxel_grid, threshold=0):
    voxel_grid = voxel_grid.squeeze()  # Remove axes of length one

    # Adjusting the shape of the coordinate arrays
    coord_shape = np.array(voxel_grid.shape) + 1
    x, y, z = np.indices(coord_shape)

    # Voxels is true wherever the voxel grid is above the threshold
    voxels = voxel_grid > threshold

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    ax.voxels(x, y, z, voxels, edgecolor='k')
    plt.show()


voxel_data_pre = load_and_preprocess_data(data_dir)
subset_voxel_data = voxel_data_pre[0:50]    # subset of data
voxel_data = normalize_voxel_grid(subset_voxel_data)

# Training loop
num_epochs = 5
batch_size = 10

# Add an extra dimension to voxel data to represent the single channel
voxel_data = np.expand_dims(voxel_data, axis=-1)

# Convert the numpy array to a TensorFlow dataset
train_dataset = tf.data.Dataset.from_tensor_slices(voxel_data)
buffer_size = len(voxel_data)
train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

# Create models
noise_dim = 100
generator = build_generator(noise_dim)
discriminator = build_discriminator()

# Define loss and optimizers
cross_entropy = tf.keras.losses.BinaryCrossentropy()
## For Intel chip
# generator_optimizer = tf.keras.optimizers.Adam(1e-4)
# discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
## For M1/M2 Macs
generator_optimizer = tf.keras.optimizers.legacy.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.legacy.Adam(1e-4)


for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    for step, voxel_batch in enumerate(train_dataset):
        print(f"  Training step {step+1}")
        # Start of a batch, so we deal with the gradient tape
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate noise for the generator
            noise = tf.random.normal([batch_size, noise_dim])

            # Generate fake images using the noise
            generated_voxels = generator(noise, training=True)

            # The discriminator's opinion on the real and fake images
            real_output = discriminator(voxel_batch, training=True)
            fake_output = discriminator(generated_voxels, training=True)

            # Calculate the generator and discriminator loss
            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)
            print(f"    Generator loss: {gen_loss.numpy()}, Discriminator loss: {disc_loss.numpy()}")

        # Calculate the gradients for both generator and discriminator
        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        # Apply the gradients to the optimizer
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


# After training, generate voxel grid and plot
test_noise = tf.random.normal([1, noise_dim])
generated_voxel = generator(test_noise, training=False).numpy()

# Save output
np.save('generated_sphere.npy', generated_voxel)

# Check the shape of the generated voxel
print("Generated voxel shape:", generated_voxel.shape)

# Plot the generated voxel grid regardless of its shape
plot_3d_voxel(generated_voxel, threshold=0.5)