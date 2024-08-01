import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# U-Net Architecture
def unet(input_shape):
    model = models.Sequential([
        # Encoder
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPooling2D((2, 2), padding='same'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), padding='same'),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), padding='same'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        # Decoder
        layers.UpSampling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.UpSampling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.UpSampling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')
    ])
    return model

# Conditional DDPM model
class ConditionalDDPM(tf.keras.Model):
    def __init__(self, input_shape):
        super(ConditionalDDPM, self).__init__()
        self.unet = unet((input_shape[0], input_shape[1], input_shape[2] * 2))
        self.timesteps = 1000
        self.beta = tf.linspace(1e-4, 0.02, self.timesteps)
        self.alpha = 1 - self.beta
        self.alpha_bar = tf.math.cumprod(self.alpha)
        
    def forward_diffusion(self, x0, t):
        sqrt_alpha_bar_t = tf.sqrt(tf.gather(self.alpha_bar, t))
        sqrt_alpha_bar_t = tf.reshape(sqrt_alpha_bar_t, [-1, 1, 1, 1])
        sqrt_one_minus_alpha_bar_t = tf.sqrt(1 - tf.gather(self.alpha_bar, t))
        sqrt_one_minus_alpha_bar_t = tf.reshape(sqrt_one_minus_alpha_bar_t, [-1, 1, 1, 1])
        epsilon = tf.random.normal(shape=tf.shape(x0))
        return sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * epsilon
    
    def reverse_diffusion(self, xt, t, x_cond):
        concatenated_input = tf.concat([xt, x_cond], axis=-1)
        predicted_noise = self.unet(concatenated_input)
        return predicted_noise
    
    def train_step(self, data):
        x0, x_cond = data
        batch_size = tf.shape(x0)[0]
        t = tf.random.uniform(minval=0, maxval=self.timesteps, shape=[batch_size], dtype=tf.int32)
        
        xt = self.forward_diffusion(x0, t)
        with tf.GradientTape() as tape:
            alpha_t = tf.gather(self.alpha, t)
            alpha_t = tf.reshape(alpha_t, [-1, 1, 1, 1])
            sqrt_alpha_t = tf.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = tf.sqrt(1 - alpha_t)
            predicted_noise = self.reverse_diffusion(xt, t, x_cond)
            loss = tf.reduce_mean(tf.square(predicted_noise - (xt - sqrt_alpha_t * x0) / sqrt_one_minus_alpha_t))
        
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {'loss': loss}
    
    def call(self, inputs):
        x0, x_cond = inputs
        xt = self.forward_diffusion(x0, self.timesteps-1)
        for t in reversed(range(self.timesteps)):
            predicted_noise = self.reverse_diffusion(xt, t, x_cond)
            xt = (xt - (1 - self.alpha[t]) * predicted_noise / tf.sqrt(1 - self.alpha[t])) / tf.sqrt(self.alpha[t])
        return xt

# Load CIFAR-10 dataset and normalize the images
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, x_train)).batch(64)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, x_test)).batch(64)

# Train the model
input_shape = (32, 32, 3)
model = ConditionalDDPM(input_shape)
model.compile(optimizer='adam')
model.fit(train_dataset, epochs=100)

# Generate high-resolution images
def generate_high_res_images(low_res_images, model, timesteps=1000):
    high_res_images = []
    for low_res in low_res_images:
        low_res = tf.expand_dims(low_res, axis=0)
        xt = model.forward_diffusion(low_res, timesteps-1)
        for t in reversed(range(timesteps)):
            predicted_noise = model.reverse_diffusion(xt, t, low_res)
            xt = (xt - (1 - model.alpha[t]) * predicted_noise / tf.sqrt(1 - model.alpha[t])) / tf.sqrt(model.alpha[t])
        high_res_images.append(xt[0])
    return tf.stack(high_res_images)

low_res_images = x_test[:10]
high_res_images = generate_high_res_images(low_res_images, model)

# Display low-resolution and high-resolution images
plt.figure(figsize=(20, 10))
for i in range(10):
    # Display low-resolution images
    plt.subplot(2, 10, i + 1)
    plt.imshow(low_res_images[i])
    plt.axis('off')
    # Display high-resolution images
    plt.subplot(2, 10, i + 11)
    plt.imshow(high_res_images[i])
    plt.axis('off')
plt.show()
