import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Deshabilitar logs de TensorFlow
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Deshabilitar GPU
import tensorflow as tf
import gradio as gr
import numpy as np
from PIL import Image

# Cargar los modelos con la función de pérdida especificada
try:
    autoencoder = tf.keras.models.load_model("mi-autoencoder_final.h5", compile=False)
    autoencoder.compile(optimizer='adam', loss='mse')
    print("Modelo DAE cargado correctamente.")
except Exception as e:
    print(f"Error al cargar el modelo DAE: {e}")

try:
    decoder = tf.keras.models.load_model("mi-decoder_final.h5", compile=False)
    decoder.compile(optimizer='adam', loss='mse')
    print("Modelo Decoder cargado correctamente.")
except Exception as e:
    print(f"Error al cargar el modelo Decoder: {e}")

# Función para el Denoising Autoencoder (DAE)
def denoise_image(input_image):
    try:
        # Convertir la imagen a un array de numpy y normalizar
        input_image = np.array(input_image) / 255.0

        # Redimensionar la imagen al tamaño esperado por el modelo (128x128)
        input_image = tf.image.resize(input_image, [128, 128])

        # Añadir dimensión de lote (batch dimension)
        input_image = input_image[np.newaxis, ...]

        # Verificar que la imagen tenga el tamaño correcto
        if input_image.shape != (1, 128, 128, 3):
            raise ValueError(f"La imagen redimensionada no tiene el tamaño esperado. Se obtuvo: {input_image.shape}")

        # Predecir la imagen sin ruido
        denoised_image = autoencoder.predict(input_image)

        # Eliminar la dimensión de lote y escalar de nuevo a [0, 255]
        denoised_image = (denoised_image[0] * 255).astype(np.uint8)

        return Image.fromarray(denoised_image)

    except Exception as e:
        print(f"Error en denoise_image: {e}")
        return None  # O devuelve una imagen de error

# Función para el Variational Autoencoder (VAE)
def generate_image():
    random_latent_vector = np.random.normal(size=(1, 128))  # Asegurar tamaño correcto
    generated_image = decoder.predict(random_latent_vector)
    generated_image = (generated_image[0] * 255).astype(np.uint8)
    return Image.fromarray(generated_image)

# Crear la interfaz de Gradio
with gr.Blocks() as demo:
    gr.Markdown("# Interfaz para DAE y VAE")

    with gr.Tab("Denoising Autoencoder (DAE)"):
        gr.Markdown("Sube una imagen con ruido para ver su reconstrucción.")
        image_input = gr.Image(label="Imagen con ruido", type="pil")
        image_output = gr.Image(label="Imagen reconstruida", type="pil")
        denoise_button = gr.Button("Eliminar ruido")

    with gr.Tab("Variational Autoencoder (VAE)"):
        gr.Markdown("Genera una imagen sintética usando el VAE.")
        vae_output = gr.Image(label="Imagen generada", type="pil")
        generate_button = gr.Button("Generar imagen")

    denoise_button.click(fn=denoise_image, inputs=image_input, outputs=image_output)
    generate_button.click(fn=generate_image, inputs=None, outputs=vae_output)

# Lanzar la interfaz con enlace público
demo.launch(share=True)
