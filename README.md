# Proyecto 1: Denoising Autoencoder (DAE) y Variational Autoencoder (VAE) para Restauración y Generación de Imágenes

### Melanie Michel Rodriguez
### Ilse Regina Flores Reyes
### Santiago Aguirre Vera 

Este proyecto tiene como objetivo desarrollar y entrenar dos modelos de aprendizaje profundo: un **Denoising Autoencoder (DAE)** para la restauración de imágenes con ruido, y un **Variational Autoencoder (VAE)** para la generación de imágenes sintéticas. El proyecto fue desarrollado por **Melanie Michel**, **Santiago Aguirre** y **Regina Flores** como parte de un curso de aprendizaje profundo.

## Descripción del Proyecto

### Parte 1: Creación del Dataset
- **Obtención de imágenes**: Se recolectaron imágenes de dos clases distintas (latas de Coca-Cola y Sprite) mediante web scraping utilizando una extensión de Chrome. Luego, se depuraron manualmente para asegurar la calidad del dataset.
- **Procesamiento de datos**: Las imágenes se normalizaron y se organizaron en conjuntos de entrenamiento, validación y prueba. Se utilizó OpenCV y PIL para el preprocesamiento, incluyendo el redimensionamiento y la aplicación de padding para mantener la proporción de las imágenes.

### Parte 2: Denoising Autoencoder (DAE)
- **Entrenamiento**: Se construyó un Autoencoder para eliminar ruido de las imágenes. Las imágenes del conjunto de entrenamiento se ensuciaron sintéticamente con ruido gaussiano.
- **Evaluación**: El modelo fue evaluado con imágenes de prueba que también fueron ensuciadas. Se utilizó Weights & Biases (W&B) para registrar el entrenamiento, métricas y pérdidas.

### Parte 3: Variational Autoencoder (VAE) para Generación de Imágenes
- **Entrenamiento**: Se implementó un VAE para la generación de imágenes sintéticas de las dos clases (Coca-Cola y Sprite). Se utilizó una métrica personalizada basada en DenseNet121 para evaluar la calidad de las imágenes generadas.
- **Evaluación**: Se compararon las imágenes generadas con las imágenes reales utilizando la métrica de distancia euclidiana basada en características extraídas por DenseNet121.

### Parte 4: Demo Interactivo
- **Interfaz gráfica**: Se creó una interfaz gráfica interactiva utilizando Gradio para probar ambos modelos (DAE y VAE). La interfaz permite subir imágenes con ruido para su restauración y generar imágenes sintéticas.
- **Hosting**: El demo está hosteado en Hugging Face Spaces.

## Herramientas y Tecnologías Utilizadas
- **Lenguaje de programación**: Python
- **Librerías principales**: TensorFlow, Keras, OpenCV, PIL, NumPy, Gradio, Weights & Biases (W&B), DenseNet121.
- **Plataformas**: Google Colab, Hugging Face Spaces, Weights & Biases.

## Enlaces Relevantes
- **Demo en Hugging Face**: [Proyect1_DAE_VAE](https://huggingface.co/spaces/melanieeemichel/Proyect1_DAE_VAE)
- **Reporte DAE Weights & Biases**: [Reporte_DAE_WAB](https://api.wandb.ai/links/melaniemichelrod-iteso/2kbint90)
- **Reporte VAE Weights & Biases**: [Reporte_VAE_WAB]()
