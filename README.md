# Plant Disease Recognition System

![Plant Disease Recognition](home_page.jpeg)

## Project Overview

The Plant Disease Recognition System is a comprehensive solution for identifying diseases in various crops using machine learning and computer vision techniques. This system enables farmers and agricultural specialists to quickly diagnose plant diseases by simply uploading an image, potentially saving crops and increasing agricultural productivity.

## Features

- **Image-based Disease Detection**: Upload images of plant leaves to detect diseases
- **Multi-crop Support**: Identifies diseases across various crops including apple, corn, grape, potato, tomato, and many others
- **User-friendly Interface**: Intuitive Streamlit web application for easy interaction
- **High Accuracy Model**: Utilizes state-of-the-art deep learning models (ResNet50) for accurate disease classification
- **Detailed Analysis**: Provides classification results for 38+ different crop-disease combinations

## Dataset

The project uses a comprehensive dataset of crop diseases consisting of approximately 87,000 RGB images of healthy and diseased crop leaves categorized into 38+ different classes. The dataset includes:

- Training set: ~70,000 images
- Validation set: ~17,500 images 
- Test set: 33 images for final evaluation

## Project Structure

```
sentra-bot/
├── main.py                           # Streamlit web application
├── trainer_notebook.py               # Model training pipeline
├── crop_disease_dataset_scraper.py   # Data collection utility
├── requirements.txt                  # Project dependencies
├── dataset/                          # Dataset directory with categorized images
├── models/                           # Saved model files
└── results/                          # Evaluation metrics and visualizations
```

## Technologies Used

- **Python**: Primary programming language
- **PyTorch**: Deep learning framework for model training and evaluation
- **TensorFlow/Keras**: For model deployment
- **Streamlit**: Web application framework
- **OpenCV & PIL**: Image processing
- **scikit-learn**: Evaluation metrics
- **Matplotlib & Seaborn**: Data visualization

## Getting Started

### Prerequisites

- Python 3.8+
- Required packages (install using `pip install -r requirements.txt`)

### Installation

1. Clone this repository
   ```bash
   git clone https://github.com/yourusername/sentra-bot.git
   cd sentra-bot
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Run the web application
   ```bash
   streamlit run main.py
   ```

### Using the Application

1. Open the web application in your browser (typically at http://localhost:8501)
2. Navigate to the "Disease Recognition" page
3. Upload an image of a plant leaf
4. Click "Predict" to get the disease classification result

## Model Training

The model training pipeline in `trainer_notebook.py` includes:

- Data loading and preprocessing
- Data augmentation techniques
- Model architecture selection (ResNet50, ResNet18, or custom CNN)
- Transfer learning implementation
- Training with early stopping
- Model evaluation with detailed metrics
- Result visualization and model saving

To train a new model:

```bash
python trainer_notebook.py
```

## Performance

The system achieves high accuracy in disease classification, with comprehensive evaluation metrics including:
- Confusion matrix visualization
- Classification reports
- Accuracy measurements
- Sample predictions

## Future Improvements

- Mobile application deployment
- Real-time disease detection using camera feed
- Additional crop disease categories
- Treatment recommendations based on detected diseases
- Integration with agricultural management systems

## License

[MIT License](LICENSE)

## Acknowledgments

- Original plant disease dataset creators
- Contributors to open-source libraries used in this project