# DermaNet

## Overview

DermaNet is a skin disease detection system using EfficientNet. It is designed to classify various skin conditions based on images of affected skin areas. The project leverages TensorFlow and EfficientNet for image preprocessing and classification.

**Note**: This project is currently a work in progress. The model's accuracy is still being improved, and results may vary depending on the dataset.

## Technologies Used
- **Python**
- **TensorFlow/Keras** (for deep learning model and image processing)
- **Streamlit** (for the web interface)

## Installation

1. Clone this repository:

    ```bash
    git clone <your-repo-url>
    ```

2. Create and activate a Python virtual environment:

    ```bash
    python -m venv myenv
    source myenv/bin/activate  # On Windows use `myenv\Scripts\activate`
    ```
   
3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Run the Streamlit app:

    ```bash
   streamlit run app.py
    ```
    
## Acknowledgments
- The skin disease dataset used in this project is sourced from DermNet.
