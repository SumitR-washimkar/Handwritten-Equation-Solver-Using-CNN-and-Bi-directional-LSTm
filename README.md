# Image-to-LaTeX Model

This repository contains a deep learning model that converts input images into LaTeX code using a CNN encoder and RNN decoder with attention. It includes modularized code for training, prediction, and tokenizer handling.

---

## 📁 Repository Structure

``` bash

.
├── Modules/                                 # Contains core Python modules (model components, utils, etc.)
│   ├── _init_.py
│   ├── custom_collate_fn.py
│   ├── Decoder.py
│   ├── EarlyStopping.py
|   ├── Encoder.py
|   ├── EquationSeqDatasat.py
|   ├── Sequence.py
|   └── Tokenizer.py
├── Model/                                    # Contains model training, prediction, and tokenizer
│   ├── Final Model.py                        # Training loop for the model
│   ├── Prediction_File.py                    # Prediction script for test images
│   └── tokenizer.pth                         # Serialized tokenizer for LaTeX tokens
├── Dataset/                                  # Dataset folder
│   ├── Handwritten Equation Dataset.zip      # Compressed dataset of all input images
|   ├── Demo Image
|   └── caption_data.csv
└── README.md                                 # Project documentation
