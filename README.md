# Multimodal Movie Genre Classification

This project predicts multiple movie genres using plot-text features and movie-poster features from the MM-IMDb dataset.

## Project Overview

Movie genre prediction is a multi-label classification problem because one movie can belong to several genres. This project compares four approaches:

1. Text-only classification
2. Poster-only classification
3. Early fusion of text and poster features
4. Late fusion of text and poster predictions

## Dataset

The project uses the MM-IMDb dataset available from Kaggle:

https://www.kaggle.com/datasets/johnarevalo/mmimdb

The dataset contains 25,959 movies and 23 possible genres.

The original dataset split was retained:

- Training: 15,552 movies
- Validation: 2,608 movies
- Test: 7,799 movies

The supplied features include:

- 300-dimensional text features
- 4,096-dimensional VGG poster features
- 23 binary genre labels

The dataset is downloaded automatically using `kagglehub` and is not included in this repository.

## Methodology

Text and poster features were standardised separately using statistics calculated only from the training data.

The neural networks used two hidden layers with ReLU activation and dropout. Binary cross-entropy with logits was used because each genre represents a separate binary prediction.

Genre-specific probability thresholds were selected using the validation set and then applied unchanged to the test set.

## Results

| Model | Test micro-F1 | Test macro-F1 | Mean per-genre AP |
|---|---:|---:|---:|
| Poster-only | 0.4240 | 0.3326 | 0.2972 |
| Early fusion | 0.5958 | 0.5072 | 0.5154 |
| Text-only | 0.6149 | 0.5363 | 0.5434 |
| Late fusion | **0.6274** | **0.5567** | **0.5707** |

Late fusion produced the strongest overall test performance.

Additional late-fusion results:

- Exact-match accuracy: 13.4%
- Hamming loss: 8.7%
- Micro precision: 58.6%
- Micro recall: 67.6%

## Running the Notebook

1. Clone or download this repository.
2. Install the required packages:

   ```bash
   pip install -r requirements.txt