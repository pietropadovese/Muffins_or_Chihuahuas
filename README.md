# Muffins_or_Chihuahuas
 Project repository as a part of a university machine learning course. This project aims to classify images into two categories: Chihuahuas and Muffins testing different neural network architectures.

## Repository Structure
- **dataset**: Contains Chihuahuas and Muffins images divided into training and validation sets.
- **models**: contain subfolders for each implemented model. They include a python script to construct the model architecture, the model saved in .keras format and CSV file containing training and validation metrics.
- **cnn_01.ipynb, cnn_02.ipynb, residual_learning.ipynb**: Notebooks where the three models are trained and tested.
- **cvv.ipynb**, a comparison between the cnn_02 and residual learning models. This analysis employs 5-fold cross-validation to assess their performance, focusing on the 0-1 loss metric.
