# shape-rotator
Generative model that learns to rotate shapes

# STAT 946 Kaggle Data Challenge Details
The following information is copy pasted from the kaggle challenge.

"""
STAT946 Data Challenge: CIFAR Image Transformation
Train a generative model to transform MNIST digits into their rotated versions—no direct rotation methods allowed!


STAT946 Data Challenge: CIFAR Image Transformation

Submit Prediction
Overview
In this competition, you will develop a generative model to transform input images into their corresponding rotated versions. The dataset consists of images of different objects from CIFAR100 dataset, where each sample contains:

An input image X: An image in its original orientation.
An output image Y: The same image, rotated by either 90° or 180°.
The goal is to build a generative model that learns this transformation and produces same-quality output images. Directly rotating images using deterministic transformations is strictly prohibited.
Students will be graded based on their performance on the leaderboard.
Note: This competition has both a public and a private leaderboard, and the final grading is determined by the private leaderboard.

Note: Please check the Rules section

Start

11 days ago
Close
7 days to go
Description
In this competition, your goal is to develop a generative model that learns to transform an input image into its corresponding rotated version, rather than applying predefined transformations. The dataset consists of images of different objects from CIFAR100 dataset, where each input image is rotated by either 90° or 180° to produce the target output. It is not clear which image is rotated 90° or 180° degree.



Goal
Train a generative model that takes an input image and generates a corresponding rotated version (90° or 180°),

Evaluation
The evaluation metric for this competition is the mean squared error (MSE) between the generated images and the ground truth, where a lower MSE indicates better performance.

Your grade is determined using the following formula:

Grade=100−1000×MSE

Example:
If MSE = 0.05, then Grade = 50

The final grade will be based on the private leaderboard score.

Please note that at the end of the data challenge, grades may be normalized based on the overall scoreboard results.
Dataset Description
The dataset is based on CIFAR100 and contains 3x32×32 images of different objects. Each sample includes:

Input Image X: An image in its original orientation.
Output Image Y: The same image, rotated by either 90° or 180°.
Dataset Split
Training Set: Provided for training, containing both X and Y.
Test Set: Used for leaderboard evaluation, containing only X (models must generate Y as predictions).
You can split part of the training set for validation purposes.
Note: This competition has both a public and a private leaderboard, and the final grading is determined by the private leaderboard.

Files
train_dataset_input_images.csv - contains the input images of the train set
train_dataset_output_images.csv - contains the output images of the train set
test_dataset_input_images.csv - contains the input images of the test set
sample_submission.csv - It's similar to test_dataset_input_images.csv and every row contains the vectored version of the test output images
loading the data
in order to load the dataset, you can use the following code to reconstruct the image matrices from the csv files:

import pandas as pd
def get_images_from_csv(csv_file_path):
    """
    Loads a CSV file, converts pixel vectors back to images, and returns a list of images.

    Args:
        csv_file_path: The path to the CSV file.

    Returns:
        A list of reshaped images. (3 x 32 x 32)
    """
    try:
        df = pd.read_csv(csv_file_path)
        images = []
        for index, row in df.iloc[:, 1:].iterrows():  # Exclude the 'ID' column
            pixel_vector = row.values
            image = pixel_vector.reshape(3, 32, 32)  # Reshape to original CIFAR image size
            images.append(image)
        return images
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_file_path}")
        return None

Competition Rules
The data challenge will run for two weeks, starting on March 3, 2025, at 11:59 PM and ending on March 17, 2025, at 11:59 PM.

Individual Participation
Students must work independently—no team submissions are allowed.

You cannot use a classifier to predict the rotation (e.g., a model that first classifies whether the digit should be rotated 90° or 180° and then applies the transformation).

You must use a generative model that directly learns the transformation, such as an autoencoder (e.g., U-Net), GAN, or denoising diffusion models.

5- Register your Kaggle username in the following google doc.
https://docs.google.com/forms/d/1EDawGS7wPrdzCIsc19LxGFSz6x19SAb3M5N-cx9Mdfo/viewform?pli=1&pli=1&edit_requested=true&fbzx=-4652525671867144894

6- At the end of the data challenge, you must upload your solution code and a brief report (maximum of 2 pages) to the designated dropbox in Learn. (Dead line is March 19, 2025, at 11:59 PM.)

7- Private leaderboard results will be released at March 19, 2025

8- There is a 5 submission limitation per day
"""