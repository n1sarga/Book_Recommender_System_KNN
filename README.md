# Collaborative Filtering for Book Recommendations

This project implements collaborative filtering using k-Nearest Neighbors (k-NN) to identify user clusters based on common book ratings. It then predicts book ratings for users based on the ratings of their nearest neighbors. Additionally, the project combines rating data with total rating count data to determine popular books and exclude less popular ones.

## Features

* k-NN Collaborative Filtering: Identifies user clusters and predicts book ratings using the top k-nearest neighbor average rating.
* Popular Book Identification: Combines rating data with total rating count data to determine popular books, excluding less popular ones.
* Recommendation Functions: Includes Python functions to:
* Show the top-10 recommended books for a given user ID.
* Predict the probable rating of a book by a given user ID and ISBN.

## Requirements
![Python 3](https://img.shields.io/badge/Python-3-blue)
![pandas](https://img.shields.io/badge/pandas-1.3.3-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24.2-blue)
