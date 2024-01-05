
# Recommender System for Amazon Customer Reviews

## Objective:

The objective of this code is to implement a recommender system for Amazon Customer Reviews using collaborative filtering algorithms. The script suggests products to users based on their preferences and behavior.

## File Structure:

- **`output.py`:** Main script for building and evaluating the recommender system.
- **`customer review.csv`:** Dataset containing Amazon Customer Reviews.

## Dependencies:

- [Pandas](https://pandas.pydata.org/): Used for data manipulation and analysis.
- [Surprise](https://surprise.readthedocs.io/): A Python scikit for building and analyzing collaborative filtering recommender systems.

## Steps to Run:

1. **Install Dependencies:**
   ```bash
   pip install pandas scikit-surprise

2. **Update File Path:**
    - Update the `file_path` variable with the correct path to your CSV file.

3. **Run the Code:**
    - Save the code in a Python file (e.g., `output.py`) and run it using:
        ```bash
        python output.py
        ```
# Review Outputs

## RMSE values for the SVD Model:

The Root Mean Squared Error (RMSE) is a measure of how well the recommender system's predictions align with the actual ratings provided by users. Lower RMSE values indicate better predictive accuracy.

- **RMSE Value:** [Insert RMSE Value]

## Top 2 Product Recommendations for Specified Users:

The script provides top 2 product recommendations for specific users based on the collaborative filtering model using the KNNBasic algorithm. These recommendations aim to suggest items that align with the user's preferences and behavior.

- **User: Truman**
  - Recommendation 1: [Product ASIN 1]
  - Recommendation 2: [Product ASIN 2]

- **User: Dave**
  - Recommendation 1: [Product ASIN 3]
  - Recommendation 2: [Product ASIN 4]

- **User: James**
  - Recommendation 1: [Product ASIN 5]
  - Recommendation 2: [Product ASIN 6]

Note: ASIN represents Amazon Standard Identification Numbers.

Adjustments and further analysis can be performed based on these recommendations to enhance the effectiveness of the recommender system.

# Author

- Anu-Shalini-12




