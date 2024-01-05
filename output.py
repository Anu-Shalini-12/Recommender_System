import pandas as pd
from surprise import Reader, Dataset
from surprise.model_selection import train_test_split
from surprise import SVD, KNNBasic
from surprise import accuracy

# Load the dataset with specified data types
file_path = r'C:\Users\0042H8744\customer review.csv'
df = pd.read_csv(file_path, dtype={'reviews.rating': float})

# Choose relevant columns
df = df[['reviews.username', 'asins', 'reviews.rating']]

# Rename columns for Surprise library
df.columns = ['user', 'item', 'rating']

# Drop rows with missing values
df = df.dropna()

# Create a Surprise dataset
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df, reader)

# Split the data into training and testing sets
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Build the collaborative filtering model (SVD algorithm)
model = SVD()
model.fit(trainset)

# Make predictions on the test set
predictions = model.test(testset)

# Evaluate the model
rmse = accuracy.rmse(predictions)
print(f"RMSE: {rmse}")

# Build a user-based collaborative filtering model (KNNBasic algorithm)
knn_model = KNNBasic(sim_options={'user_based': True})
knn_model.fit(trainset)

# Example: Recommend top 2 products for a list of users
users_to_recommend = ['truman', 'Dave', 'James']

for user_to_recommend in users_to_recommend:
    try:
        user_id = trainset.to_inner_uid(user_to_recommend)
        user_ratings = knn_model.get_neighbors(user_id, k=2)
        # Filter out invalid inner item IDs
        valid_user_ratings = [iid for iid in user_ratings if iid in trainset.all_items()]
        # Convert inner item IDs back to original ASINs
        recommended_asins = [trainset.to_raw_iid(iid) for iid in valid_user_ratings]
        print(f"Top 2 recommendations for {user_to_recommend}: {recommended_asins}")
    except ValueError as e:
        print(f"Error: {e}")
