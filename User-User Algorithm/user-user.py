import numpy as np
import pandas as pd
import sklearn as skl
import chardet as cd
from scipy.spatial.distance import pdist, squareform
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.metrics.pairwise as cos


#Read CSV Files 

with open('movie-tags.csv', 'rb') as f:
    result = cd.detect(f.read())  # or readline if the file is large

tags = pd.read_csv('movie-tags.csv', header=None, encoding=result['encoding'],names=['movie_id', 'movie_tag'])
movies = pd.read_csv('movie-titles.csv', header=None, encoding=result['encoding'],names=['movie_id', 'movie', 'genres'])
ratings = pd.read_csv('ratings.csv', header=None, encoding=result['encoding'],names=['user_id', 'movie_id', 'rating'])

#find mean rating of all Users
users_ratings = ratings.groupby('user_id')
mean_rating_by_users = users_ratings.transform('mean')

#subtract mean rating from each user rating
adjusted_ratings = ratings.copy()
adjusted_ratings['rating'] = ratings['rating'] - mean_rating_by_users['rating']

#create user item rating matrix
user_item_matrix = adjusted_ratings.pivot(index='user_id', columns='movie_id', values='rating')
user_item_matrix = user_item_matrix.fillna(0)

movie_ratings = adjusted_ratings.groupby('movie_id')

#find similar users 
similarities = squareform(1 - pdist(user_item_matrix, 'cosine'))
similarity_matrix = pd.DataFrame(similarities,
                                 index=user_item_matrix.index, columns=user_item_matrix.index)


#find top 30 similar users by creating neighbourhood of 30 users
def find_similar_users(target_user, target_item, user_limit=30):
    similar_users = similarity_matrix[target_user]
    users_rated_target_item = find_users_rated_item(target_item)
    similar_users = similar_users[users_rated_target_item]
    similar_users = similar_users.nlargest(30)
    return similar_users

#find users who rated the target item
def find_users_rated_item(item_id):
    return movie_ratings.get_group(item_id)['user_id'].values

# get adjusted scores
def get_adjusted_score(target_user, target_item):
    similarities = find_similar_users(target_user, target_item)
    r = user_item_matrix.loc[similarities.index]
    return sum(r[target_item] * similarities)/sum(abs(similarities))


# get perdeictions
def get_predictions(target_user, item_limit=10):
    if not target_user in similarity_matrix:
        print ("target user: ", target_user, "not found")
        return

    target_user_rating_mean = users_ratings.get_group(target_user)['rating'].mean()

    target_item = [x for x in movies['movie_id'].values
                       if not x in users_ratings.get_group(target_user)['movie_id'].values]

    if not isinstance(target_item, list):
        return target_user_rating_mean + get_adjusted_score(target_user, target_item)
    
    
    predictions = {}
    for item in target_item:
        mean_adjusted_rating = get_adjusted_score(target_user, item)
        predictions[item] = target_user_rating_mean + mean_adjusted_rating
    return pd.Series(predictions).nlargest(item_limit)

recommendation = get_predictions(320)
recommendation_movie_ids = recommendation.keys().tolist();
recommendation_similarity = recommendation.as_matrix()

recommendation_movie_names = []

for movie in recommendation_movie_ids:
    movie_name = movies.query('movie_id=='+ str(movie))['movie']
    str_movie_name = str(movie_name)
    str_movie_name = str_movie_name.replace("\nName: movie, dtype: object"," ")
    recommendation_movie_names.append(str_movie_name[4:])

recommendation_df = pd.DataFrame(
    {'movie_id': recommendation_movie_ids,
     'similarities': recommendation_similarity,
     'movie_name': recommendation_movie_names
    })

print(recommendation_df)