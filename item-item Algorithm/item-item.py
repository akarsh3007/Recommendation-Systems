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

#find mean rating of all movies
movies_ratings = ratings.groupby('movie_id')
mean_rating_by_movie = movies_ratings.transform('mean')
item_mean_ratings = movies_ratings.mean()

#subtract mean rating from each user rating
adjusted_ratings = ratings.copy()
adjusted_ratings['rating'] = ratings['rating'] - mean_rating_by_movie['rating']

#create user item rating matrix
user_item_matrix = adjusted_ratings.pivot(index='movie_id', columns='user_id', values='rating')
user_item_matrix = user_item_matrix.fillna(0)

user_ratings = adjusted_ratings.groupby('user_id')

#find similar users 
similarities = squareform(1 - pdist(user_item_matrix, 'cosine'))
similarity_matrix = pd.DataFrame(similarities,
                                 index=user_item_matrix.index, columns=user_item_matrix.index)


#find top 30 similar items by creating neighbourhood of 30 users
def find_similar_items(target_user, target_item, item_neighbor_limit=20):
    similar_items = similarity_matrix[target_item]
    items_rated_target_user = find_items_rated_by_user(target_user)
    similar_items = similar_items[items_rated_target_user]
    similar_items = similar_items.nlargest(item_neighbor_limit)
    return similar_items[similar_items > 0]

def find_items_rated_by_user(user_id):
    return user_ratings.get_group(user_id)['movie_id'].values

# get adjusted scores
def get_adjusted_score(target_user, target_item):
    similarities = find_similar_items(target_user, target_item)
    r = user_item_matrix.loc[similarities.index]
    return sum(r[target_user] * similarities)/sum(abs(similarities))

# get perdeictions
def get_predictions(target_user, target_item=None, top_n=10):
    if not target_user in user_ratings.groups:
        print("this user has no ratings: ", target_user)
        return
    
    if target_item == None:
        target_item = [x for x in movies['movie_id'].values
                       if not x in user_ratings.get_group(target_user)['movie_id'].values]
    
    if not isinstance(target_item, list):
        target_item_rating_mean = item_mean_ratings.loc[target_item]['rating']
        return target_item_rating_mean + get_adjusted_score(target_user, target_item)
    
    predictions = {}
    for item in target_item:
        mean_adjusted_rating = get_adjusted_score(target_user, item)
        target_item_rating_mean = item_mean_ratings.loc[item]['rating']
        predictions[item] = target_item_rating_mean + mean_adjusted_rating
    return pd.Series(predictions).nlargest(top_n)

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
