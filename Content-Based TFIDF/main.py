import numpy as np
import pandas as pd
import sklearn as skl
import seaborn as sns
import matplotlib as plt
import chardet as cd
from scipy.spatial.distance import cdist
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.metrics.pairwise as cos


#Read CSV Files 

with open('movie-tags.csv', 'rb') as f:
    result = cd.detect(f.read())  # or readline if the file is large

tags = pd.read_csv('movie-tags.csv', header=None, encoding=result['encoding'],names=['movie_id', 'movie_tag'])
movies = pd.read_csv('movie-titles.csv', header=None, encoding=result['encoding'],names=['movie_id', 'movie', 'genres'])
ratings = pd.read_csv('ratings.csv', header=None, encoding=result['encoding'],names=['user_id', 'movie_id', 'rating'])

#first initializing item_tag_dict

item_tag_dict = {}
for movie in movies['movie_id'].tolist():
    item_tag_dict[movie] = {}

# then, filling item_tag_dict

item_tags = tags.groupby('movie_id')
for item in item_tags:
    item_i_tags = item[1]['movie_tag'].value_counts().to_dict()
    item_tag_dict[item[0]] = item_i_tags
Q = pd.DataFrame.from_dict(item_tag_dict, orient='index')

# also, we need to put items with no tag count in Q, and sort it based on item_id

item_with_no_tags = [i for i in item_tag_dict.keys() if len(item_tag_dict[i]) == 0]
for item in item_with_no_tags:
    Q.loc[item] = np.nan
Q = Q.sort_index()

#Step 2: Iterate through each item again, performing the following:
#      a. Divide each term value 𝑞̂𝑖𝑡 by the log of the document frequency (𝑙𝑛 𝑑𝑡). The resulting vector 𝒒𝑖 is the TF-IDF vector.

df = Q.count()
lnn = np.log(Q.shape[0])
idf = lnn - np.log(df)
Q = Q.fillna(0)
Q = Q*idf

#  b. After dividing each term value by the log of the DF, compute the length (Euclidean norm) of the TF-IDF vector 𝒒𝑖, and divide each element of it by the length to yield a unit vector 𝒒𝑖.

Q = Q.div(np.sqrt(np.square(Q).sum(axis=1)), axis=0)
Q = Q.fillna(0)

#Build user profile for each query user
#The profile is the sum of the item-tag vectors of all items the user has rated positively (>= 3.5 stars)
#first building the user rating matrix (movies * users)

item_rating_dict = {}
item_ratings = ratings.groupby('movie_id')
for item in item_ratings:
    item_i_ratings = item[1][['user_id','rating']].set_index('user_id').to_dict()['rating']
    item_rating_dict[item[0]] = item_i_ratings
R = pd.DataFrame.from_dict(item_rating_dict, orient='index')
R = R.sort_index()
R = R.T

# then, converting ratings to 1 if ratings>=3.5 otherwise 0

R35 = R.copy()
R35[R35<3.5]=0
R35[R35>=3.5]=1
R35 = R35.fillna(0)

#finally, doing the dot product

P = R35.dot(Q)

# cosine distance

c = cdist(P.values, Q.values, 'cosine')

#cosine similarity score matrix (user * item)
S = pd.DataFrame(1 - c, index=P.index, columns=Q.index)

# Return recommendations for a user
def GetTopTenForUser(user_id):
    s = S.loc[user_id]
    s = s[s.notnull() & s != 0].sort_values()
    return s.nlargest(10)

#print(GetTopTenForUser(320))

recomm_movies = GetTopTenForUser(320)
#print(recomm_movies)
recomm_movie_ids = recomm_movies[1:]
print("Recommeded Movies unweighted")
print(recomm_movie_ids)

#Part 2: Weighted User Profile

RW = R.copy()
mu = RW.mean(axis=1)
W = RW.sub(mu, axis=0)
W = W.fillna(0)
PW = W.dot(Q)

cw = cdist(PW.values, Q.values, 'cosine')
SW = pd.DataFrame(1 - cw, index=PW.index, columns=Q.index)

def GetTopTenForUser_Weighted(user_id):
    sw = SW.loc[user_id]
    sw = sw[sw.notnull() & sw != 0].sort_values()
    return sw.nlargest(10)

recomm_weighted = GetTopTenForUser_Weighted(320)


print("Recommeded Movies weighted")
print(recomm_weighted)




