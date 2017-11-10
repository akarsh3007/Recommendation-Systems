// Databricks notebook source
val read_movies = sc.textFile("/FileStore/tables/who758r41509559668657/movies.dat")
val read_ratings = sc.textFile("/FileStore/tables/who758r41509559668657/ratings.dat")

// COMMAND ----------

read_movies.first()
read_ratings.first()

// COMMAND ----------

val formatted_ratings = read_ratings.map(_.split("::").take(3))

// COMMAND ----------

formatted_ratings.first()

// COMMAND ----------

import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.Rating

// COMMAND ----------

val ratings = formatted_ratings.map { case Array(user,movie,rating) => Rating(user.toInt, movie.toInt,rating.toDouble) }

// COMMAND ----------

ratings.first()

// COMMAND ----------

val model = ALS.train(ratings,50,10,0.01)

// COMMAND ----------

val predicted_rating = model.predict(786,123)

// COMMAND ----------

val recommended_movies = model.recommendProducts(789,10)

// COMMAND ----------

val movie_for_user = ratings.keyBy(_.user).lookup(789)

// COMMAND ----------

val titles = read_movies.map(line => line.split("::").take(2)).map(array => (array(0).toInt,array(1)))

// COMMAND ----------

val rec_movies = sc.parallelize(recommended_movies)

// COMMAND ----------

val rec_moviesKV = rec_movies.map(x=> (x.product,x))

// COMMAND ----------

rec_moviesKV.first()

// COMMAND ----------

titles.first()

// COMMAND ----------

val results = rec_moviesKV.join(titles)

// COMMAND ----------

val movies_recommendation = results.map(x => List(x._2._1.user,x._2._1.product,x._2._1.rating,x._2._2))

// COMMAND ----------

movies_recommendation.collect().foreach(println)
