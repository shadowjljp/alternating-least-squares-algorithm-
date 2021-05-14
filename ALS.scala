// Databricks notebook source
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating

// Load and parse the data
val data = sc.textFile("/FileStore/tables/ratings.dat")
val ratings = data.map(_.split("::") match { case Array(user, item, rate, timestamp) =>
  Rating(user.toInt, item.toInt, rate.toDouble)
})
//split the data 60% for training and 40% for testing
val split = ratings.randomSplit(Array(0.6,0.4))
val train = split(0)
val test = split(1)

// Build the recommendation model using ALS
val rank = 10
val numIterations = 10
val model = ALS.train(train, rank, numIterations, 0.01)

// Evaluate the model on rating data
val usersProducts = test.map { case Rating(user, product, rate) =>
  (user, product)
}
val predictions =
  model.predict(usersProducts).map { case Rating(user, product, rate) =>
    ((user, product), rate)
  }
val ratesAndPreds = test.map { case Rating(user, product, rate) =>
  ((user, product), rate)
}.join(predictions)

val MSE = ratesAndPreds.map { case ((user, product), (r1, r2)) =>
  val err = (r1 - r2)
  err * err
}.mean()
println(s"Mean Squared Error = $MSE")

// Save and load model
// model.save(sc, "target/tmp/myCollaborativeFilter")
// val sameModel = MatrixFactorizationModel.load(sc, "target/tmp/myCollaborativeFilter")

// COMMAND ----------

ratesAndPreds.take(10).foreach(println)

// COMMAND ----------

predictions.take(10).foreach(println)

// COMMAND ----------

usersProducts.take(10).foreach(println)

// COMMAND ----------

import scala.util.Random

val x: Int = Random.nextInt(10)

x match {
  case 0 => "zero"
  case 1 => "one"
  case 2 => "two"
  case _ => "other"
}

// COMMAND ----------

sealed abstract class Furniture
case class Couch() extends Furniture
case class Chair() extends Furniture
case class Zhair() extends Furniture
def findPlaceToSit(piece: Furniture): String = piece match {
  case a: Couch => "Lie on the couch"
  case b: Chair => "Sit on the chair"
}


// COMMAND ----------


