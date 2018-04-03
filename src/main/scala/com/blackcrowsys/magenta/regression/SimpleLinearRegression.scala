package com.blackcrowsys.magenta.regression

import com.blackcrowsys.crimson.common.Tolerence
import com.blackcrowsys.crimson.matrix.Matrix
import com.blackcrowsys.crimson.statistics.Stats

/**
  * Simple Linear Regression
  * A simple linear regression is of the form Y = Ax + B
  * where x is one of the columns in the matrix and Y is the final column (the result)
  * A more complicated example is where X is the vector input and A and B are coefficient
  * matrices.
  * For the purposes of this object, an augmented matrix is the one that contains the output/result
  * in the final column.
  */
object SimpleLinearRegression {

  def sumOfXDeltaTimesYDelta(xArray: Array[Double], yArray: Array[Double], xMean: Double, yMean: Double): Double = {
    var result: Array[Double] = for ((x, y) <- xArray zip yArray) yield (x - xMean) * (y - yMean)
    result.sum
  }

  def sumOfXDeltaSquared(xArray: Array[Double], xMean: Double): Double = {
    val squared: Array[Double] = for (x <- xArray) yield Math.pow(x - xMean, 2)
    squared.sum
  }

  /**
    * A simple relationship between one input and out
    *
    * @param data   augmented matrix containing data
    * @param column the column to infer relationship from
    * @return tuple containing beta0 and beta1
    */
  def simpleInference(data: Matrix.Matrix, column: Int): (Double, Double) = {
    val yMean: Double = Stats.mean(data, data.columns)
    val xMean: Double = Stats.mean(data, column)

    val x: Array[Double] = data.getColumnAsArray(column)
    val y: Array[Double] = data.getColumnAsArray(data.columns)

    val beta1: Double = sumOfXDeltaTimesYDelta(x, y, xMean, yMean) / sumOfXDeltaSquared(x, xMean)
    val beta0 = yMean - beta1 * xMean
    (beta0, beta1)
  }

}
