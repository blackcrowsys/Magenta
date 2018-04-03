package com.blackcrowsys.magenta.regression

import com.blackcrowsys.crimson.common.Tolerence
import com.blackcrowsys.crimson.matrix.Matrix._
import org.scalatest.FunSuite

class SimpleLinearRegressionTests extends FunSuite {

  val testData: Matrix = createFromResourceFile("Advertising.csv", true, true)

  test("it should give the right parameters for given column") {
    val expectedBeta0: Double = 7.032593549127705
    val expectedBeta1: Double = 0.04753664043301969

    val actual: Tuple2[Double, Double] = SimpleLinearRegression.simpleInference(testData, 1)

    assert(actual._1 == expectedBeta0)
    assert(actual._2 == expectedBeta1)
  }

}
