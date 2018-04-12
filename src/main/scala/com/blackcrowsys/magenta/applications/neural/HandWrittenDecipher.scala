package com.blackcrowsys.magenta.applications.neural

import com.blackcrowsys.crimson.common.Tolerence
import com.blackcrowsys.crimson.functions.Functions
import com.blackcrowsys.crimson.matrix.Matrix
import com.blackcrowsys.crimson.matrix.Matrix.Matrix
import com.blackcrowsys.magenta.neural.ThreeLayer._

import scala.collection.GenTraversableOnce
import scala.io.Source

object HandWrittenDecipher {

  def createMatrix(dataFile: String, inputNodes: Int, outputNodes: Int): Matrix = {
    var contents: Array[Double] = Array()
    for (line <- Source.fromFile(dataFile).getLines()) {
      contents = contents ++ convertLineToArray(line, inputNodes)
    }
    Matrix.create(contents, inputNodes + outputNodes)
  }

  private def convertLineToArray(line: String, inputNodes: Int): Array[Double] = {
    val tokens: Array[String] = line.split(",")
    val number: Int = tokens(0).toInt
    val input: Array[Double] = normaliseInputs(tokens.slice(1, tokens.length), inputNodes)
    val targets: Array[Double] = normaliseTarget(number)
    input ++ targets
  }

  private def normaliseTarget(number: Int): Array[Double] = {
    val targets: Array[Double] = (for (i <- 0 until 10) yield 0.1).toArray
    targets(number) = 0.9
    targets
  }


  private def normaliseInputs(inputs: Array[String], inputNodes: Int): Array[Double] = {
    (for (i <- 0 until inputNodes) yield convertStringToNormalised(inputs(i))).toArray
  }

  private def convertStringToNormalised(aNumber: String): Double = {
    aNumber.toDouble / 255 * 0.9
  }

  def createDecipher(inputNodes: Int, hiddenNodes: Int, outputNodes: Int, learningRate: Double): HandWrittenDecipher = {
    new HandWrittenDecipher(inputNodes, hiddenNodes, outputNodes, learningRate)
  }


  class HandWrittenDecipher(inputNodes: Int, hiddenNodes: Int, outputNodes: Int, learningRate: Double) {

    private val tolerence: Tolerence = new Tolerence(0.0001)
    private val activation = (x: Double) => Functions.sigmoid(x)
    private val delta = (error: Matrix, output: Matrix, input: Matrix) => (error * output * (Matrix.create(output.rows, 1, 1) - output)).dot(input.transpose)

    val network: ThreeLayer = create(inputNodes, hiddenNodes, outputNodes, tolerence,
      learningRate, activation, delta)

    def train(trainingData: Matrix): Tuple2[Matrix, Matrix] = {
      network.train(trainingData)
    }

    def test(testData: Matrix): Double = {
      val contents: Array[Array[Double]] = testData.rowArray()
      var success: Int = 0
      for (row <- contents) {
        if (testQuery(row)) {
          success += 1
        }
      }
      (success.toDouble / contents.length.toDouble) * 100
    }

    private def testQuery(row: Array[Double]): Boolean = {
      val io: Tuple2[Matrix, Matrix] = splitTest(row)
      val result: Matrix = network.query(io._1)
      compareResult(io._2, result)
    }

    private def splitTest(row: Array[Double]): (Matrix, Matrix) = {
      val input: Array[Double] = row.slice(0, inputNodes)
      val output: Array[Double] = row.slice(inputNodes, row.length)
      (Matrix.create(input, 1), Matrix.create(output, 1))
    }

    private def compareResult(expected: Matrix, actual: Matrix): Boolean = {
      val expectedNumber: Int = getValueWithHighestProbability(expected)
      val actualNumber: Int = getValueWithHighestProbability(actual)
      expectedNumber == actualNumber
    }

    def getValueWithHighestProbability(expected: Matrix): Int = {
      val max: Double = expected.contents.max
      val index: Int = expected.contents.indexOf(max)
      index
    }
  }

}
