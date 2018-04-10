package com.blackcrowsys.magenta.neural

import com.blackcrowsys.crimson.common.Tolerence
import com.blackcrowsys.crimson.functions.Functions
import com.blackcrowsys.crimson.matrix.Matrix
import com.blackcrowsys.crimson.matrix.Matrix.Matrix

object ThreeLayer {

  case class ThreeLayer(
                         inputNodes: Int,
                         hiddenNodes: Int,
                         outputNodes: Int,
                         tolerence: Tolerence,
                         learningRate: Double,
                         inputWeigths: Matrix,
                         outputWeights: Matrix,
                         activation: Double => Double,
                         delta: (Matrix, Matrix, Matrix) => Matrix) {

    var wih: Matrix = inputWeigths
    var who: Matrix = outputWeights

    def train(trainingSet: Matrix): (Matrix, Matrix) = {
      for (row <- trainingSet.rowArray()) {
        val io: Tuple2[Matrix, Matrix] = splitRowINtoInputOutput(row)
        val hiddenOutputs = wih.dot(io._1).apply(activation)
        val outputs: Matrix = who.dot(hiddenOutputs).apply(activation)
        val outputErrors: Matrix = io._2 - outputs
        val hiddenErrors: Matrix = who.transpose.dot(outputErrors)

        val whoDelta = delta.apply(outputErrors, outputs, hiddenOutputs)
        who += whoDelta * learningRate

        val wihDelta = delta.apply(hiddenErrors, hiddenOutputs, io._1)
        wih += wihDelta * learningRate
      }
      (wih, who)
    }

    private def splitRowINtoInputOutput(row: Array[Double]): (Matrix, Matrix) = {
      val input: Array[Double] = row.slice(0, inputNodes)
      val output: Array[Double] = row.slice(inputNodes, row.length)
      (Matrix.create(input, 1), Matrix.create(output, 1))
    }

    def query(input: Matrix): Matrix = {
      val outputHidden: Matrix = wih.dot(input).apply(activation)
      who.dot(outputHidden).apply(activation)
    }

  }

  def create(inputNodes: Int, hiddenNodes: Int, outputNodes: Int, tolerence: Tolerence, learningRate: Double,
             inputWeights: Matrix, outputWeights: Matrix, activation: Double => Double, delta: (Matrix, Matrix, Matrix) => Matrix): ThreeLayer = {

    ThreeLayer(inputNodes, hiddenNodes, outputNodes, tolerence, learningRate,
      inputWeights, outputWeights, activation, delta)
  }

}
