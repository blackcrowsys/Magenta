package com.blackcrowsys.magenta.neural

import com.blackcrowsys.crimson.common.Tolerence
import com.blackcrowsys.crimson.functions.Functions
import com.blackcrowsys.crimson.matrix.Matrix
import com.blackcrowsys.crimson.matrix.Matrix.{Matrix, createFromResourceFile}
import com.blackcrowsys.magenta.neural.ThreeLayer.ThreeLayer
import org.scalatest.{BeforeAndAfterAll, FunSuite}

class ThreeLayerTests extends FunSuite {

  val inputNodes: Int = 2
  val hiddenNodes: Int = 3
  val outputNodes: Int = 1
  val tolerence: Tolerence = new Tolerence(0.0001)
  val learningRate: Double = 0.3

  val inputWeights: Matrix = Matrix.create(hiddenNodes, inputNodes, 1)
  val outputWeights: Matrix = Matrix.create(outputNodes, hiddenNodes, 1)

  val activation = (x: Double) => Functions.sigmoid(x)
  val delta = (error: Matrix, output: Matrix, input: Matrix) => (error * output * (Matrix.create(output.rows, 1, 1) - output)).dot(input.transpose)

  val input = Matrix.create(inputNodes, 1, 1)

  val network: ThreeLayer = ThreeLayer.create(inputNodes, hiddenNodes, outputNodes, tolerence, learningRate,
    inputWeights, outputWeights, activation, delta)

  test("it should handle simple query") {
    val result: Matrix = network.query(input)

    assert(result.rows == 1 && result.columns == 1)
    assert(result.get(1, 1) == 0.9335404768183254)
  }

  test("it should train the network using the delta function and learning rate") {
    val dataSet: Matrix = Matrix.create(Array(1, 1, 6), 3)

    val result: Tuple2[Matrix, Matrix] = network.train(dataSet)

    val wih = result._1
    val who = result._2

    val restulAfterTraining: Matrix = network.query(input)

    assert(wih.equals(Matrix.create(Array(1.1595837251921755, 1.1595837251921755, 1.1595837251921755, 1.1595837251921755, 1.1595837251921755, 1.1595837251921755), 2), tolerence))
    assert(who.equals(Matrix.create(Array(1.0830600276519622, 1.0830600276519622, 1.0830600276519622), 3), tolerence))
    assert(restulAfterTraining.get(1, 1) == 0.9506506845636884)
  }
}
