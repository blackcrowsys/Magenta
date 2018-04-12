package com.blackcrowsys.magenta.applications.neural

import java.text.SimpleDateFormat
import java.util.Calendar

import com.blackcrowsys.crimson.matrix.Matrix.Matrix
import org.scalatest.{BeforeAndAfterAll, FunSuite}

class HandWrittenDecipherTests extends FunSuite with BeforeAndAfterAll {

  val trainingDataFile: String = "/Users/ramindursingh/MNIST/mnist_train_1.csv"
  val testDataFile: String = "/Users/ramindursingh/MNIST/mnist_test.csv"
  val inputNodes: Int = 784
  val outputNodes: Int = 10

  val dateFormat: SimpleDateFormat = new SimpleDateFormat("dd/MM/YYYY HH:mm:ss.SSS")
  val printTime = (msg: String) => println(msg + " at " + dateFormat.format(Calendar.getInstance().getTime))

  var trainingData: Matrix = _
  var testData: Matrix = _

  override def beforeAll(): Unit = {
    printTime("Loading training data")
    trainingData = HandWrittenDecipher.createMatrix(trainingDataFile, inputNodes, outputNodes)
    printTime("Loading test data")
    testData = HandWrittenDecipher.createMatrix(testDataFile, inputNodes, outputNodes)
    printTime("Finished creating training and test data")
  }

  test("accuracy for hidden layer of 100 nodes and learning rate of 0.4") {
    val decipher = HandWrittenDecipher.createDecipher(inputNodes, 100, outputNodes, 0.4)
    printTime("Starting training")
    decipher.train(trainingData)

    printTime("Starting testing")
    val successRate: Double = decipher.test(testData)
    printTime("Success rate of " + successRate + "% for 100/0.4")
  }

  test("accuracy for hidden layer of 100 nodes and learning rate of 0.3") {
    val decipher = HandWrittenDecipher.createDecipher(inputNodes, 100, outputNodes, 0.3)
    printTime("Starting training")
    decipher.train(trainingData)

    printTime("Starting testing")
    val successRate: Double = decipher.test(testData)
    printTime("Success rate of " + successRate + "% for 100/0.3")
  }

  test("accuracy for hidden layer of 100 nodes and learning rate of 0.2") {
    val decipher = HandWrittenDecipher.createDecipher(inputNodes, 100, outputNodes, 0.2)
    printTime("Starting training")
    decipher.train(trainingData)

    printTime("Starting testing")
    val successRate: Double = decipher.test(testData)
    printTime("Success rate of " + successRate + "% for 100/0.2")
  }

  test("accuracy for hidden layer of 150 nodes and learning rate of 0.4") {
    val decipher = HandWrittenDecipher.createDecipher(inputNodes, 150, outputNodes, 0.4)
    printTime("Starting training")
    decipher.train(trainingData)

    printTime("Starting testing")
    val successRate: Double = decipher.test(testData)
    printTime("Success rate of " + successRate + "% for 150/0.4")
  }

  test("accuracy for hidden layer of 150 nodes and learning rate of 0.3") {
    val decipher = HandWrittenDecipher.createDecipher(inputNodes, 150, outputNodes, 0.3)
    printTime("Starting training")
    decipher.train(trainingData)

    printTime("Starting testing")
    val successRate: Double = decipher.test(testData)
    printTime("Success rate of " + successRate + "% for 150/0.3")
  }

  test("accuracy for hidden layer of 150 nodes and learning rate of 0.2") {
    val decipher = HandWrittenDecipher.createDecipher(inputNodes, 150, outputNodes, 0.2)
    printTime("Starting training")
    decipher.train(trainingData)

    printTime("Starting testing")
    val successRate: Double = decipher.test(testData)
    printTime("Success rate of " + successRate + "% for 150/0.2")
  }

}
