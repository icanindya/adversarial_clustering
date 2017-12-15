package com.icanindya.adversarial

import org.apache.spark.mllib.linalg._
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.feature.PCA
import org.apache.spark.mllib.regression.LabeledPoint

object AdvUtil{  
    
  def getProjFromOrig(vector: Vector, pcMatrix: DenseMatrix): Vector = {
    var pointMat = new DenseMatrix(1, vector.size, vector.toArray)
    var projPoint = pointMat.multiply(pcMatrix);
    new DenseVector(projPoint.toArray);
  }
  
  def getProjPointsAndPcMatrix(originalPoints: RDD[Vector], numPc: Int): (RDD[Vector], DenseMatrix) = {
    var featureMatrix = new RowMatrix(originalPoints)
    val pcaModel = new PCA(numPc).fit(originalPoints)
    var pcMatrix = pcaModel.pc
    val projPoints = featureMatrix.multiply(pcMatrix).rows
    (projPoints, pcMatrix)
  }
  
  def pcToArr(pcMatrix: Matrix): Array[Array[Double]] = {
    var c = 0
    val pcArr = Array.ofDim[Double](pcMatrix.numRows, pcMatrix.numCols)

    //pc.toArray is a column major array
    for (i <- 0 to pcMatrix.numCols - 1) {
      for (j <- 0 to pcMatrix.numRows - 1) {
        pcArr(j)(i) = pcMatrix.toArray(c)
        c += 1
      }
    }
    pcArr
  }
  
  def chebyshevThreshold(mean: Double, stdDev: Double): Double = {
    mean + 3 * stdDev
  }
  
}