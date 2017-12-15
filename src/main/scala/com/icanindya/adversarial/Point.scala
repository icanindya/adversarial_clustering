package com.icanindya.adversarial

import org.apache.spark.mllib.linalg._

class Point(val features: Vector, val cls: String, val nearestCluster: Int, val nearestSqDist: Double, val classifiedAs: String, pc: Matrix) extends Serializable{  
  val projFeatures = getProjFromOrig(features, pc)
    
  def getProjFromOrig(vector: Vector, pc: Matrix): Vector = {
    var pointMat = new DenseMatrix(1, vector.size, vector.toArray)
    var projPoint = pointMat.multiply(new DenseMatrix(pc.numRows, pc.numCols, pc.toArray));
    new DenseVector(projPoint.toArray);
  }
  
  
}