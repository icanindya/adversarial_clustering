package com.icanindya.adversarial

import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.clustering.{ KMeans, KMeansModel }
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd._
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.Matrices
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.feature.PCA
import org.apache.log4j.Logger
import org.apache.log4j.Level
import scala.reflect.io.Path


object Test {
  def main(args: Array[String]): Unit = {
    
    System.setProperty("hadoop.home.dir", "C:/hadoop241/");
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)

    val conf = new SparkConf().setAppName("adversarial").setMaster("local")
    conf.set("spark.hadoop.validateOutputSpecs", "false")
    conf.set("spark.executor.memory", "10g")
    val sc = new SparkContext(conf)

    val data = Array(
      Vectors.sparse(6, Seq((1, 1.0), (3, 7.0))),
      Vectors.dense(2.0, 0.0, 3.0, 4.0, 5.0, 3.0),
      Vectors.dense(4.0, 0.0, 0.0, 6.0, 7.0, 2.0))
      

    val dataRDD = sc.parallelize(data, 2)

    val mat: RowMatrix = new RowMatrix(dataRDD)

    // Compute the top 4 principal components.
    // Principal components are stored in a local dense matrix.
    var pc: Matrix = mat.computePrincipalComponents(3)
    
    println(pc)
    // Project the rows to the linear space spanned by the top 4 principal components.
    val projected: RowMatrix = mat.multiply(pc)
    
//    val pc = new DenseMatrix(2, 3, Array(3, 2, 1, 4, 6, 1)) 
//    val pcArr = pc.toArray
//    var c = 0
//    val pc2DArr = Array.ofDim[Double](2,3)
//    
//    for(i <- 0 to pc.numCols - 1){
//      for(j <- 0 to pc.numRows - 1){
//        pc2DArr(j)(i) = pcArr(c)
//        c += 1
//      }
//    }
//    
//    for(i <- 0 to pc.numRows - 1){
//      for(j <- 0 to pc.numCols - 1){
//        print(pc2DArr(i)(j))
//      }
//      println()
//    }

    println("\n" + pc.toArray.mkString(",") + "\n")
    
    val densePc = new DenseMatrix(pc.numRows, pc.numCols, pc.toArray)
    
    println(densePc)
    
    if(Path("E:/asd").deleteRecursively()) println("true")
    
   sc.parallelize(Array(pc)).saveAsObjectFile("E:/a.txt")
   
   pc = sc.objectFile[Matrix]("E:/a.txt").collect()(0)
   
   println(pc)
  }
}