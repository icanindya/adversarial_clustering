package com.icanindya.adversarial.iscx

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd._
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.feature.PCA
import org.apache.spark.mllib.classification.NaiveBayes
import scala.reflect.io.Path
import com.icanindya.adversarial.Spark
import org.apache.spark.mllib.clustering.KMeansModel

object AnomalyDetection {
 
  val modelDir = "D:/Data/Adversarial/ISCX/model/%d/kmeans"
  val pcFile = "D:/Data/Adversarial/ISCX/model/%d/pc"

  def main(args: Array[String]) {
    val sc = Spark.getContext()
    
     val labeledPointTrainRdd = sc.textFile(FinalDataset.finalTrainFile).map{line =>
      val values = line.split(",", -1).map(_.toDouble)
      new LabeledPoint(values.last, Vectors.dense(values.dropRight(1)))
    }
    
    val labeledPointTestRdd = sc.textFile(FinalDataset.finalTestFile).map{line =>
      val values = line.split(",", -1).map(_.toDouble)
      new LabeledPoint(values.last, Vectors.dense(values.dropRight(1)))
    }

    runAnomalyDetection(sc, labeledPointTestRdd)
   
  }
  
  def saveKMeansModel(sc: SparkContext, labeledPointTrainRdd: RDD[LabeledPoint], labeledPointTestRdd: RDD[LabeledPoint], k: Int) {
    for(k <- 80 to 150 by 10){
      println(labeledPointTrainRdd.count())
  
      val numPc = 20
      var mat = new RowMatrix(labeledPointTrainRdd.map(_.features))
      val pca = new PCA(numPc).fit(labeledPointTrainRdd.map(_.features))
      var pc = pca.pc
      val projTrain = mat.multiply(pc).rows
  
      val numClusters = k
      val numIterations = 10
      val kmeans = new KMeans().setK(numClusters).setMaxIterations(numIterations)
      val model = kmeans.run(projTrain)
  
      if(Path(modelDir.format(k)).exists) Path(modelDir.format(k)).deleteRecursively()
      model.save(sc, modelDir.format(k))
  
      sc.parallelize(Array(pc)).saveAsObjectFile(pcFile.format(k))
    }
  }

  
  def runAnomalyDetection(sc: SparkContext, labeledPointTestRdd: RDD[LabeledPoint]){
    
    for(k <- 80 to 150 by 10){
      
      val model = KMeansModel.load(sc, modelDir.format(k))
      val pc = sc.objectFile[DenseMatrix](pcFile.format(k)).collect()(0)
      
      var TP = 0
      var FP = 0
      var TN = 0
      var FN = 0
      val sqDistThres = 0.012
      
      var minSqDist = Double.MaxValue
      var maxSqDist = Double.MinValue
      
      labeledPointTestRdd.collect.foreach { testPoint  => 
        val projTestPoint = AttackAnomalyDetection.getProjFromOrig(testPoint.features, pc)
        val label = testPoint.label
        
        val clusterIndex = model.predict(projTestPoint)
        val sqDist = Vectors.sqdist(projTestPoint, model.clusterCenters(clusterIndex))
        
        if(sqDist < minSqDist) minSqDist = sqDist
        if(sqDist > maxSqDist) maxSqDist = sqDist
    
        if (sqDist < sqDistThres) {
          if (label == 0.0) TN += 1
          else FN += 1
        } else {
          if (label == 0.0) FP += 1
          else TP += 1
        }
      
      }
      
      println("\n\n")
      println("k: " + k)
      println("TP is : " + TP)
      println("FP is : " + FP)
      println("TN is : " + TN)
      println("FN is : " + FN)
      println("precision is : " + TP / (TP + FP).toFloat)
      println("recall is : " + TP / (TP + FN).toFloat)
      println("Accuracy is :" + (TP + TN) / (TP + TN + FP + FN).toFloat)
    
    }
  }
}