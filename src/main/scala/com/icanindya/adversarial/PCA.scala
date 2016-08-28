package com.icanindya.adversarial

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
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

object PCA {
  def main(args: Array[String]) {
       System.setProperty("hadoop.home.dir", "C:/hadoop241/");
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)

    val conf = new SparkConf().setAppName("adversarial").setMaster("local")
    conf.set("spark.hadoop.validateOutputSpecs", "false")
    conf.set("spark.executor.memory", "10g")
    val sc = new SparkContext(conf)

    val data = sc.textFile("E:/Data/kddcupdata/kddcup.trasfrom.normal")
    val metadata = sc.textFile("E:/Data/kddcupdata/kddcup.trasfrom")
    
    val protocols = metadata.map(_.split(',')(1)).distinct().collect().zipWithIndex.toMap
    val services = metadata.map(_.split(',')(2)).distinct().collect().zipWithIndex.toMap
    val tcpStates = metadata.map(_.split(',')(3)).distinct().collect().zipWithIndex.toMap
    val labelData = data.map{line =>
            val buffer = line.split(",").toBuffer
            val protocol = buffer.remove(1)
            val service = buffer.remove(1)
            val tcpState = buffer.remove(1)
            val label = buffer.remove(buffer.length - 1)
            val vector = buffer.map(_.toDouble)

            val newProtocolFeatures = new Array[Double](protocols.size)
            newProtocolFeatures(protocols(protocol)) = 1.0
            val newServiceFeatures = new Array[Double](services.size)
            newServiceFeatures(services(service)) = 1.0
            val newTcpStateFeatures = new Array[Double](tcpStates.size)
            newTcpStateFeatures(tcpStates(tcpState)) = 1.0

            vector.insertAll(1, newTcpStateFeatures)
            vector.insertAll(1, newServiceFeatures)
            vector.insertAll(1, newProtocolFeatures)
            
            var classlabel = 0.0 
            if(label != "normal.") {
                classlabel = 1.0
            }
            // (label,Vectors.dense(vector.t
            new LabeledPoint(classlabel, Vectors.dense(vector.toArray))
          
            
    }

   val labelNewData = data.map{line =>
             val buffer = line.split(",").toBuffer
            val protocol = buffer.remove(1)
            val service = buffer.remove(1)
            val tcpState = buffer.remove(1)
            val label = buffer.remove(buffer.length - 1)
            val vector = buffer.map(_.toDouble)

            val newProtocolFeatures = new Array[Double](protocols.size)
            newProtocolFeatures(protocols(protocol.trim)) = 1.0
            val newServiceFeatures = new Array[Double](services.size)
            newServiceFeatures(services(service.trim)) = 1.0
            val newTcpStateFeatures = new Array[Double](tcpStates.size)
            newTcpStateFeatures(tcpStates(tcpState.trim)) = 1.0

            vector.insertAll(1, newTcpStateFeatures)
            vector.insertAll(1, newServiceFeatures)
            vector.insertAll(1, newProtocolFeatures)

            (Vectors.dense(vector.toArray))
        }

   // run PCA on train data to get top k PCA 
   // val pcaK =args(0).toInt 
    val pcaK  = 30
    val mat = new RowMatrix(labelNewData)

   //  Compute principal components.
   val pc = mat.computePrincipalComponents(pcaK)

   val projected = mat.multiply(pc).rows
   val numClusters = 100
   // val numClusters = args(1).toInt
   val numIterations = 10     
   val kmeans = new KMeans()
   kmeans.setK(numClusters)
   kmeans.setRuns(numIterations)
   val model2 = kmeans.run(projected)
  
   val pca = new PCA(pcaK).fit(labelData.map(_.features))

   val projectednew = labelData.map(p => p.copy(features = pca.transform(p.features)))

     
   var clusterID =0
   var centroidMap:Map[Int,Vector] = Map() 
   model2.clusterCenters.map{ line =>
        // print("cluster" + line)
        centroidMap += ( clusterID -> line )
        clusterID += 1
   }


//model2.save(sc, "/Cloud/spark-1.6.1/bin/models/approach2/")

   

//Model 1 Features











   val testData = sc.textFile("E:/Data/kddcupdata/correctednoicmp")
   
   val labelTestData = testData.map{line => 
            val buffer = line.split(",").toBuffer
            val protocol = buffer.remove(1)
            val service = buffer.remove(1)
            val tcpState = buffer.remove(1)
            val label = buffer.remove(buffer.length - 1)
            val vector = buffer.map(_.toDouble)
            
            val newProtocolFeatures = new Array[Double](protocols.size)
            newProtocolFeatures(protocols(protocol)) = 1.0
            val newServiceFeatures = new Array[Double](services.size)
            newServiceFeatures(services(service)) = 1.0
            val newTcpStateFeatures = new Array[Double](tcpStates.size)
            newTcpStateFeatures(tcpStates(tcpState)) = 1.0
            
            vector.insertAll(1, newTcpStateFeatures)
            vector.insertAll(1, newServiceFeatures)
            vector.insertAll(1, newProtocolFeatures)

            var classlabel = 0.0 
            if(label != "normal.") {
                classlabel = 1.0
            }
            new LabeledPoint(classlabel, Vectors.dense(vector.toArray))
   }

    val projectedtest = labelTestData.map(p => p.copy(features = pca.transform(p.features)))
    var threshold = 75000
   // var threshold = args(2).toDouble
    
    var TP = 0L 
    var FP = 0L
    var FN = 0L
    var TN = 0L
//    val clustersLabelCount =
      projectedtest.collect().foreach {line =>
        val cluster = model2.predict(line.features)
        println("Cluster number is " + cluster)
        var modelLabel = "attack"
        var count = 1
        centroidMap.get(cluster) match {
         case Some(i) => 
            var dist = Vectors.sqdist(line.features,centroidMap.get(cluster).get) 
            println("Dist  is " + dist)
            if( dist < threshold ) {
                modelLabel = "normal" 
             }
         case None => println("Cluster number is: " + cluster)  
        }
        if(line.label == 0.0  && modelLabel == "normal") {
            println("Inside TP")
            TP += count
        } else if(line.label != 0.0 && modelLabel == "normal") {
            println("Inside FP")
             FP += count
        } else if(line.label != 0.0  && modelLabel != "normal") {
            TN += count
            println("Inside TN")
        } else if (line.label == 0.0  && modelLabel != "normal") {
            FN += count
            println("Inside FN")
        }
        }

//        clustersLabelCount.foreach(println)

   println("TP is : " + TP)
   println("FP is : " + FP)
   println("TN is : " + TN)
   println("FN is : " + FN)
   println("precision is : " + TP/(TP+FP).toFloat)
    println("recall is : "+ TP/(TP+FN).toFloat)
    println("Accuracy is :" + (TP+TN)/(TP+TN+FP+FN).toFloat)
    sc.stop()
  }
}
