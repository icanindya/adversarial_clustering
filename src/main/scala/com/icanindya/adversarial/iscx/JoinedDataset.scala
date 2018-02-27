package com.icanindya.adversarial.iscx

import com.icanindya.adversarial.Spark
import org.apache.spark.rdd.RDD.rddToPairRDDFunctions
import java.text.SimpleDateFormat
import java.util.Date

object JoinedDataset {

  val FLOW_FEATURES_FILE = "D:/Data/ISCX_IDS/flowtbag/features_%d" //.format(int)
  val LABEL_FILE = "D:/Data/ISCX_IDS/processed_labels/labels_%d.txt" //.format(int)
  
  val JOINED_DS_FILE = "D:/Data/ISCX_IDS/joined/%d" //.format(int)
  val COMMA = ","
  
  val fStartInd = 7
  val numFeatures = 47
  

  def main(args: Array[String]): Unit = {

    val sc = Spark.getContext()
    
    val flowRecords = sc.textFile(FLOW_FEATURES_FILE.format(11))
    flowRecords.map{ x =>
      x.split(",").slice(fStartInd, numFeatures).mkString(",") + "," + "Normal"      
    }
    .coalesce(1).saveAsTextFile(JOINED_DS_FILE.format(11))
    
    for (i <- 12 to 17) {
      
      println("%d june: ".format(i))

      val flowRecords = sc.textFile(FLOW_FEATURES_FILE.format(i))

      val keyedFlowRecords = flowRecords.map { line =>
        val tokens = line.split(COMMA, -1)
        val flowKey = (tokens(0), tokens(1), tokens(2), tokens(3), tokens(4))
        (flowKey, tokens)
      }.cache()

      println(" -- flow count: " + keyedFlowRecords.count())

      val labels = sc.textFile(LABEL_FILE.format(i))

      val keyedLabels = labels.map { line =>
        val labelTokens = line.split(COMMA, -1)
        val flowKey = (labelTokens(0), labelTokens(1), labelTokens(2), labelTokens(3), labelTokens(4))
        (flowKey, labelTokens)
      }

      println(" -- label count: " + keyedLabels.count())

      val joined = keyedFlowRecords.join(keyedLabels)
      .map { kv =>
        val key = kv._1
        val features = kv._2._1
        val labelTokens = kv._2._2
        (key, (features, labelTokens))
      }
      .filter{ kv =>
        val features = kv._2._1
        val labelTokens = kv._2._2
        
        val dateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss")
        
        val fStartTime = dateFormat.parse(features(5))
        val fEndTime = dateFormat.parse(features(6))
        
        val lStartTime = dateFormat.parse(labelTokens(5))
        val lEndTime = dateFormat.parse(labelTokens(6))
        

        var acceptable = false
        
        //feature time can be greater than label time
        if(i == 12){
          if((fStartTime.getTime() - lStartTime.getTime >= 14000 && fStartTime.getTime() - lStartTime.getTime <= 20000) &&
          (fEndTime.getTime() - lEndTime.getTime >= 14000 && fEndTime.getTime() - lEndTime.getTime <= 20000)) acceptable = true
          else acceptable = false
        }
        else if(i == 13){
          
          if((fStartTime.getTime() - lStartTime.getTime >= 10000 && fStartTime.getTime() - lStartTime.getTime <= 14000) &&
          (fEndTime.getTime() - lEndTime.getTime >= 10000 && fEndTime.getTime() - lEndTime.getTime <= 14000)) acceptable = true
          else acceptable = false
          
        }
        else if(i == 14){
          
          if((fStartTime.getTime() - lStartTime.getTime >= 6000 && fStartTime.getTime() - lStartTime.getTime <= 9000) &&
          (fEndTime.getTime() - lEndTime.getTime >= 6000 && fEndTime.getTime() - lEndTime.getTime <= 9000)) acceptable = true
          else acceptable = false
          
        }
        else if(i == 15){
          
          if((fStartTime.getTime() - lStartTime.getTime >= 0 && fStartTime.getTime() - lStartTime.getTime <= 5000) &&
          (fEndTime.getTime() - lEndTime.getTime >= 0 && fEndTime.getTime() - lEndTime.getTime <= 5000)) acceptable = true
          else acceptable = false
          
        }
        //now label time can be greater than feature time
        else if( i ==16){
          
          if((lStartTime.getTime() - fStartTime.getTime >= 0 && lStartTime.getTime() - fStartTime.getTime <= 4000) &&
          (lEndTime.getTime() - fEndTime.getTime >= 0 && lEndTime.getTime() - fEndTime.getTime <= 4000)) acceptable = true
          else acceptable = false
          
        }
        else if(i == 17){
          
          if((lStartTime.getTime() - fStartTime.getTime >= 4000 && lStartTime.getTime() - fStartTime.getTime <= 9000) &&
          (lEndTime.getTime() - fEndTime.getTime >= 4000 && lEndTime.getTime() - fEndTime.getTime <= 9000)) acceptable = true
          else acceptable = false
          
        }
        
        acceptable
      }
      .map{ x =>
        ((x._2._1, 1), x._2._2(7))
      }
      .groupByKey()
      .filter{ x =>
        if(x._2.toSet.size == 2) println("lost tuple")
        x._2.toSet.size == 1
      }
      .map{ x =>
        x._1._1.slice(7, 47).mkString(",") + "," + x._2.head 
      }      
      .cache()
      
      joined.coalesce(1).saveAsTextFile(JOINED_DS_FILE.format(i))
      
      println(" -- join count:" + joined.count)

    }
  }
}