package com.icanindya.adversarial.kdd99

import com.icanindya.adversarial._

object AdversarialResult {
  
  def main(args: Array[String]): Unit = {
          
    val sc = Spark.getContext()
    
    val results = sc.textFile(DrMuratAdversarialModel.kmEvasionResultsFile)

    results.map{x =>
      val pair = x.split(":", -1)
      val settings = pair(0)
      val values = pair(1)

      val ekt = settings.split(",", -1).map(_.split("=", -1)(1).toDouble)
      val goal = values.split(",", -1).last.split("=", -1)(1).toDouble
      
//      println(ekt.mkString(",") + " " + targetVal)
      val e = ekt(0)
      val k = ekt(1)
      val t = ekt(2)

      (e, (k, t, goal))
    }
    .groupByKey()
    .sortByKey(true)
    .map{kv =>
       println("%5.2f: %s".format(kv._1, kv._2.toArray.sortWith(_._3 < _._3)(0).productIterator.mkString(",")))
    }
    .collect
    
    
  }
  
}