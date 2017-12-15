package com.icanindya.adversarial.iscx

import com.icanindya.adversarial.Spark

object Stat {
  
  val DERIVED_DS_PATH = "E:/Data/ISCX IDS/derived/derived_%d.txt" //.format(int)
  val ATTACK_LABEL = "Attack"
  val NORMAL_LABEL = "Normal"
  
  val COMMA = ","
  
  
  def main(args: Array[String]): Unit = {
    val sc = Spark.getContext()
    for(i <- 11 to 17){
      val lines = sc.textFile(DERIVED_DS_PATH.format(i)).cache()
      
      val count = lines.count()
      val distinctCount = lines.distinct().count()
      if(count != distinctCount) println("differnce : %d".format(count - distinctCount))
      val normalCount = lines.map(_.split(COMMA, -1).last).filter(_ == NORMAL_LABEL).count()
      val attackCount = count - normalCount
      
      println("total flows: " + count)
      println("normal flows: " + normalCount)
      println("attack flows: " + attackCount)
    }
  }
  
}