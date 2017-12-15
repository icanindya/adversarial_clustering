package com.icanindya.adversarial

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf

object Spark {
  def getContext(): SparkContext = {
    System.setProperty("hadoop.home.dir", "C:/hadoop241/");
    val conf = new SparkConf().setAppName("adversarial").setMaster("local")
    conf.set("spark.hadoop.validateOutputSpecs", "false")
    conf.set("spark.executor.memory", "13g")
    val sc = new SparkContext(conf)
    sc.setLogLevel("WARN")
    sc
  }
}