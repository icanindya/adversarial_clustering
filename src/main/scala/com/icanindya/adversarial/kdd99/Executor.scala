package com.icanindya.adversarial.kdd99
import com.icanindya.adversarial._

object Executor {
  def main(args: Array[String]): Unit = {
    
    val sc = Spark.getContext()
    
    FinalDataset.getFinalDatasetStat(sc, FinalDataset.finalTrainFile, FinalDataset.finalTestFile)

  
  }
}