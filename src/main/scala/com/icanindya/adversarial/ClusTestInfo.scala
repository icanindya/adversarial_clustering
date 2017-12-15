package com.icanindya.adversarial

class ClusTestInfo extends Serializable{
    var minSqDist = Double.MaxValue;
    var maxSqDist = Double.MinValue;
    var numAssocTestPoints = 0L;
    var totSqDist = 0.0;
    
    def add(sqDist: Double){
      numAssocTestPoints += 1
      totSqDist += sqDist
      if (sqDist < this.minSqDist) this.minSqDist += sqDist
      if (sqDist > this.maxSqDist) this.maxSqDist += sqDist
    }
  }