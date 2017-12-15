package com.icanindya.adversarial;

import ilog.concert.*;
import ilog.cplex.*;

public class CplexOptimizer {
	
	
	public static CplexSolution hardOptimize(double[] origX, double[] projX, double[] projCenter, double sqDistThres, double[][] pcArray, double[] attrChangeThres){
		
		//cplex solution
		CplexSolution solution = new CplexSolution();
		
		// parameters

		// number of features
		int N = origX.length;

		// number of principal components
		int M = projX.length;
		
		
		IloCplex cplex = null;
			
			
		try {
			// define new model
			
			cplex = new IloCplex();
			cplex.setOut(null);

			// variables

			// modified attack point
			IloNumVar[] origXp = cplex.numVarArray(N, 0.0, 1.0);
			
			// objective

			IloNumExpr objective = cplex.numExpr();
			IloNumExpr numExprs[] = new IloNumExpr[M];

			for (int j = 0; j < M; j++) {
				IloLinearNumExpr numExpr = cplex.linearNumExpr();
				for (int i = 0; i < N; i++) {
					numExpr.addTerm(origXp[i], pcArray[i][j]);
				}
				numExprs[j] = cplex.square(cplex.diff(numExpr, projX[j]));
			}

			objective = cplex.sum(numExprs);
//			objective = cplex.constant(0);
			cplex.addMinimize(objective);
			
			// constraints
			// constraint 1
			IloNumExpr numExprs2[] = new IloNumExpr[M];

			for (int j = 0; j < M; j++) {
				IloLinearNumExpr numExpr = cplex.linearNumExpr();
				for (int i = 0; i < N; i++) {
					numExpr.addTerm(origXp[i], pcArray[i][j]);
				}
				numExprs2[j] = cplex.square(cplex.diff(numExpr, projCenter[j]));
			}
			
			cplex.addLe(cplex.sum(numExprs2), sqDistThres);
			
			// constraint 2
			for (int i = 0; i < N; i++) {
				cplex.addLe(cplex.square(cplex.diff(origXp[i], origX[i])), Math.pow(origX[i] * attrChangeThres[i], 2));
			}

			if (cplex.solve()) {
				solution.found = true;
				solution.objValue = cplex.getObjValue();
				solution.evasivePoint = cplex.getValues(origXp);
			}
			
		} catch (Exception e) {
			e.printStackTrace();
		}
		finally{
			if(cplex != null) cplex.end();
		}
		return solution;
		
	}
	
public static CplexSolution softOptimize(double[] origX, double[] projX, double[] projCenter, double sqDistThres, double[][] pcArray, double[] attrChangeThres){
		
		//cplex solution
		CplexSolution solution = new CplexSolution();
		
		// parameters

		// number of features
		int N = origX.length;

		// number of principal components
		int M = projX.length;
		
		
		IloCplex cplex = null;
			
			
		try {
			// define new model
			
			cplex = new IloCplex();
			cplex.setOut(null);

			// variables

			// modified attack point
			IloNumVar[] origXp = cplex.numVarArray(N, 0.0, 1.0);
			
			// objective

			IloNumExpr objective = cplex.numExpr();
			
			objective = cplex.constant(0);
			cplex.addMinimize(objective);
			
			// constraints
			// constraint 1
			IloNumExpr numExprs2[] = new IloNumExpr[M];

			for (int j = 0; j < M; j++) {
				IloLinearNumExpr numExpr = cplex.linearNumExpr();
				for (int i = 0; i < N; i++) {
					numExpr.addTerm(origXp[i], pcArray[i][j]);
				}
				numExprs2[j] = cplex.square(cplex.diff(numExpr, projCenter[j]));
			}
			
			cplex.addLe(cplex.sum(numExprs2), sqDistThres);
			
			// constraint 2
			for (int i = 0; i < N; i++) {
				cplex.addLe(cplex.square(cplex.diff(origXp[i], origX[i])), Math.pow(origX[i] * attrChangeThres[i], 2));
			}

			if (cplex.solve()) {
				solution.found = true;
				solution.objValue = cplex.getObjValue();
				solution.evasivePoint = cplex.getValues(origXp);
			}
			
		} catch (Exception e) {
			e.printStackTrace();
		}
		finally{
			if(cplex != null) cplex.end();
		}
		return solution;
		
	}
}
