package com.elasticcloudservice.arima;

import java.util.Random;
import java.util.Vector;

public class ARIMA {

	double[] originalData = {};
	ARMAMath armamath = new ARMAMath();
	
	double stderrDara = 0;
	double avgsumData = 0;
	
	int period = 7;
	
	Vector<double[]> armaARMAcoe = new Vector<double[]>();
	Vector<double[]> bestarmaARMAcoe = new Vector<double[]>();
	
	public ARIMA(int period) {
		this.period = period;
	}
	
	public ARIMA(double[] originalData) {
		this.originalData = originalData;
	}
	
	public void setData(double[] originalData) {
		this.originalData = originalData;
		this.period = Math.min(this.period, originalData.length - 1);
	}

	/**
	 * 季节性差分
	 * @return
	 */
	public double[] preDealDif() {
		double[] tempData = new double[originalData.length - this.period];
		for (int i = 0; i < originalData.length - this.period; i++) {
			if (this.period == 0) tempData[i] = originalData[i];
			else tempData[i] = originalData[i + this.period] - originalData[i];
		}
		return tempData;
	}

	public double[] preDealNor(double[] tempData) {
		// Z-Score
		avgsumData = armamath.avgData(tempData);
		stderrDara = armamath.stderrData(tempData);

		for (int i = 0; i < tempData.length; i++) {
			tempData[i] = (tempData[i] - avgsumData) / stderrDara;
		}

		return tempData;
	}

	public int[] getARIMAmodel(boolean first, int[] bestModel) {
		double[] stdoriginalData = this.preDealDif();
		if (first) {
			int paraType = 0;
			double minAIC = 0x3f3f3f3f;
			int bestModelindex = -1;
			int[][] model = new int[][] { { 0, 1 }, { 1, 0 }, { 1, 1 }, { 0, 2 }, { 2, 0 }, { 2, 2 }, { 1, 2 }, { 2, 1 }};
//			int[][] model = new int[][] { { 0, 1 }, { 1, 0 }, { 1, 1 }};
//			int[][] model = {{ 0, 3 }, { 3, 0 }, { 1, 3 }, { 3, 1 }, { 2, 3 }, { 3, 2 }, { 3, 3 }};
//			int[][] model = new int[][] { { 0, 1 }, { 1, 0 }, { 1, 1 }, { 0, 2 }, { 2, 0 }, { 2, 2 }, { 1, 2 },{ 2, 1 }, 
//				{ 0, 3 }, { 3, 0 }, { 1, 3 }, { 3, 1 }, { 2, 3 }, { 3, 2 }, { 3, 3 } };
			for (int i = 0; i < model.length; i++) {
				if (model[i][0] == 0) {
					MA ma = new MA(stdoriginalData, model[i][1]);
					armaARMAcoe = ma.MAmodel();
					paraType = 1;
				} else if (model[i][1] == 0) {
					AR ar = new AR(stdoriginalData, model[i][0]);
					armaARMAcoe = ar.ARmodel();
					paraType = 2;
				} else {
					ARMA arma = new ARMA(stdoriginalData, model[i][0], model[i][1]);
					armaARMAcoe = arma.ARMAmodel();
					paraType = 3;
				}
	
				double temp = getmodelAIC(armaARMAcoe, stdoriginalData, paraType);
				if (temp < minAIC) {
					bestModelindex = i;
					minAIC = temp;
					bestarmaARMAcoe = armaARMAcoe;
				}
			}
			if (bestModelindex == -1) return null;
			return model[bestModelindex];
		}
		else {
			if (bestModel[0] == 0) {
				MA ma = new MA(stdoriginalData, bestModel[1]);
				armaARMAcoe = ma.MAmodel();
			} else if (bestModel[1] == 0) {
				AR ar = new AR(stdoriginalData, bestModel[0]);
				armaARMAcoe = ar.ARmodel();
			} else {
				ARMA arma = new ARMA(stdoriginalData, bestModel[0], bestModel[1]);
				armaARMAcoe = arma.ARMAmodel();
			}
			bestarmaARMAcoe = armaARMAcoe;
			return bestModel;
		}
	}

	public double getmodelAIC(Vector<double[]> para, double[] stdoriginalData, int type) {
		double temp = 0;
		double temp2 = 0;
		double sumerr = 0;
		int p = 0;// ar1,ar2,...,sig2
		int q = 0;// sig2,ma1,ma2...
		int n = stdoriginalData.length;
		Random random = new Random(2018);

		if (type == 1) {
			double[] maPara = para.get(0);
			q = maPara.length;
			double[] err = new double[q]; // error(t),error(t-1),error(t-2)...
			for (int k = q - 1; k < n; k++) {
				temp = 0;

				for (int i = 1; i < q; i++) {
					temp += maPara[i] * err[i];
				}

				for (int j = q - 1; j > 0; j--) {
					err[j] = err[j - 1];
				}
				err[0] = random.nextGaussian() * Math.sqrt(maPara[0]);

				sumerr += (stdoriginalData[k] - (temp)) * (stdoriginalData[k] - (temp));

			}
			return (n - (q - 1)) * Math.log(sumerr / (n - (q - 1))) + (q + 1) * 2;
		} else if (type == 2) {
			double[] arPara = para.get(0);
			p = arPara.length;
			for (int k = p - 1; k < n; k++) {
				temp = 0;
				for (int i = 0; i < p - 1; i++) {
					temp += arPara[i] * stdoriginalData[k - i - 1];
				}
				sumerr += (stdoriginalData[k] - temp) * (stdoriginalData[k] - temp);
			}
			return (n - (q - 1)) * Math.log(sumerr / (n - (q - 1))) + (p + 1) * 2;
		} else {
			double[] arPara = para.get(0);
			double[] maPara = para.get(1);
			p = arPara.length;
			q = maPara.length;
			double[] err = new double[q];

			for (int k = p - 1; k < n; k++) {
				temp = 0;
				temp2 = 0;
				for (int i = 0; i < p - 1; i++) {
					temp += arPara[i] * stdoriginalData[k - i - 1];
				}

				for (int i = 1; i < q; i++) {
					temp2 += maPara[i] * err[i];
				}

				for (int j = q - 1; j > 0; j--) {
					err[j] = err[j - 1];
				}
				err[0] = random.nextGaussian() * Math.sqrt(maPara[0]);
				sumerr += (stdoriginalData[k] - (temp2 + temp)) * (stdoriginalData[k] - (temp2 + temp));
			}
			return (n - (q - 1)) * Math.log(sumerr / (n - (q - 1))) + (p + q) * 2;
		}
	}

	public int aftDeal(int predictValue) {
		if (this.period == 0) return predictValue;
		else return (int) (predictValue + originalData[originalData.length - this.period]);
	}

	public int predictValue(int p, int q) {
		int predict = 0;
		double[] stdoriginalData = this.preDealDif(); // 预处理后的数据
		int n = stdoriginalData.length;
		
		double temp = 0, temp2 = 0;
		double[] err = new double[q + 1];

		Random random = new Random(2018);
		if (p == 0) {
			double[] maPara = bestarmaARMAcoe.get(0);
			for (int k = q; k < n; k++) {
				temp = 0;
				for (int i = 1; i <= q; i++) {
					temp += maPara[i] * err[i];
				}
				for (int j = q; j > 0; j--) {
					err[j] = err[j - 1];
				}
				err[0] = random.nextGaussian() * Math.sqrt(maPara[0]);
			}
			predict = (int) (temp);
		} else if (q == 0) {
			double[] arPara = bestarmaARMAcoe.get(0);
			for (int k = p; k < n; k++) {
				temp = 0;
				for (int i = 0; i < p; i++) {
					temp += arPara[i] * stdoriginalData[k - i - 1];
				}
			}
			predict = (int) (temp);
		} else {

			double[] arPara = bestarmaARMAcoe.get(0);
			double[] maPara = bestarmaARMAcoe.get(1);
			
			err = new double[q + 1]; // error(t),error(t-1),error(t-2)...
			for (int k = p; k < n; k++) {
				temp = 0;
				temp2 = 0;
				for (int i = 0; i < p; i++) {
					temp += arPara[i] * stdoriginalData[k - i - 1];
				}

				for (int i = 1; i <= q; i++) {
					temp2 += maPara[i] * err[i];
				}

				for (int j = q; j > 0; j--) {
					err[j] = err[j - 1];
				}

				err[0] = random.nextGaussian() * Math.sqrt(maPara[0]);
			}

			predict = (int) (temp2 + temp);

		}
		return predict >= 0 ? predict : 0;
	}

	public double[] getMApara(double[] autocorData, int q) {
		double[] maPara = new double[q + 1];
		double[] tempmaPara = maPara;
		double temp = 0;
		boolean iterationFlag = true;
		System.out.println("autocorData[0]" + autocorData[0]);
		while (iterationFlag) {
			for (int i = 1; i < maPara.length; i++) {
				temp += maPara[i] * maPara[i];
			}
			tempmaPara[0] = autocorData[0] / (1 + temp);

			for (int i = 1; i < maPara.length; i++) {
				temp = 0;
				for (int j = 1; j < maPara.length - i; j++) {
					temp += maPara[j] * maPara[j + i];
				}
				tempmaPara[i] = -(autocorData[i] / tempmaPara[0] - temp);
			}
			iterationFlag = false;
			for (int i = 0; i < maPara.length; i++) {
				if (maPara[i] != tempmaPara[i]) {
					iterationFlag = true;
					break;
				}
			}

			maPara = tempmaPara;
		}

		return maPara;
	}

}
