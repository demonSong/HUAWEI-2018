package com.elasticcloudservice.arima;

public class ARMAMath {
	public double avgData(double[] dataArray) {
		return this.sumData(dataArray) / dataArray.length;
	}

	public double sumData(double[] dataArray) {
		double sumData = 0;
		for (int i = 0; i < dataArray.length; i++) {
			sumData += dataArray[i];
		}
		return sumData;
	}

	public double stderrData(double[] dataArray) {
		return Math.sqrt(this.varerrData(dataArray));
	}

	public double varerrData(double[] dataArray) {
		double variance = 0;
		double avgsumData = this.avgData(dataArray);

		for (int i = 0; i < dataArray.length; i++) {
			dataArray[i] -= avgsumData;
			variance += dataArray[i] * dataArray[i];
		}
		return variance / dataArray.length;// variance error;
	}

	public double[] autocorData(double[] dataArray, int order) {
		double[] autoCor = new double[order + 1];
		double varData = this.varerrData(dataArray);

		for (int i = 0; i <= order; i++) {
			autoCor[i] = 0;
			for (int j = 0; j < dataArray.length - i; j++) {
				autoCor[i] += dataArray[j + i] * dataArray[j];
			}
			autoCor[i] /= dataArray.length;
			autoCor[i] /= varData;
		}
		return autoCor;
	}

	public double[] autocorGrma(double[] dataArray, int order) {
		double[] autoCor = new double[order + 1];
		for (int i = 0; i <= order; i++) {
			autoCor[i] = 0;
			for (int j = 0; j < dataArray.length - i; j++) {
				autoCor[i] += dataArray[j + i] * dataArray[j];
			}
			autoCor[i] /= (dataArray.length - i);

		}
		return autoCor;
	}

	public double[] parautocorData(double[] dataArray, int order) {
		double parautocor[] = new double[order];

		for (int i = 1; i <= order; i++) {
			parautocor[i - 1] = this.parcorrCompute(dataArray, i, 0)[i - 1];
		}
		return parautocor;
	}

	public double[][] toplize(double[] dataArray, int order) {
		double[][] toplizeMatrix = new double[order][order];
		double[] atuocorr = this.autocorData(dataArray, order);

		for (int i = 1; i <= order; i++) {
			int k = 1;
			for (int j = i - 1; j > 0; j--) {
				toplizeMatrix[i - 1][j - 1] = atuocorr[k++];
			}
			toplizeMatrix[i - 1][i - 1] = 1;
			int kk = 1;
			for (int j = i; j < order; j++) {
				toplizeMatrix[i - 1][j] = atuocorr[kk++];
			}
		}
		return toplizeMatrix;
	}

	public double[] getMApara(double[] autocorData, int q) {
		double[] maPara = new double[q + 1];
		double[] tempmaPara = maPara;
		double temp = 0;
		boolean iterationFlag = true;

		maPara[0] = 1;
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
				tempmaPara[i] = -(autocorData[i] / maPara[0] - temp);
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

	public double[] parcorrCompute(double[] dataArray, int p, int q) {
		double[][] toplizeArray = new double[p][p];

		double[] atuocorr = this.autocorData(dataArray, p + q);
		double[] autocorrF = this.autocorGrma(dataArray, p + q);
		for (int i = 1; i <= p; i++) {
			int k = 1;
			for (int j = i - 1; j > 0; j--) {
				toplizeArray[i - 1][j - 1] = atuocorr[q + k++];
			}
			toplizeArray[i - 1][i - 1] = atuocorr[q];
			int kk = 1;
			for (int j = i; j < p; j++) {
				toplizeArray[i - 1][j] = atuocorr[q + kk++];
			}
		}

		Matrix toplizeMatrix = new Matrix(toplizeArray);
		Matrix toplizeMatrixinverse = toplizeMatrix.inverse();

		double[] temp = new double[p];
		for (int i = 1; i <= p; i++) {
			temp[i - 1] = atuocorr[q +i];
		}

		Matrix autocorrMatrix = new Matrix(temp, p);
		Matrix parautocorDataMatrix = toplizeMatrixinverse.times(autocorrMatrix); // [Fi]=[toplize]x[autocorr]';

		double[] result = new double[parautocorDataMatrix.getRowDimension() + 1];
		for (int i = 0; i < parautocorDataMatrix.getRowDimension(); i++) {
			result[i] = parautocorDataMatrix.get(i, 0);
		}

		double sum2 = 0;
		for (int i = 0; i < p; i++)
			for (int j = 0; j < p; j++) {
				sum2 += result[i] * result[j] * autocorrF[Math.abs(i - j)];
			}
		result[result.length - 1] = autocorrF[0] - sum2;

		return result;
	}

}
