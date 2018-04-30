package com.elasticcloudservice.ml;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * 支持差分的线性回归
 * @author demonSong
 *
 */
public class LinearRegression {
	
	// 学习率
	private double rate;
	// w 权值向量
	private double[] weights;
	// bisa
	private double bias;
	//num_feature
	private int size;
	// 迭代最大次数
	private int iteration = 1000;
	// 随机数
	Random random = new Random(201803);
	
	public LinearRegression(int size, double rate, int iteration) {
		this.size = size;
		this.rate = rate;
		this.iteration = iteration;
		this.bias = random.nextDouble();
		weights = new double[size];
		randomWeights(weights);
	}
	
	public void randomWeights(double[] mat) {
		for (int i = 0; i < mat.length; i++) {
			double w = random.nextDouble();
			double f = random.nextDouble();
			mat[i] = f >= 0.5 ? w : (-w);
		}
	}
	
	public float wTx(float[] x) {
		float wtx = 0.0f;
		for (int i = 0; i < weights.length; i++) {
			wtx += weights[i] * x[i];
		}
		return (float) (wtx + this.bias);
	}

	public float predict(float[] x) {
		return wTx(x);
	}
	
		
	public void train(List<Instance> dataset) {
		normalization(dataset);
		for (int n = 0; n < iteration; n++) {
			int m = dataset.size();
			double bias_ = 0.0;
			double[] weights_ = new double[weights.length];
			for (int i = 0; i < dataset.size(); i++) {
				float[] x = dataset.get(i).X;
				float predicted = predict(x);
				double label = dataset.get(i).y;
				bias_ += (label - predicted) * bias_;
				for (int j = 0; j < weights.length; j++) {
					weights_[j] = weights_[j] + (label - predicted) * x[j];
				}
			}
			
			// update
			bias = bias + rate / m * bias_;
			for (int j = 0; j < weights.length; j++) {
				weights[j] = weights[j] + rate / m * weights_[j];
			}
		}
	}
	
	public void normalization(List<Instance> data) {
		float[] max = new float[this.size];
		float[] min = new float[this.size];
		
		for (int i = 0; i < this.size; ++i) {
			float _max = -0x3f3f3f3f;
			float _min =  0x3f3f3f3f;
			for (Instance instance : data) {
				_max = Math.max(_max, instance.X[i]);
				_min = Math.min(_min, instance.X[i]);
			}
			max[i] = _max;
			min[i] = _min;
		}
			
		for (int i = 0; i < this.size; ++i) {
			if (Math.abs(max[i] - min[i]) < 0.00001) {
				for (Instance instance : data) {
					instance.X[i] = 0.0f;
				}
				continue;
			}
			for (Instance instance : data) {
				instance.X[i] = (instance.X[i] - min[i]) / (max[i] - min[i]);
			}
		}
	}
	
	public double rmse(List<Instance> testData) {
		int N = testData.size();
		
		double diff   = 0;
		double sum_y  = 0;
		double sum_y_ = 0;
		
		for (int i = 0; i < N; ++i) {
			Instance instance = testData.get(i);
			float[] x = instance.X;
            double y = instance.y;
            float y_ = predict(x);
            diff   += (y - y_) * (y - y_);
            sum_y  += y * y;
            sum_y_ += y_ * y_;
		}
		
		double rmse = Math.sqrt(diff / N) / (Math.sqrt(sum_y / N) + Math.sqrt(sum_y_ / N));
		return 1 - rmse;
	}
	
	
	/**
     * 返回预测精度
     * @param testData
     * @return
     */
    public double accuracyRate(List<Instance> testData){
        int count = 0;
        for(int i=0;i<testData.size();i++){
            Instance instance = testData.get(i);
            float[] x = instance.X;
            double label = instance.y;
            int pred = predict(x) > 0.5 ? 1 : 0;
            if(Math.abs(label - pred) < 0.0000001) count++;
        }
        return count*1.0/testData.size();
    }
    
    /**
     * 返回权值
     * @return
     */
    public double[] getWeights(){
        return weights;
    }
    
	public List<Instance> loadData(String fileName, boolean isEnd) {
		ReadData readData = new ReadData();
		return readData.readDataSet(fileName, isEnd);
	}
}
