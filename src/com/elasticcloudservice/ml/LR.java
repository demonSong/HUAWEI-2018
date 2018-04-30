package com.elasticcloudservice.ml;

import java.util.List;
import java.util.Random;

import javax.swing.plaf.basic.BasicInternalFrameTitlePane.IconifyAction;

public class LR {

	// 学习率
	private double rate;
	// w 权值向量
	private double[] weights;
	// bias
	private double bias;
	// 迭代最大次数
	private int iteration = 1000;
	// 随机数
	Random random = new Random(201803);

	public LR(int size, double rate) {
		this.rate = rate;
		this.weights = new double[size];
		this.bias = random.nextDouble();
		randomWeights(weights);
	}

	public LR(int size, double rate, int iteration) {
		this(size, rate);
		this.iteration = iteration;
	}

	public void randomWeights(double[] mat) {
		for (int i = 0; i < mat.length; i++) {
			double w = random.nextDouble();
			double f = random.nextDouble();
			mat[i] = f >= 0.5 ? w : (-w);
		}
	}
	
	public static void normalization(List<Instance> data, List<Float> max, List<Float> min) {
		if (max.size() == 0 || min.size() == 0) {
			int n = data.get(0).X.length;
			for (int i = 0; i < n; ++i) {
				float _max = -0x3f3f3f3f;
				float _min =  0x3f3f3f3f;
				for (Instance instance : data) {
					_max = Math.max(_max, instance.X[i]);
					_min = Math.min(_min, instance.X[i]);
				}
				max.add(_max);
				min.add(_min);
			}
			
			for (int i = 0; i < n; ++i) {
				if (Math.abs(max.get(i) - min.get(i)) < 0.00001) continue;
				for (Instance instance : data) {
					instance.X[i] = (instance.X[i] - min.get(i)) / (max.get(i) - min.get(i));
				}
			}
		}
		else {
			int n = data.get(0).X.length;
			for (int i = 0; i < n; ++i) {
				if (Math.abs(max.get(i) - min.get(i)) < 0.00001) continue;
				for (Instance instance : data) {
					instance.X[i] = (instance.X[i] - min.get(i)) / (max.get(i) - min.get(i));
				}
			}
		}
	}

	
	/**
	 * 分批instance直接做normalization
	 * @param data
	 */
	public static void normalization(List<Instance>[] data, List<Float>[] max, List<Float>[] min) {
		for (int k = 0; k < data.length; ++k) {
			if (max[k].size() == 0 || min[k].size() == 0) {
				int n = data[k].get(0).X.length;
				for (int i = 0; i < n; ++i) {
					float _max = -0x3f3f3f3f;
					float _min =  0x3f3f3f3f;
					for (Instance instance : data[k]) {
						_max = Math.max(_max, instance.X[i]);
						_min = Math.min(_min, instance.X[i]);
					}
					max[k].add(_max);
					min[k].add(_min);
				}
				
				for (int i = 0; i < n; ++i) {
					if (Math.abs(max[k].get(i) - min[k].get(i)) < 0.00001) {
						for (Instance instance : data[k]) {
							instance.X[i] = 0.0f;
						}
						continue;
					}
					for (Instance instance : data[k]) {
						instance.X[i] = (instance.X[i] - min[k].get(i)) / (max[k].get(i) - min[k].get(i));
					}
				}
			}
			else {
				int n = data[k].get(0).X.length;
				for (int i = 0; i < n; ++i) {
					if (Math.abs(max[k].get(i) - min[k].get(i)) < 0.00001) {
						for (Instance instance : data[k]) {
							instance.X[i] = 0.0f;
						}
						continue;
					}
					for (Instance instance : data[k]) {
						instance.X[i] = (instance.X[i] - min[k].get(i)) / (max[k].get(i) - min[k].get(i));
					}
				}
			}
		}
	}
	
	public static void normalization(Instance data, List<Float> max, List<Float> min) {
		if (max.size() == 0 || min.size() == 0) System.err.println("max min 为空");
		int n = data.X.length;
		for (int i = 0; i < n; ++i) {
			if (Math.abs(max.get(i) - min.get(i)) < 0.00001) {
				data.X[i] = 0.0f;
				continue;
			}
			data.X[i] = (data.X[i] - min.get(i)) / (max.get(i) - min.get(i));
		}
	}

	public double sigmoid(double z) {
		return 1.0 / (1 + Math.pow(Math.E, -z));
	}

	/**
	 * y = wx + b
	 * @param x
	 * @return
	 */
	public double wTx(float[] x) {
		double wtx = 0.0;
		for (int i = 0; i < weights.length; i++) {
			wtx += weights[i] * x[i];
		}
		return wtx + this.bias;
	}

	public float predict(float[] x) {
		double wtx = wTx(x);
		float logit = (float) sigmoid(wtx);
		return logit;
	}

	public void train(List<Instance> instances) {
		for (int n = 0; n < iteration; n++) {
			int m = instances.size();
			double bias_ = 0.0;
			double[] weights_ = new double[weights.length];
			for (int i = 0; i < instances.size(); i++) {
				float[] x = instances.get(i).X;
				float predicted = predict(x);
				double label = instances.get(i).y;
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
	
	public float rmse(List<Instance> testData) {
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
		
		float rmse = (float) (Math.sqrt(diff / N) / (Math.sqrt(sum_y / N) + Math.sqrt(sum_y_ / N)));
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
