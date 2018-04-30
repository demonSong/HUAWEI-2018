package com.elasticcloudservice.train;

import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Stack;

import com.date.utils.DateUtil;
import com.elasticcloudservice.arima.ARIMA;
import com.elasticcloudservice.ml.GBM;
import com.elasticcloudservice.ml.Instance;
import com.elasticcloudservice.ml.LR;
import com.elasticcloudservice.ml.LSTM;
import com.elasticcloudservice.ml.LeastSquareMethod;
import com.elasticcloudservice.ml.LinearRegression;
import com.elasticcloudservice.ml.LocalWeightedLinearRegression;
import com.elasticcloudservice.ml.Xgboost;
import com.elasticcloudservice.predict.CPU;
import com.elasticcloudservice.predict.Feature;
import com.elasticcloudservice.predict.LoadData;
import com.elasticcloudservice.predict.Param;
import com.elasticcloudservice.predict.Record;
import com.filetool.util.FileUtil;

public abstract class Train {
	
	static final String trainfile = "./data/TrainData_2015.1.1_2015.2.19.txt";
	
	/**
	 * do statics with all kinds of flavor
	 * @param dataset 训练数据集
	 * @return <flavor, ratio>
	 */
	public static Map<String, Double> count2Ratio(List<Record> dataset){
		Map<String, Integer> cnt_map = count2Int(dataset);
		int day_len = DateUtil.totalDiffDayInDataset(dataset); // 得到总时间
		
		Map<String, Double> res = new HashMap<>();
		if (day_len == 0) return res;
		
		for (String key : cnt_map.keySet()) {
			res.put(key, cnt_map.get(key) * 1.0 / day_len);
		}
		return res;
	}
	
	/**
	 * xgb 训练
	 * @param features
	 * @param dataset
	 * @param param
	 * @param diff_param
	 * @param roll_param
	 * @return
	 */
	public static Map<String, Integer> batchXgboost(List<Feature> features, List<Record> dataset, Param param, int[] diff_param, int[] roll_param){
		Map<String, Integer> ret = new HashMap<>();
		long s_ = dataset.get(0).year2Day + 1; //训练数据起始时刻
		
		Map<String, Integer> flavor_map = getFlavorMap();
		Map<String, Integer> brandCode = getBrandMap(features, param);
		
		/***xgb 分批训练***/
		int[] tot = new int[30];
		float[][] sequenceSum = FeatureExtract.getSequenceSum(features, brandCode, tot);
		float[][] sequence = FeatureExtract.rollSequence(features, brandCode, flavor_map, sequenceSum, tot, roll_param);
		
		List<Instance>[] instances = FeatureExtract.lrExtract(param, features, s_, brandCode, flavor_map, sequence, diff_param, roll_param);
		GBM[] models = new GBM[brandCode.size()];
		
		for (String key : brandCode.keySet()) {
			Xgboost xgb = new Xgboost(instances[brandCode.get(key)], true, 0.1);
			models[brandCode.get(key)] = xgb.training(100, 50, false, "mse", "squareloss", 0.5, 10, 2, 0.8, 0.8, 20, 5, 2, 0, 1);
		}
		
		/***lr 预测 ***/
		long ss_ = DateUtil.year2dayLong(param.startTime) + 1;
		long ee_ = DateUtil.year2dayLong(param.endTime) + 1;
		
		for (String tag : param.virtuals.keySet()) {
			for (long day = ss_; day <= ee_; ++day) {
				Instance instance = FeatureExtract.next(param, param.virtuals, tag, s_, day, brandCode, sequence);
				
				double y_ = models[brandCode.get(tag)].predict(instance);
				double val = y_ + sequence[brandCode.get(tag)][(int) (day - s_) - diff_param[flavor_map.get(tag)]];
				int rp = roll_param[flavor_map.get(tag)];
				double tmp = val * rp;
				double pre_sum = FeatureExtract.querySumRange(sequenceSum, brandCode.get(tag), tot[brandCode.get(tag)] - rp + 1, tot[brandCode.get(tag)] - 1);
				double real_val = tmp - pre_sum;
				int cnt = (int) real_val;
				cnt = cnt >= 0 ? cnt : 0;
				
				ret.put(instance.tag, ret.getOrDefault(instance.tag, 0) + cnt);
				
				tot[brandCode.get(tag)]++;
				sequenceSum[brandCode.get(tag)][tot[brandCode.get(tag)]] = sequenceSum[brandCode.get(tag)][tot[brandCode.get(tag)] - 1] + cnt;
				sequence[brandCode.get(tag)][tot[brandCode.get(tag)] - 1] = 
						FeatureExtract.querySumRange(sequenceSum, brandCode.get(tag), tot[brandCode.get(tag)] - rp, tot[brandCode.get(tag)] - 1) / rp;
				
			}
		}
		
		return ret;
	}
	
	/**
	 * 支持差分平滑训练 fix some bugs such as index problem
	 * @param features
	 * @param dataset
	 * @param param
	 * @param diff
	 * @return
	 */
	public static Map<String, Integer> batchDiffRollLinearRegression(List<Feature> features, 
			List<Record> dataset, Param param, int[] diff_param, int[] roll_param){
		
		Map<String, Integer> ret = new HashMap<>();
		long s_ = dataset.get(0).year2Day + 1; //训练数据起始时刻
		
		Map<String, Integer> flavor_map = getFlavorMap();
		Map<String, Integer> brandCode = getBrandMap(features, param);
		
		/***lr 分批训练***/
		int[] tot = new int[30];
		float[][] sequenceSum = FeatureExtract.getSequenceSum(features, brandCode, tot);
		float[][] sequence = FeatureExtract.rollSequence(features, brandCode, flavor_map, sequenceSum, tot, roll_param);
		
		List<Instance>[] instances = FeatureExtract.lrExtract(param, features, s_, brandCode, flavor_map, sequence, diff_param, roll_param);
		
		List<Float>[] max = new ArrayList[brandCode.size()];
		List<Float>[] min = new ArrayList[brandCode.size()];
		
		for (int i = 0; i < brandCode.size(); ++i) {
			max[i] = new ArrayList<>();
			min[i] = new ArrayList<>();
		}
		LR.normalization(instances, max, min);
		
		double rate = 0.001; // 学习率
		int iteration = 10000; // 迭代次数
		LinearRegression[] lr = new LinearRegression[brandCode.size()];
		
		for (String key : brandCode.keySet()) {
			int size = instances[brandCode.get(key)].get(0).X.length; // 特征数
			lr[brandCode.get(key)] = new LinearRegression(size, rate, iteration);
			lr[brandCode.get(key)].train(instances[brandCode.get(key)]);
		}
		
		/***lr 预测 ***/
		long ss_ = DateUtil.year2dayLong(param.startTime) + 1;
		long ee_ = DateUtil.year2dayLong(param.endTime) + 1;
		
		for (String tag : param.virtuals.keySet()) {
			for (long day = ss_; day <= ee_; ++day) {
				Instance instance = FeatureExtract.next(param, param.virtuals, tag, s_, day, brandCode, sequence);
				LR.normalization(instance, max[brandCode.get(tag)], min[brandCode.get(tag)]);
				
				float y_ = lr[brandCode.get(tag)].predict(instance.X);
				float val = y_ + (diff_param[flavor_map.get(tag)] != 0 ? sequence[brandCode.get(tag)][(int) (day - s_) - diff_param[flavor_map.get(tag)]] : 0);
				int rp = roll_param[flavor_map.get(tag)];
				float tmp = val * rp;
				float pre_sum = FeatureExtract.querySumRange(sequenceSum, brandCode.get(tag), tot[brandCode.get(tag)] - rp + 1, tot[brandCode.get(tag)] - 1);
				float real_val = tmp - pre_sum;
				int cnt = (int) real_val;
				cnt = cnt >= 0 ? cnt : 0;
				
				ret.put(instance.tag, ret.getOrDefault(instance.tag, 0) + cnt);
				
				tot[brandCode.get(tag)]++;
				sequenceSum[brandCode.get(tag)][tot[brandCode.get(tag)]] = sequenceSum[brandCode.get(tag)][tot[brandCode.get(tag)] - 1] + cnt;
				sequence[brandCode.get(tag)][tot[brandCode.get(tag)] - 1] = 
						FeatureExtract.querySumRange(sequenceSum, brandCode.get(tag), tot[brandCode.get(tag)] - rp, tot[brandCode.get(tag)] - 1) / rp;
				
			}
		}
		return ret;
	}
	
	
	/**
	 * 支持差分训练
	 * @param features
	 * @param dataset
	 * @param param
	 * @param diff
	 * @return
	 */
	public static Map<String, Integer> batchDiffLinearRegression(List<Feature> features, List<Record> dataset, Param param, int[] diff){
		Map<String, Integer> ret = new HashMap<>();
		long s_ = dataset.get(0).year2Day + 1; //训练数据起始时刻
		
		Map<String, Integer> flavor_map = getFlavorMap();
		
		/***lr 分批训练***/
		Map<String, Integer> brandCode = new HashMap<>();
		float[][] sequence = new float[30][2400];
		List<Instance>[] instances = FeatureExtract.lrExtract(param, features, s_, brandCode, flavor_map, sequence, diff);
		
		List<Float>[] max = new ArrayList[brandCode.size()];
		List<Float>[] min = new ArrayList[brandCode.size()];
		
		for (int i = 0; i < brandCode.size(); ++i) {
			max[i] = new ArrayList<>();
			min[i] = new ArrayList<>();
		}
		LR.normalization(instances, max, min);
		
		double rate = 0.001; // 学习率
		int iteration = 10000; // 迭代次数
		LinearRegression[] lr = new LinearRegression[brandCode.size()];
		
		for (String key : brandCode.keySet()) {
			int size = instances[brandCode.get(key)].get(0).X.length; // 特征数
			lr[brandCode.get(key)] = new LinearRegression(size, rate, iteration);
			lr[brandCode.get(key)].train(instances[brandCode.get(key)]);
		}
		
		/***lr 预测 ***/
		long ss_ = DateUtil.year2dayLong(param.startTime) + 1;
		long ee_ = DateUtil.year2dayLong(param.endTime) + 1;
		
		for (String tag : param.virtuals.keySet()) {
			for (long day = ss_; day <= ee_; ++day) {
				Instance instance = FeatureExtract.next(param, param.virtuals, tag, s_, day, brandCode, sequence);
				LR.normalization(instance, max[brandCode.get(tag)], min[brandCode.get(tag)]);
				
				float y_ = lr[brandCode.get(tag)].predict(instance.X);
				int cnt = (int) (y_ + sequence[brandCode.get(tag)][(int) (day - s_) - diff[flavor_map.get(tag)]]);
				cnt = cnt >= 0 ? cnt : 0;
				ret.put(instance.tag, ret.getOrDefault(instance.tag, 0) + cnt);
				sequence[brandCode.get(tag)][(int) (day - s_)] = cnt;
			}
		}
		return ret;
	}
	
	public static Map<String, Integer> batchLinearRegression(List<Feature> features, List<Record> dataset, Param param){
		Map<String, Integer> ret = new HashMap<>();
		long s_ = dataset.get(0).year2Day + 1; //训练数据起始时刻
		
		/***lr 分批训练***/
		Map<String, Integer> brandCode = new HashMap<>();
		float[][] minmax_y = new float[20][2];
		for (int i = 0; i < 20; ++i) {
			minmax_y[i][0] = -0x3f3f3f3f;
			minmax_y[i][1] =  0x3f3f3f3f;
		}
		
		float[][] sequence = new float[30][1200];
		List<Instance>[] instances = FeatureExtract.lrExtract(param, features, s_, param.virtuals, brandCode, minmax_y, sequence, true);
		
		List<Float>[] max = new ArrayList[brandCode.size()];
		List<Float>[] min = new ArrayList[brandCode.size()];
		
		for (int i = 0; i < brandCode.size(); ++i) {
			max[i] = new ArrayList<>();
			min[i] = new ArrayList<>();
		}
		LR.normalization(instances, max, min);
		
		double rate = 0.001; // 学习率
		int iteration = 10000; // 迭代次数
		LinearRegression[] lr = new LinearRegression[brandCode.size()];
		
		for (String key : brandCode.keySet()) {
			int size = instances[brandCode.get(key)].get(0).X.length; // 特征数
			lr[brandCode.get(key)] = new LinearRegression(size, rate, iteration);
			lr[brandCode.get(key)].train(instances[brandCode.get(key)]);
		}
		
		/***lr 预测 ***/
		long ss_ = DateUtil.year2dayLong(param.startTime) + 1;
		long ee_ = DateUtil.year2dayLong(param.endTime) + 1;
		
		for (String tag : param.virtuals.keySet()) {
			for (long day = ss_; day <= ee_; ++day) {
				Instance instance = FeatureExtract.next(param, param.virtuals, tag, s_, day, brandCode, sequence);
				LR.normalization(instance, max[brandCode.get(tag)], min[brandCode.get(tag)]);
				
				float y_ = lr[brandCode.get(tag)].predict(instance.X);
				int cnt = (int) y_;
				ret.put(instance.tag, ret.getOrDefault(instance.tag, 0) + cnt);
				sequence[brandCode.get(tag)][(int) (day - s_)] = cnt;
			}
		}
		return ret;
	}
	
	public static Map<String, Integer> linearRegression(List<Feature> features, List<Record> dataset, Param param){
		Map<String, Integer> ret = new HashMap<>();
		long s_ = dataset.get(0).year2Day + 1; //训练数据起始时刻
		
		/***LinearRegression 训练***/
		Map<String, Integer> brandCode = new HashMap<>();
		float[] minmax_y = {-0x3f3f3f3f, 0x3f3f3f3f};
		float[][] sequence = new float[30][2400];
		List<Instance> instances = FeatureExtract.lrExtract(param, features, s_, param.virtuals, brandCode, minmax_y, sequence, true);
		
		List<Float> max = new ArrayList<>();
		List<Float> min = new ArrayList<>();
		LR.normalization(instances, max, min);
		
		double rate = 0.001; // 学习率
		int iteration = 10000; // 迭代次数
		
		int size = instances.get(0).X.length; // 特征数
		LinearRegression logistic = new LinearRegression(size, rate, iteration);
		
		logistic.train(instances);
		System.out.println(logistic.rmse(instances));
		
		/***lr 预测 ***/
		
		long ss_ = DateUtil.year2dayLong(param.startTime) + 1;
		long ee_ = DateUtil.year2dayLong(param.endTime) + 1;
		
		for (String tag : param.virtuals.keySet()) {
			for (long day = ss_; day <= ee_; ++day) {
				Instance instance = FeatureExtract.next(param, param.virtuals, tag, s_, day, brandCode, sequence);
				LR.normalization(instance, max, min);
				
				float y_ = logistic.predict(instance.X);
				int cnt = (int) y_;
				ret.put(instance.tag, ret.getOrDefault(instance.tag, 0) + cnt);
				sequence[brandCode.get(tag)][(int) (day - s_)] = cnt;
			}
		}
		return ret;
	}
	
	/**
	 * 复赛最多支持1 - 18
	 * @return
	 */
	public static Map<String, Integer> getFlavorMap(){
		Map<String, Integer> flavor_map = new HashMap<>();
		for (int i = 1; i <= 18; ++i) {
			flavor_map.put("flavor" + i, i - 1);
		}
		return flavor_map;
	}
	
	/**
	 * brand code
	 * @param features
	 * @param param
	 * @return
	 */
	private static Map<String, Integer> getBrandMap(List<Feature> features, Param param){
		Map<String, Integer> brand_code = new HashMap<>();
		for (Feature record : features) {
			String tag = record.tag;
			if (!param.virtuals.containsKey(tag)) continue;
			if (!brand_code.containsKey(tag)) {
				brand_code.put(tag, brand_code.size());
			}
		}
		return brand_code;
	}
	
	
	private static double[] list2array(List<Double> arra) {
		double[] ret = new double[arra.size()];
		for (int i = 0; i < arra.size(); ++i) {
			ret[i] = arra.get(i);
		}
		return ret;
	}
	
	private static Map<String, Double> getLocalWeightedPrediction(List<Feature> features, Param param, Map<String, Stack<Double>> _Xy, 
			Map<String, Integer> flavor_map,double alpha, double sigma, int T, int span, double fixed, int recent_day){
		
		Map<String, Double> ret = new HashMap<>();
		int week = span / T;
		int days = span % T;
		
		Map<String, Integer> replace = recentDecreseMean(features, param, recent_day, alpha, 0.66, 0.34, fixed, week * T);
		
		for (String flavor : _Xy.keySet()) {
			if (!param.virtuals.containsKey(flavor)) continue;
			Stack<Double> val =_Xy.get(flavor);
			int len = val.size();
			double[][] X = new double[len][1];
			for (int i = 0; i < len; ++i) X[i][0] = i + 1;
			double[] y = new double[len];
			for (int i = 0; i < len; ++i) y[i] = val.pop();
			for (int i = len - 1; i >= 0; --i) val.push(y[i]); // 送回去
			LocalWeightedLinearRegression lwlr = new LocalWeightedLinearRegression(X, y, sigma);
			for (int i = 1; i <= week; ++i) {
				double pred = lwlr.fit(new double[] {len + i});
				if (Double.compare(pred, -0x3f3f3f3f) == 0) {
					pred = replace.getOrDefault(flavor, 0);
				}
				pred = pred > 0 ? pred : 0;
				ret.put(flavor, ret.getOrDefault(flavor, 0.0) + pred);
			}
		}
		
		Map<String, Integer> remain = recentDecreseMean(features, param, recent_day, alpha, 0.66, 0.34, fixed, days);
		
		Map<String, Double> ans = new HashMap<>();
		for (String flavor : flavor_map.keySet()) {
			if (!param.virtuals.containsKey(flavor)) continue;
			if (remain.containsKey(flavor) && ret.containsKey(flavor)) {
				ans.put(flavor, ret.get(flavor) + remain.get(flavor));
			}
			else if (remain.containsKey(flavor)) {
				int re = remain.get(flavor);
				ans.put(flavor, re * 1.0);
			}
			else if (ret.containsKey(flavor)){
				double val = ret.get(flavor);
				ans.put(flavor, val);
			}
		}
		return ans;
	}
	
	public static Map<String, Integer> localWeightedLR(List<Feature> features, Param param, double alpha, double sigma, int T, int gap, double fixed, int recent_day){
		Map<String, Integer> flavor_map = getFlavorMap();
		int[] tot = new int[30];
		
		float[][] sequence = FeatureExtract.getSequence(features, flavor_map, tot);
		
		Map<String, Stack<Double>> mem = new HashMap<>();
		for (String flavor : flavor_map.keySet()) {
			int idx = flavor_map.get(flavor);
			double sum = 0;
			for (int i = tot[idx] - 1, c = 1; i >= 0; --i, ++c) {
				if (c % T == 0) {
					sum += sequence[idx][i];
					mem.computeIfAbsent(flavor, k -> new Stack<>()).push(sum);
					sum = 0;
				}
				else {
					sum += sequence[idx][i];
				}
			}
		}
		
		Map<String, Double> all = getLocalWeightedPrediction(features, param, mem, flavor_map, alpha, sigma, T, param.span + gap, fixed, recent_day);
		Map<String, Double> sub = getLocalWeightedPrediction(features, param, mem, flavor_map, alpha, sigma, T, gap, fixed, recent_day);
		
		Map<String, Integer> ans = new HashMap<>();
		for (String flavor : flavor_map.keySet()) {
			if (!param.virtuals.containsKey(flavor)) continue;
			double a = all.getOrDefault(flavor, 0.0);
			double b = sub.getOrDefault(flavor, 0.0);
			double s = a - b;
			s = s > 0 ? s : 0;
			ans.put(flavor, (int)s);
		}
		return ans;
	}
	
	public static List<Feature> localWeightedLR(List<Feature> features, Param param, double sigma){
		List<Feature> ret = new ArrayList<>();
		Map<String, CPU> virtuals = param.virtuals;
		int span = param.span;
		long sday = DateUtil.year2dayLong(param.startTime);
		
		Map<String, List<Double>> sequence_y = new HashMap<>();
		Map<String, List<Double>> sequence_x = new HashMap<>();
		for (Feature f : features) {
			if (virtuals.containsKey(f.tag)) {
				double cnt = f.count * 1.0;
				sequence_y.computeIfAbsent(f.tag, k -> new ArrayList<>()).add(cnt);
				sequence_x.computeIfAbsent(f.tag, k -> new ArrayList<>()).add(sequence_y.get(f.tag).size() * 1.0);
			}
		}
		
		Map<String, Integer> replace = recentDecreseMean(features, param, 10, 0.966, 0.66, 0.34, 1.0, 1);
		
		for (String key : virtuals.keySet()) {
			double[] x = list2array(sequence_x.get(key));
			double[][] X = new double[x.length][1];
			for (int i = 0; i < x.length; ++i) X[i][0] = x[i];
			double[] y = list2array(sequence_y.get(key));
			
			LocalWeightedLinearRegression lwlr = new LocalWeightedLinearRegression(X, y, sigma);
			
//			List<Double> ans = new ArrayList<>();
//			for (int i = 1; i <= x.length; ++i) ans.add(lwlr.fit(new double[] {i}));
//			System.out.println(key);
//			System.out.println(sequence_y.get(key));
//			System.out.println(ans);
			
			for (int i = 0; i < span; ++i) {
				double predict = lwlr.fit(new double[] {x.length + 1 + i});
				if (Double.compare(predict, -0x3f3f3f3f) == 0) {
					predict = replace.getOrDefault(key, 0);
				}
				predict = predict > 0 ? predict : 0;
				Feature feature = new Feature();
				feature.year2Day = sday + i;
				feature.tag = key;
				feature.date = DateUtil.long2Date((feature.year2Day + 1) * 1000 * 24 * 3600);
				feature.isHoliday = param.longHolidaySet.contains(feature.year2Day);
				feature.dayOfWeek = DateUtil.week(feature.date);
				feature.count = (int) (predict > 0 ? predict : 0);
				ret.add(feature);
			}
		}
		return ret;
	}
	
	public static List<Feature> leastSquareMethod(List<Feature> features, Param param, int order){
		List<Feature> ret = new ArrayList<>();
		Map<String, CPU> virtuals = param.virtuals;
		int span = param.span;
		long sday = DateUtil.year2dayLong(param.startTime);
		
		Map<String, List<Double>> sequence_y = new HashMap<>();
		Map<String, List<Double>> sequence_x = new HashMap<>();
		for (Feature f : features) {
			if (virtuals.containsKey(f.tag)) {
				double cnt = f.count * 1.0;
				sequence_y.computeIfAbsent(f.tag, k -> new ArrayList<>()).add(cnt);
				sequence_x.computeIfAbsent(f.tag, k -> new ArrayList<>()).add(sequence_y.get(f.tag).size() * 1.0);
			}
		}
		
		for (String key : virtuals.keySet()) {
			double[] x = list2array(sequence_x.get(key));
			double[] y = list2array(sequence_y.get(key));
			
			LeastSquareMethod eastSquareMethod = new LeastSquareMethod(x, y, order); 
			
//			List<Double> ans = new ArrayList<>();
//			for (int i = 1; i <= x.length; ++i) ans.add(eastSquareMethod.fit(i));
//			System.out.println(key);
//			System.out.println(sequence_y.get(key));
//			System.out.println(ans);
			
			for (int i = 0; i < span; ++i) {
				double predict = eastSquareMethod.fit(x.length + 1 + i);
				Feature feature = new Feature();
				feature.year2Day = sday + i;
				feature.tag = key;
				feature.date = DateUtil.long2Date((feature.year2Day + 1) * 1000 * 24 * 3600);
				feature.isHoliday = param.longHolidaySet.contains(feature.year2Day);
				feature.dayOfWeek = DateUtil.week(feature.date);
				feature.count = (int) (predict > 0 ? predict : 0);
				ret.add(feature);
			}
		}
		return ret;
	}
	
	public static List<Feature> arimaForMock(List<Feature> features, List<Record> dataset, Param param, int[] flavor_param, double alpha){
		List<Feature> ret = new ArrayList<>();
		
		/***arima***/
		Map<String, CPU> virtuals = param.virtuals;
		int span = param.span;
		long sday = DateUtil.year2dayLong(param.startTime);
		
//		Map<String, Double> mean_map = Train.train(dataset, false);
		Map<String, Double> mean_map = Train.recentMeanDouble(features, param, 10);
		Map<String, List<Double>> train_map = FeatureExtract.extract(features, virtuals.keySet(), mean_map, param);
		
		Map<String, Integer> flavor_map = getFlavorMap();
		
		for (String key : virtuals.keySet()) {
			List<Double> train = train_map.get(key);
			train = getExpoSequence(train, alpha);
			ARIMA arima = new ARIMA(flavor_param[flavor_map.get(key)]);
			int[] bestModel = {1, 0};
			for (int i = 0; i < span; ++i) {
				double[] data = new double[train.size()];
				boolean same = true;
				double _same = 0.0;
				for (int j = 0; j < train.size(); ++j) {
					data[j] = train.get(j);
					if (j >= 1 && data[j] != data[j - 1]) {
						same = false;
					}
					_same = data[j];
				}
				
				if (same) {
					train.add(_same);
					
					Feature feature = new Feature();
					feature.count = (int) _same;
					feature.year2Day = sday + i;
					feature.tag = key;
					feature.date = DateUtil.long2Date((feature.year2Day + 1) * 1000 * 24 * 3600);
					feature.isHoliday = param.longHolidaySet.contains(feature.year2Day);
					feature.dayOfWeek = DateUtil.week(feature.date);
					ret.add(feature);
				}
				else {
					arima.setData(data);
					int[] model = arima.getARIMAmodel(i == 0, bestModel); // 可调
					if (model == null) {
						System.err.println("can not predict!!!");
						break;
					}
					int predict = arima.aftDeal(arima.predictValue(model[0], model[1]));
					train.add(predict * 1.0);
					bestModel = model;
					
					Feature feature = new Feature();
					feature.count = predict;
					feature.year2Day = sday + i;
					feature.tag = key;
					feature.date = DateUtil.long2Date((feature.year2Day + 1) * 1000 * 24 * 3600);
					feature.isHoliday = param.longHolidaySet.contains(feature.year2Day);
					feature.dayOfWeek = DateUtil.week(feature.date);
					ret.add(feature);
				}
			}
		}
		return ret;
	}
	
	public static Map<String, Integer> arima(List<Feature> features, List<Record> dataset, Param param, int[] flavor_param, double alpha){
		Map<String, Integer> ret = new HashMap<>();
		
		/***arima***/
		Map<String, CPU> virtuals = param.virtuals;
		int span = param.span;
		
//		Map<String, Double> mean_map = Train.train(dataset, false);
		Map<String, Double> mean_map = Train.recentMeanDouble(features, param, 5);
		Map<String, List<Double>> train_map = FeatureExtract.extract(features, virtuals.keySet(), mean_map, param);
		
		Map<String, Integer> flavor_map = getFlavorMap();
		
		for (String key : virtuals.keySet()) {
			
			List<Double> train = train_map.get(key);
			train = getExpoSequence(train, alpha);
			int _sum = 0;
			boolean cannotPredict = false;
			ARIMA arima = new ARIMA(flavor_param[flavor_map.get(key)]);
			int[] bestModel = {1, 0};
			for (int i = 0; i < span; ++i) {
				double[] data = new double[train.size()];
				boolean same = true;
				double _same = 0.0;
				for (int j = 0; j < train.size(); ++j) {
					data[j] = train.get(j);
					if (j >= 1 && data[j] != data[j - 1]) {
						same = false;
					}
					_same = data[j];
				}
				
				if (same) {
					_sum += _same;
					train.add(_same);
				}
				else {
					arima.setData(data);
					int[] model = arima.getARIMAmodel(i == 0, bestModel); // 可调
					if (model == null) {
						cannotPredict = true;
						System.err.println("can not predict!!!");
						break;
					}
					int predict = arima.aftDeal(arima.predictValue(model[0], model[1]));
					_sum += predict;
					train.add(predict * 1.0);
					bestModel = model;
				}
			}
			
			if (cannotPredict) ret.put(key, (int)(mean_map.get(key) * span));
			else ret.put(key, _sum);
		}
		
		return ret;
	}
	
	/**
	 * 近期一段时间内每个flavor的情况
	 * @param features
	 * @param param
	 * @return
	 */
	private static Map<String, Double> recentMeanDouble(List<Feature> features, Param param, int latest){
		long startday = DateUtil.year2dayLong(param.startTime);
		long endday = startday - latest;
		int n = features.size();
		
		Map<String, Double> cnt = new HashMap<>();
		for (int i = n - 1; i >= 0; --i) {
			Feature f = features.get(i);
			long now = f.year2Day;
			if (now >= endday) {
				cnt.put(f.tag, cnt.getOrDefault(f.tag, 0.0) + f.count);
			}
			else {
				break;
			}
		}
		for (String key : cnt.keySet()) {
			cnt.put(key, cnt.get(key) / latest);
		}
		return cnt;
	}
	
	private static List<Double> getExpoSequence(List<Double> train, double alpha){
		int n = train.size();
		List<Double> ret = new ArrayList<>();
		for (int i = 0; i < n; ++i) {
			if (i == 0) {
				ret.add(train.get(i));
			}
			else {
				ret.add(alpha * train.get(i) + (1 - alpha) * ret.get(i - 1));
			}
		}
		return ret;
	}
	
	
	private static float[][] getExpoSequence(float[][] sequence, int[] tot, double alpha){
		int n = sequence.length;
		int m = sequence[0].length;
		
		float[][] exp_sequence = new float[n][m];
		for (int idx = 0; idx < n; ++idx) {
			for (int i = 0; i < tot[idx]; ++i) {
				if (i == 0) {
					exp_sequence[idx][i] = sequence[idx][i];
					continue;
				}
				exp_sequence[idx][i] = (float) (alpha * sequence[idx][i] + (1 - alpha) * exp_sequence[idx][i - 1]);
			}
		}
		
		return exp_sequence;
	}
	
	private static double bestAlpha(float[] expSequence, float[] sequence, int len, int latest){
		int train_len = len - latest;
		double all = 0.0;
		for (int i = train_len - 1, k = 0; i >= 0 && k < latest; --i, ++k) {
			all += expSequence[i];
		}
		double pred = all;
		double real = 0.0;
		for (int i = train_len; i < len; ++i) {
			real += sequence[i];
		}
		return Math.abs(pred - real);
	}
	
	public static float[][] normalize(float[][] sequence, float[] max, float[] min){
		int n = sequence.length;
		int m = sequence[0].length;
		float[][] ret = new float[n][m];
		int INF = 0x3f3f3f3f;
		Arrays.fill(max, -INF);
		Arrays.fill(min,  INF);
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < m; ++j) {
				max[i] = Math.max(max[i], sequence[i][j]);
				min[i] = Math.min(min[i], sequence[i][j]);
			}
		}
		
		for (int i = 0; i < n; ++i) {
			if (max[i] == min[i]) {
				for (int j = 0; j < m; ++j) {
					if (Double.compare(max[i], 0) == 0) break;
					ret[i][j] = sequence[i][j] / max[i];
				}
			}
			else {
				for (int j = 0; j < m; ++j) {
					ret[i][j] = (sequence[i][j] - min[i]) / (max[i] - min[i]);
				}
			}
		}
		return ret;
	}
	
//	public static Map<String, Integer> lstm(List<Feature> features, Param param, int x_dim){
//		Map<String, Integer> flavor_map = getFlavorMap();
//		int[] tot = new int[30];
//		
//		double[][] sequence = FeatureExtract.getSequence(features, flavor_map, tot);
//		double[] max = new double[30];
//		double[] min = new double[30];
//		sequence = normalize(sequence, max, min);
//		
//		LSTM[] lstms = new LSTM[30];
//		
//		Map<String, Double> ret = new HashMap<>();
//		for (String key : param.virtuals.keySet()) {
//			int idx = flavor_map.get(key);
//			double[] seq = sequence[idx];
//			int len = tot[idx];
//			
//			List<List<Double>> y = new ArrayList<>();
//			List<List<Double>> X = new ArrayList<>();
//			
//			List<Double> x = new ArrayList<>();
//			for (int i = 0; i < x_dim; ++i) {
//				x.add(seq[i]);
//			}
//			
//			for (int i = x_dim; i < len - 1; ++i) {
//				X.add(new ArrayList<>(x));
//				x.remove(0);
//				x.add(seq[i]);
//				y.add(new ArrayList<>(x));
//			}
//			
//			lstms[idx] = new LSTM(10, 1, 0.1);
//			lstms[idx].train(X, y, 100);
//			
//			x.remove(0);
//			x.add(seq[len - 1]);
//			
//			double pred_ = lstms[idx].predict(x);
//			ret.put(key, ret.getOrDefault(key, 0.0) + pred_ * (max[idx] - min[idx]) + min[idx]);
//			
//			double prev = pred_;
//			for (int i = 1; i < param.span; ++i) {
//				x.remove(0);
//				x.add(prev);
//				pred_ = lstms[idx].predict(x);
//				ret.put(key, ret.getOrDefault(key, 0.0) + pred_ * (max[idx] - min[idx]) + min[idx]);
//				prev = pred_;
//			}
//		}
//		return doubleMap2IntMapUseCeil(ret);
//	}
	
	public static Map<String, Integer> lstm(List<Feature> features, Param param, int x_dim, int cell){
		Map<String, Integer> flavor_map = getFlavorMap();
		int[] tot = new int[30];
		
		float[][] sequence = FeatureExtract.getSequence(features, flavor_map, tot);
		float[] max = new float[30];
		float[] min = new float[30];
		sequence = normalize(sequence, max, min);
		
		LSTM[] lstms = new LSTM[30];
		
		Map<String, Double> ret = new HashMap<>();
		for (String key : param.virtuals.keySet()) {
			int idx = flavor_map.get(key);
			float[] seq = sequence[idx];
			int len = tot[idx];
			
			List<List<List<Float>>> train = new ArrayList<>();
			List<List<Float>> label = new ArrayList<>();
			
			// get one feature
			List<Float> x = new ArrayList<>();
			for (int i = 0; i < x_dim; ++i) {
				x.add(seq[i]);
			}
			
			// get all features
			List<List<Float>> X = new ArrayList<>();
			List<Float> y = new ArrayList<>();
			for (int i = x_dim; i < len - 1; ++i) {
				X.add(new ArrayList<>(x));
				x.remove(0);
				x.add(seq[i]);
				y.add(seq[i]);
			}
			
			// split to cell
			List<List<Float>> X_ = new ArrayList<>();
			List<Float> y_ = new ArrayList<>();
			
			for (int i = 0; i < cell; ++i) {
				X_.add(X.get(i));
				y_.add(y.get(i));
			}
			train.add(new ArrayList<>(X_));
			label.add(new ArrayList<>(y_));
			
			for (int i = cell; i < X.size(); ++i) {
				X_.remove(0);
				y_.remove(0);
				X_.add(X.get(i));
				y_.add(y.get(i));
				train.add(new ArrayList<>(X_));
				label.add(new ArrayList<>(y_));
			}
			
			lstms[idx] = new LSTM(10, x_dim, 0.1);
			lstms[idx].train(train, label, 100);
			
			
			X_.remove(0);
			x.remove(0);
			x.add(seq[len - 1]);
			X_.add(new ArrayList<>(x));
			
			float pred_ = lstms[idx].predict(X_);
			ret.put(key, ret.getOrDefault(key, 0.0) + pred_ * (max[idx] - min[idx]) + min[idx]);
			
			float prev = pred_;
			for (int i = 1; i < param.span; ++i) {
				X_.remove(0);
				x.remove(0);
				x.add(prev);
				X_.add(new ArrayList<>(x));
				pred_ = lstms[idx].predict(X_);
				ret.put(key, ret.getOrDefault(key, 0.0) + pred_ * (max[idx] - min[idx]) + min[idx]);
				prev = pred_;
			}
		}
		return doubleMap2IntMapUseCeil(ret);
	}
	
	/**
	 * 支持一阶指数平滑法
	 * @param features
	 * @param param
	 * @return
	 */
	private static Map<String, Float> recentMeanDoubleUsedExponent(List<Feature> features, Param param, int latest, double alpha){
		Map<String, Integer> flavor_map = getFlavorMap();
		int[] tot = new int[30];
		
		float[][] sequence = FeatureExtract.getSequence(features, flavor_map, tot);
		float[][] exp_sequence = getExpoSequence(sequence, tot, alpha);

		Map<String, Float> ans = new HashMap<>();
		Map<String, Integer> cnt = new HashMap<>();
		for (String key : param.virtuals.keySet()) {
			int idx = flavor_map.get(key);
			for (int i = tot[idx] - 1, cnt_ = 0; i >= 0 && cnt_ < latest; --i, ++cnt_) {
				ans.put(key, ans.getOrDefault(key, 0.0f) + exp_sequence[idx][i]);
				cnt.put(key, cnt.getOrDefault(key, 0) + 1);
			}
		}
		for (String key : ans.keySet()) {
			ans.put(key, ans.get(key) / cnt.get(key));
		}
		return ans;
	}
	
	private static Map<String, Float> recentMeanDecreseDoubleUsedExponent(List<Feature> features, Param param, int latest, double alpha, double p1, double p2){
		Map<String, Integer> flavor_map = getFlavorMap();
		int[] tot = new int[30];
		
		float[][] sequence = FeatureExtract.getSequence(features, flavor_map, tot);
		float[][] exp_sequence = getExpoSequence(sequence, tot, alpha);

		Map<String, Float> a1 = new HashMap<>();
		Map<String, Float> a2 = new HashMap<>();
		Map<String, Integer> c1 = new HashMap<>();
		Map<String, Integer> c2 = new HashMap<>();
		for (String key : param.virtuals.keySet()) {
			int idx = flavor_map.get(key);
			for (int i = tot[idx] - 1, cnt_ = 0; i >= 0 && cnt_ < 2 * latest; --i, ++cnt_) {
				if (cnt_ < latest) {
					c1.put(key, c1.getOrDefault(key, 0) + 1);
					a1.put(key, a1.getOrDefault(key, 0f) + exp_sequence[idx][i]);
				}
				else {
					c2.put(key, c2.getOrDefault(key, 0) + 1);
					a2.put(key, a2.getOrDefault(key, 0f) + exp_sequence[idx][i]);
				}
			}
		}
		for (String key : a1.keySet()) {
			a1.put(key, a1.get(key) / c1.get(key));
		}
		for (String key : a2.keySet()) {
			a2.put(key, a2.get(key) / c2.get(key));
		}
		
		Map<String, Float> ans = new HashMap<>();
		for (String key : a1.keySet()) {
			if (a2.containsKey(key)) {
				ans.put(key, (float)(p1 * a1.get(key) + p2 * a2.get(key)));
			}
			else {
				ans.put(key, 1.0f * a1.get(key));
			}
		}
		return ans;
	}
	
	private static Map<String, Float> allMeanDoubleUsedExponent(List<Feature> features, Param param, double alpha){
		Map<String, Integer> flavor_map = getFlavorMap();
		int[] tot = new int[30];
		
		float[][] sequence = FeatureExtract.getSequence(features, flavor_map, tot);
		float[][] exp_sequence = getExpoSequence(sequence, tot, alpha);

		Map<String, Float> a1 = new HashMap<>();
		Map<String, Integer> c1 = new HashMap<>();
		
		for (String key : param.virtuals.keySet()) {
			int idx = flavor_map.get(key);
			for (int i = tot[idx] - 1; i >= 0; --i) {
				c1.put(key, c1.getOrDefault(key, 0) + 1);
				a1.put(key, a1.getOrDefault(key, 0f) + exp_sequence[idx][i]);
			}
		}
		for (String key : a1.keySet()) {
			a1.put(key, a1.get(key) / c1.get(key));
		}
		return a1;
	}

	
	public static float findRatio(List<Feature> features, Param param, int latest, double alpha){
		Map<String, Integer> flavor_map = getFlavorMap();
		int[] tot = new int[30];
		
		float[][] sequence = FeatureExtract.getSequence(features, flavor_map, tot);
		float[][] exp_sequence = getExpoSequence(sequence, tot, alpha);

		float k1 = 0;
		float k2 = 0;
		int c1 = 0;
		int c2 = 0;
		for (String key : param.virtuals.keySet()) {
			int idx = flavor_map.get(key);
			for (int i = tot[idx] - 1, cnt_ = 0; i >= 0 && cnt_ < 2 * latest; --i, ++cnt_) {
				if (cnt_ < latest) {
					c1 ++;
					k1 += exp_sequence[idx][i];
				}
				else {
					c2 ++;
					k2 += exp_sequence[idx][i];
				}
			}
		}
		if (c1 > 0) k1 /= c1;
		if (c2 > 0) k2 /= c2;
		return k2 != 0 ? k1 / k2 : 1;
	}
	
	
	/**
	 * 支持一阶指数平滑法
	 * @param features
	 * @param param
	 * @return
	 */
	private static Map<String, Double> recentMeanDoubleUsedExponent(List<Feature> features, Param param, int latest, boolean eval){
		Map<String, Integer> flavor_map = getFlavorMap();
		int[] tot = new int[30];
		
		float[][] sequence = FeatureExtract.getSequence(features, flavor_map, tot);
		float[] alpha = { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1 };
		float[][][] exps = new float[alpha.length][20][2400];

		for (int i = 0; i < alpha.length; ++i) {
			exps[i] = getExpoSequence(sequence, tot, alpha[i]);
		}
		
		Map<String, Double> ans = new HashMap<>();
		for (String key : param.virtuals.keySet()) {
			int idx = flavor_map.get(key);
			
			int best_ = alpha.length - 1;
			if (tot[idx] >= 2 * 7 && eval) {
				
				// 支持 超惨搜索功能
				double minRmse = 0x3f3f3f3f;
				for (int j = 0; j < alpha.length; ++j) {
					double score = bestAlpha(exps[j][idx], sequence[idx], tot[idx], 7);
					if (minRmse > score) {
						minRmse = score;
						best_ = j;
					}
				}
//				System.out.println(key + "支持超参搜索，最佳参数为: " + alpha[best_] + ", 最高分数为： " + minRmse);
			}
			
			float[][] exp_sequence = exps[best_];
			for (int i = tot[idx] - 1, cnt_ = 0; i >= 0 && cnt_ < latest; --i, ++cnt_) {
				ans.put(key, ans.getOrDefault(key, 0.0) + exp_sequence[idx][i]);
			}
		}
		for (String key : ans.keySet()) {
			ans.put(key, ans.get(key) / latest);
		}
		return ans;
	}
	
	public static List<Feature> twoExpoSequence(List<Feature> features, Param param, double first_alpha, double second_alpha){
		Map<String, Integer> flavor_map = getFlavorMap();
		int[] tot = new int[30];
		
		// 一级指数平滑 平滑数据
		float[][] sequence = FeatureExtract.getSequence(features, flavor_map, tot);
		float[][] exp_sequence = getExpoSequence(sequence, tot, first_alpha);
		
		// 二级指数平滑 预测数据
		Map<String, float[]> deq = new HashMap<>();
		for (String type : flavor_map.keySet()) {
			int idx = flavor_map.get(type);
			int len = tot[idx];
			deq.put(type, new float[] {sequence[idx][len - 1], exp_sequence[idx][len - 2]}); // yt , s_{t - 1}
		}
		
		long sday = DateUtil.year2dayLong(param.startTime);
		int span = param.span;
		
		List<Feature> ret = new ArrayList<>();
		for (int i = 0; i < span; ++i) {
			for (String type : flavor_map.keySet()) {
				float[] bucket = deq.get(type);
				float y = bucket[0]; float s = bucket[1];
				float p = (float) (y * second_alpha + s * (1 - second_alpha));
				
				Feature feature = new Feature();
				feature.count = (int) p;
				feature.year2Day = sday + i;
				feature.tag = type;
				feature.date = DateUtil.long2Date((feature.year2Day + 1) * 1000 * 24 * 3600);
				feature.isHoliday = param.longHolidaySet.contains(feature.year2Day);
				feature.dayOfWeek = DateUtil.week(feature.date);
				ret.add(feature);
				
				//update
				int idx = flavor_map.get(type);
				int len = tot[idx];
				sequence[idx][len] = feature.count;
				exp_sequence[idx][len] = (float) (first_alpha * sequence[idx][len] + (float)(1.2f - first_alpha) * exp_sequence[idx][len - 1]);
				len ++;
				deq.put(type, new float[] {sequence[idx][len - 1], exp_sequence[idx][len - 2]});
			}
		}
		
		return ret;
	}
	
	public static List<Feature> slidingWindow(List<Feature> features, Param param, int window_size, double alpha){
		Map<String, Integer> flavor_map = getFlavorMap();
		int[] tot = new int[30];
		
		// 一级指数平滑 平滑数据
		float[][] sequence = FeatureExtract.getSequence(features, flavor_map, tot);
		float[][] exp_sequence = getExpoSequence(sequence, tot, alpha);
		
		
		
		Map<String, Float> deq_mem = new HashMap<>();
		for (String type : flavor_map.keySet()) {
			int idx = flavor_map.get(type);
			int len = tot[idx];
			for (int i = len - window_size; i < len; ++i) {
				deq_mem.put(type, deq_mem.getOrDefault(type, 0.0f) + (i >= 0 ? exp_sequence[idx][i] : 0));
			}
		}
		
		long sday = DateUtil.year2dayLong(param.startTime);
		int span = param.span;
		
		List<Feature> ret = new ArrayList<>();
		for (int i = 0; i < span; ++i) {
			for (String type : flavor_map.keySet()) {
				float predict = deq_mem.get(type) / window_size;
				Feature feature = new Feature();
				feature.count = (int) predict;
				feature.year2Day = sday + i;
				feature.tag = type;
				feature.date = DateUtil.long2Date((feature.year2Day + 1) * 1000 * 24 * 3600);
				feature.isHoliday = param.longHolidaySet.contains(feature.year2Day);
				feature.dayOfWeek = DateUtil.week(feature.date);
				ret.add(feature);
				
				// update sequence
				int idx = flavor_map.get(type);
				deq_mem.put(type, deq_mem.get(type) - (tot[idx] >= window_size ? exp_sequence[idx][tot[idx] - window_size] : 0));
				exp_sequence[idx][tot[idx]] = predict;
				tot[idx]++;
				deq_mem.put(type, deq_mem.get(type) + predict);
			}
		}
		return ret;
	}
	
	static class Pair{
		String type;
		int val;
		
		Pair(String type, int val){
			this.type = type;
			this.val = val;
		}
	}
	
	public static Map<String, Integer> add(Map<String, Integer> model, int week){
		for (String flavor : model.keySet()) {
			int idx = Integer.parseInt(flavor.substring(6));
			if (idx >= 1 && idx <= 2 || idx >= 3 && idx <= 9) {
				model.put(flavor, model.get(flavor) + 15 * week);
			}
		}
		return model;
	}
	
	public static Map<String, Integer> trick(Map<String, Integer> model){
		int size = model.size();
		List<Pair> ps = new ArrayList<>();
		for (String type : model.keySet()) {
			ps.add(new Pair(type, model.get(type)));
		}
		boolean[] vis = new boolean[size];
		
		for (int i = 0; i < ps.size(); ++i) {
			for (int j = i + 1; j < ps.size(); ++j) {
				if (vis[i] || vis[j]) continue;
				if (Math.abs(ps.get(i).val - ps.get(j).val) == 2) { // ps.get(i).val + 4 == ps.get(j).val
					int all = ps.get(i).val; all += ps.get(j).val;
					model.put(ps.get(i).type, all / 2);
					model.put(ps.get(j).type, all / 2);
					vis[i] = true;
					vis[j] = true;
					break;
				}
			}
		}
		return model;
	}
	
	public static Map<String, Integer> trick_reduce369121518(Map<String, Integer> model){
		for (String type : model.keySet()) {
			int idx = Integer.parseInt(type.substring(6));
			if (idx % 3 == 0) {
				model.put(type, Math.max(0, model.get(type) - 1));
			}
		}
		return model;
	}
	
	public static Map<String, Integer> fixedModel(List<Feature> features, Param param, int latest, float alpha, int gap){
		Map<String, Float> recent_map = recentMeanDoubleUsedExponent(features, param, latest, alpha);
		Map<String, Integer> ret = new HashMap<>();
		for (String type : recent_map.keySet()) {
			double a = 1;
			if (gap == 0) {
				a = 0.81;
			}
			else {
				a = 1.51;
			}
			ret.put(type, (int) Math.floor(recent_map.get(type) * param.span * a));
		}
		return ret;
	}
	
	/**
	 * 
	 * @param features
	 * @param param
	 * @param latest
	 * @param alpha
	 * @param fixed
	 * @return
	 */
	public static Map<String, Double> findRatio(List<Feature> features, Param param, double alpha){
		Map<String, Integer> flavor_map = getFlavorMap();
		int[] tot = new int[30];
		
		float[][] sequence = FeatureExtract.getSequence(features, flavor_map, tot);
		
		Map<String, List<Double>> week_val = new HashMap<>();
		for (String flavor : flavor_map.keySet()) {
			int idx = flavor_map.get(flavor);
			double sum = 0.0;
			for (int i = tot[idx] - 1, c = 1; i >= 0; --i, ++c) {
				if (c % 10 == 0) {
					week_val.computeIfAbsent(flavor, k -> new ArrayList<>()).add(sum);
					sum = 0.0;
				}
				else {
					sum += sequence[idx][i];
				}
			}
		}
		
		return null;
	}
	
	private static Map<String, Integer> recentDecreseMean(List<Feature> features, Param param, int latest, double alpha, double p1, double p2, double fixed, int day){
		Map<String, Float> recent_map = recentMeanDecreseDoubleUsedExponent(features, param, latest, alpha, p1, p2);
		Map<String, Integer> ret = new HashMap<>();
		for (String key : recent_map.keySet()) {
			int y_ = (int) Math.floor(recent_map.get(key) * day * fixed);
			ret.put(key, y_);
		}
		return ret;
	}
	
	
	public static Map<String, Integer> recentDecreseMean(List<Feature> features, Param param, int latest, double alpha, double p1, double p2, double fixed){
		Map<String, Float> recent_map = recentMeanDecreseDoubleUsedExponent(features, param, latest, alpha, p1, p2);
		Map<String, Integer> ret = new HashMap<>();
		for (String key : recent_map.keySet()) {
			int y_ = (int) Math.floor(recent_map.get(key) * param.span * fixed);
			ret.put(key, y_);
		}
		return ret;
	}
	
	public static Map<String, Integer> allMean(List<Feature> features, Param param, double alpha, double fixed){
		Map<String, Float> recent_map = allMeanDoubleUsedExponent(features, param, alpha);
		Map<String, Integer> ret = new HashMap<>();
		for (String key : recent_map.keySet()) {
			int y_ = (int) Math.floor(recent_map.get(key) * param.span * fixed);
			ret.put(key, y_);
		}
		return ret;
	}
	
	
	/**
	 * 支持 指数平滑法,没有自动搜参
	 * @param features
	 * @param param
	 * @param latest
	 * @return
	 */
	public static Map<String, Integer> recentMean(List<Feature> features, Param param, int latest, double alpha){
		Map<String, Float> recent_map = recentMeanDoubleUsedExponent(features, param, latest, alpha);
		Map<String, Integer> ret = new HashMap<>();
		for (String key : recent_map.keySet()) {
			int y_ = (int) Math.floor(recent_map.get(key) * param.span);
			ret.put(key, y_);
		}
		return ret;
	}
	
	/**
	 * 支持 指数平滑法
	 * @param features
	 * @param param
	 * @param latest
	 * @return
	 */
	public static Map<String, Integer> recentMean(List<Feature> features, Param param, int latest, boolean eval){
		Map<String, Double> recent_map = recentMeanDoubleUsedExponent(features, param, latest, eval);
		Map<String, Integer> ret = new HashMap<>();
		for (String key : recent_map.keySet()) {
			int y_ = (int) Math.floor(recent_map.get(key) * param.span);
			ret.put(key, y_);
		}
		return ret;
	}
	
	
	
	
	@Deprecated
	public static Map<String, Double>[] meanTrain(List<Record> dataset, Param param){
		List<Feature> features = FeatureExtract.groupby(dataset, param, false, 0);
		Map<String, Integer>[] counter = new HashMap[8];
		for (int i = 0; i < 8; ++i) counter[i] = new HashMap<>();
		int[] week = new int[8];
		for (Feature f : features) {
			week[f.dayOfWeek] ++;
			counter[f.dayOfWeek].put(f.tag, counter[f.dayOfWeek].getOrDefault(f.tag, 0) + f.count);
		}
		Map<String, Double>[] ans = new HashMap[8];
		for (int i = 0; i < 8; ++i) ans[i] = new HashMap<>();
		
		for (int i = 0; i < 8; ++i) {
			for (String key : counter[i].keySet()) {
				ans[i].put(key, counter[i].get(key) * 1.0 / week[i]);
			}
		}
		return ans;
	}
	
	public static Map<String, Integer> doubleMap2IntMapUseCeil(Map<String, Double> map){
		Map<String, Integer> ret = new HashMap<>();
		for (String key : map.keySet()) ret.put(key, (int)Math.ceil(map.get(key)));
		return ret;
	}
	
	public static Map<String, Integer> transfer(Map<String, Double> counter, Param param){
		Map<String, Integer> ans = new HashMap<>();
		int span = param.span;
		for (String key : counter.keySet()) {
			ans.put(key, (int) Math.ceil((counter.get(key) * span)));
		}
		return ans;
	}
	
	public static Map<String, Integer> transfer(Map<String, Double>[] counter, Param param){
		try {
			long start_time = DateUtil.year2dayLong(param.startTime, new SimpleDateFormat("yyyy-MM-dd HH:mm:ss"));
			long end_time = DateUtil.year2dayLong(param.endTime, new SimpleDateFormat("yyyy-MM-dd HH:mm:ss"));
			Map<String, Double> ans = new HashMap<>();
			for (long i = start_time; i <= end_time; ++i) {
				int week = DateUtil.long2week(i * 1000 * 24 * 3600);
				for (String key : counter[week].keySet()) {
					Double _cnt = counter[week].get(key);
					ans.put(key, ans.getOrDefault(key, 0.0) + _cnt);
				}
			}
			Map<String, Integer> nxt = new HashMap<>();
			for (String key : ans.keySet()) {
				nxt.put(key, (int)Math.ceil(ans.get(key)));
			}
			return nxt;
		} catch (ParseException e) {
			System.err.println("解析失败，请检查date的格式是否为\"yyyy-MM-dd HH:mm:ss\"");
		}
		return null;
	}
	
	
	
	/**
	 * do statics with all kinds of flavor
	 * @param dataset 训练数据集
	 * @return <flavor, ratio>
	 */
	public static Map<String, Integer> count2Int(List<Record> dataset){
		Map<String, Integer> cnt_map = new HashMap<>();
		for (Record instance : dataset) {
			cnt_map.put(instance.tag, cnt_map.getOrDefault(instance.tag, 0) + 1);
		}
		return cnt_map;
	}
	
	public static Map<String, Integer> list2Map(List<Feature> features){
		Map<String, Integer> cnt_map = new HashMap<>();
		for (Feature instance : features) {
			cnt_map.put(instance.tag, cnt_map.getOrDefault(instance.tag, 0) + instance.count);
		}
		return cnt_map;
	}
	
	public static Map<String, Double> loadLocalInfo(){
		String[] tags = {"flavor1","flavor10", "flavor11", "flavor12", "flavor13", "flavor14", "flavor15", "flavor17", "flavor18",
						 "flavor2","flavor21","flavor22","flavor23","flavor3","flavor4","flavor5","flavor6","flavor7","flavor8","flavor9"};
		double[] ratios = {0.06593406593406594,0.04395604395604396,0.24615384615384617,0.14725274725274726,0.06813186813186813,0.3054945054945055,0.09230769230769231,0.03076923076923077,
				0.008791208791208791,0.2021978021978022,0.004395604395604396,0.02197802197802198,0.004395604395604396,0.08571428571428572,0.03956043956043956,
				0.4747252747252747,0.2175824175824176,0.09010989010989011,0.7978021978021979,0.26373626373626374};
		
		Map<String, Double> ratio_map = new HashMap<>();
		for (int i = 0; i < tags.length; ++i) {
			ratio_map.put(tags[i], ratios[i]);
		}
		return ratio_map;
	}
	
	public static Map<String, Double> train(List<Record> dataset, boolean useHistory){
		if (useHistory) {
			return loadLocalInfo();
		}
		else {
			return count2Ratio(dataset);
		}
	}
	
	public static Map<String, Integer> poly_fixed(Map<String, Integer> model, double alpha){
		 for (String type : model.keySet()) model.put(type, (int) (model.get(type) * alpha));
		 return model;
	}
	
	public static Map<String, Integer> trainMock(int seed){
		Map<String, Integer> cnt_map = new HashMap<>();
		for (int i = 1; i <= 18; ++i) {
//			int[] a = {0, 0, 0, 255, 0, 213, 0, 0, 248, 0, 0, 232, 76, 281, 210, 0, 235, 93};
//			int[] a = {0, 181, 0, 142, 0, 0, 99, 0, 0, 0, 48, 53, 198, 0, 0, 216, 130, 0};
//			cnt_map.put("flavor" + i, a[i - 1]);
//			cnt_map.put("flavor" + i, 100); //268
			cnt_map.put("flavor" + i, new Random(seed + i).nextInt(100));
		}
		return cnt_map;
	}
	
	
	
	
	public static void main(String[] args) {
		String[] trainContent = FileUtil.read(trainfile, null);
		List<Record> train = LoadData.loadData(trainContent);
		Train.count2Ratio(train);
	}
	
}
