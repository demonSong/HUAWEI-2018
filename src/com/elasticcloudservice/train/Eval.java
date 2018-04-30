package com.elasticcloudservice.train;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.date.utils.DateUtil;
import com.elasticcloudservice.predict.CPU;
import com.elasticcloudservice.predict.Feature;
import com.elasticcloudservice.predict.LoadData;
import com.elasticcloudservice.predict.Manifest;
import com.elasticcloudservice.predict.Param;
import com.elasticcloudservice.predict.Predict;
import com.elasticcloudservice.predict.Record;
import com.filetool.util.FileUtil;

public class Eval {
	
	public static final int MEAN_MODEL_4 = 0;  // 简单的均值模型，需要超参搜索
	public static final int MEAN_MODEL_5 = 1;
	public static final int MEAN_MODEL_6 = 2;
	public static final int ARIMA_MEAN_AVG_4 = 3;
	public static final int ARIMA_MEAN_AVG_5 = 4;
	public static final int ARIMA_MEAN_AVG_6 = 5;
	
	
	
	public static class Pair{
		public Param param;
		public List<Record> train;
		public List<Record> valid;
		
		public Pair(Param param, List<Record> train, List<Record> valid) {
			this.param = param;
			this.train = train;
			this.valid = valid;
		}
	}
	
	public static Map<String, Integer> average(Map<String, Integer> a, Map<String, Integer> b, Param param){
		Map<String, Integer> model = new HashMap<>();
		
		for (String key : param.virtuals.keySet()) {
			double a_ = a.containsKey(key) ? a.get(key) : 0;
			double b_ = b.containsKey(key) ? b.get(key) : 0;
			
			int sum = (int) Math.ceil((a_ * 1 / 2 + b_ * 1 / 2));
			model.put(key, sum >= 0 ? sum : 0);
		}
		return model;
	}
	
	public static int bestModel(Param param, List<Record> train, List<Record> valid) {
		// init
		Map<String, Integer> y = Train.count2Int(valid);
		List<Feature> features = FeatureExtract.groupby(train, param, true, 0);
		
		// mean model
		Map<String, Integer> mean_4 = Train.recentMean(features, param, 4, true);
		Map<String, Integer> mean_5 = Train.recentMean(features, param, 5, true);
		Map<String, Integer> mean_6 = Train.recentMean(features, param, 6, true);
		
		// arima model
		Map<String, Integer> arima = Train.arima(features, train, param, Predict.flavor_param(5), 1);
		
		double maxRmse = Double.MIN_VALUE;
		int MODEL = -1;
		
		// mean model best
		double score = rmse(y, mean_4);
		if (maxRmse < score) {
			maxRmse = score;
			MODEL = MEAN_MODEL_4;
		}
		
		score = rmse(y, mean_5);
		if (maxRmse < score) {
			maxRmse = score;
			MODEL = MEAN_MODEL_5;
		}
		
		score = rmse(y, mean_6);
		if (maxRmse < score) {
			maxRmse = score;
			MODEL = MEAN_MODEL_6;
		}
		
		// arima + mean_4 avg
		score = rmse(y, average(mean_4, arima, param));
		if (maxRmse < score) {
			maxRmse = score;
			MODEL = ARIMA_MEAN_AVG_4;
		}
		
		score = rmse(y, average(mean_5, arima, param));
		if (maxRmse < score) {
			maxRmse = score;
			MODEL = ARIMA_MEAN_AVG_5;
		}
		
		score = rmse(y, average(mean_6, arima, param));
		if (maxRmse < score) {
			maxRmse = score;
			MODEL = ARIMA_MEAN_AVG_6;
		}
		
		return MODEL;
	}
	
	@Deprecated
	public static Pair transfer(String[] inputContent, String[] ecsContent) {
		String startDay = LoadData.loadData(ecsContent[0]).date;
		String endDay = LoadData.loadData(ecsContent[ecsContent.length - 1]).date;
		long e_ = DateUtil.year2dayLong(endDay);
		long s_ = e_ - 6;
		int len = (int) (DateUtil.year2dayLong(endDay) - DateUtil.year2dayLong(startDay));
		if (len < 14) return null; // do not support eval
		Param param = Predict.inputParam(inputContent);
		param.startTime = DateUtil.long2Date((s_ + 1) * 24 * 3600 * 1000);
		param.endTime   = DateUtil.long2Date((e_ + 1) * 24 * 3600 * 1000);
		
		List<Record> train = new ArrayList<>();
		List<Record> valid = new ArrayList<>();
		
		for (String content : ecsContent) {
			Record rec = LoadData.loadData(content);
			if (rec.year2Day < s_) train.add(rec);
			else valid.add(rec);
		}
		return new Pair(param, train, valid);
	}
	
	public static double rmse(Map<String, Integer> y, Map<String, Integer> y_) {
		int N = y_.size();
		
		double diff   = 0;
		double sum_y  = 0;
		double sum_y_ = 0;
		for (String key : y_.keySet()) {
			int p  = y.containsKey(key) ? y.get(key) : 0;
			int p_ = y_.get(key);
			diff   += (p - p_) * (p - p_);
			sum_y  += p * p;
			sum_y_ += p_ * p_;
		}
		
		double rmse = Math.sqrt(diff / N) / (Math.sqrt(sum_y / N) + Math.sqrt(sum_y_ / N));
		return 1 - rmse;
	}
	
	public static double usedRatio(List<Manifest> menu, Map<String, CPU> map, String target, Map<String, CPU> virtuals) {
		System.out.println("---------------------------------" + "ALL-INFO" + "---------------------------------");
		double ratio = 0.0;
		{
			double sum_core = 0;
			double used = 0;
			
			for (Manifest list : menu) {
				sum_core += map.get(list.type).core;
				
				Map<String, Integer> flavors = list.getList();
				for (String key : flavors.keySet()) {
					int cnt = flavors.get(key);
					used += cnt * virtuals.get(key).core;
				}
			}
			ratio += used / sum_core;
			System.out.println("CPU ratio: " + ratio);
		}
		{
			double sum_memo = 0;
			double used = 0;
			
			for (Manifest list : menu) {
				sum_memo += map.get(list.type).memory * 1024;
				
				Map<String, Integer> flavors = list.getList();
				for (String key : flavors.keySet()) {
					int cnt = flavors.get(key);
					used += cnt * virtuals.get(key).memory;
				}
			}
			ratio += used / sum_memo;
			System.out.println("MEM ratio: " + used / sum_memo);
		}

		System.out.println("ALL ratio: " + ratio / 2);
		return ratio / 2;
	}
	
	@Deprecated
	public static double usedRatio(List<Manifest> menu, CPU phys, String target, Map<String, CPU> virtuals) {
		double ratio = 0.0;
		if (target.equals("CPU")) {
			double sum_core = 0;
			double used = 0;
			
			for (Manifest list : menu) {
				sum_core += phys.core;
				
				Map<String, Integer> flavors = list.getList();
				for (String key : flavors.keySet()) {
					int cnt = flavors.get(key);
					used += cnt * virtuals.get(key).core;
				}
			}
			ratio = used / sum_core;
		}
		else {
			double sum_memo = 0;
			double used = 0;
			
			for (Manifest list : menu) {
				sum_memo += phys.memory * 1024;
				
				Map<String, Integer> flavors = list.getList();
				for (String key : flavors.keySet()) {
					int cnt = flavors.get(key);
					used += cnt * virtuals.get(key).memory;
				}
			}
			ratio = used / sum_memo;
		}
		return ratio;
	}
	
	public static double eachPhysScore(String tag, Param params, String type, Map<String, Integer> flavorCnt) {
//		CPU cpu = params.physicalCpusMap.get(type);
//		System.out.println("---------------------------------" + tag + "---------------------------------");
//		double sum_core = 0.0;
//		double sum_memo = 0.0;
//		for (String key : flavorCnt.keySet()) {
//			CPU flavor = params.virtuals.get(key);
//			sum_core += flavorCnt.get(key) * flavor.core;
//			sum_memo += flavorCnt.get(key) * flavor.memory;
//		}
//		
//		double ratio = 0.0;
//		ratio += sum_core / cpu.core;
//		ratio += sum_memo / (cpu.memory * 1024);
//		
//		System.out.println("CPU ratio: " + sum_core / cpu.core);
//		System.out.println("MEM ratio: " + sum_memo / (cpu.memory * 1024));
//		System.out.println("ALL ratio: " + ratio / 2);
//		return ratio / 2;
		return 0;
	}
	
	public static boolean allFlavorDone(Map<String, Integer> f, Map<String, Integer> m) {
		for (String t : f.keySet()) {
			if (m.containsKey(t)) {
				if (Integer.compare(f.get(t), m.get(t)) != 0) {
					return false;
				}
			}
			else {
				if (f.get(t) != 0) return false;
			}
		}
		return true;
	}
	
	public static double score(String testfile, String paramfile, String outfile) {
		//1. 加载真实测试集
		String[] testContent = FileUtil.read(testfile, null);
		List<Record> test = LoadData.loadData(testContent);
		Map<String, Integer> test_count = Train.count2Int(test);
		
		//2. 加载物理信息
		String[] inputParams = FileUtil.read(paramfile, null);
		Param params = Predict.inputMoreParam(inputParams);
		
		//3. 加载预测结果
		String[] output = FileUtil.read(outfile, null);
		
		// 1. 解析 flavor 及 个数
		Map<String, Integer> flavors = new HashMap<>();
		
		int _index = 1;
		while(!output[_index].equals("")) {
			String[] l = output[_index].split("\\s+");
			flavors.put(l[0], Integer.parseInt(l[1]));
			_index++;
		}
		
		_index++;
		
		// eval
		Map<String, Integer> evalFlavors = new HashMap<>();
		
		// 2. 解析 General
		int numOfPhy = Integer.parseInt(output[_index++].split(" ")[1]);
		List<Manifest> menu = new ArrayList<>();
		for (int i = 0; i < numOfPhy; ++i, _index++) {
			String[] l = output[_index].split("\\s+");
			int lst_ = l[0].lastIndexOf("-");
			Manifest v = new Manifest(l[0].substring(lst_), l[0].substring(0, lst_));
			Map<String, Integer> flavorCnt = new HashMap<>();
			for (int j = 1; j < l.length; j += 2) {
				v.add(l[j], Integer.parseInt(l[j + 1]));
				flavorCnt.put(l[j], flavorCnt.getOrDefault(l[j], 0) + Integer.parseInt(l[j + 1]));
				evalFlavors.put(l[j], evalFlavors.getOrDefault(l[j], 0) + Integer.parseInt(l[j + 1]));
			}
			menu.add(v);
			eachPhysScore(l[0], params, l[0].substring(0, lst_), flavorCnt);
		}
		
		//3. 解析 LARGE_MEMORY
		_index ++;
		numOfPhy = Integer.parseInt(output[_index++].split(" ")[1]);
		for (int i = 0; i < numOfPhy; ++i, _index++) {
			String[] l = output[_index].split("\\s+");
			int lst_ = l[0].lastIndexOf("-");
			Manifest v = new Manifest(l[0].substring(lst_), l[0].substring(0, lst_));
			Map<String, Integer> flavorCnt = new HashMap<>();
			for (int j = 1; j < l.length; j += 2) {
				v.add(l[j], Integer.parseInt(l[j + 1]));
				flavorCnt.put(l[j], flavorCnt.getOrDefault(l[j], 0) + Integer.parseInt(l[j + 1]));
				evalFlavors.put(l[j], evalFlavors.getOrDefault(l[j], 0) + Integer.parseInt(l[j + 1]));
			}
			menu.add(v);
			eachPhysScore(l[0], params, l[0].substring(0, lst_), flavorCnt);
		}
		
		//4. 解析 HIGH_PERFORMANCE
		_index ++;
		numOfPhy = Integer.parseInt(output[_index++].split(" ")[1]);
		for (int i = 0; i < numOfPhy; ++i, _index++) {
			String[] l = output[_index].split("\\s+");
			int lst_ = l[0].lastIndexOf("-");
			Manifest v = new Manifest(l[0].substring(lst_), l[0].substring(0, lst_));
			Map<String, Integer> flavorCnt = new HashMap<>();
			for (int j = 1; j < l.length; j += 2) {
				v.add(l[j], Integer.parseInt(l[j + 1]));
				flavorCnt.put(l[j], flavorCnt.getOrDefault(l[j], 0) + Integer.parseInt(l[j + 1]));
				evalFlavors.put(l[j], evalFlavors.getOrDefault(l[j], 0) + Integer.parseInt(l[j + 1]));
			}
			menu.add(v);
			eachPhysScore(l[0], params, l[0].substring(0, lst_), flavorCnt);
		}
		
		double acc = rmse(test_count, flavors);
		double ratio = usedRatio(menu, params.physicalCpusMap, params.target, params.virtuals);
		
		System.out.printf("Accuracy: %.5f \n", acc);
		System.out.printf("Ratio:  %.5f \n", ratio);
		System.out.printf("Score: %.3f \n", acc * ratio * 100);
		
		System.out.println(allFlavorDone(flavors, evalFlavors) ? "all flavor are allocated!" : "not all are allocated!");
		return ratio;
	}
	
	public static void main(String[] args) {
		
	}
}
