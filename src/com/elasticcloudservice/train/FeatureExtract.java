package com.elasticcloudservice.train;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.date.utils.DateUtil;
import com.elasticcloudservice.ml.Instance;
import com.elasticcloudservice.ml.LimitQueue;
import com.elasticcloudservice.predict.CPU;
import com.elasticcloudservice.predict.Feature;
import com.elasticcloudservice.predict.Param;
import com.elasticcloudservice.predict.Record;

public abstract class FeatureExtract {
	
	public static double sigmoid(double x) {
		return 1 / (1 + Math.exp(-x));
	}
	
	public static double sigmoid_inverse(double x) {
		return 1 / (1 + Math.exp(-x));
	}
	
	public static List<Feature> meanFilter(List<Feature> features, int filterSize){
		List<Feature> ret = new ArrayList<>();
		Map<String, Integer> flavor_map = Train.getFlavorMap();
		
		int[] tot = new int[30];
		float[][] sequenceSum = FeatureExtract.getSequenceSum(features, flavor_map, tot);
		
		long s = features.get(0).year2Day;
		for (Feature f : features) {
			if (flavor_map.containsKey(f.tag)) {
				int idx = flavor_map.get(f.tag);
				if (f.year2Day - s - filterSize >= 0) {
					float mean = querySumRange(sequenceSum, idx, (int)(f.year2Day - s - filterSize), (int)(f.year2Day - s)) / (filterSize + 1);
					f.count = (int) mean;
				}
			}
			ret.add(f);
		}
		return ret;
	}
	
	public static List<Feature> denoise(List<Feature> features){
		List<Feature> ret = new ArrayList<>();
		
		Map<String, Integer> cnt = new HashMap<>();
		Map<String, Integer> sum = new HashMap<>();
		Map<String, Double> mean = new HashMap<>();
		Map<String, Double> variance = new HashMap<>();
		Map<String, Double> stddev = new HashMap<>();
		
		for (Feature f : features) {
			sum.put(f.tag, sum.getOrDefault(f.tag, 0) + f.count);
			cnt.put(f.tag, cnt.getOrDefault(f.tag, 0) + 1);
		}
		
		for (String tag : sum.keySet()) {
			mean.put(tag, sum.get(tag) * 1.0 / cnt.get(tag));
		}
		
		for (Feature f : features) {
			variance.put(f.tag, variance.getOrDefault(f.tag, 0.0) + Math.pow((mean.get(f.tag) - f.count), 2));
		}
		
		for (String tag : variance.keySet()) {
			variance.put(tag, variance.get(tag) / cnt.get(tag));
			stddev.put(tag, Math.sqrt(variance.get(tag)));
		}
		
		for (Feature f : features) {
			if (Math.abs(f.count - mean.get(f.tag)) > 5 * stddev.get(f.tag)) { // 异常
				f.count = (int) Math.floor(stddev.get(f.tag) * 5);
				ret.add(f);
			}
			else {
				ret.add(f);
			}
		}
		
		return ret;
	}
	
	/**
	 * 根据year2Day分组
	 * @param dataset
	 * @return
	 */
	public static List<Feature> groupby(List<Record> dataset, Param param, boolean fullfill, int limitLen){
		long index = -1;
		int _dayOfWeek = -1;
		String date = "";
		
		Map<String, Integer> counter = new HashMap<>();
		List<Feature> data = new ArrayList<>();
		
		Set<String> vis = new HashSet<>();
		Map<String, Integer> mem = new HashMap<>();
		Set<Long> seen = new HashSet<>();
		
		for (Record rec : dataset) {
			vis.add(rec.year2Day + "," + rec.tag);
			if (index == -1 || rec.year2Day == index) {
				counter.put(rec.tag, counter.getOrDefault(rec.tag, 0) + 1);
			}
			else {
				for (String key : counter.keySet()) {
					Feature f = new Feature();
					f.date = date;
					f.year2Day = index;
					f.dayOfWeek = _dayOfWeek;
					f.tag = key;
					f.count = counter.get(key);
					f.isHoliday = param.longHolidaySet.contains(f.year2Day);
					
					seen.add(f.year2Day);
					mem.put(f.year2Day + "," + f.tag, f.count);
					data.add(f);
				}
				counter = new HashMap<>();
				counter.put(rec.tag, counter.getOrDefault(rec.tag, 0) + 1);
			}
			date = rec.date;
			_dayOfWeek = rec.dayOfWeek;
			index = rec.year2Day;
		}
		
		for (String key : counter.keySet()) {
			Feature f = new Feature();
			f.date = date;
			f.year2Day = index;
			f.dayOfWeek = _dayOfWeek;
			f.tag = key;
			f.count = counter.get(key);
			f.isHoliday = param.longHolidaySet.contains(f.year2Day);
			seen.add(f.year2Day);
			mem.put(f.year2Day + "," + f.tag, f.count);
			data.add(f);
		}
		
		Map<String, LimitQueue> lm_map = new HashMap<>();
		boolean append = limitLen != 0;
		
		if (fullfill) {
			long s_ = data.get(0).year2Day;
			long e_ = data.get(data.size() - 1).year2Day;
			for (long day = s_; day <= e_; ++day) {
				for (String key : param.virtuals.keySet()) {
					if (append) {
						if (!lm_map.containsKey(key)) lm_map.put(key, new LimitQueue(limitLen));
						if (mem.containsKey(day + "," + key)) lm_map.get(key).offer(mem.get(day + "," + key));
						else if (seen.contains(day)) lm_map.get(key).offer(0);
					}
					if (!vis.contains(day + "," + key)) {
						Feature f = new Feature();
						f.date = DateUtil.long2Date((day + 1) * 1000 * 24 * 3600);
						f.year2Day = day;
						f.dayOfWeek = DateUtil.week(f.date);
						f.tag = key;
						f.count = append && !seen.contains(day) ? (int)lm_map.get(key).getMean() : 0;
						f.isHoliday = param.longHolidaySet.contains(f.year2Day);
						data.add(f);
						if (append && !seen.contains(day)) lm_map.get(key).offer(f.count);
					}
				}
			}
			Collections.sort(data, (a, b) -> ((int)(a.year2Day - b.year2Day)));
		}
		
		return data;
	}
	
	/**
	 * 得到 每个sequences 的 roll sequences （平滑法）
	 * @param features
	 * @param flavor_map
	 * @param roll_parma
	 * @return
	 */
	public static float[][] rollSequence(List<Feature> features, Map<String, Integer> brand_code, 
			Map<String, Integer> flavor_map, float[][] sums, int[] tot, int[] roll_param){ // 1表示不进行平滑处理
		float[][] rolling = new float[20][2400];
		for (String key : brand_code.keySet()) {
			int idx = brand_code.get(key);
			int param = flavor_map.containsKey(key) ? roll_param[flavor_map.get(key)] - 1 : 0;
			for (int j = param; j < tot[idx]; ++j) {
				rolling[idx][j] = querySumRange(sums, idx, j - param, j) / (param + 1);
			}
		}
		return rolling;
	}
	
	/**
	 * 得到每个flavor sequences 序列
	 * @param features
	 * @return
	 */
	public static float[][] getSequence(List<Feature> features, Map<String, Integer> brand_code, int[] tot){
		float[][] sequence = new float[20][2400];
		for (Feature feature : features) {
			String tag = feature.tag;
			if (!brand_code.containsKey(tag)) continue;
			sequence[brand_code.get(tag)][tot[brand_code.get(tag)]] = feature.count;
			tot[brand_code.get(tag)] ++;
		}
		return sequence;
	}
	
	/**
	 * 构造累加和数组，方便查询
	 * @param features
	 * @param flavor_map
	 * @return
	 */
	public static float[][] getSequenceSum(List<Feature> features, Map<String, Integer> brand_code, int[] tot){
		float[][] sequence = getSequence(features, brand_code, tot);
		float[][] sum = new float[20][2400];
		
		for (String key : brand_code.keySet()) {
			int idx = brand_code.get(key);
			for (int i = 0; i < tot[idx]; ++i) {
				sum[idx][i + 1] = sum[idx][i] + sequence[idx][i];
			}
		}
		return sum;
	}
	
	/**
	 * 查询区间[i, j]中的和
	 * @param sum
	 * @param idx
	 * @param i
	 * @param j
	 * @return
	 */
	public static float querySumRange(float[][] sum, int idx, int i, int j) {
		if (i > j) return 0;
		return sum[idx][j + 1] - sum[idx][i];
	}
	
	
	public static Instance next(Param param, Map<String, CPU> virtuals, String tag, long ss_, long day, Map<String, Integer> brand_code, float[][] sequence) {
		String date = DateUtil.long2Date(day * 1000 * 24 * 3600);
		long year2Day = day - 1;
		int weekday = DateUtil.week(date);
		
		int dummy_len = brand_code.size() + 12;
		float[] features = new float[dummy_len + 40];
		
		//1. brand
		features[brand_code.get(tag)] = 1.0f;
		
		//dummy month
		features[brand_code.size() + DateUtil.month(date) - 1] = 1.0f;
		
		//2. day of week
		features[dummy_len] = weekday;
		
		//3. day of month
		features[dummy_len + 1] = DateUtil.month(date);
		
		//4. 是否为工作日
		features[dummy_len + 2] = (DateUtil.week(date) == 1 || DateUtil.week(date) == 7) ? 1 : 0;
		
		//5. 一个月中的第几天
		features[dummy_len + 3] = DateUtil.dayOfMonth(date);
		
		//6. 一个月中的第几周
		features[dummy_len + 4] = DateUtil.weekOfMonth(date);
		
		//7. 前一天是否为假期
		features[dummy_len + 5] = isHolidayWithGap(weekday, -1) ? 1 : 0;
		features[dummy_len + 6] = isHolidayWithGap(weekday, -2) ? 1 : 0;
		features[dummy_len + 7] = isHolidayWithGap(weekday,  1) ? 1 : 0;
		features[dummy_len + 8] = isHolidayWithGap(weekday,  2) ? 1 : 0;
		
		//8. CPU 的信息特征
		features[dummy_len + 9]  = virtuals.get(tag).core;
		features[dummy_len + 10] = virtuals.get(tag).memory;
		
		//9. 前七天的数据
		features[dummy_len + 11] = sequence[brand_code.get(tag)][(int) (day - ss_) - 7];
		features[dummy_len + 12] = sequence[brand_code.get(tag)][(int) (day - ss_) - 6];
		features[dummy_len + 13] = sequence[brand_code.get(tag)][(int) (day - ss_) - 5];
		features[dummy_len + 14] = sequence[brand_code.get(tag)][(int) (day - ss_) - 4];
		features[dummy_len + 15] = sequence[brand_code.get(tag)][(int) (day - ss_) - 3];
		features[dummy_len + 16] = sequence[brand_code.get(tag)][(int) (day - ss_) - 2];
		features[dummy_len + 17] = sequence[brand_code.get(tag)][(int) (day - ss_) - 1];
		
		//10. 是否是节假日
		features[dummy_len + 18] = param.holidaySet.contains(date.split("\\s+")[0]) ? 1 : 0;
		
		//11. 前一天 后一天是否为节假日
		features[dummy_len + 19] = param.holiday_end.contains(year2Day - 1) ? 1 : 0;
		features[dummy_len + 20] = param.holiday_start.contains(year2Day + 1) ? 1 : 0;
		features[dummy_len + 21] = param.holiday_end.contains(year2Day - 2) ? 1 : 0;
		features[dummy_len + 22] = param.holiday_start.contains(year2Day + 2) ? 1 : 0;

		//12. 差分数据
		features[dummy_len + 23] = sequence[brand_code.get(tag)][(int) (day - ss_) - 1] - sequence[brand_code.get(tag)][(int) (day - ss_) - 4];
		features[dummy_len + 24] = sequence[brand_code.get(tag)][(int) (day - ss_) - 2] - sequence[brand_code.get(tag)][(int) (day - ss_) - 5];
		features[dummy_len + 25] = sequence[brand_code.get(tag)][(int) (day - ss_) - 3] - sequence[brand_code.get(tag)][(int) (day - ss_) - 6];
		features[dummy_len + 26] = sequence[brand_code.get(tag)][(int) (day - ss_) - 4] - sequence[brand_code.get(tag)][(int) (day - ss_) - 7];
		
		//13. dummy of week
		features[dummy_len + 27] = 0;
		features[dummy_len + 28] = 0;
		features[dummy_len + 29] = 0;
		features[dummy_len + 30] = 0;
		features[dummy_len + 31] = 0;
		features[dummy_len + 32] = 0;
		features[dummy_len + 33] = 0;
		
		features[dummy_len + weekday - 1 + 27] = 1;
		
		//14. 上一周和下一周是否为假期
		features[dummy_len + 34] = param.holiday_end.contains(year2Day - 7) ? 1 : 0;
		features[dummy_len + 35] = param.holiday_start.contains(year2Day + 7) ? 1 : 0;
		
		//15. 前七天的最大值，最小值，均值
//		double _sum_cnt = 0.0;
//		double _max_cnt = -0x3f3f3f3f;
//		double _min_cnt =  0x3f3f3f3f;
//		for (int j = 1; j <= 7; ++j) {
//			_sum_cnt += sequence[brand_code.get(tag)][(int) (day - ss_) - j] ;
//			_max_cnt = Math.max(_max_cnt, sequence[brand_code.get(tag)][(int) (day - ss_) - j]);
//			_min_cnt = Math.min(_min_cnt, sequence[brand_code.get(tag)][(int) (day - ss_) - j]);
//		}
//		features[dummy_len + 36] = _sum_cnt / 7;
//		features[dummy_len + 37] = _max_cnt;
//		features[dummy_len + 38] = _min_cnt;
		
		//16. 是否为长假
		features[dummy_len + 36] = param.spring.contains(year2Day) ? 1 : 0;
		features[dummy_len + 37] = param.national.contains(year2Day) ? 1 : 0;
		features[dummy_len + 38] = param.holiday_start.contains(year2Day + 3) ? 1 : 0;
		features[dummy_len + 39] = param.holiday_end.contains(year2Day - 3) ? 1 : 0;
		
		return new Instance(tag, features);
	}
	
	@Deprecated
	public static List<Instance> generateTestData(Param params, long ss_, Map<String, Integer> brand_code, float[][] sequece){
		Map<String, CPU> virtuals = params.virtuals;
		List<Instance> test = new ArrayList<>();
		
		long s_ = DateUtil.year2dayLong(params.startTime) + 1;
		long e_ = DateUtil.year2dayLong(params.endTime) + 1;
		
		for (String key : virtuals.keySet()) {
			for (long day = s_; day <= e_; ++day) {
				String date = DateUtil.long2Date(day * 1000 * 24 * 3600);
				int weekday = DateUtil.week(date);
				
				int dummy_len = brand_code.size() + 12;
				float[] features = new float[dummy_len + 11];
				
				//1. brand
				features[brand_code.get(key)] = 1.0f;
				
				//dummy month
				features[brand_code.size() + DateUtil.month(date) - 1] = 1.0f;
				
				//2. day of week
				features[dummy_len] = weekday;
				
				//3. day of month
				features[dummy_len + 1] = DateUtil.month(date);
				
				//4. 是否为工作日
				features[dummy_len + 2] = (DateUtil.week(date) == 1 || DateUtil.week(date) == 7) ? 1 : 0;
				
				//5. 一个月中的第几天
				features[dummy_len + 3] = DateUtil.dayOfMonth(date);
				
				//6. 一个月中的第几周
				features[dummy_len + 4] = DateUtil.weekOfMonth(date);
				
				//7. 前一天是否为假期
				features[dummy_len + 5] = isHolidayWithGap(weekday, -1) ? 1 : 0;
				features[dummy_len + 6] = isHolidayWithGap(weekday, -2) ? 1 : 0;
				features[dummy_len + 7] = isHolidayWithGap(weekday,  1) ? 1 : 0;
				features[dummy_len + 8] = isHolidayWithGap(weekday,  2) ? 1 : 0;
				
				//8. CPU 的信息特征
				features[dummy_len + 9]  = virtuals.get(key).core;
				features[dummy_len + 10] = virtuals.get(key).memory;
				
				test.add(new Instance(key, features));
			}
		}
		return test;
	}
	
	private static boolean isHolidayWithGap(int day, int gap) {
		int prev_day = (day - 1 + gap + 7) % 7 + 1;
		return prev_day == 1 || prev_day == 7;
	}
	
	/**
	 * 不分tag训练
	 * @param dataset
	 * @param s_
	 * @param virtuals
	 * @param brand_code
	 * @param maxmin_y
	 * @param sequence
	 * @return
	 */
	public static List<Instance> lrExtract(Param param, List<Feature> dataset, long s_, Map<String, CPU> virtuals, 
			Map<String, Integer> brand_code, float[] maxmin_y, float[][] sequence, boolean useLinear){
		
		for (Feature record : dataset) {
			String tag = record.tag;
			if (!virtuals.containsKey(tag)) continue;
			if (!brand_code.containsKey(tag)) {
				brand_code.put(tag, brand_code.size());
			}
			maxmin_y[0] = Math.max(maxmin_y[0], record.count);
			maxmin_y[1] = Math.min(maxmin_y[1], record.count);
		}
		
		List<Instance> ret = new ArrayList<>();
		
		for (Feature record : dataset) {
			if (brand_code.containsKey(record.tag)) sequence[brand_code.get(record.tag)][(int) (record.year2Day + 1 - s_)] = record.count;
			if (!virtuals.containsKey(record.tag) || (int) (record.year2Day + 1 - s_) - 7 < 0) continue;
			
			int dummy_len = brand_code.size() + 12;
			float[] features = new float[dummy_len + 19];
			
			//1. brand
			String tag = record.tag;
			features[brand_code.get(tag)] = 1.0f;
			
			// dummy month
			features[brand_code.size() + DateUtil.month(record.date) - 1] = 1.0f;
			
			//2. day of week
			features[dummy_len] = record.dayOfWeek;
			
			//3. day of month
			features[dummy_len + 1] = DateUtil.month(record.date);
			
			//4. 是否为工作日
			features[dummy_len + 2] = (record.dayOfWeek == 1 || record.dayOfWeek == 7) ? 1 : 0;
			
			//5. 一个月中的第几天
			features[dummy_len + 3] = DateUtil.dayOfMonth(record.date);
			
			//6. 一个月中的第几周
			features[dummy_len + 4] = DateUtil.weekOfMonth(record.date);
			
			//7. 前一天是否为周末
			features[dummy_len + 5] = isHolidayWithGap(record.dayOfWeek, -1) ? 1 : 0;
			features[dummy_len + 6] = isHolidayWithGap(record.dayOfWeek, -2) ? 1 : 0;
			features[dummy_len + 7] = isHolidayWithGap(record.dayOfWeek,  1) ? 1 : 0;
			features[dummy_len + 8] = isHolidayWithGap(record.dayOfWeek,  2) ? 1 : 0;
			
			//8. CPU 的信息特征
			features[dummy_len + 9]  = virtuals.get(tag).core;
			features[dummy_len + 10] = virtuals.get(tag).memory;
			
			//9. 前7天这个时刻的count
			features[dummy_len + 11] = sequence[brand_code.get(tag)][(int) (record.year2Day + 1 - s_) - 7];
			features[dummy_len + 12] = sequence[brand_code.get(tag)][(int) (record.year2Day + 1 - s_) - 6];
			features[dummy_len + 13] = sequence[brand_code.get(tag)][(int) (record.year2Day + 1 - s_) - 5];
			features[dummy_len + 14] = sequence[brand_code.get(tag)][(int) (record.year2Day + 1 - s_) - 4];
			features[dummy_len + 15] = sequence[brand_code.get(tag)][(int) (record.year2Day + 1 - s_) - 3];
			features[dummy_len + 16] = sequence[brand_code.get(tag)][(int) (record.year2Day + 1 - s_) - 2];
			features[dummy_len + 17] = sequence[brand_code.get(tag)][(int) (record.year2Day + 1 - s_) - 1];
			
			//10. 是否是节假日
			features[dummy_len + 18] = param.holidaySet.contains(record.date.split("\\s+")[0]) ? 1 : 0;
			
			//label
			float y = record.count;
			if (!useLinear) y = (y - maxmin_y[1]) / (maxmin_y[0] - maxmin_y[1]);
			
			Instance instance = new Instance(tag, y, features);
			ret.add(instance);
		}
		
		return ret;
	}
	
	/**
	 * 分批训练
	 * @param dataset
	 * @param s_
	 * @param virtuals
	 * @param brand_code
	 * @param maxmin_y
	 * @param sequence
	 * @return
	 */
	public static List<Instance>[] lrExtract(Param param, List<Feature> dataset, long s_, Map<String, CPU> virtuals, Map<String, Integer> brand_code, float[][] maxmin_y, float[][] sequence, boolean useLinear){
		
		for (Feature record : dataset) {
			String tag = record.tag;
			if (!virtuals.containsKey(tag)) continue;
			if (!brand_code.containsKey(tag)) {
				brand_code.put(tag, brand_code.size());
			}
			maxmin_y[brand_code.get(tag)][0] = Math.max(maxmin_y[brand_code.get(tag)][0], record.count);
			maxmin_y[brand_code.get(tag)][1] = Math.min(maxmin_y[brand_code.get(tag)][1], record.count);
		}
		
		List<Instance>[] ret = new ArrayList[brand_code.size()];
		for (int i = 0; i < brand_code.size(); ++i) ret[i] = new ArrayList<>();
		
		for (Feature record : dataset) {
			if (brand_code.containsKey(record.tag)) sequence[brand_code.get(record.tag)][(int) (record.year2Day + 1 - s_)] = record.count;
			if (!virtuals.containsKey(record.tag) || (int) (record.year2Day + 1 - s_) - 7 < 0) continue;
			
			int dummy_len = brand_code.size() + 12;
			float[] features = new float[dummy_len + 40];
			
			//1. brand
			String tag = record.tag;
			features[brand_code.get(tag)] = 1.0f;
			
			// dummy month
			features[brand_code.size() + DateUtil.month(record.date) - 1] = 1.0f;
			
			//2. day of week
			features[dummy_len] = record.dayOfWeek;
			
			//3. day of month
			features[dummy_len + 1] = DateUtil.month(record.date);
			
			//4. 是否为工作日
			features[dummy_len + 2] = (record.dayOfWeek == 1 || record.dayOfWeek == 7) ? 1 : 0;
			
			//5. 一个月中的第几天
			features[dummy_len + 3] = DateUtil.dayOfMonth(record.date);
			
			//6. 一个月中的第几周
			features[dummy_len + 4] = DateUtil.weekOfMonth(record.date);
			
			//7. 前一天是否为假期
			features[dummy_len + 5] = isHolidayWithGap(record.dayOfWeek, -1) ? 1 : 0;
			features[dummy_len + 6] = isHolidayWithGap(record.dayOfWeek, -2) ? 1 : 0;
			features[dummy_len + 7] = isHolidayWithGap(record.dayOfWeek,  1) ? 1 : 0;
			features[dummy_len + 8] = isHolidayWithGap(record.dayOfWeek,  2) ? 1 : 0;
			
			//8. CPU 的信息特征
			features[dummy_len + 9]  = virtuals.get(tag).core;
			features[dummy_len + 10] = virtuals.get(tag).memory;
			
			//9. 前7天这个时刻的count
			features[dummy_len + 11] = sequence[brand_code.get(tag)][(int) (record.year2Day + 1 - s_) - 7];
			features[dummy_len + 12] = sequence[brand_code.get(tag)][(int) (record.year2Day + 1 - s_) - 6];
			features[dummy_len + 13] = sequence[brand_code.get(tag)][(int) (record.year2Day + 1 - s_) - 5];
			features[dummy_len + 14] = sequence[brand_code.get(tag)][(int) (record.year2Day + 1 - s_) - 4];
			features[dummy_len + 15] = sequence[brand_code.get(tag)][(int) (record.year2Day + 1 - s_) - 3];
			features[dummy_len + 16] = sequence[brand_code.get(tag)][(int) (record.year2Day + 1 - s_) - 2];
			features[dummy_len + 17] = sequence[brand_code.get(tag)][(int) (record.year2Day + 1 - s_) - 1];
			
			//10. 是否是节假日
			features[dummy_len + 18] = param.holidaySet.contains(record.date.split("\\s+")[0]) ? 1 : 0;
			
			//11. 前一天 后一天是否为节假日
			features[dummy_len + 19] = param.holiday_end.contains(record.year2Day - 1) ? 1 : 0;
			features[dummy_len + 20] = param.holiday_start.contains(record.year2Day + 1) ? 1 : 0;
			features[dummy_len + 21] = param.holiday_end.contains(record.year2Day - 2) ? 1 : 0;
			features[dummy_len + 22] = param.holiday_start.contains(record.year2Day + 2) ? 1 : 0;
			
			//12. 差分数据
			features[dummy_len + 23] = sequence[brand_code.get(tag)][(int) (record.year2Day + 1 - s_) - 1] - sequence[brand_code.get(tag)][(int) (record.year2Day + 1 - s_) - 4];
			features[dummy_len + 24] = sequence[brand_code.get(tag)][(int) (record.year2Day + 1 - s_) - 2] - sequence[brand_code.get(tag)][(int) (record.year2Day + 1 - s_) - 5];
			features[dummy_len + 25] = sequence[brand_code.get(tag)][(int) (record.year2Day + 1 - s_) - 3] - sequence[brand_code.get(tag)][(int) (record.year2Day + 1 - s_) - 6];
			features[dummy_len + 26] = sequence[brand_code.get(tag)][(int) (record.year2Day + 1 - s_) - 4] - sequence[brand_code.get(tag)][(int) (record.year2Day + 1 - s_) - 7];
			
			//13. dummy of week
			features[dummy_len + 27] = 0;
			features[dummy_len + 28] = 0;
			features[dummy_len + 29] = 0;
			features[dummy_len + 30] = 0;
			features[dummy_len + 31] = 0;
			features[dummy_len + 32] = 0;
			features[dummy_len + 33] = 0;
			
			features[dummy_len + record.dayOfWeek - 1 + 27] = 1;
			
			//14. 上一周和下一周是否为假期
			features[dummy_len + 34] = param.holiday_end.contains(record.year2Day - 7) ? 1 : 0;
			features[dummy_len + 35] = param.holiday_start.contains(record.year2Day + 7) ? 1 : 0;
			
			//15. 前七天的最大值，最小值，均值
//			double _sum_cnt = 0.0;
//			double _max_cnt = -0x3f3f3f3f;
//			double _min_cnt =  0x3f3f3f3f;
//			for (int j = 1; j <= 7; ++j) {
//				_sum_cnt += sequence[brand_code.get(tag)][(int) (record.year2Day + 1 - s_) - j];
//				_max_cnt = Math.max(_max_cnt, sequence[brand_code.get(tag)][(int) (record.year2Day + 1 - s_) - j]);
//				_min_cnt = Math.min(_min_cnt, sequence[brand_code.get(tag)][(int) (record.year2Day + 1 - s_) - j]);
//			}
//			features[dummy_len + 36] = _sum_cnt / 7;
//			features[dummy_len + 37] = _max_cnt;
//			features[dummy_len + 38] = _min_cnt;
			
			//16. 是否为放假后的第123天
			features[dummy_len + 36] = param.spring.contains(record.year2Day) ? 1 : 0;
			features[dummy_len + 37] = param.national.contains(record.year2Day) ? 1 : 0;
			features[dummy_len + 38] = param.holiday_start.contains(record.year2Day + 3) ? 1 : 0;
			features[dummy_len + 39] = param.holiday_end.contains(record.year2Day - 3) ? 1 : 0;
			
			//label
			float y = record.count;
			if (!useLinear) y = (y - maxmin_y[brand_code.get(tag)][1]) / (maxmin_y[brand_code.get(tag)][0] - maxmin_y[brand_code.get(tag)][1]);
			
			ret[brand_code.get(tag)].add(new Instance(tag, y, features));
		}
		
		return ret;
	}
	
	/**
	 * 差分 + 分批 + 平滑 + 训练
	 * @param dataset
	 * @param s_
	 * @param virtuals
	 * @param brand_code
	 * @param maxmin_y
	 * @param sequence
	 * @return
	 */
	public static List<Instance>[] lrExtract(Param param, List<Feature> dataset, long s_, Map<String, Integer> brand_code, 
			Map<String, Integer> flavor_map, float[][] sequence, int[] diff, int[] roll_param){
		
		List<Instance>[] ret = new ArrayList[brand_code.size()];
		for (int i = 0; i < brand_code.size(); ++i) ret[i] = new ArrayList<>();
		
		for (Feature record : dataset) {
			if (!param.virtuals.containsKey(record.tag) || (int) (record.year2Day + 1 - s_) - 7 < 0
		|| (int) (record.year2Day + 1 - s_) - 7 - roll_param[flavor_map.get(record.tag)] + 1 < 0) continue;
			
			int dummy_len = brand_code.size() + 12;
			float[] features = new float[dummy_len + 40];
			
			//1. brand
			String tag = record.tag;
			features[brand_code.get(tag)] = 1.0f;
			
			// dummy month
			features[brand_code.size() + DateUtil.month(record.date) - 1] = 1.0f;
			
			//2. day of week
			features[dummy_len] = record.dayOfWeek;
			
			//3. day of month
			features[dummy_len + 1] = DateUtil.month(record.date);
			
			//4. 是否为工作日
			features[dummy_len + 2] = (record.dayOfWeek == 1 || record.dayOfWeek == 7) ? 1 : 0;
			
			//5. 一个月中的第几天
			features[dummy_len + 3] = DateUtil.dayOfMonth(record.date);
			
			//6. 一个月中的第几周
			features[dummy_len + 4] = DateUtil.weekOfMonth(record.date);
			
			//7. 前一天是否为假期
			features[dummy_len + 5] = isHolidayWithGap(record.dayOfWeek, -1) ? 1 : 0;
			features[dummy_len + 6] = isHolidayWithGap(record.dayOfWeek, -2) ? 1 : 0;
			features[dummy_len + 7] = isHolidayWithGap(record.dayOfWeek,  1) ? 1 : 0;
			features[dummy_len + 8] = isHolidayWithGap(record.dayOfWeek,  2) ? 1 : 0;
			
			//8. CPU 的信息特征
			features[dummy_len + 9]  = param.virtuals.get(tag).core;
			features[dummy_len + 10] = param.virtuals.get(tag).memory;
			
			//9. 前7天这个时刻的count
			features[dummy_len + 11] = sequence[brand_code.get(tag)][(int) (record.year2Day + 1 - s_) - 7];
			features[dummy_len + 12] = sequence[brand_code.get(tag)][(int) (record.year2Day + 1 - s_) - 6];
			features[dummy_len + 13] = sequence[brand_code.get(tag)][(int) (record.year2Day + 1 - s_) - 5];
			features[dummy_len + 14] = sequence[brand_code.get(tag)][(int) (record.year2Day + 1 - s_) - 4];
			features[dummy_len + 15] = sequence[brand_code.get(tag)][(int) (record.year2Day + 1 - s_) - 3];
			features[dummy_len + 16] = sequence[brand_code.get(tag)][(int) (record.year2Day + 1 - s_) - 2];
			features[dummy_len + 17] = sequence[brand_code.get(tag)][(int) (record.year2Day + 1 - s_) - 1];
			
			//10. 是否是节假日
			features[dummy_len + 18] = param.holidaySet.contains(record.date.split("\\s+")[0]) ? 1 : 0;
			
			//11. 前一天 后一天是否为节假日
			features[dummy_len + 19] = param.holiday_end.contains(record.year2Day - 1) ? 1 : 0;
			features[dummy_len + 20] = param.holiday_start.contains(record.year2Day + 1) ? 1 : 0;
			features[dummy_len + 21] = param.holiday_end.contains(record.year2Day - 2) ? 1 : 0;
			features[dummy_len + 22] = param.holiday_start.contains(record.year2Day + 2) ? 1 : 0;
			
			//12. 差分数据
			features[dummy_len + 23] = sequence[brand_code.get(tag)][(int) (record.year2Day + 1 - s_) - 1] - sequence[brand_code.get(tag)][(int) (record.year2Day + 1 - s_) - 4];
			features[dummy_len + 24] = sequence[brand_code.get(tag)][(int) (record.year2Day + 1 - s_) - 2] - sequence[brand_code.get(tag)][(int) (record.year2Day + 1 - s_) - 5];
			features[dummy_len + 25] = sequence[brand_code.get(tag)][(int) (record.year2Day + 1 - s_) - 3] - sequence[brand_code.get(tag)][(int) (record.year2Day + 1 - s_) - 6];
			features[dummy_len + 26] = sequence[brand_code.get(tag)][(int) (record.year2Day + 1 - s_) - 4] - sequence[brand_code.get(tag)][(int) (record.year2Day + 1 - s_) - 7];
			
			//13. dummy of week
			features[dummy_len + 27] = 0;
			features[dummy_len + 28] = 0;
			features[dummy_len + 29] = 0;
			features[dummy_len + 30] = 0;
			features[dummy_len + 31] = 0;
			features[dummy_len + 32] = 0;
			features[dummy_len + 33] = 0;
			
			features[dummy_len + record.dayOfWeek - 1 + 27] = 1;
			
			//14. 上一周和下一周是否为假期
			features[dummy_len + 34] = param.holiday_end.contains(record.year2Day - 7) ? 1 : 0;
			features[dummy_len + 35] = param.holiday_start.contains(record.year2Day + 7) ? 1 : 0;
			
			//15. 前七天的最大值，最小值，均值
//			double _sum_cnt = 0.0;
//			double _max_cnt = -0x3f3f3f3f;
//			double _min_cnt =  0x3f3f3f3f;
//			for (int j = 1; j <= 7; ++j) {
//				_sum_cnt += sequence[brand_code.get(tag)][(int) (record.year2Day + 1 - s_) - j];
//				_max_cnt = Math.max(_max_cnt, sequence[brand_code.get(tag)][(int) (record.year2Day + 1 - s_) - j]);
//				_min_cnt = Math.min(_min_cnt, sequence[brand_code.get(tag)][(int) (record.year2Day + 1 - s_) - j]);
//			}
//			features[dummy_len + 36] = _sum_cnt / 7;
//			features[dummy_len + 37] = _max_cnt;
//			features[dummy_len + 38] = _min_cnt;
			
			//16. 是否为放假后的第123天
			features[dummy_len + 36] = param.spring.contains(record.year2Day) ? 1 : 0;
			features[dummy_len + 37] = param.national.contains(record.year2Day) ? 1 : 0;
			features[dummy_len + 38] = param.holiday_start.contains(record.year2Day + 3) ? 1 : 0;
			features[dummy_len + 39] = param.holiday_end.contains(record.year2Day - 3) ? 1 : 0;
			
			//label
			float y = sequence[brand_code.get(tag)][(int) (record.year2Day + 1 - s_)] - (diff[flavor_map.get(tag)] != 0 
					? sequence[brand_code.get(tag)][(int) (record.year2Day + 1 - s_) - diff[flavor_map.get(tag)]] : 0);
			
			ret[brand_code.get(tag)].add(new Instance(tag, y, features));
		}
		
		return ret;
	}
	
	
	/**
	 * 差分 + 分批 + 训练
	 * @param dataset
	 * @param s_
	 * @param virtuals
	 * @param brand_code
	 * @param maxmin_y
	 * @param sequence
	 * @return
	 */
	public static List<Instance>[] lrExtract(Param param, List<Feature> dataset, long s_, Map<String, Integer> brand_code, 
			Map<String, Integer> flavor_map, float[][] sequence, int[] diff){
		
		// 编码
		for (Feature record : dataset) {
			String tag = record.tag;
			if (!param.virtuals.containsKey(tag)) continue;
			if (!brand_code.containsKey(tag)) {
				brand_code.put(tag, brand_code.size());
			}
		}
		
		List<Instance>[] ret = new ArrayList[brand_code.size()];
		for (int i = 0; i < brand_code.size(); ++i) ret[i] = new ArrayList<>();
		
		for (Feature record : dataset) {
			if (brand_code.containsKey(record.tag)) sequence[brand_code.get(record.tag)][(int) (record.year2Day + 1 - s_)] = record.count;
			if (!param.virtuals.containsKey(record.tag) || (int) (record.year2Day + 1 - s_) - 7 < 0) continue;
			
			int dummy_len = brand_code.size() + 12;
			float[] features = new float[dummy_len + 40];
			
			//1. brand
			String tag = record.tag;
			features[brand_code.get(tag)] = 1.0f;
			
			// dummy month
			features[brand_code.size() + DateUtil.month(record.date) - 1] = 1.0f;
			
			//2. day of week
			features[dummy_len] = record.dayOfWeek;
			
			//3. day of month
			features[dummy_len + 1] = DateUtil.month(record.date);
			
			//4. 是否为工作日
			features[dummy_len + 2] = (record.dayOfWeek == 1 || record.dayOfWeek == 7) ? 1 : 0;
			
			//5. 一个月中的第几天
			features[dummy_len + 3] = DateUtil.dayOfMonth(record.date);
			
			//6. 一个月中的第几周
			features[dummy_len + 4] = DateUtil.weekOfMonth(record.date);
			
			//7. 前一天是否为假期
			features[dummy_len + 5] = isHolidayWithGap(record.dayOfWeek, -1) ? 1 : 0;
			features[dummy_len + 6] = isHolidayWithGap(record.dayOfWeek, -2) ? 1 : 0;
			features[dummy_len + 7] = isHolidayWithGap(record.dayOfWeek,  1) ? 1 : 0;
			features[dummy_len + 8] = isHolidayWithGap(record.dayOfWeek,  2) ? 1 : 0;
			
			//8. CPU 的信息特征
			features[dummy_len + 9]  = param.virtuals.get(tag).core;
			features[dummy_len + 10] = param.virtuals.get(tag).memory;
			
			//9. 前7天这个时刻的count
			features[dummy_len + 11] = sequence[brand_code.get(tag)][(int) (record.year2Day + 1 - s_) - 7];
			features[dummy_len + 12] = sequence[brand_code.get(tag)][(int) (record.year2Day + 1 - s_) - 6];
			features[dummy_len + 13] = sequence[brand_code.get(tag)][(int) (record.year2Day + 1 - s_) - 5];
			features[dummy_len + 14] = sequence[brand_code.get(tag)][(int) (record.year2Day + 1 - s_) - 4];
			features[dummy_len + 15] = sequence[brand_code.get(tag)][(int) (record.year2Day + 1 - s_) - 3];
			features[dummy_len + 16] = sequence[brand_code.get(tag)][(int) (record.year2Day + 1 - s_) - 2];
			features[dummy_len + 17] = sequence[brand_code.get(tag)][(int) (record.year2Day + 1 - s_) - 1];
			
			//10. 是否是节假日
			features[dummy_len + 18] = param.holidaySet.contains(record.date.split("\\s+")[0]) ? 1 : 0;
			
			//11. 前一天 后一天是否为节假日
			features[dummy_len + 19] = param.holiday_end.contains(record.year2Day - 1) ? 1 : 0;
			features[dummy_len + 20] = param.holiday_start.contains(record.year2Day + 1) ? 1 : 0;
			features[dummy_len + 21] = param.holiday_end.contains(record.year2Day - 2) ? 1 : 0;
			features[dummy_len + 22] = param.holiday_start.contains(record.year2Day + 2) ? 1 : 0;
			
			//12. 差分数据
			features[dummy_len + 23] = sequence[brand_code.get(tag)][(int) (record.year2Day + 1 - s_) - 1] - sequence[brand_code.get(tag)][(int) (record.year2Day + 1 - s_) - 4];
			features[dummy_len + 24] = sequence[brand_code.get(tag)][(int) (record.year2Day + 1 - s_) - 2] - sequence[brand_code.get(tag)][(int) (record.year2Day + 1 - s_) - 5];
			features[dummy_len + 25] = sequence[brand_code.get(tag)][(int) (record.year2Day + 1 - s_) - 3] - sequence[brand_code.get(tag)][(int) (record.year2Day + 1 - s_) - 6];
			features[dummy_len + 26] = sequence[brand_code.get(tag)][(int) (record.year2Day + 1 - s_) - 4] - sequence[brand_code.get(tag)][(int) (record.year2Day + 1 - s_) - 7];
			
			//13. dummy of week
			features[dummy_len + 27] = 0;
			features[dummy_len + 28] = 0;
			features[dummy_len + 29] = 0;
			features[dummy_len + 30] = 0;
			features[dummy_len + 31] = 0;
			features[dummy_len + 32] = 0;
			features[dummy_len + 33] = 0;
			
			features[dummy_len + record.dayOfWeek - 1 + 27] = 1;
			
			//14. 上一周和下一周是否为假期
			features[dummy_len + 34] = param.holiday_end.contains(record.year2Day - 7) ? 1 : 0;
			features[dummy_len + 35] = param.holiday_start.contains(record.year2Day + 7) ? 1 : 0;
			
			//15. 前七天的最大值，最小值，均值
//			double _sum_cnt = 0.0;
//			double _max_cnt = -0x3f3f3f3f;
//			double _min_cnt =  0x3f3f3f3f;
//			for (int j = 1; j <= 7; ++j) {
//				_sum_cnt += sequence[brand_code.get(tag)][(int) (record.year2Day + 1 - s_) - j];
//				_max_cnt = Math.max(_max_cnt, sequence[brand_code.get(tag)][(int) (record.year2Day + 1 - s_) - j]);
//				_min_cnt = Math.min(_min_cnt, sequence[brand_code.get(tag)][(int) (record.year2Day + 1 - s_) - j]);
//			}
//			features[dummy_len + 36] = _sum_cnt / 7;
//			features[dummy_len + 37] = _max_cnt;
//			features[dummy_len + 38] = _min_cnt;
			
			//16. 是否为放假后的第123天
			features[dummy_len + 36] = param.spring.contains(record.year2Day) ? 1 : 0;
			features[dummy_len + 37] = param.national.contains(record.year2Day) ? 1 : 0;
			features[dummy_len + 38] = param.holiday_start.contains(record.year2Day + 3) ? 1 : 0;
			features[dummy_len + 39] = param.holiday_end.contains(record.year2Day - 3) ? 1 : 0;
			
			//label
			float y = record.count - sequence[brand_code.get(tag)][(int) (record.year2Day + 1 - s_) - diff[flavor_map.get(tag)]];
			
			ret[brand_code.get(tag)].add(new Instance(tag, y, features));
		}
		
		return ret;
	}
	
	
	public static List<Double> extract(List<Feature> dataset){
		List<Double> ret = new ArrayList<>();
		
		int n = dataset.size();
		long s_ = dataset.get(0).year2Day;
		long e_ = dataset.get(n - 1).year2Day;
		
		for (int i = 0; i < e_ - s_ + 1; ++i) {
			ret.add(0.0);
		}
		
		for (Feature feature : dataset) {
			ret.set((int)(feature.year2Day - s_), feature.count * 1.0);
		}
		
		return ret;
	}
	
	public static Map<String, List<Double>> meanWinow(Map<String, List<Double>> law, int window){
		Map<String, List<Double>> ret = new HashMap<>();
		for (String key : law.keySet()) {
			List<Double> list = law.get(key);
			List<Double> next = new ArrayList<>();
			int n = list.size();
			double _sum = 0;
			for (int i = 0; i < Math.min(window, n) - 1; ++i) {
				_sum += list.get(i);
			}
			for (int i = Math.min(window, n); i < n; ++i) {
				_sum += list.get(i);
				next.add(_sum / Math.min(window, n));
				_sum -= list.get(i - Math.min(window, n));
			}
			ret.put(key, next);
		}
		return ret;
	}
	
	public static Map<String, List<Double>> extract(List<Feature> dataset, Set<String> keySet, Map<String, Double> mean, Param param){
		Map<String, List<Double>> mem = new HashMap<>();
		
		int n = dataset.size();
		long s_ = dataset.get(0).year2Day;
		long e_ = dataset.get(n - 1).year2Day;
		
		for (String key : keySet) {
			List<Double> ret = new ArrayList<>();
			for (int i = 0; i < e_ - s_ + 1; ++i) {
				ret.add(0.0);
			}
			mem.put(key, ret);
		}
		
		for (Feature feature : dataset) {
			if (mem.containsKey(feature.tag)) {
				mem.get(feature.tag).set((int)(feature.year2Day - s_), feature.count * 1.0);
			}
		}
		
		return mem;
	}
	
	
	public static void main(String[] args) {
		System.out.println(DateUtil.year2dayLong("2012-04-02"));
		System.out.println(DateUtil.year2dayLong("2012-04-03") - 1);
	}
}
