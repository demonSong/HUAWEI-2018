package com.elasticcloudservice.train;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Random;
import java.util.Set;

import com.elasticcloudservice.predict.CPU;
import com.elasticcloudservice.predict.Goods;
import com.elasticcloudservice.predict.Manifest;
import com.elasticcloudservice.predict.Param;

public abstract class Allocation {
	
	private static List<Manifest> doAllocate(int all_core, int all_memo, List<Goods> goods){
		List<Manifest> mani = new ArrayList<>();
		if (goods.size() == 0) return mani;
		
		int sum_core = 0;
		int sum_memo = 0;
		
		int uuid = 1;
		Manifest one = new Manifest(uuid + "");
		
		for (int i = 0; i < goods.size(); ++i) {
			sum_core += goods.get(i).core;
			sum_memo += goods.get(i).mem;
			
			if (sum_core <= all_core && sum_memo <= all_memo) {
				one.add(goods.get(i));
			}
			else {
				mani.add(one);
				
				uuid += 1;
				one = new Manifest(uuid + "");
				
				sum_core = goods.get(i).core;
				sum_memo = goods.get(i).mem;
				one.add(goods.get(i));
			}
		}
		
		mani.add(one);
		return mani;
	}

	static class Node{
		int remain;
		Node(int remain){
			this.remain = remain;
		}
	}
	
	
	/**
	 * 模拟退火算法 CPU
	 * @param all_core
	 * @param all_memo
	 * @param goods
	 * @return
	 */
	private static List<Manifest> annealingCPU(int all_core, int all_memo, List<Goods> goods){
		double max_score = Double.MIN_VALUE;
		List<Manifest> manifests = null;
		
		double T =  100.0;
		double Tmin = 0.1;
		double r = 0.999;
		List<Integer> dice = new ArrayList<>();
		Random rnd = new Random(2018);
		for (int i = 0; i < goods.size(); ++i) dice.add(i);
		
		while (T > Tmin) {
			Collections.shuffle(dice, rnd);
			List<Goods> shuffled = new ArrayList<>(goods);
			
			Goods swap = new Goods(shuffled.get(dice.get(0)));
			shuffled.set(dice.get(0), new Goods(shuffled.get(dice.get(1))));
			shuffled.set(dice.get(1), swap);
			
			List<Manifest> tmp = CPUAllocate(all_core, all_memo, shuffled);
			
			//评估函数
			double score = 0;
			for (Manifest m : tmp) score += m.scoreCPU(all_core);
			
			if (score > max_score) {
				max_score = score;
				manifests = new ArrayList<>(tmp);
				goods = new ArrayList<>(shuffled);
			}
			else {
				double p = 1 / (1 + Math
                        .exp(-(score - max_score) / T));
                if (rnd.nextDouble() < p) {
                	max_score = score;
                    manifests = new ArrayList<>(tmp);
                    goods = new ArrayList<>(shuffled);
                }
			}
			T = r * T;
		}
		return manifests;
	}
	
	
	private static List<Manifest> CPUAllocate(int all_core, int all_memo, List<Goods> goods){
		int core_sum = 0;
		int memo_sum = 0;
		for (Goods good : goods) {
			core_sum += good.core;
			memo_sum += good.mem;
		}
		int atLeast = Math.max((int)Math.ceil(core_sum * 1.0 / all_core), (int)Math.ceil(memo_sum * 1.0 / all_memo));
		
		Queue<Manifest> queue = new PriorityQueue<>((a, b) -> (b.core == a.core ? b.memo - a.memo : b.core - a.core));
		for (int i = 0; i < atLeast; ++i) {
			queue.offer(new Manifest((i + 1) + "", all_core, all_memo));
		}
		
		List<Manifest> manifests = new ArrayList<>();
		boolean[] done = new boolean[goods.size()];
		
		int isAllocate = 0;
		while (!queue.isEmpty()) {
			if (isAllocate == goods.size()) {
				while(!queue.isEmpty()) manifests.add(queue.poll());
				break;
			}
			
			Manifest best = queue.poll();
			boolean noGoodsFit = true;
			
			for (int i = 0; i < goods.size(); ++i) {
				if (done[i]) continue;
				Goods good = goods.get(i);
				if (best.core >= good.core && best.memo >= good.mem) {
					best.addAndRemain(good, good.core, good.mem);
					queue.offer(best);
					done[i] = true;
					noGoodsFit = false;
					isAllocate ++;
					break;
				}
			}
			if (noGoodsFit) manifests.add(best);
		}
		
		List<Goods> remain = new ArrayList<>();
		for (int i = 0; i < goods.size(); ++i) {
			if (!done[i]) remain.add(goods.get(i));
		}
		
		if (remain.size() != 0) {
			manifests.addAll(cpuDP(remain, all_core, all_memo, manifests.size() + 1));
		}
		
		Collections.sort(manifests, (a, b) -> (a.id - b.id));
		return manifests;
	}
	
	
	/**
	 * 模拟退火算法 MEM
	 * @param all_core
	 * @param all_memo
	 * @param goods
	 * @return
	 */
	private static List<Manifest> annealingMEM(int all_core, int all_memo, List<Goods> goods){
		double max_score = Double.MIN_VALUE;
		
		int memo_sum = 0;
		for (Goods good : goods) {
			memo_sum += good.mem;
		}
		
		List<Manifest> manifests = null;
		
		double T =  100.0;
		double Tmin = 0.1;
		double r = 0.999;
		List<Integer> dice = new ArrayList<>();
		Random rnd = new Random(2018);
		for (int i = 0; i < goods.size(); ++i) dice.add(i);
		
		while (T > Tmin) {
			Collections.shuffle(dice, rnd);
			List<Goods> shuffled = new ArrayList<>(goods);
			
			Goods swap = new Goods(shuffled.get(dice.get(0)));
			shuffled.set(dice.get(0), new Goods(shuffled.get(dice.get(1))));
			shuffled.set(dice.get(1), swap);
			
			List<Manifest> tmp = MEMAllocate(all_core, all_memo, shuffled);
			
			//评估函数
			double score = 0;
			for (Manifest m : tmp) score += m.scoreMEM(all_memo);
			
			if (score > max_score) {
				max_score = score;
				manifests = new ArrayList<>(tmp);
				goods = new ArrayList<>(shuffled);
			}
			else {
				double p = 1 / (1 + Math
                        .exp(-(score - max_score) / T));
                if (rnd.nextDouble() < p) {
                	max_score = score;
                    manifests = new ArrayList<>(tmp);
                    goods = new ArrayList<>(shuffled);
                }
			}
			T = r * T;
		}
		return manifests;
	}
	
	
	private static List<Manifest> MEMAllocate(int all_core, int all_memo, List<Goods> goods){
		int core_sum = 0;
		int memo_sum = 0;
		for (Goods good : goods) {
			core_sum += good.core;
			memo_sum += good.mem;
		}
		int atLeast = Math.max((int)Math.ceil(core_sum * 1.0 / all_core), (int)Math.ceil(memo_sum * 1.0 / all_memo));
		
		Queue<Manifest> queue = new PriorityQueue<>((a, b) -> (b.memo == a.memo ? b.core - a.core : b.memo - a.memo));
		for (int i = 0; i < atLeast; ++i) {
			queue.offer(new Manifest((i + 1) + "", all_core, all_memo));
		}
		
		List<Manifest> manifests = new ArrayList<>();
		boolean[] done = new boolean[goods.size()];
		
		int isAllocate = 0;
		while (!queue.isEmpty()) {
			if (isAllocate == goods.size()) {
				while(!queue.isEmpty()) manifests.add(queue.poll());
				break;
			}
			Manifest best = queue.poll();
			boolean noGoodsFit = true;
			
			for (int i = 0; i < goods.size(); ++i) {
				if (done[i]) continue;
				Goods good = goods.get(i);
				if (best.core >= good.core && best.memo >= good.mem) {
					best.addAndRemain(good, good.core, good.mem);
					queue.offer(best);
					done[i] = true;
					noGoodsFit = false;
					isAllocate ++;
					break;
				}
			}
			if (noGoodsFit) manifests.add(best);
		}
		
		List<Goods> remain = new ArrayList<>();
		for (int i = 0; i < goods.size(); ++i) {
			if (!done[i]) remain.add(goods.get(i));
		}
		
		if (remain.size() != 0) {
			manifests.addAll(memDP(remain, all_core, all_memo, manifests.size() + 1));
		}
		Collections.sort(manifests, (a, b) -> (a.id - b.id));
		return manifests;
	}
	
	
	
	private static List<Manifest> priorityCPUAllocate(int all_core, int all_memo, List<Goods> goods){
		int core_sum = 0;
		int memo_sum = 0;
		for (Goods good : goods) {
			core_sum += good.core;
			memo_sum += good.mem;
		}
		int atLeast = Math.max((int)Math.ceil(core_sum * 1.0 / all_core), (int)Math.ceil(memo_sum * 1.0 / all_memo));
		Queue<Manifest> queue = new PriorityQueue<>((a, b) -> (b.core == a.core ? b.memo - a.memo : b.core - a.core));
		for (int i = 0; i < atLeast; ++i) {
			queue.offer(new Manifest((i + 1) + "", all_core, all_memo));
		}
		
		Collections.sort(goods, (a, b) -> (b.core == a.core ? b.mem - a.mem : b.core - a.core));
		List<Manifest> manifests = new ArrayList<>();
		boolean[] done = new boolean[goods.size()];
		
		int isAllocate = 0;
		while (!queue.isEmpty()) {
			if (isAllocate == goods.size()) {
				while(!queue.isEmpty()) manifests.add(queue.poll());
				break;
			}
			
			Manifest best = queue.poll();
			boolean noGoodsFit = true;
			
			for (int i = 0; i < goods.size(); ++i) {
				if (done[i]) continue;
				Goods good = goods.get(i);
				if (best.core >= good.core && best.memo >= good.mem) {
					best.addAndRemain(good, good.core, good.mem);
					queue.offer(best);
					done[i] = true;
					noGoodsFit = false;
					isAllocate ++;
					break;
				}
			}
			if (noGoodsFit) manifests.add(best);
		}
		
		List<Goods> remain = new ArrayList<>();
		for (int i = 0; i < goods.size(); ++i) {
			if (!done[i]) remain.add(goods.get(i));
		}
		
		if (remain.size() != 0) {
			manifests.addAll(cpuDP(remain, all_core, all_memo, manifests.size() + 1));
		}
		
		Collections.sort(manifests, (a, b) -> (a.id - b.id));
		return manifests;
	}
	
	private static List<Manifest> priorityMEMAllocate(int all_core, int all_memo, List<Goods> goods){
		int core_sum = 0;
		int memo_sum = 0;
		for (Goods good : goods) {
			core_sum += good.core;
			memo_sum += good.mem;
		}
		
		int atLeast = Math.max((int)Math.ceil(core_sum * 1.0 / all_core), (int)Math.ceil(memo_sum * 1.0 / all_memo));
		
		Queue<Manifest> queue = new PriorityQueue<>((a, b) -> (b.memo == a.memo ? b.core - a.core : b.memo - a.memo));
		for (int i = 0; i < atLeast; ++i) {
			queue.offer(new Manifest((i + 1) + "", all_core, all_memo));
		}
		
		Collections.sort(goods, (a, b) -> (b.mem == a.mem ? b.core - a.core : b.mem - a.mem));
		List<Manifest> manifests = new ArrayList<>();
		boolean[] done = new boolean[goods.size()];
		
		int isAllocate = 0;
		while (!queue.isEmpty()) {
			if (isAllocate == goods.size()) {
				while(!queue.isEmpty()) manifests.add(queue.poll());
				break;
			}
			Manifest best = queue.poll();
			boolean noGoodsFit = true;
			
			for (int i = 0; i < goods.size(); ++i) {
				if (done[i]) continue;
				Goods good = goods.get(i);
				if (best.core >= good.core && best.memo >= good.mem) {
					best.addAndRemain(good, good.core, good.mem);
					queue.offer(best);
					done[i] = true;
					noGoodsFit = false;
					isAllocate ++;
					break;
				}
			}
			if (noGoodsFit) manifests.add(best);
		}
		
		List<Goods> remain = new ArrayList<>();
		for (int i = 0; i < goods.size(); ++i) {
			if (!done[i]) remain.add(goods.get(i));
		}
		
		if (remain.size() != 0) {
			manifests.addAll(memDP(remain, all_core, all_memo, manifests.size() + 1));
		}
		Collections.sort(manifests, (a, b) -> (a.id - b.id));
		return manifests;
	}
	
	static class Info{
		int val;
		int cpu;
		int mem;
		double ratio;
		List<Integer> path;
	}
	
	/**
	 * ratio 最大动规分配
	 * @param goods
	 * @param all_core
	 * @param all_memo
	 * @param idx
	 * @return
	 */
	private static List<Manifest> ratioDP(List<Goods> goods, int all_core, int all_memo, int idx){
		List<Manifest> mani = new ArrayList<>();
		while (goods.size() != 0) {
			Manifest one = new Manifest(idx + "");
			mem_ratio = new HashMap<>();
			Info next = dynamicalAllocateCoreAndMem(all_core, all_memo, goods, 0);
			for (int pos : next.path) {
				one.add(goods.get(pos));
			}
			for (int pos : next.path) {
				goods.remove(pos);
			}
			
			mani.add(one);
			idx++;
		}
		return mani;
	}
	
	/**
	 * 使得 ratio 最大
	 * @param all_core
	 * @param all_memo
	 * @param goods
	 * @param pos
	 * @return
	 */
	static Map<String, Info> mem_ratio;
	private static Info dynamicalAllocateCoreAndMem(int all_core, int all_memo, List<Goods> goods, int pos) {
		String[] keys = {all_core + "", all_memo + "", pos + ""};
		String key = String.join(",", keys);
		if (mem_ratio.containsKey(key)) return mem_ratio.get(key);
		
		Info info = new Info();
		if (pos == goods.size()) {
			info.cpu = 0;
			info.mem = 0;
			info.ratio = 0;
			info.path = new ArrayList<>();
		}
		else if (all_core < goods.get(pos).core || all_memo < goods.get(pos).mem){ //无法分配
			info = dynamicalAllocateCoreAndMem(all_core, all_memo, goods, pos + 1);
		}
		else {
			Info left = dynamicalAllocateCoreAndMem(all_core, all_memo, goods, pos + 1);
			Info right = dynamicalAllocateCoreAndMem(all_core - goods.get(pos).core, all_memo - goods.get(pos).mem, goods, pos + 1);
			
			double cmp_ratio = (right.cpu + goods.get(pos).core) * 1.0 / all_core;
			cmp_ratio += (right.mem + goods.get(pos).mem) * 1.0 / all_memo;
			cmp_ratio /= 2;
			
			if (left.ratio >= cmp_ratio) {
				info.cpu = left.cpu;
				info.mem = left.mem;
				info.ratio = left.ratio;
				info.path = new ArrayList<Integer>(left.path);
			}
			else {
				info.cpu = right.cpu + goods.get(pos).core;
				info.mem = right.mem + goods.get(pos).mem;
				info.ratio = cmp_ratio;
				info.path = new ArrayList<Integer>(right.path);
				info.path.add(pos);
			}
		}
		
		mem_ratio.put(key, info);
		return info;
	}
	
	
	
	/**
	 * 在 核数 和 内存的总限制下，使得核数最大
	 * @param all_core
	 * @param all_memo
	 * @param goods
	 * @param pos
	 * @return
	 */
	static Map<String, Info> mem_core;
	private static Info dynamicalAllocateCore(int all_core, int all_memo, List<Goods> goods, int pos) {
		String[] keys = {all_core + "", all_memo + "", pos + ""};
		String key = String.join(",", keys);
		if (mem_core.containsKey(key)) return mem_core.get(key);
		
		Info info = new Info();
		if (pos == goods.size()) {
			info.val = 0;
			info.path = new ArrayList<>();
		}
		else if (all_core < goods.get(pos).core || all_memo < goods.get(pos).mem){ //无法分配
			info = dynamicalAllocateCore(all_core, all_memo, goods, pos + 1);
		}
		else {
			Info left = dynamicalAllocateCore(all_core, all_memo, goods, pos + 1);
			Info right = dynamicalAllocateCore(all_core - goods.get(pos).core, all_memo - goods.get(pos).mem, goods, pos + 1);
			
			if (left.val >= right.val + goods.get(pos).core + goods.get(pos).mem) {
				info.val = left.val;
				info.path = new ArrayList<Integer>(left.path);
			}
			else {
				info.val = right.val + goods.get(pos).core + goods.get(pos).mem;
				info.path = new ArrayList<Integer>(right.path);
				info.path.add(pos);
			}
		}
		
		mem_core.put(key, info);
		return info;
	}
	
	
	static Map<String, Info> mem_memo;
	private static Info dynamicalAllocateMemo(int all_core, int all_memo, List<Goods> goods, int pos) {
		String[] keys = {all_core + "", all_memo + "", pos + ""};
		String key = String.join(",", keys);
		if (mem_memo.containsKey(key)) return mem_memo.get(key);
		
		Info info = new Info();
		if (pos == goods.size()) {
			info.val = 0;
			info.path = new ArrayList<>();
		}
		else if (all_core < goods.get(pos).core || all_memo < goods.get(pos).mem){ //无法分配
			info = dynamicalAllocateMemo(all_core, all_memo, goods, pos + 1);
		}
		else {
			Info left = dynamicalAllocateMemo(all_core, all_memo, goods, pos + 1);
			Info right = dynamicalAllocateMemo(all_core - goods.get(pos).core, all_memo - goods.get(pos).mem, goods, pos + 1);
			
			if (left.val >= right.val + goods.get(pos).core + goods.get(pos).mem) {
				info.val = left.val;
				info.path = new ArrayList<Integer>(left.path);
			}
			else {
				info.val = right.val + goods.get(pos).core + goods.get(pos).mem;
				info.path = new ArrayList<Integer>(right.path);
				info.path.add(pos);
			}
		}
		
		mem_memo.put(key, info);
		return info;
	}
	
	@Deprecated
	private static Map<String, List<Manifest>> __ployDP__(Map<Goods, Queue<Goods>> cagos, Map<Goods, Integer> cagosCnt, Map<String, CPU> phys){
		Map<String, List<Manifest>> res = new HashMap<>();
		for (String type : phys.keySet()) res.put(type, new ArrayList<>());
		List<Goods> remain = new ArrayList<>();
		while (cagosCnt.size() != 0) {
			Manifest one = new Manifest();
			double best = 0;
			String best_type = "";
			int N = cagos.size();
			int[] MEM = new int[N];
			int[] CPU = new int[N];
			int[] CNT = new int[N];
			int[] best_path = new int[N];
			
			Goods[] KEY = new Goods[N];
			int idx = 0;
			for (Goods key : cagosCnt.keySet()) {
				MEM[idx] = key.mem;
				CPU[idx] = key.core;
				CNT[idx] = cagosCnt.get(key);
				KEY[idx] = key;
				idx ++;
			}
			
			for (String type : phys.keySet()) {
				int C = phys.get(type).core;
				int M = phys.get(type).memory * 1024;
				int[] path = __knapsack__binary(C, M, MEM, CPU, CNT);
				double sum_c = 0;
				double sum_m = 0;
				for (int i = 0; i < N; ++i) {
					sum_c += CPU[i] * path[i];
					sum_m += MEM[i] * path[i];
				}
				double ratio = sum_c / C;
				ratio += sum_m / M;
				ratio /= 2;
				if (ratio > best) {
					best = ratio;
					best_path = path;
					best_type = type;
					if (Double.compare(ratio, 1) == 0) break;
				}
			}
			
			if (Double.compare(best, 1) == 0) {
				System.out.println("find...");
				for (int i = 0; i < N; ++i) {
					if (best_path[i] > 0) {
						int add = best_path[i];
						Goods key = KEY[i];
						Queue<Goods> goods = cagos.get(key);
						for (int j = 0; j < add; ++j) {
							one.add(goods.poll());
						}
						if (goods.isEmpty()) {
							cagos.remove(key);
							cagosCnt.remove(key);
						}
						else {
							cagosCnt.put(key, goods.size());
						}
					}
				}
				
				one.uuid = "" + (res.get(best_type).size() + 1);
				one.id = Integer.parseInt(one.uuid);
				
				res.computeIfAbsent(best_type, k -> new ArrayList<>()).add(one);
			}
			else {
				for (int i = 0; i < N; ++i) {
					if (best_path[i] > 0) {
						int add = best_path[i];
						Goods key = KEY[i];
						Queue<Goods> goods = cagos.get(key);
						for (int j = 0; j < add; ++j) {
							remain.add(goods.poll());
						}
						if (goods.isEmpty()) {
							cagos.remove(key);
							cagosCnt.remove(key);
						}
						else {
							cagosCnt.put(key, goods.size());
						}
					}
				}
			}
		}
		
		Map<String, List<Manifest>> other = powerfulDP(remain, phys, null, null, null);
		for (String key : other.keySet()) {
			if (!res.containsKey(key)) {
				res.put(key, new ArrayList<>());
			}
			res.get(key).addAll(other.get(key));
		}
		return res;
	}
	
	/**
	 * 分配还需要保证物品多样性
	 * @param cagos
	 * @param cagosCnt
	 * @param phys
	 * @return
	 */
	private static Map<String, List<Manifest>> __powerfulDP__(Map<Goods, Queue<Goods>> cagos, Map<Goods, Integer> cagosCnt, Map<String, CPU> phys){
		Map<String, List<Manifest>> res = new HashMap<>();
		for (String type : phys.keySet()) res.put(type, new ArrayList<>());
		while (cagosCnt.size() != 0) {
			Manifest one = new Manifest();
			double best = 0;
			String best_type = "";
			int N = cagos.size();
			int[] MEM = new int[N];
			int[] CPU = new int[N];
			int[] CNT = new int[N];
			int[] best_path = new int[N];
			
			Goods[] KEY = new Goods[N];
			int idx = 0;
			for (Goods key : cagosCnt.keySet()) {
				MEM[idx] = key.mem;
				CPU[idx] = key.core;
				CNT[idx] = cagosCnt.get(key);
				KEY[idx] = key;
				idx ++;
			}
			
			for (String type : phys.keySet()) {
				int C = phys.get(type).core;
				int M = phys.get(type).memory * 1024;
				int[] path = __knapsack__binary(C, M, MEM, CPU, CNT);
				double sum_c = 0;
				double sum_m = 0;
				for (int i = 0; i < N; ++i) {
					sum_c += CPU[i] * path[i];
					sum_m += MEM[i] * path[i];
				}
				double ratio = sum_c / C;
				ratio += sum_m / M;
				ratio /= 2;
				if (ratio > best) {
					best = ratio;
					best_path = path;
					best_type = type;
					if (Double.compare(ratio, 1) == 0) break;
				}
			}
			
			for (int i = 0; i < N; ++i) {
				if (best_path[i] > 0) {
					int add = best_path[i];
					Goods key = KEY[i];
					Queue<Goods> goods = cagos.get(key);
					for (int j = 0; j < add; ++j) {
						one.add(goods.poll());
					}
					if (goods.isEmpty()) {
						cagos.remove(key);
						cagosCnt.remove(key);
					}
					else {
						cagosCnt.put(key, goods.size());
					}
				}
			}
			
			one.uuid = "" + (res.get(best_type).size() + 1);
			one.id = Integer.parseInt(one.uuid);
			
			res.computeIfAbsent(best_type, k -> new ArrayList<>()).add(one);
		}
		return res;
	}
	
	private static boolean[] __ff__(int C, int M, List<Goods> goods) {
		int N = goods.size();
		boolean[] path = new boolean[N];
		int J = 0;
		while (J < N && C >= 0 && M >= 0) {
			int cpu = goods.get(J).core;
			int mem = goods.get(J).mem;
			
			if (cpu < C && mem < M) {
				C -= cpu;
				M -= mem;
				path[J] = true;
			}
			J ++;
		}
		return path;
	}
	
	
	/**
	 * If use allocate ff, please make sure all goods sent into this function should be sorted.
	 * @param goods
	 * @param phys
	 * @param res
	 */
	private static void allocate_ff(List<Goods> goods, Map<String, CPU> phys, Map<String, List<Manifest>> res) {
		Manifest one = new Manifest();
		int N = goods.size();
		double best = 0;
		String best_type = "";
		boolean[] best_path = new boolean[N];
		for (String type : phys.keySet()) {
			int C = phys.get(type).core;
			int M = phys.get(type).memory * 1024;
			boolean[] path = __ff__(C, M, goods);
			double sum_c = 0;
			double sum_m = 0;
			for (int i = 0; i < N; ++i) {
				if (path[i]) {
					sum_c += goods.get(i).core;
					sum_m += goods.get(i).mem;
				}
			}
			double ratio = sum_c / C;
			ratio += sum_m / M;
			ratio /= 2;
			if (ratio > best) {
				best = ratio;
				best_path = path;
				best_type = type;
				if (Double.compare(ratio, 1) == 0) break;
			}
		}
		
		List<Goods> remove = new ArrayList<>();
		for (int i = 0; i < N; ++i) {
			if (best_path[i]) {
				one.add(goods.get(i));
				remove.add(goods.get(i));
			}
		}
		
		one.uuid = "" + (res.get(best_type).size() + 1);
		one.id = Integer.parseInt(one.uuid);
		
		res.computeIfAbsent(best_type, k -> new ArrayList<>()).add(one);
		
		for (Goods obj : remove) goods.remove(obj);
	}
	
	private static Map<String, List<Manifest>> powerfulGreedy(List<Goods> goods, Map<String, CPU> phys){
		Map<String, List<Manifest>> res = new HashMap<>();
		for (String type : phys.keySet()) res.put(type, new ArrayList<>());
		Collections.sort(goods, (a, b) -> (b.core == a.core ? b.mem - a.mem : b.core - a.core));
		while (goods.size() != 0) {
			allocate_ff(goods, phys, res);
		}
		return res;
	}
	
	private static List<Goods> topBottomPoly(List<Goods> cpu_goods, Map<String, CPU> phys){
		Collections.sort(cpu_goods, (a, b) -> Double.compare(a.ratio, b.ratio));
		List<Goods> ret = new ArrayList<>();
		int i = 0, j = cpu_goods.size() - 1;
		int l = cpu_goods.size();
		int c1 = 0;
		int c2 = 0;
		for (int k = 0; k < l; ++k) {
			Goods g = cpu_goods.get(k);
			if (g.ratio <= 1) {
				c1 ++;
			}
			if (g.ratio >= 3) {
				c2 ++;
			}
		}
		System.out.println("ratio < 1 cnt : " + c1);
		System.out.println("ratio > 3 cnt : " + c2);
		
		while (i <= j) {
			if (i == j) {
				ret.add(cpu_goods.get(i));
			}
			else {
				Goods g1 = cpu_goods.get(i);
				Goods g2 = cpu_goods.get(j);
				boolean valid = true;
				for (String type : phys.keySet()) {
					int C = phys.get(type).core;
					int M = phys.get(type).memory * 1024;
					if (g1.core + g2.core > C || g1.mem + g2.mem > M) {
						valid = false;
						break;
					}
				}
				
				if (g1.ratio != g2.ratio && valid) {
					Goods g = new Goods(true);
					g.group(g1);
					g.group(g2);
					ret.add(g);
				}
				else {
					ret.add(g1);
					ret.add(g2);
				}
			}
			i ++;
			j --;
		}
		return ret;
	}
	
	private static String[] getFlavors() {
		String[] f = new String[18];
		for (int i = 1; i <= 18; ++i) {
			f[i - 1] = "flavor" + i;
		}
		return f;
	}
	
	private static Map<String, List<Manifest>> powerfulDP__test(List<Goods> goods, Map<String, CPU> phys, Param param){
		Map<String, List<Manifest>> res = new HashMap<>();
		for (String type : phys.keySet()) res.put(type, new ArrayList<>());
		
		List<Goods> cpu_goods = new ArrayList<>();
		while (goods.size() != 0) {
			double best = 0;
			String best_type = "";
			List<Integer> best_path = new ArrayList<>();
			
			double cpu_ratio = 0;
			for (String type : phys.keySet()) {
				int C = phys.get(type).core;
				int M = phys.get(type).memory * 1024;
				List<Integer> path = __knapsack__01(C, M, goods); //非常耗时
				double sum_c = 0;
				double sum_m = 0;
				for (int pos : path) {
					sum_c += goods.get(pos).core;
					sum_m += goods.get(pos).mem;
				}
				double ratio = sum_c / C;
				ratio += sum_m / M;
				ratio /= 2;
				if (ratio > best) {
					cpu_ratio = sum_m / M;
					best = ratio;
					best_path = path;
					best_type = type;
					if (Double.compare(ratio, 1) == 0) break;
				}
			}
			
			if (cpu_ratio < 0.93) {
				for (int pos : best_path) {
					cpu_goods.add(goods.get(pos)); // collections
				}
				
				for (int pos : best_path) {
					goods.remove(pos);
				}
			}
			else {
				Manifest one = new Manifest("" + (res.get(best_type).size() + 1), 
						phys.get(best_type).core, phys.get(best_type).memory * 1024);
				
				for (int pos : best_path) {
					one.add(goods.get(pos));
				}
				
				res.computeIfAbsent(best_type, k -> new ArrayList<>()).add(one);
				
				for (int pos : best_path) {
					goods.remove(pos);
				}
			}
		}
		
		if (cpu_goods.size() > 0) {
			System.out.println(cpu_goods.size());
			String[] flavors = getFlavors();
			for (String type : res.keySet()){
				int C = phys.get(type).core;
				int M = phys.get(type).memory * 1024;
				for (Manifest val : res.get(type)) {
					for (String flavor : flavors) {
						int cpu = param.virtuals.get(flavor).core;
						int mem = param.virtuals.get(flavor).memory;
						if (val.remove(flavor, cpu, mem)) {
							double ratio = val.scoreCPU(C);
							ratio += val.scoreMEM(M);
							if (ratio / 2 >= 0.98) {
								cpu_goods.add(new Goods(flavor, cpu, mem));
							}
							else {
								val.add(new Goods(flavor, cpu, mem));
							}
						}
					}
				}
			}
		}
		
		System.out.println(cpu_goods.size());
		List<Goods> grouped = topBottomPoly(cpu_goods, phys);
//		List<Goods> grouped = cpu_goods;
		
		while (grouped.size() != 0) {
			Manifest one = new Manifest();
			double best = 0;
			String best_type = "";
			List<Integer> best_path = new ArrayList<>();
			
			for (String type : phys.keySet()) {
				int C = phys.get(type).core;
				int M = phys.get(type).memory * 1024;
				List<Integer> path = __knapsack__nopt(C, M, grouped, true); //非常耗时
				double sum_c = 0;
				double sum_m = 0;
				for (int pos : path) {
					sum_c += grouped.get(pos).core;
					sum_m += grouped.get(pos).mem;
				}
				double ratio = sum_c / C;
				ratio += sum_m / M;
				ratio /= 2;
				if (ratio > best) {
					best = ratio;
					best_path = path;
					best_type = type;
					if (Double.compare(ratio, 1) == 0) break;
				}
			}
			
			for (int pos : best_path) {
				one.add(grouped.get(pos));
			}
			
			one.uuid = "" + (res.get(best_type).size() + 1);
			one.id = Integer.parseInt(one.uuid);
			
			res.computeIfAbsent(best_type, k -> new ArrayList<>()).add(one);
			
			for (int pos : best_path) {
				grouped.remove(pos);
			}
		}
		
		return res;
	}
	
	static class ManiAndScore implements Comparable<ManiAndScore>{
		
		Manifest mani;
		double score;
		ManiAndScore(Manifest mani, double score){
			this.mani = mani;
			this.score = score;
		}
		@Override
		public int compareTo(ManiAndScore that) {
			return Double.compare(this.score, that.score);
		}
	}
	
	private static Map<String, List<Manifest>> powerfulDP(List<Goods> goods, Map<String, CPU> phys, Param param, Map<String, Integer> other, Map<String, Integer> fixed){
		Map<String, List<Manifest>> res = new HashMap<>();
		for (String type : phys.keySet()) res.put(type, new ArrayList<>());
		Collections.sort(goods, (a, b) -> (a.core == b.core ? b.mem - a.mem : b.core - a.core));
		
		List<Goods> canBeChoosen =new ArrayList<>();
		for (String key : other.keySet()) {
			for (int i = 0; i < other.get(key); ++i) {
				int cpu = param.virtuals.get(key).core;
				int mem = param.virtuals.get(key).memory;
				canBeChoosen.add(new Goods(key, cpu, mem));
			}
		}
		Collections.sort(canBeChoosen, (a, b) -> (a.core == b.core ? b.mem - a.mem : b.core - a.core));
		while (goods.size() != 0) {
			double best = 0;
			String best_type = "";
			List<Integer> best_path = new ArrayList<>();
			
			int len = goods.size();
			for (String type : phys.keySet()) {
				int C = phys.get(type).core;
				int M = phys.get(type).memory * 1024;
				List<Integer> path = __knapsack__01(C, M, goods, canBeChoosen); //非常耗时
				double sum_c = 0;
				double sum_m = 0;
				for (int pos : path) {
					if (pos < len) {
						sum_c += goods.get(pos).core;
						sum_m += goods.get(pos).mem;
					}
					else {
						sum_c += canBeChoosen.get(pos - len).core;
						sum_m += canBeChoosen.get(pos - len).mem;
					}
				}
				
//				double ratio = sum_c / C;
//				ratio += sum_m / M;
//				ratio = ratio / 2;
				
				double ratio = sum_c / C;
				ratio *= sum_m / M;
				ratio = Math.sqrt(ratio);
				
				if (ratio > best) {
					best = ratio;
					best_path = path;
					best_type = type;
					if (Double.compare(ratio, 1) == 0) break;
				}
			}
			
			Manifest one = new Manifest("" + (res.get(best_type).size() + 1), 
					phys.get(best_type).core, phys.get(best_type).memory * 1024);
			one.type = best_type;
			
			for (int pos : best_path) {
				if (pos < len)
					one.add(goods.get(pos));
				else {
					Goods tmp = canBeChoosen.get(pos - len);
					one.add(tmp);
					fixed.put(tmp.tag, fixed.getOrDefault(tmp.tag, 0) + 1);
				}
			}
			
			res.computeIfAbsent(best_type, k -> new ArrayList<>()).add(one);
			
			for (int pos : best_path) {
				if (pos < goods.size()) {
					goods.remove(pos);
				}
				else {
					canBeChoosen.remove(pos - len);
				}
			}
		}
		
		Set<String> all = new HashSet<>();
		for (String flavor : fixed.keySet()) all.add(flavor);
		Queue<ManiAndScore> queue = new PriorityQueue<>();
		
		for (String type : res.keySet()) {
			int C = phys.get(type).core;
			int M = phys.get(type).memory * 1024;
			for (Manifest mani : res.get(type)) {
				double ratio = mani.scoreCPU(C);
				ratio *= mani.scoreMEM(M);
				ratio = Math.sqrt(ratio);
				queue.offer(new ManiAndScore(mani, ratio));
			}
		}
		
		while (!queue.isEmpty()) {
			Manifest mani = queue.poll().mani;
			
			int C = phys.get(mani.type).core;
			int M = phys.get(mani.type).memory * 1024;
			
			double best_ratio = 0;
			String best_flavor = "";
			
			for (String flavor : all) {
				if (mani.memo == 0 || mani.core == 0) continue;
				int cpu = param.virtuals.get(flavor).core;
				int mem = param.virtuals.get(flavor).memory;
				
				double sum_core = C - mani.core + cpu;
				double sum_memo = M - mani.memo + mem;
				
				if (sum_core > C || sum_memo > M) continue;
				
				double ratio = sum_core / C;
				ratio *= sum_memo / M;
				ratio = Math.sqrt(ratio);
				
				if (ratio > best_ratio) {
					best_ratio = ratio;
					best_flavor = flavor;
				}
			}
			
			if (!best_flavor.isEmpty()) {
				int cpu = param.virtuals.get(best_flavor).core;
				int mem = param.virtuals.get(best_flavor).memory;
				mani.add(new Goods(best_flavor, cpu, mem));
				queue.offer(new ManiAndScore(mani, best_ratio));
				fixed.put(best_flavor, fixed.getOrDefault(best_flavor, 0) + 1);
				all.remove(best_flavor);
			}
		}
		return res;
	}
	
	/**
	 * CPU 动规分配
	 * @param goods
	 * @param all_core
	 * @param all_memo
	 * @param idx
	 * @return
	 */
	private static List<Manifest> cpuDP(List<Goods> goods, int all_core, int all_memo, int idx){
		List<Manifest> mani = new ArrayList<>();
		while (goods.size() != 0) {
			Manifest one = new Manifest(idx + "");
			mem_core = new HashMap<>();
//			Info next = dynamicalAllocateCore(all_core, all_memo, goods, 0);
			List<Integer> path = knapsack(all_core, all_memo, goods);
			for (int pos : path) {
				one.add(goods.get(pos));
			}
			for (int pos : path) {
				goods.remove(pos);
			}
			
			mani.add(one);
			idx++;
		}
		return mani;
	}
	
	/**
	 * MEM 动规分配
	 * @param goods
	 * @param all_core
	 * @param all_memo
	 * @param idx
	 * @return
	 */
	private static List<Manifest> memDP(List<Goods> goods, int all_core, int all_memo, int idx){
		List<Manifest> mani = new ArrayList<>();
		while (goods.size() != 0) {
			Manifest one = new Manifest(idx + "");
			mem_memo = new HashMap<>();
			List<Integer> path = knapsack(all_core, all_memo, goods);
			for (int pos : path) {
				one.add(goods.get(pos));
			}
			for (int pos : path) {
				goods.remove(pos);
			}
			mani.add(one);
			idx++;
		}
		return mani;
	}
	
	
	
	public static int[][] getCagos(Map<String, Integer> goods, Map<String, CPU> virtuals) {
		int[][] cagos = new int[20][3];  // 0 cpu, 1 mem, 2 cnt
		for (String key : goods.keySet()) {
			int id = Integer.parseInt(key.substring(6));
			cagos[id] = new int[] {virtuals.get(key).core, virtuals.get(key).memory, goods.get(key)};
		}
		return cagos;
	}
	
	public static List<Goods> transfer(Map<String, Integer> goods, Map<String, CPU> virtuals){
		List<Goods> all = new ArrayList<>();
		for (String tag : goods.keySet()) {
			if (virtuals.containsKey(tag)) {
				CPU info = virtuals.get(tag);
				for (int i = 0; i < goods.get(tag); ++i) {
					all.add(new Goods(tag, info.core, info.memory));
				}
			}
		}
		return all;
	}
	
	static class Error{
		String type;
		double error;

		public Error(String type, double error) {
			this.type  = type;
			this.error = error;
		}
		
		public boolean terminate(double eps) {
			return Double.compare(error, eps) < 0;
		}
	}
	
	public static Error choosePhys(double ratio, String[] sorted_index, Map<String, CPU> physicals) {
		int n = sorted_index.length;
		for (int i = n - 1; i >= 0; --i) {
			if (ratio < physicals.get(sorted_index[i]).ratio) {
				return new Error(sorted_index[i], physicals.get(sorted_index[i]).ratio - ratio);
			}
		}
		return null;
	}
	
	
	public static String choosePhysType(double ratio, String[] sorted_index, Map<String, CPU> physicals) {
		int n = sorted_index.length;
		double min = 0x3f3f3f3f;
		String ans = "";
		for (int i = 0; i < n; ++i) {
			double diff = Math.abs(physicals.get(sorted_index[i]).ratio - ratio);
			if (diff < min) {
				min = diff;
				ans = sorted_index[i];
			}
		}
		
		
//		min = 0x3f3f3f3f;
//		for (int i = 0; i < n; ++i) {
//			double diff = Math.abs(physicals.get(sorted_index[i]).ratio - ratio);
//			if (diff < min && Double.compare(physicals.get(sorted_index[i]).ratio, ratio) >= 0) {
//				min = diff;
//				ans = sorted_index[i];
//			}
//		}
		
//		return "General";
//		return "High-Performance";
//		return "Large-Memory";
		return ans;
	}
	
	
	public static String[] physRatioSortedIndex(Map<String, CPU> physicals){	
		String[] index = new String[physicals.size()];
		List<CPU> vals = new ArrayList<>();
		for (String key : physicals.keySet()) vals.add(physicals.get(key));
		Collections.sort(vals, (a, b) -> (Double.compare(a.ratio, b.ratio)));
		int _idx = 0;
		for (CPU cpu : vals) {
			index[_idx++] = cpu.type;
		}
		return index;
	}
	
	public static Map<String, Double> medianErrorRange(Map<String, CPU> physicals, String[] sorted_index) {
		Map<String, Double> mem = new HashMap<>();
		for (int i = 0; i < sorted_index.length; ++i) {
			mem.put(sorted_index[i], physicals.get(sorted_index[i]).ratio - 
					(physicals.get(sorted_index[i]).ratio + 
					(i >= 1 ? physicals.get(sorted_index[i - 1]).ratio : 0.0)) / 2);
		}
		return mem;
	}
	
	
	static class Bucket{
		int one;
		int two;
		Goods good;
		String type;
		
		public Bucket(int one, int two, Goods good, String type) {
			this.one = one;
			this.two = two;
			this.good = good;
			this.type = type;
		}
	}
	
	private static boolean _valid_or(int core, int memo, Map<String, CPU> physicals) {
		for (String type : physicals.keySet()) {
			int C = physicals.get(type).core;
			int M = physicals.get(type).memory * 1024;
			if (core <= C && memo <= M) return true;
		}
		return false;
	}
	
	private static boolean _valid_and(int core, int memo, Map<String, CPU> physicals) {
		for (String type : physicals.keySet()) {
			int C = physicals.get(type).core;
			int M = physicals.get(type).memory * 1024;
			if (core > C && memo > M) return false;
		}
		return true;
	}
	
	/**
	 * 同等级交叉
	 * @param q1
	 * @param q2
	 * @param q4
	 * @param sorted_index
	 * @param physicals
	 * @return
	 */
	public static Bucket bestGrouped(Goods q1, Goods q2, Goods q4, String[] sorted_index, Map<String, CPU> physicals) {
		double ratio = 100;
		Bucket ret = null;
		
		// 2 1
		if (q1 != null && q2 != null) {
			double download = (q1.mem + q2.mem) / 1024 / (q1.core + q2.core);
			String choosen = choosePhysType(download, sorted_index, physicals);
			int phy_core = physicals.get(choosen).core;
			int pyh_memo = physicals.get(choosen).memory;
//			if (_valid_and(q1.core + q2.core, q1.mem + q2.mem, physicals) && q1.core == q2.core) {
			if (q1.mem + q2.mem <= pyh_memo * 1024 && q1.core + q2.core <= phy_core && q1.core == q2.core) {
				if (download < ratio) {
					ratio = download;
					Goods grouped = new Goods(true);
					grouped.group(q1);
					grouped.group(q2);
					ret = new Bucket(1, 2, grouped, choosen);
				}
			}
		}
		
		// 4 2
		if (q4 != null && q2 != null) {
			double download = (q4.mem + q2.mem) / 1024 / (q4.core + q2.core);
			String choosen = choosePhysType(download, sorted_index, physicals);
			int phy_core = physicals.get(choosen).core;
			int pyh_memo = physicals.get(choosen).memory;

//			if (_valid_and(q2.core + q4.core, q2.mem + q4.mem, physicals) && q2.core == q4.core) {
			if (q4.mem + q2.mem <= pyh_memo * 1024 && q4.core + q2.core <= phy_core && q4.core == q2.core) {
				if (download < ratio) {
					ratio = download;
					Goods grouped = new Goods(true);
					grouped.group(q4);
					grouped.group(q2);
					ret = new Bucket(4, 2, grouped, choosen);
				}
			}
			
		}
		
		// 4 1
		if (q4 != null && q1 != null) {
			double download = (q4.mem + q1.mem) / 1024 / (q4.core + q1.core);
			String choosen = choosePhysType(download, sorted_index, physicals);
			int phy_core = physicals.get(choosen).core;
			int pyh_memo = physicals.get(choosen).memory;
			

//			if (_valid_and(q1.core + q4.core, q1.mem + q4.mem, physicals) && q1.core == q4.core) {
			if (q4.mem + q1.mem <= pyh_memo * 1024 && q4.core + q1.core <= phy_core && q1.core == q4.core) {
				if (download < ratio) {
					ratio = download;
					Goods grouped = new Goods(true);
					grouped.group(q4);
					grouped.group(q1);
					ret = new Bucket(1, 4, grouped, choosen);
				}
			}
			
		}
		return ret;
	}
	
	// 思路1 同等级交叉
	public static Map<String, List<Goods>> groupWithTarget(Map<String, CPU> physicals, List<Goods> goods) {
		Map<String, List<Goods>> mem = new HashMap<>();
		
		// cpu sorted index
		String[] sorted_index = physRatioSortedIndex(physicals);
		
		// grouped
		Queue<Goods> queue4 = new PriorityQueue<>((a, b) -> (b.core - a.core));
		Queue<Goods> queue2 = new PriorityQueue<>((a, b) -> (b.core - a.core));
		Queue<Goods> queue1 = new PriorityQueue<>((a, b) -> (b.core - a.core));
		for (Goods cago : goods) {
			int ratio = cago.mem / 1024 / cago.core;
			if (ratio == 4) {
				queue4.offer(cago);
			}
			else if (ratio == 2) {
				queue2.offer(cago);
			}
			else if (ratio == 1) {
				queue1.offer(cago);
			}
		}
		
		
		Set<Double> set = new HashSet<>();
		// 1. 尽量降低 ratio
		// 2. 尽量同级别进行ratio, 4 - > 1, 2 -> 1
		// 3. 尽量选不同的ratio
		outer: while (!(queue4.isEmpty() && queue2.isEmpty() && queue1.isEmpty())) {
			Goods q4 = queue4.isEmpty() ? null : queue4.peek();
			Goods q2 = queue2.isEmpty() ? null : queue2.peek();
			Goods q1 = queue1.isEmpty() ? null : queue1.peek();
			
			Bucket ret = bestGrouped(q1, q2, q4, sorted_index, physicals);
			
			if (ret == null) {
				if (q4 != null) {
					String type = choosePhysType(q4.ratio, sorted_index, physicals);
					mem.computeIfAbsent(type, k -> new ArrayList<>()).add(q4);
					queue4.poll();
					continue outer;
				}
				
				if (q2 != null) {
					String type = choosePhysType(q2.ratio, sorted_index, physicals);
					mem.computeIfAbsent(type, k -> new ArrayList<>()).add(q2);
					queue2.poll();
					continue outer;
				}
				if (q1 != null) {
					String type = choosePhysType(q1.ratio, sorted_index, physicals);
					mem.computeIfAbsent(type, k -> new ArrayList<>()).add(q1);
					queue1.poll();
					continue outer;
				}
			}
			else {
				int one = ret.one;
				int two = ret.two;
				if (one == 1) {
					queue1.poll();
				}
				else if (one == 2) {
					queue2.poll();
				}
				else {
					queue4.poll();
				}
				
				if (two == 1) {
					queue1.poll();
				}
				else if (two == 2) {
					queue2.poll();
				}
				else {
					queue4.poll();
				}
				
				// 采用多级group
				set.add(ret.good.ratio);
				if (ret.good.ratio > physicals.get(sorted_index[sorted_index.length - 1]).ratio) queue4.offer(ret.good);
				else if (ret.good.ratio < physicals.get(sorted_index[0]).ratio) queue1.offer(ret.good);
				else mem.computeIfAbsent(ret.type, k -> new ArrayList<>()).add(ret.good);
			}
		}
		return mem;
	}
	
	
	public static List<Goods> groupedGoods(Map<String, CPU> physicals, List<Goods> goods) {
		List<Goods> ans = new ArrayList<>();
		
		// cpu sorted index
		String[] sorted_index = physRatioSortedIndex(physicals);
		
		// grouped
		Queue<Goods> queue4 = new PriorityQueue<>((a, b) -> (b.core - a.core));
		Queue<Goods> queue2 = new PriorityQueue<>((a, b) -> (b.core - a.core));
		Queue<Goods> queue1 = new PriorityQueue<>((a, b) -> (b.core - a.core));
		for (Goods cago : goods) {
			int ratio = cago.mem / 1024 / cago.core;
			if (ratio == 4) {
				queue4.offer(cago);
			}
			else if (ratio == 2) {
				queue2.offer(cago);
			}
			else if (ratio == 1) {
				queue1.offer(cago);
			}
		}
		
		
		Set<Double> set = new HashSet<>();
		// 1. 尽量降低 ratio
		// 2. 尽量同级别进行ratio, 4 - > 1, 2 -> 1
		// 3. 尽量选不同的ratio
		outer: while (!(queue4.isEmpty() && queue2.isEmpty() && queue1.isEmpty())) {
			Goods q4 = queue4.isEmpty() ? null : queue4.peek();
			Goods q2 = queue2.isEmpty() ? null : queue2.peek();
			Goods q1 = queue1.isEmpty() ? null : queue1.peek();
			
			Bucket ret = bestGrouped(q1, q2, q4, sorted_index, physicals);
			
			if (ret == null) {
				if (q4 != null) {
					ans.add(q4);
					queue4.poll();
					continue outer;
				}
				
				if (q2 != null) {
					ans.add(q2);
					queue2.poll();
					continue outer;
				}
				if (q1 != null) {
					ans.add(q1);
					queue1.poll();
					continue outer;
				}
			}
			else {
				int one = ret.one;
				int two = ret.two;
				if (one == 1) {
					queue1.poll();
				}
				else if (one == 2) {
					queue2.poll();
				}
				else {
					queue4.poll();
				}
				
				if (two == 1) {
					queue1.poll();
				}
				else if (two == 2) {
					queue2.poll();
				}
				else {
					queue4.poll();
				}
				
				// 采用多级group
				set.add(ret.good.ratio);
				if (ret.good.ratio > physicals.get(sorted_index[sorted_index.length - 1]).ratio) queue4.offer(ret.good);
				else if (ret.good.ratio < physicals.get(sorted_index[0]).ratio) queue1.offer(ret.good);
				else ans.add(ret.good);
			}
		}
		return ans;
		
	}
	
	private static double eval(Param param, List<Manifest> allocate, int core, int memo) {
		double ratio = 0;
		int sum_core = allocate.size() * core;
		double evl_core = 0;
		for (Manifest menu : allocate) {
			Map<String, Integer> list = menu.getList();
			for (String flavor : list.keySet()) {
				evl_core += param.virtuals.get(flavor).core * list.get(flavor);
			}
		}
		ratio += evl_core / sum_core;
		int sum_memo = allocate.size() * memo;
		double evl_memo = 0;
		for (Manifest menu : allocate) {
			Map<String, Integer> list = menu.getList();
			for (String flavor : list.keySet()) {
				evl_memo += param.virtuals.get(flavor).memory * list.get(flavor);
			}
		}
		ratio += evl_memo / sum_memo;
		return ratio / 2;
	}
	
	/**
	 * 1. 核数总和不能超过一块物理机核数
	 * 2. 内存总和不能超过一块物理机内存
	 * @param physical
	 * @param goods
	 * @param target
	 * @return
	 */
	public static Map<String, List<Manifest>> allocateCPUAndMEM(Param param, List<Goods> goods, Map<String, Integer> other, Map<String, Integer> fixed){
		List<Goods> grouped = groupedGoods(param.physicalCpusMap, goods);
		return powerfulDP(grouped, param.physicalCpusMap, param, other, fixed);
		
//		if (grouped.size() <= 4500) {
//			return powerfulDP(grouped, param.physicalCpusMap);
//		}
//		else {
//			return powerfulGreedy(grouped, param.physicalCpusMap);
//		}
	}
	
	public static int[] __knapsack__binary(int all_cpu, int all_mem, int[] MEM, int[] CPU, int[] CNT){
		all_mem /= 1024;
		int N = MEM.length;
		int[] ret = new int[N];
		int[][] tbl = new int[all_cpu + 1][all_mem + 1];
		int[][][] path = new int[N + 1][all_cpu + 1][all_mem + 1];
		
		for (int i = 1; i <= N; ++i) {
			int cpu = CPU[i - 1];
			int mem = MEM[i - 1] / 1024;
			int num = CNT[i - 1];
			
			for (int k = 1; num > 0; k <<= 1) {
				int mul = Math.min(k, num);
				for (int u = all_cpu; u >= cpu * mul; --u) {
					for (int v = all_mem; v >= mem * mul; --v) {
						if (tbl[u][v] < tbl[u - cpu * mul][v - mem * mul] + cpu * mul + mem * mul) {
							tbl[u][v] = tbl[u - cpu * mul][v - mem * mul] + cpu * mul + mem * mul;
							path[i - 1][u][v] = mul;
						}
					}
				}
				num -= mul;
			}
			
		}
		for (int i = N - 1, u = all_cpu, v = all_mem; i >= 0 && u > 0 && v > 0; --i) {
			if (path[i][u][v] > 0) {
				int mul = path[i][u][v];
				int cpu = CPU[i];
				int mem = MEM[i] / 1024;
				u -= mul * cpu; v -= mul * mem;
				ret[i] += mul;
			}
		}
		return ret;
	}
	
	public static List<Integer> __knapsack__nopt(int all_cpu, int all_mem, List<Goods> goods, boolean useRate){
		all_mem /= 1024;
		int N = goods.size();
		List<Integer> ans = new ArrayList<>();
		
		float rate = 1.0f;
		if (useRate) rate = all_mem * 1.0f / all_cpu;
		
		float[][] tbl = new float[all_cpu + 1][all_mem + 1];
		boolean[][][] path = new boolean[N][all_cpu + 1][all_mem + 1];
		
		for (int i = 1; i <= N; ++i) {
			Goods g = goods.get(i - 1);
			int cpu = g.core;
			int mem = g.mem / 1024;
			for (int u = all_cpu; u >= cpu; --u) {
				for (int v = all_mem; v >= mem; --v) {
					if (tbl[u][v] < tbl[u - cpu][v - mem] + cpu * rate + mem) {
						tbl[u][v] = tbl[u - cpu][v - mem] + cpu * rate + mem;
						path[i - 1][u][v] = true;
					}
				}
			}
		}
		
//		while (all_cpu >= 0 && all_mem >= 0) {
//			int pos = path[all_cpu][all_mem];
//			if (pos == -1) break;
//			Goods g = goods.get(pos);
//			int cpu = g.core;
//			int mem = g.mem / 1024;
//			all_cpu -= cpu; all_mem -= mem;
//			ret.add(pos);
//		}
		
		for (int i = N - 1, u = all_cpu, v = all_mem; i >= 0; --i) {
			if (path[i][u][v]) {
				Goods g = goods.get(i);
				int mem = g.mem / 1024;
				u = u - g.core;
				v = v - mem;
				ans.add(i);
			}
		}
		return ans;
	}
	
	public static List<Integer> __knapsack__01(int all_cpu, int all_mem, List<Goods> goods, List<Goods> canBeChoosen){
		all_mem /= 1024;
		int N = goods.size() + canBeChoosen.size();
		Set<Integer> ret = new HashSet<>();
		
		int[][] tbl = new int[all_cpu + 1][all_mem + 1];
		int[][] path = new int[all_cpu + 1][all_mem + 1];
		for (int i = 0; i < path.length; ++i) Arrays.fill(path[i], -1);
		
		int full = all_cpu + all_mem;
		
		int max = 0;
		int cnt = 0;
		
		for (int i = 1; i <= N; ++i) {
			Goods g = null;
			if (i - 1 < goods.size())
				g = goods.get(i - 1);
			else 
				g = canBeChoosen.get(i - 1 - goods.size());
			
			int cpu = g.core;
			int mem = g.mem / 1024;
			for (int u = all_cpu; u >= cpu; --u) {
				for (int v = all_mem; v >= mem; --v) {
					if (tbl[u][v] < tbl[u - cpu][v - mem] + cpu + mem) {
						tbl[u][v] = tbl[u - cpu][v - mem] + cpu + mem;
						path[u][v] = i - 1;
					}
					if (u == all_cpu && v == all_mem && full == tbl[u][v]) {
						u = cpu - 1;
						v = mem - 1;
						i = N + 1;
						break;
					}
					if (u == all_cpu && v == all_mem) {
						int val = tbl[u][v];
						if (val >= max) {
							if (val > max) {
								max = val;
								cnt = 0;
							}
							else {
								cnt ++;
								if (cnt == 2 * N / 3) {
									u = cpu - 1;
									v = mem - 1;
									i = N + 1;
									break;
								}
							}
						}
					}
				}
			}
		}
		
		while (all_cpu >= 0 && all_mem >= 0) {
			int pos = path[all_cpu][all_mem];
			if (pos == -1) break;
			Goods g = null;
			if (pos < goods.size())
				g = goods.get(pos);
			else
				g = canBeChoosen.get(pos - goods.size());
			int cpu = g.core;
			int mem = g.mem / 1024;
			all_cpu -= cpu; all_mem -= mem;
			ret.add(pos);
		}
		
		List<Integer> ans = new ArrayList<>(ret);
		Collections.sort(ans, (a, b) -> (b - a));
		return ans;
	}
	
	public static List<Integer> __knapsack__01(int all_cpu, int all_mem, List<Goods> goods){
		all_mem /= 1024;
		int N = goods.size();
		Set<Integer> ret = new HashSet<>();
		
		int[][] tbl = new int[all_cpu + 1][all_mem + 1];
		int[][] path = new int[all_cpu + 1][all_mem + 1];
		for (int i = 0; i < path.length; ++i) Arrays.fill(path[i], -1);
		
		int full = all_cpu + all_mem;
		
		int max = 0;
		int cnt = 0;
		
		for (int i = 1; i <= N; ++i) {
			Goods g = goods.get(i - 1);
			int cpu = g.core;
			int mem = g.mem / 1024;
			for (int u = all_cpu; u >= cpu; --u) {
				for (int v = all_mem; v >= mem; --v) {
					if (tbl[u][v] < tbl[u - cpu][v - mem] + cpu + mem) {
						tbl[u][v] = tbl[u - cpu][v - mem] + cpu + mem;
						path[u][v] = i - 1;
					}
					if (u == all_cpu && v == all_mem && full == tbl[u][v]) {
						u = cpu - 1;
						v = mem - 1;
						i = N + 1;
						break;
					}
					if (u == all_cpu && v == all_mem) {
						int val = tbl[u][v];
						if (val >= max) {
							if (val > max) {
								max = val;
								cnt = 0;
							}
							else {
								cnt ++;
								if (cnt == 2 * goods.size() / 3) {
									u = cpu - 1;
									v = mem - 1;
									i = N + 1;
									break;
								}
							}
						}
					}
				}
			}
		}
		
		while (all_cpu >= 0 && all_mem >= 0) {
			int pos = path[all_cpu][all_mem];
			if (pos == -1) break;
			Goods g = goods.get(pos);
			int cpu = g.core;
			int mem = g.mem / 1024;
			all_cpu -= cpu; all_mem -= mem;
			ret.add(pos);
		}
		
		List<Integer> ans = new ArrayList<>(ret);
		Collections.sort(ans, (a, b) -> (b - a));
		return ans;
	}
	
	
	@Deprecated
	public static List<Integer> knapsack(int all_cpu, int all_mem, List<Goods> goods){
		all_mem /= 1024;
		int N = goods.size();
		List<Integer> path = new ArrayList<>();
		
		int[][][] tbl = new int[N + 1][all_cpu + 1][all_mem + 1];
		for (int i = 0; i <= N; ++i) {
			for (int u = 0; u <= all_cpu; ++u) {
				for (int v = 0; v <= all_mem; ++v) {
					if (i == 0 || u == 0 || v == 0) {
						tbl[i][u][v] = 0;
					}
					else {
						Goods g = goods.get(i - 1);
						int cpu = g.core;
						int mem = g.mem / 1024;
						if (cpu <= u && mem <= v) {
							tbl[i][u][v] = Math.max(cpu + mem + tbl[i - 1][u - cpu][v - mem], tbl[i - 1][u][v]);
						}
						else {
							tbl[i][u][v] = tbl[i - 1][u][v];
						}
						
					}
				}
			}
		}
		
		for (int i = N, u = all_cpu, v = all_mem, res = tbl[N][all_cpu][all_mem]; i > 0 && res > 0; --i) {
			if (res == tbl[i - 1][u][v]) continue; // 说明当前物品没有被选择
			else {
				Goods g = goods.get(i - 1);
				int mem = g.mem / 1024;
				res = res - g.core - mem;
				u = u - g.core;
				v = v - mem;
				path.add(i - 1);
			}
		}
		
		return path;
	}
	
	/**
	 * 1. 核数总和不能超过一块物理机核数
	 * 2. 内存总和不能超过一块物理机内存
	 * @param physical
	 * @param goods
	 * @param target
	 * @return
	 */
	@Deprecated
	public static List<Manifest> allocate(CPU physical, List<Goods> goods, String target){
		int all_core = physical.core;
		int all_memo = physical.memory * 1024;
		
		
		if (target.equals("CPU")) { // CPU Optimization
//			Collections.sort(goods, (a, b) -> (b.core - a.core));
//			return doAllocate(all_core, all_memo, goods);
//			return cpuDP(goods, all_core, all_memo, 1);
			
			return priorityCPUAllocate(all_core, all_memo, goods);
			
//			List<Manifest> manifests1 = priorityCPUAllocate(all_core, all_memo, goods);
//			List<Manifest> manifests2 = annealingCPU(all_core, all_memo, goods);
//			
//			double score1 = core_sum * 1.0 / (manifests1.size() * all_core);
//			double score2 = core_sum * 1.0 / (manifests2.size() * all_core);
//			
//			return score1 > score2 ? manifests1 : manifests2;
		}
		else { // MEM Optimization
//			Collections.sort(goods, (a, b) -> (b.mem - a.mem));
//			return doAllocate(all_core, all_memo, goods);
//			return memDP(goods, all_core, all_memo, 1);
			
			return priorityMEMAllocate(all_core, all_memo, goods);
			
//			List<Manifest> manifests1 = priorityMEMAllocate(all_core, all_memo, goods);
//			List<Manifest> manifests2 = annealingMEM(all_core, all_memo, goods);
//			
//			double score1 = memo_sum * 1.0 / (manifests1.size() * all_memo);
//			double score2 = memo_sum * 1.0 / (manifests2.size() * all_memo);
//			
//			return score1 > score2 ? manifests1 : manifests2;
		}
	}
}
