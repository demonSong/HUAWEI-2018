package com.elasticcloudservice.predict;

import java.util.HashMap;
import java.util.Map;

public class GroupedGoods implements Comparable<GroupedGoods>{
	
	private static int uuid = 0;
	
	
	private String type;
	private boolean done;
	
	public String tag;
	public int core;
	public int memo;
	public double ratio;
	public Map<String, Integer> flavorCnt;
	
	public GroupedGoods(){
		uuid ++;
		this.tag = "grouped" + uuid;
		this.core = 0;
		this.memo = 0;
		this.ratio = 0;
		this.done = false;
		flavorCnt = new HashMap<>();
	}
	
	public GroupedGoods(GroupedGoods clone){
		this.tag = clone.tag;
		this.core = clone.core;
		this.memo = clone.memo;
		this.ratio = clone.ratio;
		this.done = clone.done;
		this.flavorCnt = new HashMap<>();
		for (String key : clone.flavorCnt.keySet()) this.flavorCnt.put(key, clone.flavorCnt.get(key));
	}
	
	public GroupedGoods(Goods goods){
		uuid ++;
		this.tag = "grouped" + uuid;
		this.core = goods.core;
		this.memo = goods.mem;
		this.ratio = memo * 1.0 / (core * 1024);
		this.done = false;
		
		flavorCnt = new HashMap<>();
		flavorCnt.put(goods.tag, 1);
	}
	
	public void group(Goods goods) {
		this.core += goods.core;
		this.memo += goods.mem;
		this.ratio = memo * 1.0 / (core * 1024);
		flavorCnt.put(goods.tag, flavorCnt.getOrDefault(goods.tag, 0) + 1);
	}
	
	public void done(String type) {
		this.done = true;
		this.type = type;
	}
	
	public String type() {
		return this.type;
	}

	@Override
	public int compareTo(GroupedGoods o) {
		return Double.compare(o.ratio, this.ratio);
	}

	@Override
	public String toString() {
		return "GroupedGoods [type=" + type + ", tag=" + tag + ", core=" + core + ", memo=" + memo + ", ratio=" + ratio
				+ ", flavorCnt=" + flavorCnt + "]";
	}
}
