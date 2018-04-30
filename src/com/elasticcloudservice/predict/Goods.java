package com.elasticcloudservice.predict;

import java.util.HashMap;
import java.util.Map;

import javax.swing.plaf.basic.BasicInternalFrameTitlePane.IconifyAction;

public class Goods{
	
	private static int uuid = 0;
	private boolean grouped;
	public Map<String, Integer> flavorCnt;
	
	public String tag;
	public int core;
	public int mem;
	
	public double ratio;
	
	public Goods(boolean grouped) {
		uuid ++;
		this.grouped = true;
		this.tag = "grouped" + uuid;
		this.flavorCnt = new HashMap<>();
	}
	
	public void group(Goods goods) {
		this.core += goods.core;
		this.mem += goods.mem;
		this.ratio = mem * 1.0 / (core * 1024);
		if (goods.isGrouped()) {
			for (String key : goods.flavorCnt.keySet()) {
				this.flavorCnt.put(key, 
					this.flavorCnt.getOrDefault(key, 0) + goods.flavorCnt.get(key));
			}
		}
		else {
			this.flavorCnt.put(goods.tag, this.flavorCnt.getOrDefault(goods.tag, 0) + 1);
		}
	}
	
	public boolean isGrouped() {
		return this.grouped;
	}
	
	public Goods(String tag, int core, int mem){
		this.tag = tag;
		this.core = core;
		this.mem = mem;
		this.ratio = mem * 1.0  / (core * 1024);
	}
	
	public Goods(Goods good) {
		this.tag = good.tag;
		this.core = good.core;
		this.mem = good.mem;
		this.ratio = good.ratio;
	}
	
	@Override
	public boolean equals(Object obj) {
		if (!(obj instanceof Goods)) return false;
		Goods o = (Goods) obj;
		return o.mem == this.mem && o.core == this.core;
	}
	
	@Override
	public int hashCode() {
		int result = 17;
		result = 31 * result + this.core;
		result = 31 * result + this.mem;
		return result;
	}

	@Override
	public String toString() {
		return "Goods [tag=" + tag + ", core=" + core + ", mem=" + mem + ", ratio=" + ratio + "]";
	}
}