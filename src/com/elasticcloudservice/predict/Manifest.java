package com.elasticcloudservice.predict;

import java.util.HashMap;
import java.util.Map;

public class Manifest{
	
	public String type;
	
	public String uuid;
	public int id;
	public Map<String, Integer> menu;
	public int core;
	public int memo;
	
	public Manifest() {
		this.menu = new HashMap<>();
	}
	
	public Manifest(String idx) {
		this.uuid = idx;
		this.id = Integer.parseInt(idx);
		this.menu = new HashMap<>();
	}
	
	public Manifest(String idx, String type) {
		this(idx);
		this.type = type;
	}
	
	public Manifest(String idx, int core, int memo) {
		this(idx);
		this.core = core;
		this.memo = memo;
	}
	
	public Manifest(String idx, int core, int memo, String type) {
		this(idx, core, memo);
		this.type = type;
	}
	
	public void add(String tag, int count) {
		menu.put(tag, menu.getOrDefault(tag, 0) + count);
	}
	
	public boolean remove(String tag, int core, int memo) {
		if (!menu.containsKey(tag)) {
			return false;
		}
		else {
			this.core += core;
			this.memo += memo;
			menu.put(tag, menu.get(tag) - 1);
			return true;
		}
	}
	
	public void add(Goods info) {
		if (info.isGrouped()) {
			Map<String, Integer> cnt = info.flavorCnt;
			this.core -= info.core;
			this.memo -= info.mem;
			for (String key : cnt.keySet()) {
				menu.put(key, menu.getOrDefault(key, 0) + cnt.get(key));
			}
		}
		else {
			this.core -= info.core;
			this.memo -= info.mem;
			menu.put(info.tag, menu.getOrDefault(info.tag, 0) + 1);
		}
	}
	
	public void addAndRemain(Goods info, int core, int memo) {
		if (info.isGrouped()) {
			Map<String, Integer> cnt = info.flavorCnt;
			for (String key : cnt.keySet()) {
				menu.put(key, menu.getOrDefault(key, 0) + cnt.get(key));
			}
		}
		else {
			menu.put(info.tag, menu.getOrDefault(info.tag, 0) + 1);
		}
		this.core -= core;
		this.memo -= memo;
	}
	
	public int flavor_cnt() {
		int cnt = 0;
		for (String flavor : menu.keySet()) cnt += menu.get(flavor);
		return cnt;
	}
	
	public double scoreCPU(int all_core) {
		return (all_core - this.core) * 1.0 / all_core;
	}
	
	public double scoreMEM(int all_memo) {
		return (all_memo - this.memo) * 1.0 / all_memo;
	}
	
	public Map<String, Integer> getList(){
		return this.menu;
	}

	@Override
	public String toString() {
		return "Manifest [uuid=" + uuid + ", menu=" + menu + "]";
	}
}
