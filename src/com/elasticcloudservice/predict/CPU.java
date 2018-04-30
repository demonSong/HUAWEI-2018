package com.elasticcloudservice.predict;

public class CPU{
	
	public String type;
	public int core;
	public int memory;
	public int hdd;
	
	public double ratio;
	
	public CPU(){
		
	}
	
	/**
	 * ratio is supported on physical machine
	 * @param core
	 * @param memory
	 */
	public CPU(int core, int memory){
		this.core = core;
		this.memory = memory;
	}
	
	public CPU(int core, int memory, int hdd){
		this(core, memory);
		this.hdd = hdd;
	}
	
	public CPU(String type, int core, int memory, int hdd) {
		this(core, memory, hdd);
		this.type = type;
		this.ratio = this.memory * 1.0 / this.core;
	}

	@Override
	public String toString() {
		return "CPU [type=" + type + ", core=" + core + ", memory=" + memory + ", hdd=" + hdd + ", ratio=" + ratio
				+ "]";
	}
}
