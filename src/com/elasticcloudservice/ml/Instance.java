package com.elasticcloudservice.ml;

public class Instance {
	
	public String date;
	public String tag;
    public float y;
    public float[] X;
    private int feature_dim;
    
    public Instance(){}
    
    public Instance(String tag, float[] X) {
    	this.tag = tag;
    	this.X = X;
    	this.feature_dim = X.length;
    }
    
    public Instance(String tag, float y,float[] X){
    	this(tag, X);
    	this.y = y;
    }
    
    public int getFeature_dim() {
		return this.feature_dim;
	}
    
    public String toHead() {
    	int n = X.length;
    	String[] head = new String[n + 1];
    	for (int i = 0; i < n; ++i) head[i] = "feature_" + i;
    	head[n] = "label";
    	return "date," + String.join(",", head);
    }
    
    @Override
    public String toString() {
    	int n = X.length;
    	String[] content = new String[n + 1];
    	for (int i = 0; i < n; ++i) content[i] = X[i] + "";
    	content[n] = y + "";
    	return date + "," + String.join(",", content);
    }
}
