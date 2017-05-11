package xilodyne.machinelearning.test.weka;

import xilodyne.util.G;
import xilodyne.util.Logger;


public class Run_weka_IRIS {

	//moved logging to other class at it appears to effect
	//timestamp logging
	//private static Logger log = new Logger("results", "IRIS-weka");

	//.csv is added automatically to filename
	public static String CSV_Filename = "IRIS_Data_NaiveBayes-weka";
	public static String delimiter = ",";
	public static String[] header = {"timestamp", "class name", 
		"accuracy", "CV Fold", "# of lines", "# trained", "date time", 
		"train time", "predict time","total time"};

	
	public static void main(String[] args) throws Exception {

	//G.setLoggerLevel(G.LOG_FINE);
	//G.setLoggerLevel(G.LOG_INFO);
	//G.setLoggerLevel(G.LOG_DEBUG);
	// G.setLoggerShowDate(false);

	//log.logln_withClassName(G.lF, "Running Weka Naive Bayes classification for Iris Data.");
	
	Weka_IRIS_NB_NoCV wekaNoCV = new Weka_IRIS_NB_NoCV();
	
//	wekaNoCV.runClassifier("weka.classifiers.bayes.NaiveBayes");
//	wekaNoCV.runClassifier("weka.classifiers.bayes.NaiveBayesMultinomial");
//	wekaNoCV.runClassifier("weka.classifiers.bayes.NaiveBayesMultinomialText");
//	wekaNoCV.runClassifier("weka.classifiers.bayes.NaiveBayesMultinomialUpdateable");
//	wekaNoCV.runClassifier("weka.classifiers.bayes.NaiveBayesUpdateable");

	Weka_IRIS_NB_CV wekaCV = new Weka_IRIS_NB_CV();
//	wekaCV.runClassifier("weka.classifiers.bayes.NaiveBayes");
//	wekaCV.runClassifier("weka.classifiers.bayes.NaiveBayesMultinomial");
//	wekaCV.runClassifier("weka.classifiers.bayes.NaiveBayesMultinomialText");
//	wekaCV.runClassifier("weka.classifiers.bayes.NaiveBayesMultinomialUpdateable");
	wekaCV.runClassifier("weka.classifiers.bayes.NaiveBayesUpdateable");
	
	}
}
