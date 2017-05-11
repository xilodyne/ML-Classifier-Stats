package xilodyne.machinelearning.test.weka;

import java.time.Duration;
import java.time.Instant;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instances;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AddClassification;
import weka.filters.unsupervised.instance.RemovePercentage;
//import xilodyne.udacity.intro_ml.miniproj_NB_Enron.weka.WEKA_GNB_Enron;
import xilodyne.util.G;
import xilodyne.util.Logger;
import xilodyne.util.LoggerCSV;
import xilodyne.util.metrics.OutputResults;
import xilodyne.util.metrics.TestResultsDataML;
import xilodyne.util.weka.WekaARFFUtils;
import xilodyne.util.weka.WekaUtils;


public class Weka_IRIS_NB_NoCV {

	 //private Logger log = new Logger("results", "IRIS-weka-NB-noCV");
	private Logger log;
	private LoggerCSV logCSV;
	//private Instant startData, endData, startFit, endFit, startPredict, endPredict = null;
	private Instances data, trainingData, testingData;

	private Classifier cls;
	private AddClassification filter;
	private FilteredClassifier fc;
	private Evaluation eval;
	

	private double accuracy;


	public void runClassifier(String className) throws Exception {
		this.clearValues();
		 TestResultsDataML resultsData = new TestResultsDataML();
		 OutputResults results = new OutputResults();

		resultsData.setClassMLName(className);
		
//		log = new Logger("results", "IRIS-weka-NB-noCV" + "_" + className.substring(className.lastIndexOf(".") + 1));
		log = new Logger("results", "IRIS-weka-NB-noCV" + "_" + resultsData.getClassMLNameWithoutDomain());
		logCSV = new LoggerCSV("results", Run_weka_IRIS.CSV_Filename, 
				Run_weka_IRIS.delimiter, Run_weka_IRIS.header);
		 G.setLoggerLevel(G.LOG_FINE);
		//G.setLoggerLevel(G.LOG_INFO);
		 //G.setLoggerLevel(G.LOG_DEBUG);
		// G.setLoggerShowDate(false);

		log.logln_withClassName(G.lF, "Running Weka Naive Bayes classification for Iris Data.");

		logCSV.log_CSV_Timestamp();
		//logCSV.log_CSV_Entry(className);
		logCSV.log_CSV_Entry(resultsData.getClassMLName());
		
		//startData = Instant.now();
		resultsData.setStartData();

		log.logln(G.lF, "\n\n\n\n");
		log.logln("==================================================");
		//log.logln(className);
		log.logln(resultsData.getClassMLName());
		log.logln("Generating test data");

		//initClassifier(className);
		initClassifier(resultsData.getClassMLName());
		generateTestData();

		//endData = Instant.now();
		resultsData.setEndData();
		

		// weka.generateTestData(data);

		// trainingData = weka.getTrainingSet();
		// testingData = weka.getTestingSet();

		// convtTrainData = weka.convertToTdidfVector(trainingData);
		// convtTestData = weka.convertToTdidfVector(testingData);
		// WekaUtils.printInstancesLabelsAndData(convertedData, log);

		//startFit = Instant.now();
		resultsData.setStartFit();
		log.logln(G.lF, "Fit data...");
		fit();
		//endFit = Instant.now();
		resultsData.setEndFit();

		//startPredict = Instant.now();
		resultsData.setStartPredict();
		log.logln("Test data...");
		testAndPredict();
		//endPredict = Instant.now();
		resultsData.setEndPredict();

		//getStats(className);
		
		resultsData.setAccuracy(this.accuracy);
		resultsData.setTrainingDataSize(WekaUtils.getCountFromInstances(trainingData));
		resultsData.setTestingDataSize(WekaUtils.getCountFromInstances(testingData));

		results.getMLStats(log, logCSV, resultsData);
		logCSV.log_CSV_EOL();
	}

	private void clearValues() {
		cls = null;
		filter = null;
		fc = null;
		eval = null;
		trainingData = null;
		testingData = null;
		data = null;
	/*	startData = null;
		endData = null;
		startFit = null;
		endFit = null;
		startPredict = null;
		endPredict = null;
*/
	}

	private void initClassifier(String className) {
		// Create a naïve bayes classifier

		String[] tmpOptions = new String[1];
		// String classname;
		// classname = "weka.classifiers.bayes.NaiveBayes";
		tmpOptions[0] = "";
		try {
			cls = (Classifier) Utils.forName(Classifier.class, className, tmpOptions);
		} catch (Exception e) {
			e.printStackTrace();
		}
		filter = new AddClassification();
		fc = new FilteredClassifier();

	}

	private void generateTestData() throws Exception {
		int seed = 1;
		int folds = 10;

		data = WekaARFFUtils.wekaReadARFF("data/iris.arff");
		// WEKA_GNB_Enron weka = new WEKA_GNB_Enron();

		String clsIndex = "last";
		if (clsIndex.length() == 0)
			clsIndex = "last";
		if (clsIndex.equals("first"))
			data.setClassIndex(0);
		else if (clsIndex.equals("last"))
			data.setClassIndex(data.numAttributes() - 1);
		else
			data.setClassIndex(Integer.parseInt(clsIndex) - 1);

		log.logln(G.lF, "Generating test data from size: " + data.numInstances());
		// randomize data
		Random rand = new Random(seed);
//		Instances randData = new Instances(data);
//		randData.randomize(rand);

		data.randomize(rand);
//		if (randData.classAttribute().isNominal())
//			randData.stratify(folds);

		RemovePercentage rmvp = new RemovePercentage();
		rmvp.setInputFormat(data);
		rmvp.setPercentage(90);
		rmvp.setInvertSelection(true);
		// perform cross-validation
//		trainingData = randData.trainCV(folds, 0);
//		testingData = randData.testCV(folds, 0);
		trainingData = Filter.useFilter(data,  rmvp);
		rmvp.setInputFormat(data);
		rmvp.setPercentage(90);
		rmvp.setInvertSelection(false);
		testingData = Filter.useFilter(data,  rmvp);
		
		//WekaUtils.printInstancesData(trainingData, log);
		//WekaUtils.printInstancesData(testingData, log);

		log.logln("Sizes train/test: " + trainingData.numInstances() + "/" + testingData.numInstances());
	}

	private void fit() {
		try {

			cls.buildClassifier(trainingData);
			eval = new Evaluation(trainingData);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private void testAndPredict() {

		try {
			eval.evaluateModel(cls, testingData);
		} catch (Exception e) {
			e.printStackTrace();
		}
		log.logln(G.LOG_FINE, eval.toSummaryString("\nResults\n======\n", false));
		this.accuracy = eval.pctCorrect();
	}

/*	public void getStats(String className) {
		long dataTime = Duration.between(startData, endData).toMillis();
		long trainingTime = Duration.between(startFit, endFit).toMillis();
		long predictTime = Duration.between(startPredict, endPredict).toMillis();
		double dDataTime = dataTime / (double) 1000;
		double dTrainTime = trainingTime / (double) 1000;
		double dPredictTime = predictTime / (double) 1000;

		double totalDuration = dDataTime + dTrainTime + dPredictTime;
		long dataPercent = Math.round((dDataTime / totalDuration) * 100);
		long trainPercent = Math.round((dTrainTime / totalDuration) * 100);
		long predictPercent = Math.round((dPredictTime / totalDuration) * 100);
		log.logln(G.lF, "Class tested: " + className);
		log.logln("Accuracy: " + accuracy + "%");
		log.logln("Total lines training: " + WekaUtils.getCountFromInstances(trainingData));
		log.logln("Total lines predicted: " + WekaUtils.getCountFromInstances(testingData));
		// log.logln("Training time: " + trainingTime + " milliseconds.");
		// log.logln("Predict time: " + predictTime + " milliseconds.");
		log.logln("Activity\tTime (in seconds)\t% of Total Duration");
		log.logln("--------\t-----------------\t-------------------");
		// log.logln("Data setup\t" + dDataTime + "\t" +
		// System.out.format("%fn", dataPercent));
		log.logln("Data setup\t" + dDataTime + "\t\t\t" + dataPercent + "%");
		log.logln("Training\t" + dTrainTime + "\t\t\t" + trainPercent + "%");
		log.logln("Predict\t\t" + dPredictTime + "\t\t\t" + predictPercent + "%" );
		log.logln("Total Time\t" + totalDuration);
		logCSV.log_CSV_Entry(String.valueOf(accuracy));
		logCSV.log_CSV_Entry("-");
		logCSV.log_CSV_Entry(String.valueOf(WekaUtils.getCountFromInstances(trainingData)));
		logCSV.log_CSV_Entry(String.valueOf(WekaUtils.getCountFromInstances(testingData)));
		logCSV.log_CSV_Entry(String.valueOf(dDataTime));
		logCSV.log_CSV_Entry(String.valueOf(dTrainTime));
		logCSV.log_CSV_Entry(String.valueOf(dPredictTime));
		logCSV.log_CSV_Entry(String.valueOf(totalDuration));


		// double acc = (double)
		// ArrayUtils.getNumberOfCorrectMatches(predictResults,
		// labels)/predictResults.size();
		// System.out.println("Accuracy: " +
		// ArrayUtils.getAccuracyOfLabels(predictResults, labels));

	}
	*/

}
