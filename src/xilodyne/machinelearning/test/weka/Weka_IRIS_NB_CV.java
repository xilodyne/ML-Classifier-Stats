package xilodyne.machinelearning.test.weka;

import java.time.Duration;
import java.time.Instant;
import java.util.Random;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSink;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AddClassification;
import weka.knowledgeflow.LogManager;


import xilodyne.util.G;
import xilodyne.util.Logger;
import xilodyne.util.LoggerCSV;
import xilodyne.util.metrics.OutputResults;
import xilodyne.util.metrics.TestResultsDataML;
import xilodyne.util.weka.WekaARFFUtils;
import xilodyne.util.weka.WekaUtils;

//import weka.core.logging.Logger;
/**
 * Performs a single run of cross-validation and adds the prediction on the test
 * set to the dataset.
 *
 * Command-line parameters:
 * <ul>
 * <li>-t filename - the dataset to use</li>
 * <li>-o filename - the output file to store dataset with the predictions in</li>
 * <li>-x int - the number of folds to use</li>
 * <li>-s int - the seed for the random number generator</li>
 * <li>-c int - the class index, "first" and "last" are accepted as well; "last"
 * is used by default</li>
 * <li>-W classifier - classname and options, enclosed by double quotes; the
 * classifier to cross-validate</li>
 * </ul>
 *
 * Example command-line:
 * 
 * <pre>
 * java wekaexamples.classifiers.CrossValidationAddPrediction -t anneal.arff -c last -o predictions.arff -x 10 -s 1 -W "weka.classifiers.trees.J48 -C 0.25"
 * </pre>
 *
 * @author FracPete (fracpete at waikato dot ac dot nz)
 * @version $Revision: 5937 $
 */

public class Weka_IRIS_NB_CV {
	/**
	 * Performs the cross-validation. See Javadoc of class for information on
	 * command-line parameters.
	 *
	 * @param args
	 *            the command-line parameters
	 * @throws Exception
	 *             if something goes wrong
	 */
	//private Logger log = new Logger("results", "IRIS-weka-NB-CV");
	private Logger log;
	private LoggerCSV logCSV;


	private Instances data, randData;
	//private Instant startData, endData, startFit, endFit, 
	//		startPredict, endPredict = null;
	private Classifier cls;
	private double accuracy;
	
	// other options
	int seed = 1;
	//int folds = 10;

	
	public void runClassifier(String classNamex) throws Exception {
		 G.setLoggerLevel(G.LOG_FINE);
			//G.setLoggerLevel(G.LOG_INFO);
			// G.setLoggerLevel(G.LOG_DEBUG);
			// G.setLoggerShowDate(false);

	this.clearValues();
	
	 TestResultsDataML resultsData = new TestResultsDataML();
	 resultsData.setFoldSize(10);
	 //OutputResults results = new OutputResults();
		resultsData.setClassMLName(classNamex);

	
	//	log = new Logger("results", "IRIS-weka-NB-CV" + "_" + className.substring(className.lastIndexOf(".") + 1));
		log = new Logger("results", "IRIS-weka-NB-CV" + "_" + resultsData.getClassMLNameWithoutDomain());

	log.logln_withClassName(G.lF, "Running Weka Naive Bayes classification for Iris Data.");


	logCSV = new LoggerCSV("results", Run_weka_IRIS.CSV_Filename, 
			Run_weka_IRIS.delimiter, Run_weka_IRIS.header);
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
	this.loadData();
	//this.randData(className);
	this.randData(resultsData.getClassMLName(), resultsData.getFoldSize());
	
	//endData = Instant.now();
	resultsData.setEndData();



	//startPredict = Instant.now();
	resultsData.setStartPredict();
	log.logln("Test data...");
	this.predict(resultsData.getFoldSize());
	//endPredict = Instant.now();
	resultsData.setEndPredict();

	resultsData.setAccuracy(this.accuracy);
	
	OutputResults.getMLStats(log, logCSV, resultsData);
	//getStats(className);
	logCSV.log_CSV_EOL();
}
	private void clearValues() {
		cls = null;
		data = null;
		randData = null;


	}

	private void initClassifier(String className) {
		String[] tmpOptions = new String[1];
		tmpOptions[0] = "";
		try {
			cls = (Classifier) Utils.forName(Classifier.class, className, tmpOptions);
		} catch (Exception e) {
			e.printStackTrace();
		}

	}


	private void loadData() throws Exception {

		data = WekaARFFUtils.wekaReadARFF("data/iris.arff");
		// loads data and set class index
		data = DataSource.read("data/iris.arff");
		log.logln(G.lI, "Generating test data from size: " + data.numInstances());

		String clsIndex = "last";
		if (clsIndex.length() == 0)
			clsIndex = "last";
		if (clsIndex.equals("first"))
			data.setClassIndex(0);
		else if (clsIndex.equals("last"))
			data.setClassIndex(data.numAttributes() - 1);
		else
			data.setClassIndex(Integer.parseInt(clsIndex) - 1);
	}

	public void randData(String classname, int folds) throws Exception {
		// randomize data
		Random rand = new Random(seed);
		randData = new Instances(data);
		randData.randomize(rand);
		if (randData.classAttribute().isNominal())
			randData.stratify(folds);
	}
		
	private void predict(int folds) throws Exception {

		// perform cross-validation and add predictions
		Instances predictedData = null;
		Evaluation eval = new Evaluation(randData);
		for (int n = 0; n < folds; n++) {
			Instances train = randData.trainCV(folds, n);
			Instances test = randData.testCV(folds, n);
			// the above code is used by the StratifiedRemoveFolds filter, the
			// code below by the Explorer/Experimenter:
			// Instances train = randData.trainCV(folds, n, rand);

			// build and evaluate classifier
			Classifier clsCopy = AbstractClassifier.makeCopy(cls);
			clsCopy.buildClassifier(train);
			eval.evaluateModel(clsCopy, test);

			// add predictions
			AddClassification filter = new AddClassification();
			filter.setClassifier(cls);
			filter.setOutputClassification(true);
			filter.setOutputDistribution(true);
			filter.setOutputErrorFlag(true);
			filter.setInputFormat(train);
			Filter.useFilter(train, filter); // trains the classifier
			Instances pred = Filter.useFilter(test, filter); // perform
																// predictions
																// on test set
			if (predictedData == null)
				predictedData = new Instances(pred, 0);
			for (int j = 0; j < pred.numInstances(); j++)
				predictedData.add(pred.instance(j));
		}

		accuracy = eval.pctCorrect();


		// output evaluation
		log.logln(G.lF, "");
		log.logln("=== Setup ===");
		if (cls instanceof OptionHandler)
			log.logln("Classifier: " + cls.getClass().getName() + " "
					+ Utils.joinOptions(((OptionHandler) cls).getOptions()));
		else
			log.logln("Classifier: " + cls.getClass().getName());
		log.logln("Dataset: " + data.relationName());
		log.logln("Folds: " + folds);
		log.logln("Seed: " + seed);
		log.logln("");
		log.logln(eval.toSummaryString("=== " + folds + "-fold Cross-validation ===", false));


	}
	
	
/*	public void getStats(String className) {
		long dataTime = Duration.between(startData, endData).toMillis();
	//	long trainingTime = Duration.between(startFit, endFit).toMillis();
		long predictTime = Duration.between(startPredict, endPredict).toMillis();
		double dDataTime = dataTime / (double) 1000;
	//	double dTrainTime = trainingTime / (double) 1000;
		double dPredictTime = predictTime / (double) 1000;

		double totalDuration = dDataTime +  dPredictTime;
		long dataPercent = Math.round((dDataTime / totalDuration) * 100);
	//	long trainPercent = Math.round((dTrainTime / totalDuration) * 100);
		long predictPercent = Math.round((dPredictTime / totalDuration) * 100);
		log.logln(G.lF, "Class tested: " + className);
		log.logln("Accuracy: " + accuracy + "%");
//		log.logln("Total lines training: " + WekaUtils.getCountFromInstances(trainingData));
//		log.logln("Total lines predicted: " + WekaUtils.getCountFromInstances(testingData));
		log.logln("Activity\tTime (in seconds)\t% of Total Duration");
		log.logln("--------\t-----------------\t-------------------");
		log.logln("Data setup\t" + dDataTime + "\t\t\t" + dataPercent + "%");
	//	log.logln("Training\t" + dTrainTime + "\t\t\t" + trainPercent + "%");
		log.logln("Predict\t\t" + dPredictTime + "\t\t\t" + predictPercent + "%");
		log.logln("Total Time\t" + totalDuration);
		
		logCSV.log_CSV_Entry(String.valueOf(accuracy));
		logCSV.log_CSV_Entry(String.valueOf(folds));
		logCSV.log_CSV_Entry("-");
		logCSV.log_CSV_Entry("-");
		logCSV.log_CSV_Entry(String.valueOf(dDataTime));
		logCSV.log_CSV_Entry("-");
		logCSV.log_CSV_Entry(String.valueOf(dPredictTime));
		logCSV.log_CSV_Entry(String.valueOf(totalDuration));

	}
*/

}
