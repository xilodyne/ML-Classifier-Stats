package xilodyne.machinelearning.test.weka;

import java.time.Duration;
import java.time.Instant;
import java.util.Random;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSink;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AddClassification;
import xilodyne.util.G;
import xilodyne.util.Logger;
import xilodyne.util.weka.WekaUtils;

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

public class Weka_CrossValidationAddPrediction_Iris {
	/**
	 * Performs the cross-validation. See Javadoc of class for information on
	 * command-line parameters.
	 *
	 * @param args
	 *            the command-line parameters
	 * @throws Exception
	 *             if something goes wrong
	 */
	private static Logger log = new Logger("results", "IRIS-weka-NB-CV");

	private static Instances data;
	private static Instant startCV, endCV, startFit, endFit, 
			startPredict, endPredict = null;


	public static void main(String[] args) throws Exception {
		
		//G.setLoggerLevel(G.LOG_FINE);
		G.setLoggerLevel(G.LOG_INFO);
		//G.setLoggerLevel(G.LOG_DEBUG);
		// G.setLoggerShowDate(false);

		log.logln_withClassName(G.lF, "Running Weka Gaussian Naive Bayes classification for Iris Data.");

		// loads data and set class index
		data = DataSource.read("data/iris.arff");
		String clsIndex = "last";
		if (clsIndex.length() == 0)
			clsIndex = "last";
		if (clsIndex.equals("first"))
			data.setClassIndex(0);
		else if (clsIndex.equals("last"))
			data.setClassIndex(data.numAttributes() - 1);
		else
			data.setClassIndex(Integer.parseInt(clsIndex) - 1);

		runClassifier("weka.classifiers.bayes.NaiveBayes");
		runClassifier("weka.classifiers.bayes.NaiveBayesMultinomial");
		runClassifier("weka.classifiers.bayes.NaiveBayesMultinomialText");
		runClassifier("weka.classifiers.bayes.NaiveBayesMultinomialUpdateable");
		runClassifier("weka.classifiers.bayes.NaiveBayesUpdateable");
	}

	public static void runClassifier(String classname) throws Exception {
		// classifier
		String[] tmpOptions = new String[1];
		// String classname;
		// classname = "weka.classifiers.bayes.NaiveBayes";
		tmpOptions[0] = "";
		Classifier cls = (Classifier) Utils.forName(Classifier.class, classname, tmpOptions);

		// other options
		int seed = 1;
		int folds = 10;

		// randomize data
		startCV = Instant.now();
		Random rand = new Random(seed);
		Instances randData = new Instances(data);
		randData.randomize(rand);
		if (randData.classAttribute().isNominal())
			randData.stratify(folds);

		endCV = Instant.now();
		
		startPredict = Instant.now();
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

		endPredict = Instant.now();
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

		getStats();
	}
	
	private static void getStats() {
		//long trainingTime = Duration.between(startFit, endFit).toMillis();

		
		long cvTime = Duration.between(startCV, endCV).toMillis();
		long predictTime = Duration.between(startPredict, endPredict).toMillis();
		
		//double dTrainTime = trainingTime / (double) 1000;
		double dcvTime = cvTime / (double) 1000;
		double dPredictTime = predictTime / (double) 1000;
	//	log.logln(G.lF,"Total lines training: " + WekaUtils.getCountFromInstances(trainingData));
	//	log.logln("Total lines predicted: " + WekaUtils.getCountFromInstances(testingData));
		//log.logln("Training time: " + trainingTime + " milliseconds.");
		//log.logln("Predict time: " + predictTime + " milliseconds.");
		log.logln("Data setup time: " + dcvTime + " seconds.");
		log.logln("Predict time: " + dPredictTime + " seconds.");
//		double acc = (double) ArrayUtils.getNumberOfCorrectMatches(predictResults,  labels)/predictResults.size();
//		System.out.println("Accuracy: "  + ArrayUtils.getAccuracyOfLabels(predictResults,  labels));

	}

}
