package xilodyne.machinelearning.test.weka;

import java.util.Random;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AddClassification;

public class Weka_CrossValidationAddPrediction_EnronSubset {
	/**
	 * Performs the cross-validation. See Javadoc of class for information on
	 * command-line parameters.
	 *
	 * @param args
	 *            the command-line parameters
	 * @throws Exception
	 *             if something goes wrong
	 */

	private static Instances data;

	public static void main(String[] args) throws Exception {
		//convert udacity enron pkl (pickle file) data to arff format
		
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
		Random rand = new Random(seed);
		Instances randData = new Instances(data);
		randData.randomize(rand);
		if (randData.classAttribute().isNominal())
			randData.stratify(folds);

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

		// output evaluation
		System.out.println();
		System.out.println("=== Setup ===");
		if (cls instanceof OptionHandler)
			System.out.println("Classifier: " + cls.getClass().getName() + " "
					+ Utils.joinOptions(((OptionHandler) cls).getOptions()));
		else
			System.out.println("Classifier: " + cls.getClass().getName());
		System.out.println("Dataset: " + data.relationName());
		System.out.println("Folds: " + folds);
		System.out.println("Seed: " + seed);
		System.out.println();
		System.out.println(eval.toSummaryString("=== " + folds + "-fold Cross-validation ===", false));

	}

}
