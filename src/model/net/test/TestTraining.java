package model.net.test;

import model.dataReader.DataReader;
import model.net.domain.Net;

public class TestTraining {

	// test1: test training
	public static void test1() throws Exception {
		// create net
		Net net = new Net();

		// set up
		net.setNumberInput(28 * 28);
		net.addHiddenLayer(32);
		net.addHiddenLayer(32);
		net.setNumberOutput(10);
		net.createNet();
		System.out.println("net okay!");
		// load

		// dataSet and labelSet
		DataReader dataReader = new DataReader();
		double[][] dataSet = dataReader.readDataSet("data/train-images.idx3-ubyte");
		double[][] labelSet = dataReader.convertLabelSet("data/train-labels.idx1-ubyte", 10);
		System.out.println("data okay!");

		// training
		System.out.println("Training...");
		int epoch = 3;
		int miniBatch = 10;
		net.train(dataSet, labelSet, epoch, miniBatch, 1);
		System.out.println("training completed!");

		// saving
		System.out.print("saved: ");
		net.saveCurrentNet("trained/32-32-Net");
		System.out.println();

		// test
		DataReader dataReader1 = new DataReader();
		double[][] dataTestSet = dataReader1.readDataSet("data/t10k-images.idx3-ubyte");
		double[] labelTestSet = dataReader1.readLabelSet("data/t10k-labels.idx1-ubyte");

		// test
		System.out.println("Testing...");
		System.out.println("Correct answer in 10000 test: " + net.test(dataTestSet, labelTestSet));
	}

	// run test
	public static void main(String[] args) throws Exception {
		test1();
	}
}
