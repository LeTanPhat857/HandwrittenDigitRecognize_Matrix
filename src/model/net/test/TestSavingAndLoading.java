package model.net.test;

import java.io.IOException;

import model.dataReader.DataReader;
import model.net.domain.Net;

public class TestSavingAndLoading {

	// test1: test saving
	public static void test1() throws IOException {
		// create net
		Net net = new Net();
		net.setNumberInput(28 * 28);
		net.addHiddenLayer(15);
		net.setNumberOutput(10);
		net.createNet();

		// saving
		net.saveCurrentNet("trained/randomNet");
	}

	// test2: load trained net
	public static void test2() throws Exception {
		// create net
		Net net = new Net();

		// load trained net
		net.loadTrainedNet("trained/32-32-Net");

		// data test
		DataReader dataReader1 = new DataReader();
		double[][] dataTestSet = dataReader1.readDataSet("data/t10k-images.idx3-ubyte");
		double[] labelTestSet = dataReader1.readLabelSet("data/t10k-labels.idx1-ubyte");

		// test
		System.out.println("predict correction: " + net.test(dataTestSet, labelTestSet));
	}

	// run test
	public static void main(String[] args) throws Exception {
		test2();
	}
}
