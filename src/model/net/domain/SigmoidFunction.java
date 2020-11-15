package model.net.domain;

import model.net.iActivationFunction;

public class SigmoidFunction implements iActivationFunction {

	// fileds
	private static SigmoidFunction instance;

	// constructor
	private SigmoidFunction() {
	}

	// methods
	public static SigmoidFunction getInstance() {
		if (instance == null) {
			return new SigmoidFunction();
		} else {
			return instance;
		}
	}

	@Override
	public double[] active(double[] sum) {
		double[] result = new double[sum.length];

		for (int i = 0; i < sum.length; i++) {
			result[i] = 1 / (1 + Math.pow(Math.E, -sum[i]));
		}

		return result;
	}

	@Override
	public double[] computeDerived(double[] output) {
		double[] result = new double[output.length];

		for (int i = 0; i < output.length; i++) {
			result[i] = output[i] * (1 - output[i]);
		}

		return result;
	}

}
