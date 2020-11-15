package model.net.domain;

import model.net.iOutputLayer;

public class OutputLayer extends AbstractLayer implements iOutputLayer {

	// fields

	// ------------------------------------
	// constructor
	public OutputLayer() {

	}

	// ------------------------------------
	// methods
	@Override
	public void computeError(double[] predictVec, double[] desiredVec) {
		this.errorVector = new double[this.size];
		
		for (int i = 0; i < predictVec.length; i++) {
			this.errorVector[i] = (predictVec[i] - desiredVec[i]) * predictVec[i] * (1 - predictVec[i]);
		}
	}
}
