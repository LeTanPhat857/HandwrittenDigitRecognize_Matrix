package model.net.domain;

import model.net.iInputLayer;

public class InputLayer extends AbstractLayer implements iInputLayer {

	// fields

	// ------------------------------------
	// constructor
	public InputLayer() {
	}

	// ------------------------------------
	// methods
	@Override
	public void createLayer() {
		this.outputVector = new double[this.size];
	}
	
	@Override
	public void setOutputVector(double[] outputVector) {
		this.outputVector = outputVector;
	}
}
