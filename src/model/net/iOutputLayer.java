package model.net;

public interface iOutputLayer extends iLayer {

	public void computeError(double[] predictVec, double[] desiredVec);
}
