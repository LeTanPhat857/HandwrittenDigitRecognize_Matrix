package model.net;

public interface iHiddenLayer extends iLayer {

	public void setActiveFunction(iActivationFunction activationFunction);

	public void computeError() throws Exception;
}
