package kaggle;

import java.io.Serializable;

public abstract class Classifier implements Cloneable, Serializable {

    public abstract void train(boolean[] isCategory, double[][] features, double[] labels);

    public abstract double predict(double[] features);
}
