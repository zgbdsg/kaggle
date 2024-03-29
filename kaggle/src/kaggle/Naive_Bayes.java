package kaggle;

import java.util.*;

class Bundle  {
    int count;                 //记录关联的样本数量
    double probability;        //保存概率值
}

public class Naive_Bayes extends Classifier {
    private double[][] _features;
    private boolean[] _isCategory;
    private double[] _labels;
    private double[] _defaults;
    private int _attrCount;                               //样本的属性数量
    
    private HashMap<Double, Bundle> _labelCounter;        //记录P(C)
    private HashMap<Double,Bundle>[][] _attrCounter;      //记录P(X|C),
    private double[] _labelList;                          //所有不同标签的列表
    private double[][] _mean, _mse;                       //为连续属性存储均值和标准差
    private int _virtual;                                 //拉普拉斯校准中添加的假样本数量
    

    public Naive_Bayes() {
    }

    @Override
    public void train(boolean[] isCategory, double[][] features, double[] labels) {
        _features = features;
        _isCategory = isCategory;
        _labels = labels;
        _attrCount = _isCategory.length - 1;
        _virtual = 0;
        
        //处理缺失属性
        _defaults = kill_missing_data();
        
        countForLabels();        //统计P(C)的count值
        countForAttributes();    //统计P(X|C)的count值
        updateProbability();     //计算概率
    }

    @Override
    public double predict(double[] features) {
        //处理缺失属性
        for (int i = 0; i < features.length; ++i) {
            if (Double.isNaN(features[i])) {
                features[i] = _defaults[i];
            }
        }
        
        //拉普拉斯校准
        laplace(features);
        
        double probability = -1;
        double anser = 0;
        for (int i = 0; i < _labelList.length; ++i) {
            double temp = calculateProbabilityForLabelIndex(i, features);
            if (temp > probability) {
                probability = temp;
                anser = _labelList[i];
            }
        }
        
        return anser;
    }
    
    private double[] kill_missing_data() {
        int num = _isCategory.length - 1;
        double[] defaults = new double[num];
        
        for (int i = 0; i < defaults.length; ++i) {
            if (_isCategory[i]) {
                //离散属性取最多的值
                HashMap<Double, Integer> counter = new HashMap<Double, Integer>();
                for (int j = 0; j < _features.length; ++j) {
                    double feature = _features[j][i];
                    if (!Double.isNaN(feature)) {
                        if (counter.get(feature) == null) {
                            counter.put(feature, 1);
                        } else {
                            int count = counter.get(feature) + 1;
                            counter.put(feature, count);
                        }
                    }
                }
                
                int max_time = 0;
                double value = 0;
                Iterator<Double> iterator = counter.keySet().iterator();
                while (iterator.hasNext()) {
                    double key = iterator.next();
                    int count = counter.get(key);
                    if (count > max_time) {
                        max_time = count;
                        value = key;
                    }
                }
                defaults[i] = value;
            } else {
                //连续属性取平均值
                int count = 0;
                double total = 0;
                for (int j = 0; j < _features.length; ++j) {
                    if (!Double.isNaN(_features[j][i])) {
                        count++;
                        total += _features[j][i];
                    }
                }
                defaults[i] = total / count;
            }
        }
        
        //代换
        for (int i = 0; i < _features.length; ++i) {
            for (int j = 0; j < defaults.length; ++j) {
                if (Double.isNaN(_features[i][j])) {
                    _features[i][j] = defaults[j];
                }
            }
        }
        return defaults;
    }
    
    private void countForLabels() {
        _labelCounter = new HashMap<Double, Bundle>();
        int count = 0;
        
        for (double label : _labels) {
            Bundle bundle = _labelCounter.get(label);
            if (bundle == null) {
                bundle = new Bundle();
                bundle.count = 1;
                _labelCounter.put(label, bundle);
                count++;
            } else {
                bundle.count++;
            }
        }
        
        _labelList = new double[count];
        Iterator<Double> iterator = _labelCounter.keySet().iterator();
        int index = 0;
        while (iterator.hasNext()) {
            _labelList[index++] = iterator.next();
        }
    }
    
    private void countForAttributes() {
        _attrCounter = new HashMap[_labelList.length][];
        _mean = new double[_labelList.length][_attrCount];
        _mse = new double[_labelList.length][_attrCount];
        
        for (int i = 0; i < _labelList.length; ++i) {
            HashMap<Double, Bundle>[] temp = new HashMap[_attrCount];
            _attrCounter[i] = temp;
            for (int j = 0; j < _attrCount; ++j) {
                if (_isCategory[j]) {
                    HashMap<Double, Bundle> counter = new HashMap<Double, Bundle>();
                    temp[j] = counter;
                }
            }
        }
        
        for (int i = 0; i < _features.length; ++i) {
            HashMap<Double, Bundle>[] temp = _attrCounter[indexForLabel(_labels[i])];
            for (int j = 0; j < _attrCount; ++j) {
                if (_isCategory[j]) {
                    HashMap<Double , Bundle> counter = temp[j];
                    Bundle bundle = counter.get(_features[i][j]);
                    
                    if (bundle == null) {
                        bundle = new Bundle();
                        bundle.count = 1;
                        counter.put(_features[i][j], bundle);
                        
                        //拉普拉斯校准
                        bundle.count++;
                        _virtual++;
                        _labelCounter.get(_labels[i]).count++;
                    } else {
                        bundle.count++;
                    }
                }
            }
        }
        
        //为连续属性计算平均值
        for (int i = 0; i < _labelList.length; ++i) {
            double label = _labelList[i];
            for (int j = 0; j < _attrCount; ++j) {
                if (_isCategory[j]) continue;
                
                int count = 0;
                double temp = 0;
                for (int k = 0; k < _features.length; ++k) {
                    if (_labels[k] == label) {
                        temp += _features[k][j];
                        count++;
                    }
                }
                _mean[i][j] = temp / count;
            }
        }
        //为连续属性计算标准差
        for (int i = 0; i < _labelList.length; ++i) {
            double label = _labelList[i];
            for (int j = 0; j < _attrCount; ++j) {
                if (_isCategory[j]) continue;
                
                int count = 0;
                double temp = 0;
                double mean = _mean[i][j];
                for (int k = 0; k < _features.length; ++k) {
                    if (_labels[k] == label) {
                        double sub = _features[k][j] - mean;
                        temp += sub * sub;
                        count++;
                    }
                }
                _mse[i][j] = Math.sqrt(temp / count);
            }
        }
    }
    
    private int indexForLabel(double label) {
        for (int i = 0; i < _labelList.length; ++i) {
            if (_labelList[i] == label) {
                return i;
            }
        }
        return 0;
    }
    
    //计算待预测元组的P(X|C)
    private double calculateProbabilityForLabelIndex(int index, double[] features) {
        double label = _labelList[index];
        double temp = _labelCounter.get(label).probability;
        for (int i = 0; i < features.length; ++i) {
            if (_isCategory[i]) {
                HashMap<Double, Bundle> counter = _attrCounter[indexForLabel(label)][i];
                Bundle bundle = counter.get(features[i]);
                temp *= bundle.probability;
            } else {
                //连续属性计算概率
                double mean = _mean[index][i];
                double mse = _mse[index][i];
                double var = features[i];
                
                if (mse != 0) {
                    //使用高斯高斯分布
                    double gauss = 1 / (Math.sqrt(2*Math.PI) * mse);
                    double t = var - mean;
                    gauss *= Math.exp(-(t*t) / (2*mse*mse));
                    temp *= gauss;
                } else {
                    //只有一个值，无法使用高斯分布
                    temp *= 1;
                }
            }
        }
        
        return temp;
    }
    
    //拉普拉斯校准
    private void laplace(double[] features) {
        for (int i = 0; i < _labelList.length; ++i) {
            double label = _labelList[i];
            HashMap<Double, Bundle>[] temp = _attrCounter[i];
            for (int j = 0; j < _attrCount; ++j) {
                if (_isCategory[j]) {
                    HashMap<Double, Bundle> counter = temp[j];
                    Bundle bundle = counter.get(features[j]);
                    
                    if (bundle == null) {
                        bundle = new Bundle();
                        bundle.count = 1;
                        counter.put(features[j], bundle);
                        _virtual++;
                        _labelCounter.get(label).count++;
                    }
                }
            }
        }
        updateProbability();
    }
    
    private void updateProbability() {
        for (int i = 0; i < _labelList.length; ++i) {
            Bundle bundle = _labelCounter.get(_labelList[i]);
            bundle.probability = (double)bundle.count / (_labels.length + _virtual);
        }
        
        for (int i = 0; i < _labelList.length; ++i) {
            double label = _labelList[i];
            HashMap<Double, Bundle>[] temp = _attrCounter[i];
            for (int j = 0; j < _attrCount; ++j) {
                if (_isCategory[j]) {
                    HashMap<Double, Bundle> counter = temp[j];
                    
                    Iterator<Double> iterator = counter.keySet().iterator();
                    while (iterator.hasNext()) {
                        Bundle bundle = counter.get(iterator.next());
                        bundle.probability = (double)bundle.count / _labelCounter.get(label).count;
                    }
                }
            }
        }
    }
}
