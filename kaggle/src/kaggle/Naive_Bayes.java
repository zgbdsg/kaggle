package kaggle;

import java.util.*;

class Bundle  {
    int count;                 //��¼��������������
    double probability;        //�������ֵ
}

public class Naive_Bayes extends Classifier {
    private double[][] _features;
    private boolean[] _isCategory;
    private double[] _labels;
    private double[] _defaults;
    private int _attrCount;                               //��������������
    
    private HashMap<Double, Bundle> _labelCounter;        //��¼P(C)
    private HashMap<Double,Bundle>[][] _attrCounter;      //��¼P(X|C),
    private double[] _labelList;                          //���в�ͬ��ǩ���б�
    private double[][] _mean, _mse;                       //Ϊ�������Դ洢��ֵ�ͱ�׼��
    private int _virtual;                                 //������˹У׼����ӵļ���������
    

    public Naive_Bayes() {
    }

    @Override
    public void train(boolean[] isCategory, double[][] features, double[] labels) {
        _features = features;
        _isCategory = isCategory;
        _labels = labels;
        _attrCount = _isCategory.length - 1;
        _virtual = 0;
        
        //����ȱʧ����
        _defaults = kill_missing_data();
        
        countForLabels();        //ͳ��P(C)��countֵ
        countForAttributes();    //ͳ��P(X|C)��countֵ
        updateProbability();     //�������
    }

    @Override
    public double predict(double[] features) {
        //����ȱʧ����
        for (int i = 0; i < features.length; ++i) {
            if (Double.isNaN(features[i])) {
                features[i] = _defaults[i];
            }
        }
        
        //������˹У׼
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
                //��ɢ����ȡ����ֵ
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
                //��������ȡƽ��ֵ
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
        
        //����
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
                        
                        //������˹У׼
                        bundle.count++;
                        _virtual++;
                        _labelCounter.get(_labels[i]).count++;
                    } else {
                        bundle.count++;
                    }
                }
            }
        }
        
        //Ϊ�������Լ���ƽ��ֵ
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
        //Ϊ�������Լ����׼��
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
    
    //�����Ԥ��Ԫ���P(X|C)
    private double calculateProbabilityForLabelIndex(int index, double[] features) {
        double label = _labelList[index];
        double temp = _labelCounter.get(label).probability;
        for (int i = 0; i < features.length; ++i) {
            if (_isCategory[i]) {
                HashMap<Double, Bundle> counter = _attrCounter[indexForLabel(label)][i];
                Bundle bundle = counter.get(features[i]);
                temp *= bundle.probability;
            } else {
                //�������Լ������
                double mean = _mean[index][i];
                double mse = _mse[index][i];
                double var = features[i];
                
                if (mse != 0) {
                    //ʹ�ø�˹��˹�ֲ�
                    double gauss = 1 / (Math.sqrt(2*Math.PI) * mse);
                    double t = var - mean;
                    gauss *= Math.exp(-(t*t) / (2*mse*mse));
                    temp *= gauss;
                } else {
                    //ֻ��һ��ֵ���޷�ʹ�ø�˹�ֲ�
                    temp *= 1;
                }
            }
        }
        
        return temp;
    }
    
    //������˹У׼
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
