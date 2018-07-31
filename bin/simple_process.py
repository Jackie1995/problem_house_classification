# 导包
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
import pylab as pl
from scipy.stats.stats import pearsonr
import re
import math
import os
import operator
#from catboost import Pool


from statsmodels.graphics.factorplots import interaction_plot
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from datetime import datetime, date, timedelta
import pickle

plt.rcParams['axes.unicode_minus']=False## 显示负号
plt.rcParams['font.sans-serif']=['SimHei'] ## 显示中文

#读入数据：model_data.pkl
model_data0 = pickle.load(open('D:\jikang001\Desktop\problem_house_classification\data\model_data.pkl','rb'))
model_data0.info()

#调整原始数据集 model_data0 中各列的数据类型：
# 对数据集中的数据类型进行调整：
# 需要转化成：category 类型的变量有：app_source_brand ，hdic_bizcircle_id，rent_type
def convert_2_catg(x): return x.astype('int').astype('category')
model_data0[['app_source_brand', 'hdic_bizcircle_id', 'rent_type']] = model_data0[['app_source_brand', 'hdic_bizcircle_id', 'rent_type']].apply(convert_2_catg)
# 需要转化成int类型的变量有：frame_bedroom_num
model_data0['frame_bedroom_num'] = model_data0['frame_bedroom_num'].astype('int')


### 【获取数据集】得到能直接用于catboost建模的数据集:
# X_CAT , Y_CAT ,
# CAT_features,
# X_CAT_train, Y_CAT_train,
# X_CAT_test, Y_CAT_test

from sklearn.model_selection import train_test_split
def convert_0_1(scalar_value):
    if scalar_value == 100:
        return 1
    else:
        return 0
Y_CAT = model_data0['audit_result'].apply(convert_0_1)
X_CAT = model_data0.drop('audit_result',axis = 1)
CAT_features = [0,1,3]
X_CAT_train, X_CAT_test, Y_CAT_train, Y_CAT_test = train_test_split( X_CAT, Y_CAT, test_size=0.33, random_state=42,stratify = Y_CAT)
print('用于catboost建模的X_CAT的数据结构如下：\n',X_CAT.dtypes)
print('用于catboost建模的X_CAT的类别变量所在索引位置如下：\n',CAT_features)
print('用于catboost建模的Y_CAT的数据结构如下：\n',Y_CAT.value_counts())
print('训练集 X_CAT_train 的shape是',X_CAT_train.shape)
print('测试集 X_CAT_test  的shape是',X_CAT_test.shape)
print('训练集 Y_CAT_train 的shape是',Y_CAT_train.shape)
print('测试集 Y_CAT_test  的shape是',Y_CAT_test.shape)
print('训练集中label的正负比例分布如下：\n',Y_CAT_train.value_counts())
print('测试集中label的正负比例分布如下：\n',Y_CAT_test.value_counts())
print('可以看出在划分训练集和测试集时设定strtify参数为Y_CAT；使得在训练测试集中正负例所占比例一致。')

## catboost建模
### Step1: Pool Initialize
from catboost import Pool
pool_data = Pool(data = X_CAT,
           label = Y_CAT,
           cat_features = CAT_features)
print('pool_data的 type 是：', type(pool_data))
print('pool_data的 shpe 是：', pool_data.shape)
print('pool_data.get_features()返回的是list类型，其长度是：',len(pool_data.get_features()))
print('pool_data.get_label()返回的是list类型，其长度是：', len(pool_data.get_label()))
print('pool_data中类别变量所在的索引位置是 pool_data.get_cat_feature_indices() ：', pool_data.get_cat_feature_indices())
#print('生成的pool_data的各观测的weight：', pool_data.get_weight())
#print('生成的pool_data的各观测的baseline：', pool_data.get_baseline())


#### Step2.1 自定义metric类。用以做最优模型选择和过拟合检测

# **************Custom metric for overfitting detector and best model selection******
import math
from catboost import Pool, CatBoostClassifier


class Recall_1_Metric(object):
    def get_final_error(self, error, weight):
        return error / (weight + 1e-38)

    def is_max_optimal(self):
        return True

    def evaluate(self, approxes, target, weight):
        # approxes is list of indexed containers (containers with only __len__ and __getitem__ defined), one container
        # per approx dimension. Each container contains floats.
        # weight is one dimensional indexed container.
        # target is float.

        # weight parameter can be None.
        # Returns pair (error, weights sum)

        assert len(approxes) == 1
        assert len(target) == len(approxes[0])

        approx = approxes[0]

        weight_sum = 0.0
        TP_and_FN = 0
        TP = 0
        for i in range(len(approx)):
            w = 1.0 if weight is None else weight[i]
            weight_sum += w
            TP_and_FN += target[i]
            if target[i] == 1 and approx[i] == 1:
                TP += 1

        error_sum = float(1 - TP / TP_and_FN)
        return error_sum, weight_sum

#### Step2.2 自定义LossFunction类。用来做模型参数的学习。
import math
from catboost import Pool, CatBoostClassifier

class Recall_1_objective(object):
    def calc_ders_range(self, approxes, targets, weights):
        # approxes, targets, weights are indexed containers of floats (containers with only __len__ and __getitem__ defined).
        # weights parameter can be None.
        # Returns list of pairs (der1, der2)
        assert len(approxes) == len(targets)
        if weights is not None:
            assert len(weights) == len(approxes)

        exponents = []
        for index in xrange(len(approxes)):
            exponents.append(math.exp(approxes[index]))

        result = []
        for index in range(len(targets)):
            p = exponents[index] / (1 + exponents[index])
            der1 = (1 - p) if targets[index] > 0.0 else -p
            der2 = -p * (1 - p)

            if weights is not None:
                der1 *= weights[index]
                der2 *= weights[index]

            result.append((der1, der2))

        return result

from catboost import CatBoostClassifier
# Initialize data：
# X_CAT , Y_CAT , CAT_features,
# X_CAT_train, Y_CAT_train,
# X_CAT_test, Y_CAT_test
# Initialize CatBoostClassifier
model = CatBoostClassifier(loss_function = 'Logloss',
                            custom_metric = 'Accuracy',
                            eval_metric = Recall_1_Metric(),
                            bootstrap_type = 'No',
                            use_best_model = True,
                            class_weights = [100,1],
                            iterations=1000,
                            learning_rate=1,
                            depth=10,
                            od_type = None,#'IncToDec',
                            #od_pval = 0.0001,
                            #od_wait = 20,
                            logging_level = 'Verbose')
# Fit model
model.fit(X = X_CAT_train,
          y = Y_CAT_train,
          cat_features = CAT_features,
          eval_set = (X_CAT_test,Y_CAT_test),
          sample_weight = None,
          verbose = True,
          plot = False)
# Get predicted classes
#preds_class = model.predict(X_CAT_test)
# Get predicted probabilities for each class
#preds_proba = model.predict_proba(X_CAT_test)
# Get predicted RawFormulaVal
#preds_raw = model.predict(X_CAT_test, prediction_type='RawFormulaVal')

### 【分组统计】将数据框按商圈和品牌分别groupby

#看一下商圈的分布: 共517个商圈。
len(model_data0['hdic_bizcircle_id'].unique())
##看一下品牌的分布: 共399个品牌。
len(model_data0['app_source_brand'].unique())
#按商圈进行 groupby
model_group0 = model_data0.groupby('hdic_bizcircle_id')
#for name, value in model_group0:
#    print(name)
#    print(value)
# selecting a group
# 提取出商圈id为18335623的所有样本的信息：
model_group0.get_group(18335623)

### 【分组统计】将数据框按租赁类型groupby
#看一下租赁类型的分布:
#model_data0['rent_type'].unique()
print('租赁类型rent_type的频数分布表（value_counts）是：\n',model_data0['rent_type'].value_counts())
# model_group_type = model_data0.groupby('rent_type')
#for name, value in model_group0:
#    print(name)
#    print(value)
# selecting a group
# 提取出商圈id为18335623的所有样本的信息：
# model_group0.get_group(18335623)

### 【数据预处理1】: 将3个categoory列 做LabelEncode；并将新列添加到原数据框中。
# 数据预处理: 将3个categoory column 做 LabelEncode
from sklearn import preprocessing

le_brand = preprocessing.LabelEncoder()
model_data0['app_source_brand_le'] = le_brand.fit_transform(model_data0[['app_source_brand']])

le_bizcircle = preprocessing.LabelEncoder()
model_data0['hdic_bizcircle_id_le'] = le_bizcircle.fit_transform(model_data0[['hdic_bizcircle_id']])

le_type = preprocessing.LabelEncoder()
model_data0['rent_type_le'] = le_type.fit_transform(model_data0[['rent_type']])
#le.transform([1, 1, 2, 6])
#le.inverse_transform([0, 0, 1, 2])

### 【数据预处理2】：用对响应变量audit_result 进行0-1化得到：audit_result_0_1，并将其添加到数据框中：
#数据预处理：用对结果变量audit_result 进行0-1化：保存在新变量：audit_result_0_1 中。
#from sklearn.preprocessing import label_binarize
#model_data0['audit_result_0_1'] = label_binarize(model_data0['audit_result'], classes=[101])
from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()
model_data0['audit_result_0_1'] = lb.fit_transform(model_data0['audit_result'])
#model_data0['audit_result_0_1'].value_counts()

### 【数据预处理3】: 将3个经过LabelEncoder编码之后的类别列；再做 OneHotEncode，保存在稀疏矩阵model_data1中。
# 数据预处理: 将3个经过LabelEncoder编码之后的categoory_le 列再做 OneHotEncode。
# 将包含one-hot编码的稀疏矩阵 用变量：model_data1 来指代。
from sklearn import preprocessing
#one_hot_encode = preprocessing.OneHotEncoder(categorical_features=['app_source_brand_le','hdic_bizcircle_id_le','rent_type_le'])
one_hot_encode = preprocessing.OneHotEncoder(categorical_features = [0,1,2])
model_data1 = one_hot_encode.fit_transform(model_data0[['app_source_brand_le','hdic_bizcircle_id_le','rent_type_le','frame_bedroom_num','rent_price_listing','audit_result_0_1']])
#这样生成的 model_data1 是‘scipy.sparse.coo.coo_matrix’ 类型的稀疏矩阵。
#输出one-hot之后稀疏矩阵的各个性质：
print('稀疏矩阵model_data1的type是：',type(model_data1))
print('稀疏矩阵的dtype是：',model_data1.dtype)
print('稀疏矩阵的shape是：',model_data1.shape)
print('稀疏矩阵的ndim是：',model_data1.ndim)
print('稀疏矩阵的nnz是：',model_data1.nnz)
print('稀疏矩阵的data是：',model_data1.data)
print('稀疏矩阵的row是：',model_data1.row)
print('稀疏矩阵的col是：',model_data1.col)
print('稀疏矩阵调用toarray()方法得',model_data1.toarray())
print('稀疏矩阵调用todense()方法得密集矩阵model_data2',model_data1.todense())
# 稀疏矩阵可通过如下语句转化为密集矩阵：
model_data2 = model_data1.todense()
print('稀疏矩阵的特征变量范围是：',one_hot_encode.feature_indices_)
print('稀疏矩阵model_data1的[0,399]列对应特征app_source_brand_le ；[399,916]对应特征hdic_bizcircle_id_le ；[916,919]对应特征rent_type_le\
919对应特征frame_bedroom_num；920对应特征rent_price_listing；最后一列921对应响应分类值：audit_result_0_1')

print('\n将coo_matrix类型的稀疏矩阵[model_data1],转化为csc_matrix类型的稀疏矩阵[model_data3],这样转化的原因有两个：\n\
(1): csc格式支持列切片efficient column slicing\n（2）：sklearn中的机器学习方法只支持对csc格式的稀疏矩阵fit')
model_data3 = model_data1.tocsc()

### 稀疏矩阵的切片操作
#print('稀疏矩阵的切片：',model_data1[:,-[920,921]])
print('CSC稀疏矩阵model_data3的shape是：',model_data3.shape)
print('\nCSC稀疏矩阵的列切片：\n')
print('CSC稀疏矩阵model_data3的列切片当col=0时：\n',model_data3[:,0])
print('CSC稀疏矩阵model_data3的列切片当col=[3:5]时：\n',model_data3[:,3:5])
print('注意区别：onehot之后时对应某个特征，每一行只有一个值取1，其余取0，但某一列缺不一定是只有一个1')

print('\nCSC稀疏矩阵的行切片：\n')
print('CSC稀疏矩阵model_data3的行切片: \n',model_data3[0,:])

### 数据预处理完成后，再看数据框model_data0的各变量类型：
### 【获取数据集】得到能够被树模型直接使用的数据集。
X, Y,
X_CSC, Y,
x_train,  x_test,  y_train,  y_test,
x_csc_train, x_csc_test, y_csc_train, y_csc_test

X = model_data0[['app_source_brand','hdic_bizcircle_id','rent_type','rent_price_listing','frame_bedroom_num']]
Y = model_data0['audit_result_0_1']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = c( X, Y, test_size=0.33, random_state=42,stratify = Y)
print('训练集 x_train 的shape是',x_train.shape)
print('测试集 x_test  的shape是',x_test.shape)
print('训练集 y_train 的shape是%s:',y_train.shape)
print('测试集 y_test  的shape是%s:',y_test.shape)
print('训练集中label的正负比例分布如下：\n',y_train.value_counts())
print('测试集中label的正负比例分布如下：\n',y_test.value_counts())
print('可以看出在划分训练集和测试集时设定strtify参数为true；使得在训练测试集中正负例所占比例一致。')

print('\nCSC稀疏矩阵model_data3去掉最后一列响应变量列之后：得到稀疏矩阵自变量 X_CSC ')
X_CSC = model_data3[:,:-1]
print('稀疏矩阵自变量 X_CSC的shape是：',X_CSC.shape)
x_csc_train, x_csc_test, y_csc_train, y_csc_test = train_test_split( X_CSC, Y, test_size=0.33, random_state=42,stratify = Y)

### 【定义模型训练函数】model_fit_process（model_name,clf,X_train,Y_train）

# 函数需要传入的参数分别是：estimator, 训练集X，训练集Y。
def model_fit_process(model_name,clf,X_train,Y_train):
    clf.fit(X_train, Y_train)
    print('\n',model_name+'的训练过程的信息如下：')
    print(model_name+'的超参数设置如下:')
    for key_value in clf.get_params().items():
         print(key_value)
    print('训练集的样本数目和特征数目分别是：',X_train.shape)
    #print('训练集上的特征顺序和数据类型如下:\n',X.dtypes)
    #print('这个决策树在训练集上学习到的特征重要性排序是：\n',clf.feature_importances_)
    #print(model_name+'在训练集上学习到的特征重要性从大到小的排序是：\n',\
    #pd.DataFrame(clf.feature_importances_,index=X_train.columns,columns = ['feature_importances']).sort_values(by = ['feature_importances'],ascending = False)
    #)
    print(model_name+'的预测变量（audit_result）：是否价格异常房源。其取值为：',clf.classes_)
    #print('这个模型包含的叶子节点的数目是：',clf.tree_.node_count)

### 【定义模型预测函数】model_predict_process(model_name,clf,X_test,Y_test)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score


def model_predict_process(model_name, clf, X_test, Y_test):
    '''
    clf: estimator,并且已经调用过fit方法。
    X： 测试集X
    '''
    print('\n', model_name, '的预测过程的信息如下：')
    Y_predict = clf.predict(X_test)
    predict_mask = (Y_predict != Y_test)
    # X['predict_mask'] = predict_mask
    # print(X[X['predict_mask']==True])
    print('根据预测结果的正确与否，生成一个indicator变量：predict_mask')
    print(model_name + '在测试集上预测错误的样本总数是：', sum(predict_mask))
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(Y_test, Y_predict)
    print(model_name + '在测试集上预测的混淆矩阵是：\n', cnf_matrix)
    print(model_name + '在测试集上预测的normalized混淆矩阵是：\n', cnf_matrix / cnf_matrix.sum(axis=1)[:, np.newaxis])
    print(model_name + '在测试集上预测的recall rate [average=None] 是：', recall_score(Y_test, Y_predict, average=None))
    # print(model_name+'在测试集上预测的recall rate [average=binary] 是：',recall_score(Y_test, Y_predict, average='binary'))
    # print(model_name+'在测试集上预测的recall rate [average=micro] 是：',recall_score(Y_test, Y_predict, average='micro'))
    # print(model_name+'在测试集上预测的recall rate [average=macro] 是：',recall_score(Y_test, Y_predict, average='macro'))
    # print(model_name+'在测试集上预测的recall rate [average=weighted] 是：',recall_score(Y_test, Y_predict, average='weighted'))
    # print('这个决策树在测试集上预测的recall rate [average=samples] 是：',recall_score(Y_test, Y_predict, average='samples'))
    print(model_name + '在测试集上预测的平均准确率是：%.2f' % clf.score(X_test, Y_test))
    # print(model_name+'将测试集上的前10个样本预测为各个target-label的probability是：','\n',clf.predict_proba(X_test.iloc[:10]))
    # print(model_name+'将测试集上的前10个样本归入的叶子节点序号是：\n',clf.apply(X_test)[:10])
    # print('这个决策树将训练集X中各个样本点的决策路径（存储为稀疏矩阵） shape = [n_samples, n_nodes]：','\n',clf.decision_path(X))


### 【定义模型交叉验证函数】cross_validate_process(model_name,clf)
# 定义模型交叉验证函数：
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
#from sklearn.model_selection import learning_curve
#from sklearn import tree
#from sklearn.metrics import SCORERS
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
## 这里定义（封装）了使用交叉验证进行训练和预测的函数：
def cross_validate_process(model_name,clf):
    #注意：函数内部用到的 X 和 Y 是 全局变量。
    #for key_value in cross_validate(clf, X, Y, cv=10).items():
    #    print(key_value)
    y_pred = cross_val_predict(clf, X, Y,cv =10)
    print('\n',model_name+'交叉验证的recall_rate是：\n',recall_score(Y, y_pred, average=None))
    print(model_name+'交叉验证的confusion_matrix是：\n',confusion_matrix(Y, y_pred))
    #print('recall:',cross_val_score(clf, X, Y, cv=10, scoring ='recall'))
    #print('recall_macro:',cross_val_score(clf, X, Y, cv=10, scoring ='recall_macro'))
    #len(cross_val_predict(clf, X, Y, cv=10))
    #for value in learning_curve(clf, X, Y, cv=10):
    #    print(value,'\n\n')
    #SCORERS

### 【模型学习+预测】单个决策树+全数据集
# 首先使用默认树+数据集全集(X,Y)作为训练集来fit模型。
from sklearn import tree
model_name = '单个决策树模型'
clf = tree.DecisionTreeClassifier(class_weight={1:1,0:15})
model_fit_process(model_name,clf,X,Y)
model_predict_process(model_name,clf,X,Y)
print(model_name+'的节点数目共：',clf.tree_.node_count)


### 【模型学习+预测】单个决策树+稀疏全数据集
# 注意decisontree的fit方法：接受的sparsematrix  is provided to a sparse csc_matrix.
# csc_matrix : compressed-sparse-column matrix的意思
from sklearn import tree
model_name = '单个决策树模型+稀疏训练集'
clf = tree.DecisionTreeClassifier(class_weight={1:1,0:15})
model_fit_process(model_name,clf,X_CSC,Y)
model_predict_process(model_name,clf,X_CSC,Y)
print(model_name+'的节点数目共：',clf.tree_.node_count)

### 【决策树可视化】单个决策树可视化--保存在个人工作区dot文件和png文件
#coding=utf-8
from sklearn.externals.six import StringIO
import pydot
from sklearn import tree
model_name = '单个决策树模型'
clf = tree.DecisionTreeClassifier(class_weight={1:1,0:15})
clf.fit(X, Y)
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph[0].write_dot('decision_tree.dot')
graph[0].write_png('decision_tree.png')

### 【模型学习和预测】单个决策树+训练/测试数据集
from sklearn import tree
model_name = '单个决策树模型'
clf = tree.DecisionTreeClassifier(class_weight={1:1,0:15})
#关于超参数的调整：调整class_weight 这个参数。以提高在测试集上面负例预测的准确率。
#model_fit_process(clf.set_params(class_weight=None),x_train,y_train)
#model_fit_process(clf.set_params(class_weight='balanced'),x_train,y_train)
model_fit_process(model_name,clf,x_train,y_train)
model_predict_process(model_name,clf,x_test,y_test)
print('通过模型在测试集上预测的混淆矩阵可知：模型倾向于将样本预测为正例，效果并不好')

### 【模型学习和预测】单个决策树+稀疏训练/测试数据集
from sklearn import tree
model_name = '单个决策树模型稀疏+训练测试'
clf = tree.DecisionTreeClassifier(class_weight={1:1,0:15})
#关于超参数的调整：调整class_weight 这个参数。以提高在测试集上面负例预测的准确率。
#model_fit_process(clf.set_params(class_weight=None),x_train,y_train)
#model_fit_process(clf.set_params(class_weight='balanced'),x_train,y_train)
model_fit_process(model_name,clf,x_csc_train,y_csc_train)
model_predict_process(model_name,clf,x_csc_test,y_csc_test)
print('通过模型在测试集上预测的混淆矩阵可知：模型倾向于将样本预测为正例，效果并不好')

### 【模型学习和预测】单个决策树+十折交叉验证
model_name = '单个决策树'
clf = tree.DecisionTreeClassifier(class_weight={1:1,0:15},random_state=0)
# 调用交叉验证的函数：
cross_validate_process(model_name,clf)

### 【模型学习和预测】随机森林+全数据集训练预测&十折交叉验证
# 随机森林分类器
from sklearn.ensemble import RandomForestClassifier
# 注意随机森林超参数的选择：森林特有的参数：bootstrap=False。决策树的参数： class_weight={1:1,0:15}
model_name = '10个树的随机森林'
# 随机森林在全数据集上面学习和预测
clf = RandomForestClassifier(bootstrap=False,class_weight={1:1,0:15})
model_fit_process(model_name,clf,X_CSC,Y)
model_predict_process(model_name,clf,X_CSC,Y)
# 随机森林做十折交叉验证。输出 混淆矩阵 和 recall-rate
clf = RandomForestClassifier(bootstrap=False,class_weight={1:1,0:15})
cross_validate_process(model_name,clf)

### 【模型学习和预测】Adaboost+全数据集训练预测&十折交叉验证
# Create and fit an AdaBoosted decision tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
model_name = '200个树的Adaboost'
clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1,class_weight={1:1,0:15}),
                         algorithm="SAMME",
                         n_estimators=200)
model_fit_process(model_name,clf,X_CSC,Y)
model_predict_process(model_name,clf,X_CSC,Y)

clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1,class_weight={1:1,0:15}),
                         algorithm="SAMME",
                         n_estimators=200)
cross_validate_process(model_name,clf)

