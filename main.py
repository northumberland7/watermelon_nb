"""西瓜数据集 3.0：GaussianNB 二分类。
唯一数据集：watermelon3.0_source.csv
主模型仅使用「密度」和「含糖率」两个连续特征。
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = BASE_DIR / 'figures'
FIG_DIR.mkdir(exist_ok=True)

df = pd.read_csv(BASE_DIR / 'watermelon3.0_source.csv', encoding='utf-8')

#加载与统计
if __name__ == '__main__':
    print('形状:', df.shape)
    print('特征数:', df.shape[1] - 1) #取第二个变量，是列数（特征加上标签），减去一是因为最后一列是标签
    print('类别分布:', df['好瓜'].value_counts().to_dict())
    print('缺失值总数:', df.isna().sum().sum()) 
    #第一个sum是对每一列求和，第二个sum是对所有列的和求和，得到缺失值总数

#画类别分类图
counts = df['好瓜'].value_counts().sort_index()
'''sort_index() 按标签值排序。中文按 Unicode 码位比大小，「否」排在「是」前面，
所以 counts.values 是 [9, 8]，和下面横轴写的 bad、good 顺序对得上。   '''

plt.figure()
plt.bar(['bad (0)', 'good (1)'], counts.values)
plt.title('Class distribution')
plt.xlabel('Class')
plt.ylabel('Number of samples')
plt.savefig(FIG_DIR / 'class_distribution.png')
plt.show()

#画密度-含糖率散点图
plt.figure()
plt.scatter(df.loc[df['好瓜'] == '是', '密度'],
            df.loc[df['好瓜'] == '是', '含糖率'],
            label='good', marker='o')
            #索引好瓜的密度和含糖率，用圆点画出好瓜的散点图

plt.scatter(df.loc[df['好瓜'] == '否', '密度'],
            df.loc[df['好瓜'] == '否', '含糖率'],
            label='bad', marker='x')
            #索引坏瓜的密度和含糖率，用叉号画出坏瓜的散点图

plt.title('Density vs Sugar ratio')
plt.xlabel('Density')
plt.ylabel('Sugar ratio')
plt.legend()
plt.savefig(FIG_DIR / 'scatter.png')
plt.show()

#预处理和pipeline，包括选出两列特征、把中文标签换成0和1、划分训练集和测试集，并封装Pipeline
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

X = df[['密度', '含糖率']]
y = (df['好瓜'] == '是').astype(int) #逐行比较，得到一列True/False，再用 astype(int) 转成 0/1，其中 True 对应 1，False 对应 0

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
)  #stratify=y 表示按 y 的类别比例划分训练集和测试集，保证训练集和测试集的类别分布一致

pipe = Pipeline([('nb', GaussianNB())])
pipe.fit(X_train, y_train)
#这次训练没有进行特征缩放，也就是标准化，因为 GaussianNB 对特征的尺度不敏感。特征缩放是对梯度下降类的模型有帮助，比如线性回归、逻辑回归、神经网络等。

print('训练集:', X_train.shape, ' 测试集:', X_test.shape)
print('训练集类别:', y_train.value_counts().to_dict()) #得到训练集里好瓜和坏瓜的数量，出现顺序是 0、1，转成字典后，字典的键顺序是按出现次数排的
print('测试集类别:', y_test.value_counts().to_dict())

#看训练好的gaussianNB模型的参数
nb = pipe.named_steps['nb']
#named_steps 是 Pipeline 的一个属性，它是一个字典，键是步骤名，值是对应的模型对象。这里我们取出名为 'nb' 的步骤，也就是 GaussianNB 模型。

print('theta_ (每类每特征的均值):') 
print(nb.theta_)
#theta_ 是 GaussianNB 模型的一个属性，它是一个二维数组，表示每个类别每个特征的均值。第一维是类别，第二维是特征。对于二分类问题， theta_ 的形状是 (2, n_features)，其中 n_features 是特征数。

print('var_ (每类每特征的方差):')
print(nb.var_)
#var_ 是 GaussianNB 模型的一个属性，它是一个二维数组，表示每个类别每个特征的方差。

print('class_prior_ (类先验):', nb.class_prior_)
'''class_prior_ 是 GaussianNB 模型的一个属性，它是一个一维数组，表示每个类别的先验概率。
对于二分类问题， class_prior_ 的形状是 (2,)，其中第一个元素是类别 0 的先验概率，第二个元素是类别 1 的先验概率。
先验概率就是训练集中每个类别的样本数占总样本数的比例。'''

print('classes_:', nb.classes_)
#classes_ 是 GaussianNB 模型的一个属性，它是一个一维数组，表示类别标签。其中第一个元素是类别 0 的标签，第二个元素是类别 1 的标签。
#注意，这些模型的属性后面都有一个下划线，表示它们是训练好的模型的属性，而不是初始化模型时的参数。

y_pred = pipe.predict(X_test)
#这里预测时比的是贝叶斯公式的分子，predict 对每个样本分别算「似然 × 先验」，得到两个未归一化的后验（分子），谁大判给谁。

y_prob = pipe.predict_proba(X_test)
#这下是把上一步的结果各自除以两类分子之和，即归一化（让他们加起来等于1），输出的是后验概率（即假设他是这一类的基础上，它拥有这些特征的概率有多大？）

print('测试集真实标签:', y_test.tolist())
print('测试集预测标签:', y_pred.tolist())
print('测试集概率:')
print(y_prob.round(4))

test1 = pd.DataFrame({'密度': [0.697], '含糖率': [0.460]})
#这里是构造了一个新的样本，注意值要用列表包起来，列名必须和训练集对应

print('测1 预测:', pipe.predict(test1))
print('测1 概率:', pipe.predict_proba(test1).round(6))

#对模型进行评估
#先是准确率、混淆矩阵、分类报告
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print('准确率:', round(accuracy_score(y_test, y_pred), 4))

cm = confusion_matrix(y_test, y_pred)
print('混淆矩阵:')
print(cm)

print(classification_report(y_test, y_pred, digits=4))
#分类报告给出每类的精确率、召回率、F1

#留一法：因为样本太少，粒度太粗，说明不了模型水平。留一法让每个样本都当一次测试集（其余16个训练）
from sklearn.model_selection import LeaveOneOut, cross_val_score

loo_scores = cross_val_score(GaussianNB(), X, y, cv=LeaveOneOut())
#这里用的交叉验证函数，参数填了留一法的切分器。注意这里每轮都是一个全新的模型，返回十七次的准确率得分

print('留一法每折得分:', loo_scores.astype(int).tolist())
print('留一法平均准确率:', round(loo_scores.mean(), 4))
print('留一法方差:', round(loo_scores.var(), 4))

#在密度——含糖率平面画出决策边界（边界是两个后验概率相等的情况，即在这样的特征条件下，它是好瓜和坏瓜的概率是等可能的）
from sklearn.inspection import DecisionBoundaryDisplay

mis = y_pred != y_test

plt.figure()
DecisionBoundaryDisplay.from_estimator(
    pipe, X_train, response_method='predict', alpha=0.25, grid_resolution=300,
    ax=plt.gca())  
''' 传的参数有训练好的模型，训练集特征（决定画多大的范围，取训练集的密度，含糖率的最小最大值；
    response_method参数是指用什么标准来划分区域，这里是预测的标签。如果改成predict_proba就会按概率深浅渐变
    alpha设定底色透明度；grid_resolution是设定网格密度，plt.gca()是当前坐标轴的意思'''

plt.scatter(X_train.loc[y_train == 1, '密度'], X_train.loc[y_train == 1, '含糖率'],
            label='train good', marker='o', s=25)
#画出训练集里面好瓜的密度——含糖率点

plt.scatter(X_train.loc[y_train == 0, '密度'], X_train.loc[y_train == 0, '含糖率'],
            label='train bad', marker='x', s=25)
#画出训练集里坏瓜的密度——含糖率点

plt.scatter(X_test.loc[~mis, '密度'], X_test.loc[~mis, '含糖率'],
            label='test (correct)', marker='s', s=70, facecolors='none', edgecolors='black')
#预测对的点

plt.scatter(X_test.loc[mis, '密度'], X_test.loc[mis, '含糖率'],
            label='test (misclassified)', marker='s', s=90, color='red')
''' mis = y_pred != y_test，即得到一个列表，如果预测对了就是false，预测错了就是true
    现在对它取反，那么预测对了就是true,预测错了就是false，又因为loc是保留行索引条件为true的，
    所以得到的是预测对的。那么同理，对于行索引是mis的，就是取的预测错的'''

plt.xlim(0.18, 0.85)
plt.ylim(0.0, 0.52)
#限定坐标范围

plt.title('GaussianNB decision boundary')
plt.xlabel('Density') 
plt.ylabel('Sugar ratio')
plt.legend()
plt.tight_layout()  #避免图例压住数据点
plt.savefig(FIG_DIR / 'decision_boundary.png')
plt.show()

#先验与平滑分析，改参数看输出怎么变。先验是根据生活经验人为修正两类的比例（比如抛硬币的情况就是正反各一半），防止训练集因为抽样偶然偏了
#平滑分析是给每个方差加上一个很小的数，以防某个特征上方差恰好为零，高斯密度公式会出错（除以零）

nb_def = GaussianNB().fit(X_train, y_train) #写成一行是因为 fit 会交回模型自己（返回 self），所以能接着赋值。
nb_half = GaussianNB(priors=[0.5, 0.5]).fit(X_train, y_train) #又创了一个新的模型，手动调先验

print('默认先验:', nb_def.class_prior_)
print('手动先验:', nb_half.class_prior_)
print('默认预测:', nb_def.predict(X_test))
print('手动预测:', nb_half.predict(X_test))
print('默认好瓜概率:', nb_def.predict_proba(X_test)[:, 1].round(4))
print('手动好瓜概率:', nb_half.predict_proba(X_test)[:, 1].round(4))
'''手动调整先验之后发现后验概率升高，因为好瓜那一侧的权重变大了，似然没变，那么乘上去好瓜的贝叶斯分子变大，概率升高'''

for vs in [1e-9, 1e-3, 1e-2, 0.1, 0.5]:
    m = GaussianNB(var_smoothing=vs).fit(X_train, y_train) #注意是每轮都要新建模型并传vs进去
    prob = m.predict_proba(X_test)[:, 1]  #取索引为1的列（即第二列）的所有行，得到预测它是好瓜样本的概率
    print(f'vs={vs:<8} 预测={m.predict(X_test).tolist()} '
          f'好瓜概率={prob.round(3).tolist()}')

#交叉验证，既用留一法也用5折法，以评估模型的泛化能力
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_score

loo_scores = cross_val_score(GaussianNB(), X, y, cv=LeaveOneOut())
skf_scores = cross_val_score(
    GaussianNB(), X, y,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
)
'''这里的5折和以前不同的是，为了避免某一折完全是好瓜，另一折全是坏瓜，stratified的意思是切的时候保证好坏瓜的比例，而且传参数shuffle即切之前打乱顺序'''

print('留一法 平均:', round(loo_scores.mean(), 4), ' 方差:', round(loo_scores.var(), 4))
print('5 折   平均:', round(skf_scores.mean(), 4), ' 方差:', round(skf_scores.var(), 4))
print('5 折每折得分:', skf_scores.round(4).tolist())

#同样做不同的varsmoothing值与交叉验证的准确率的曲线
vs_list = [1e-9, 1e-7, 1e-5, 1e-3, 1e-2, 0.05, 0.1, 0.5, 1.0]
means = []

for vs in vs_list:
    s = cross_val_score(GaussianNB(var_smoothing=vs), X, y, cv=LeaveOneOut())
    #每一轮以不同的varsmoothing值赋给全新的模型，得到留一法交叉验证模型下的准确率
    means.append(s.mean())

plt.figure()
plt.plot(vs_list, means, marker='o') 
plt.xscale('log') #同样地，因为vs值范围太大，取对数让他分布均匀
plt.title('LOO accuracy vs var_smoothing')
plt.xlabel('var_smoothing (log scale)')
plt.ylabel('Mean LOO accuracy')
plt.tight_layout()
plt.savefig(FIG_DIR / 'vs_curve.png')
plt.show()

#离散特征拓展，将 6 个离散特征用 OrdinalEncoder 编码为整数，使用 CategoricalNB 单独训练并评估；
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import CategoricalNB

disc_cols = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']
enc = OrdinalEncoder(dtype=int) #创建一个能编码成整数的编码器
Xd = enc.fit_transform(df[disc_cols]) #fit 先学每列有哪些取值怎么排序编号，transform再按学到的表把文字转成编号，得到一个编码后的特征矩阵
#ordinalencoder会把每一列的文字映射成整数，按字符的unicode编码排，比如对于色泽列，乌黑=0、浅白=1、青绿=2

print('编码后形状:', Xd.shape)
print('每列的映射:', enc.categories_)
print('前 3 行:')
print(Xd[:3])

#对编码后的离散数据训练并评估
Xd_train, Xd_test, yd_train, yd_test = train_test_split(
    Xd, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
)

cnb = CategoricalNB(alpha=1.0).fit(Xd_train, yd_train)
#alpha是拉普拉斯修正的强度，即计算每个特征的似然的时候，分子上每个特征的取值都加上一（同时分母要加上这个特征的可取值数）

print('离散特征模型预测:', cnb.predict(Xd_test).tolist()) 
#sklearn的机制和手写不一样在于它把特征似然的连乘取对数，变成相加避免数太小溢出。然后比较分子（似然连乘 乘以 先验），哪类大就判成哪一类

print('离散特征模型准确率:', round(cnb.score(Xd_test, yd_test), 4))
#.score交回准确率，即判对的样本数/样本总数，注意它接受的参数是测试特征和测试真实的标签，它自己会算并比对。要和accuracy_score区分开，后者是接收预测标签和真实标签

loo_cnb = cross_val_score(CategoricalNB(), Xd, y, cv=LeaveOneOut())
print('离散特征留一法:', round(loo_cnb.mean(), 4))














































































