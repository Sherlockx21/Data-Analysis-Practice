#导入基本的库
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")#忽略警告

#读取数据
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#去除离群值
outliers=np.abs(train['count']-train['count'].mean()) > (3*train['count'].std()) #std为标准差
train.drop(index=train[outliers].index)


# from datetime import datetime
# import calendar
#对时间进行提取
def time_process(df):
    #年、月、日、小时特征提取
    #DatetimeIndex为一组Timestamp构成的Index,作为Series或DataFrame的索引
    df['year'] = pd.DatetimeIndex(df['datetime']).year
    df['month'] = pd.DatetimeIndex(df['datetime']).month
    df['day'] = pd.DatetimeIndex(df['datetime']).day
    df['hour'] = pd.DatetimeIndex(df['datetime']).hour
    df['minute'] = pd.DatetimeIndex(df['datetime']).minute
    #将日期的礼拜数标出，以探究工作日、双休日的特征
    df['week'] = pd.DatetimeIndex(df['datetime']).weekofyear
    df['weekday'] = pd.DatetimeIndex(df['datetime']).dayofweek
    #df['weekday'] = pd.DatetimeIndex(df['datetime']).weekday
    return df

train = time_process(train)
test = time_process(test)

#风速有很多缺失值，尝试使用随机森林基于气温、季节插值
from sklearn.ensemble import RandomForestClassifier
#sklearn为python的机器学习库，分为分类、回归、聚类、降维、模型选择和预处理六个模块
#补充风速缺失值
def wind_0_fill(df):
    wind_0 = df[df['windspeed']==0]
    wind_not0 = df[df['windspeed']!=0]
    y_label = wind_not0['windspeed']
    #猜测风速和天气以及时间都有关
    #RandomForestClassifier,n_estimators树木数量,max_depth树最大深度，random_state样本引导随机性
    clf = RandomForestClassifier(n_estimators=1000,max_depth=10,random_state=0)
    windcolunms = ['season', 'weather', 'temp', 'atemp', 'humidity', 'hour', 'month']
    #在非0的风速列构建森林
    clf.fit(wind_not0[windcolunms], y_label.astype('int'))
    #对风速缺失处进行预测
    pred_y = clf.predict(wind_0[windcolunms])
    #预测结果填充
    wind_0['windspeed'] = pred_y
    df_rfw = wind_not0.append(wind_0)
    df_rfw.reset_index(inplace=True)
    return df_rfw

train = wind_0_fill(train)
test = wind_0_fill(test)
#train

# 将行索引改为datetime
dt = pd.DatetimeIndex(train['datetime'])
#改变索引，对源数据生效
train.set_index(dt, inplace=True)
dtt = pd.DatetimeIndex(test['datetime'])
test.set_index(dtt, inplace=True)
def get_day(day_start):
    day_end = day_start + pd.offsets.DateOffset(hours=23)  #设置23h的偏移量
    return pd.date_range(day_start, day_end, freq="H")    #遍历一天的时间，以小时为间隔

# 纳税日，仍需工作
train.loc[get_day(pd.datetime(2011, 4, 15)), "workingday"] = 1
train.loc[get_day(pd.datetime(2012, 4, 16)), "workingday"] = 1
# 感恩节，不需要工作
test.loc[get_day(pd.datetime(2011, 11, 25)), "workingday"] = 0
test.loc[get_day(pd.datetime(2012, 11, 23)), "workingday"] = 0
#圣诞节，不工作
test.loc[get_day(pd.datetime(2011, 12, 24)), "workingday"] = 0
test.loc[get_day(pd.datetime(2011, 12, 31)), "workingday"] = 0
test.loc[get_day(pd.datetime(2012, 12, 26)), "workingday"] = 0
test.loc[get_day(pd.datetime(2012, 12, 31)), "workingday"] = 0

# 纳税日，不放假
train.loc[get_day(pd.datetime(2011, 4, 15)), "holiday"] = 0
train.loc[get_day(pd.datetime(2012, 4, 16)), "holiday"] = 0

# 感恩节，放假
test.loc[get_day(pd.datetime(2011, 11, 25)), "holiday"] = 1
test.loc[get_day(pd.datetime(2012, 11, 23)), "holiday"] = 1
#圣诞节，放假
test.loc[get_day(pd.datetime(2011, 12, 24)), "holiday"] = 1
test.loc[get_day(pd.datetime(2011, 12, 31)), "holiday"] = 1
test.loc[get_day(pd.datetime(2012, 12, 31)), "holiday"] = 1

#暴雨
test.loc[get_day(pd.datetime(2012, 5, 21)), "holiday"] = 1
#海啸
train.loc[get_day(pd.datetime(2012, 6, 1)), "holiday"] = 1

def name_process(df):
    #季节、天气重命名，后续模型将用独热编码，易于可视化
    df['season2'] = df['season']
    df['weather2'] = df['weather']
    df['month2'] = df['month']
    df['season2'] = df['season2'].map({1:'Spring',2:'Summer',3:'Fall',4:'Winter'})
    df['weather2'] = df['weather2'].map({1: 'Clear', 2: 'Mist', 3: 'Light_Snow', 4: 'Heavy_Rain'})
    df['month2'] = df['month2'].map({1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'})
    return df

train = name_process(train)
test = name_process(test)



#样本正态分布情况一般
train['count'].plot(kind='kde')
sns.distplot(train['count'],color='b')
plt.savefig(r'C:\Users\hang\Desktop\all\programming\sf1\pics\count1.png')
plt.show()



#进行log1p变换
import math
train['count_log']=train['count'].apply(lambda x: math.log(x+1))
train['count_log'].plot(kind='kde')
sns.distplot(train['count_log'],color='b')
# plt.savefig(r'C:\Users\hang\Desktop\all\programming\sf1\pics\count2.png')
plt.show()





#可视化分析
#不同时间段--使用量
sns.boxplot(x='hour',y='count',data=train)
plt.title('Users Count per Hour')
#plt.savefig(r'C:\Users\hang\Desktop\all\programming\sf1\pics\hour.png')
plt.show()

#根据数据，解译出使用量高峰时段
train['peak'] = train[['hour', 'workingday']].apply(lambda x: (0, 1)[(x['workingday'] == 1 and  ( x['hour'] == 8 or 17 <= x['hour'] <= 18 or 11 <= x['hour'] <= 12)) or (x['workingday'] == 0 and  10 <= x['hour'] <= 19)], axis = 1)
test['peak'] = test[['hour', 'workingday']].apply(lambda x: (0, 1)[(x['workingday'] == 1 and  ( x['hour'] == 8 or 17 <= x['hour'] <= 18 or 11 <= x['hour'] <= 12)) or (x['workingday'] == 0 and  10 <= x['hour'] <= 19)], axis = 1)


#不同月份--使用量
fig = sns.boxplot(x='month2',y='count',data=train)
plt.title('Users Count per Month')
#plt.savefig(r'C:\Users\hang\Desktop\all\programming\sf1\pics\month.png')
plt.show()

#每月平均使用量
monthAggregated = pd.DataFrame(train.groupby('month2')['count'].mean()).reset_index()
#monthSorted = monthAggregated.sort_values(by='count',ascending=False)
sortOrder = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
sns.barplot(x="month2",y="count",data=monthAggregated,order=sortOrder)
plt.title('Average Users Count per Month')
#plt.savefig(r'C:\Users\hang\Desktop\all\programming\sf1\pics\avemonth.png')
plt.show()

#不同季节--使用量
sns.boxplot(x='season2',y='count',data=train)
plt.title('Users Count per Season')
#plt.savefig(r'C:\Users\hang\Desktop\all\programming\sf1\pics\season.png')
plt.show()

#不同工作日--使用量
sns.boxplot(x='weekday',y='count',data=train)
plt.title('Users Count on Weekdays')
#plt.savefig(r'C:\Users\hang\Desktop\all\programming\sf1\pics\weekday.png')
plt.show()


#是否工作日--使用量
sns.boxplot(x='workingday',y='count',data=train)
plt.title('Users Count on Workingdays')
#plt.savefig(r'C:\Users\hang\Desktop\all\programming\sf1\pics\workingday.png')
plt.show()


#季节--使用量
sns.pointplot(x='hour',y='count',hue='season2',join=True,data=train)
plt.title('Users Count per Hour in Different Seasons')
#plt.savefig(r'C:\Users\hang\Desktop\all\programming\sf1\pics\seasonph.png')
plt.show()


#工作日--使用量
sns.pointplot(x='hour',y='count',hue='weekday',join=True,data=train) #join在点间绘制线条
plt.title('Users Count per Hour on Weekdays')
#plt.savefig(r'C:\Users\hang\Desktop\all\programming\sf1\pics\weekdayph.png')
plt.show()


#注册用户--使用量
hourTransformed = pd.melt(train[["hour","casual","registered"]], id_vars=['hour'], value_vars=['casual', 'registered']) #hour用作标识变量
hourAggregated = pd.DataFrame(hourTransformed.groupby(["hour","variable"],sort=True)["value"].mean()).reset_index()
sns.pointplot(x=hourAggregated["hour"], y=hourAggregated["value"],hue=hourAggregated["variable"],hue_order=["casual","registered"], data=hourAggregated, join=True)
plt.xlabel('Hour of the Day')
plt.ylabel('Users Count')
plt.title('Average Users Count by Hour of the Day Across User Type')
#plt.savefig(r'C:\Users\hang\Desktop\all\programming\sf1\usertype.png')
plt.show()

#温度--使用量
fig = plt.subplots(figsize=(12,4))  #单位英寸
sns.regplot(x='temp',y='count',data=train,line_kws={'linestyle':'-','color':'r'})
plt.title('Users Count Associated with Temperature')
#plt.savefig(r'C:\Users\hang\Desktop\all\programming\sf1\pics\temp.png')
plt.show()

#体感温度--使用量
fig = plt.subplots(figsize=(12,4))
sns.regplot(x='atemp',y='count',data=train,line_kws={'linestyle':'-','color':'r'})
plt.title('Users Count Associated with Feel Like Temperature')
#plt.savefig(r'C:\Users\hang\Desktop\all\programming\sf1\pics\atemp.png')
plt.show()

#湿度--使用量
fig = plt.subplots(figsize=(12,4))
sns.regplot(x='humidity',y='count',data=train,line_kws={'linestyle':'-','color':'r'})
plt.title('Users Count Associated with Humidity')
#plt.savefig(r'C:\Users\hang\Desktop\all\programming\sf1\pics\humidity.png')
plt.show()

#风速--使用量
fig = plt.subplots(figsize=(12,4))
sns.regplot(x='windspeed',y='count',data=train,line_kws={'linestyle':'-','color':'r'})
plt.title('Users Count Associated with Windspeed')
#plt.savefig(r'C:\Users\hang\Desktop\all\programming\sf1\pics\windspeed.png')
plt.show()

#连续变量的相关性分析
corr = train[['season','month','temp','atemp','humidity','windspeed','casual','registered','count']].corr()
mask = np.array(corr) #保存相关性结果
mask[np.tril_indices_from(mask)] = False #下三角
fig,ax = plt.subplots()
fig.set_size_inches(15,8)
sns.heatmap(corr,mask=mask,vmax=.8,square=True,annot=True) #vmax锚定色彩映射值，square=True单元格设为方形，annot写入数值
plt.title('Correlation between Variables')
#plt.savefig(r'C:\Users\hang\Desktop\all\programming\sf1\pics\correlation.png')
plt.show()






#对季节、天气、weekday等进行独热编码,并保留原属性编码，后续进行特征挑选
train=pd.get_dummies(train,columns=['season2'])
train=pd.get_dummies(train,columns=['weather2'])

test=pd.get_dummies(test,columns=['season2'])
test=pd.get_dummies(test,columns=['weather2'])

All_feature_columns = ['season','weather','temp','atemp','humidity','windspeed',
                       'year','holiday','workingday','month','day','hour','week','weekday',
                       'season2_Fall','season2_Spring','season2_Summer','season2_Winter',
                       'weather2_Clear','weather2_Heavy_Rain','weather2_Light_Snow','weather2_Mist']

RFR_feature_columns = ['weather','temp','atemp','windspeed',
                       'workingday','season','holiday','hour','weekday','week',
                       'season2_Fall','season2_Spring','season2_Summer','season2_Winter',
                       'weather2_Clear','weather2_Heavy_Rain','weather2_Light_Snow','weather2_Mist']

GBR_feature_columns =['weather','temp','atemp','humidity','windspeed',
                      'holiday','workingday','season','hour','weekday','year',
                      'season2_Fall','season2_Spring','season2_Summer','season2_Winter',
                      'weather2_Clear','weather2_Heavy_Rain','weather2_Light_Snow','weather2_Mist']

#拆分出训练数据
RFR_X_train=train[RFR_feature_columns].values
RFR_X_test=test[RFR_feature_columns].values

GBR_X_train=train[GBR_feature_columns].values
GBR_X_test=test[GBR_feature_columns].values

y_casual=train['casual'].apply(lambda x: np.log1p(x)).values
y_registered=train['registered'].apply(lambda x: np.log1p(x)).values
y_count=train['count'].apply(lambda x: np.log1p(x)).values

X_date=test['datetime'].values

#评价标准RMSLE 1/n*sum((log(y1+1)-log(y2+1))^2)
def rmsle(y_real, y_pre):
    log1 = np.log(y_real+1)
    log2 = np.log(y_pre+1)
    calc = (log1 - log2) ** 2
    return np.sqrt(np.mean(calc))

#拆分模型训练集
from sklearn.model_selection import train_test_split
X_train = train[All_feature_columns].values
xd_train,xd_test,yd_train,yd_test = train_test_split(X_train,y_count,random_state=0)

#对各种回归模型进行训练、调参、测试
##LGBM
from lightgbm import LGBMRegressor
def LGBM_model():
    #num_leaves最大树叶,learning_rate学习率，n_estimators树的数量,
    LGBM = LGBMRegressor(boosting_type='gbdt', objective='regression', num_leaves=1200,
                                learning_rate=0.17, n_estimators=1000, max_depth=10)
                                #metric='rmse', bagging_fraction=0.8, feature_fraction=0.8, reg_lambda=0.9)
    LGBM.fit(xd_train, yd_train)
    # 给出训练数据的预测值
    pre_test = LGBM.predict(xd_test)
    # 计算RMSLE
    score = rmsle(yd_test,pre_test)
    return score

##随机森林
from sklearn.ensemble import RandomForestRegressor
def RandomForest_model():
    #n_job并行作业数，n_job=-1使用所有处理器
    RFR = RandomForestRegressor(n_estimators = 1000, max_depth=15, random_state=0,n_jobs = -1)
    RFR.fit(xd_train,yd_train)
    # 给出训练数据的预测值
    pre_test = RFR.predict(xd_test)
    # 计算RMSLE
    score = rmsle(yd_test,pre_test)
    return score

##决策树
from sklearn.tree import DecisionTreeRegressor
def DecisionTree_model():
    #考虑sqrt（n_features）特征，拆分内部节点最少要4个样本
    DTR = DecisionTreeRegressor(max_features='sqrt', splitter='random', min_samples_split=4, max_depth=10)
    DTR.fit(xd_train,yd_train)
    # 给出训练数据的预测值
    pre_test = DTR.predict(xd_test)
    # 计算RMSLE
    score = rmsle(yd_test,pre_test)
    return score

##集成学习梯度提升决策树
from sklearn.ensemble import GradientBoostingRegressor
def GradientBoosting_model():
    GBR = GradientBoostingRegressor(n_estimators = 1000, max_depth = 5, random_state = 0)
    GBR.fit(xd_train,yd_train)
    # 给出训练数据的预测值
    pre_test = GBR.predict(xd_test)
    # 计算RMSLE
    score = rmsle(yd_test,pre_test)
    return score

# ##逻辑斯蒂回归
# from sklearn.linear_model import LogisticRegression
# def Logisic_model():
#     LG = LogisticRegression(penalty="l2",tol=0.0001, C=1.0, solver= "lbfgs", max_iter=3000,multi_class='ovr', verbose=0)
#     LG.fit(xd_train,yd_train)
#     # 给出训练数据的预测值
#     pre_test = LG.predict(xd_test)
#     # 计算RMSLE
#     score = rmsle(yd_test,pre_test)
#     return score

##AdaBoost
from sklearn.ensemble import AdaBoostRegressor
def AdaBoost_model():
    ABR = AdaBoostRegressor(learning_rate=0.1, loss='square', n_estimators=1000) #学习器最大数量为1000
    ABR.fit(xd_train,yd_train)
    # 给出训练数据的预测值
    pre_test = ABR.predict(xd_test)
    # 计算RMSLE
    score = rmsle(yd_test,pre_test)
    return score

#各种模型的RMSLE评价
print("各种模型RMSLE：")
print("LGBM_model:             ",LGBM_model())
print("RandomForest_model:     ",RandomForest_model())
print("DecisionTree_model:     ",DecisionTree_model())
print("GradientBoosting_model: ",GradientBoosting_model())
print("AdaBoost_model:         ",AdaBoost_model())

#随机森林模型
from sklearn.ensemble import RandomForestRegressor
params = {'n_estimators': 1000,
          'max_depth': 15,
          'random_state': 0,
          'min_samples_split' : 5,
          'n_jobs': -1}

RFR1 = RandomForestRegressor(**params)
RFR1.fit(RFR_X_train,y_casual)
print("随机森林临时租赁拟合程度:",RFR1.score(RFR_X_train,y_casual))

RFR2 = RandomForestRegressor(**params)
RFR2.fit(RFR_X_train,y_registered)
print("随机森林注册租赁拟合程度:",RFR2.score(RFR_X_train,y_registered))

RFR3 = RandomForestRegressor(**params)
RFR3.fit(RFR_X_train,y_count)
print("随机森林租赁数拟合程度:",RFR3.score(RFR_X_train,y_count))

#集成学习梯度提升决策树
from sklearn.ensemble import GradientBoostingRegressor

params2 = {'n_estimators': 150,
           'max_depth': 5,
           'random_state': 0,
           'min_samples_leaf' : 10,
           'learning_rate': 0.1,
           'subsample': 0.7,
           'loss': 'ls'}

GBR1 = GradientBoostingRegressor(**params2)
GBR1.fit(GBR_X_train,y_casual)
print("GBDT临时租赁拟合程度:",GBR1.score(GBR_X_train,y_casual))

GBR2 = GradientBoostingRegressor(**params2)
GBR2.fit(GBR_X_train,y_registered)
print("GBDT注册租赁拟合程度:",GBR2.score(GBR_X_train,y_registered))

GBR3 = GradientBoostingRegressor(**params2)
GBR3.fit(GBR_X_train,y_count)
print("GBDT租赁数拟合程度:",GBR3.score(GBR_X_train,y_count))


RFR_pre_casual = RFR1.predict(RFR_X_test)
RFR_pre_casual=np.exp(RFR_pre_casual)-1
RFR_pre_registered = RFR2.predict(RFR_X_test)
RFR_pre_registered=np.exp(RFR_pre_registered)-1
RFR_pre = RFR_pre_casual+RFR_pre_registered

GBR_pre_casual = GBR1.predict(GBR_X_test)
GBR_pre_casual=np.exp(GBR_pre_casual)-1
GBR_pre_registered = GBR2.predict(GBR_X_test)
GBR_pre_registered=np.exp(GBR_pre_registered)-1
GBR_pre = GBR_pre_casual+GBR_pre_registered

submit1 = pd.DataFrame({'datetime':X_date,'count':np.round(0.2*RFR_pre+0.8*GBR_pre+0.5)})
submit1.to_csv('submisssion_1.csv',index=False)

RFR_pre_count = RFR3.predict(RFR_X_test)
RFR_pre_count = np.exp(RFR_pre_count)-1
GBR_pre_count = GBR3.predict(GBR_X_test)
GBR_pre_count = np.exp(GBR_pre_count)-1

pre_count = 0.2*RFR_pre_count+0.8*GBR_pre_count+0.5
pre_count = np.round(pre_count)
submit2 = pd.DataFrame({'datetime':X_date,'count':pre_count})
submit2.to_csv('submisssion_2.csv',index=False)