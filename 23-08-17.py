#!/usr/bin/env python
# coding: utf-8

# In[21]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import platform
from matplotlib import font_manager, rc

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

#그래프에서 한글을 사용하기 위해서 사용
if platform.system() == 'Darwin':
    rc('font', family='AppleGothic')
#윈도우의 경우
elif platform.system() == 'Windows':
    font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
    rc('font', family=font_name)
    
#그래프에 음수를 사용하기 위해서 사용
plt.rcParams['axes.unicode_minus'] = False


# In[22]:


#student.csv 파일을 읽어오기
#이름을 인덱스로 사용
df=pd.read_csv('C:\\Users\\USER\\Desktop\\수업 자료\\ETL\\5.Date_preprocessing\\data\\data\\student.csv', encoding='ms949', index_col='이름')
#df

df.plot(kind='bar')


# In[24]:


#위 데이터의 경우 단순한 표준화 작업만으로는 성적을 비교하는 것이 어려울 수 있음
#최대값 이나 최대값-최솟값으로 나눈 데이터로는 비교하기가 어려움
#이런 경우에는 표준값이나 편차값을 구해서 비교하는 것이 좋습니다

#평균과 표준 편차 구하기
kormean, korstd = df['국어'].mean(), df['국어'].std()
engmean, engstd = df['영어'].mean(), df['영어'].std()
matmean, matstd = df['수학'].mean(), df['수학'].std()

#표준값 구하기
df['국어표준값'] = (df['국어'] - kormean)/korstd
df['영어표준값'] = (df['영어'] - engmean)/engstd
df['수학표준값'] = (df['수학'] - matmean)/matstd

#편차값 구하기
df['국어편차값'] = df['국어표준값'] * 10 + 50
df['영어편차값'] = df['영어표준값'] * 10 + 50
df['수학편차값'] = df['수학표준값'] * 10 + 50

df

df[['국어편차값','영어편차값','수학편차값']].plot(kind='bar')


# # 표준화

# In[66]:


#데이터 읽어오기
auto_mpg=pd.read_csv('C:\\Users\\USER\\Desktop\\수업 자료\\ETL\\5.Date_preprocessing\\data\\data\\auto-mpg.csv', header=None)

#컬럼 이름 설정
auto_mpg.columns = ['mpg','cylinders','displacement','horsepower','weight',
              'acceleration','model year','origin','name']

# horsepower 열의 누락 데이터('?') 삭제하고 실수형으로 변환
auto_mpg['horsepower'].replace('?', np.nan, inplace=True)      # '?'을 np.nan으로 변경
auto_mpg.dropna(subset=['horsepower'], axis=0, inplace=True)   # 누락데이터 행을 삭제
auto_mpg['horsepower'] = auto_mpg['horsepower'].astype('float') 
auto_mpg.head()


# In[37]:


#horsepower 열의 표준화
auto_mpg['maxhorsepower']=auto_mpg['horsepower'] / auto_mpg['horsepower'].max()
auto_mpg['minmaxhorsepower']=(auto_mpg['horsepower'] - auto_mpg['horsepower'].min())
(auto_mpg['horsepower'].max() - auto_mpg['horsepower'].min())

auto_mpg.describe()


# In[42]:


from sklearn import preprocessing
#스케일링을 수행할 데이터를 가져오기
x = auto_mpg[['horsepower']].values
print(type(x))
print('평균:', np.mean(x))
print('표준편차:', np.std(x))
print('최대값:', np.max(x))
print('최소값:', np.min(x))
print()

scaler = preprocessing.StandardScaler()
scaler.fit(x)
x_scaled = scaler.transform(x)
print('평균:', np.mean(x_scaled))
print('표준편차:', np.std(x_scaled))
print('최대값:', np.max(x_scaled))
print('최소값:', np.min(x_scaled))
print()


# # 정규화

# In[50]:


features = np.array([[1,2], [2,3], [3,8], [4,6], [7,2]])

#정규화 객체
#l1을 norm에 적용하면 맨하튼 거리 - 합치면 1
#l2를 적용하면 유클리드 거리 - 각 값을 전체 데이터를 제곱해서 더한 값의 제곱근으로 나누기
normalizer = preprocessing.Normalizer(norm='l2')
l2_norm = normalizer.transform(features)
print(l2_norm)

#4, 21-5=16, 30, 10(현금), 300, 300



# In[51]:


#다항과 교차항 생성
features=np.array([[1,2], [2,3], [3,4], [5,6], [8,3]])
#제곱항까지의 다항을 생성 - 열의 개수가 늘어나게 되는데
#회귀 분석을 할 때 시간의 흐름에 따라 변화가 급격하게 일어나는 경우 또는
#데이터가 부족할 때 샘플 데이터를 추가하기 위해서 사용
#제곱을 하거나 곱하기를 하게되면 데이터의 특성 자체는 크게 변화하지 않기 때문에 사용

polynomialer = preprocessing.PolynomialFeatures(degree=2)
result=polynomialer.fit_transform(features)
print(result)


# In[64]:


from sklearn.compose import ColumnTransformer

features=np.array([[1,2], [2,3], [3,4], [5,6], [8,3]])

#위의 데이터에 함수 적용
result1 = preprocessing.FunctionTransformer(lambda x : x + 1).transform(features)
print(result1)

df=pd.DataFrame(features, columns=['feature1', 'feature2'])
df.apply(lambda x : x+1).values

def add_one(x):
    return x + 1

def sub_one(x):
    return x -1

result2 = ColumnTransformer([('add_one', preprocessing.FunctionTransformer(add_one, validate=True), 
                              ['feature1']),('sub_one',preprocessing.FunctionTransformer(sub_one, validate=True), ['feature2'])]).fit_transform(df)
result2






# In[68]:


#auto_mpg의 horsepower 를 3개의 구간으로 분할
#auto_mpg['horsepower'].describe()

# 경계값 찾기
count, bin_dividers = np.histogram(auto_mpg['horsepower'], bins=3)
print(bin_dividers)
print()

# 각 그룹에 할당할 값의 리스트
bin_names = ['저출력', '보통출력', '고출력']

# pd.cut 함수로 각 데이터를 3개의 bin에 할당
auto_mpg['hp_bin'] = pd.cut(x=auto_mpg['horsepower'],     # 데이터 배열
                      bins=bin_dividers,      # 경계 값 리스트
                      labels=bin_names,       # bin 이름
                      include_lowest=True)    # 첫 경계값 포함 

# horsepower 열, hp_bin 열의 첫 10행을 출력
print(auto_mpg[['horsepower', 'hp_bin']].head(10))



# In[69]:


# 라이브러리를 임포트합니다.
import numpy as np

# 특성을 만듭니다.
age = np.array([[13],
                [30],
                [67],
                [36],
                [64],
                [24]])
# 30을 기준으로 분할
result = np.digitize(age, bins=[30])
print(result)
print()

#0-19, 20-29, 30-63, 64이상의 구간으로 분할
result = np.digitize(age, bins=[20,30,64])
print(result)
print()

#0-20, 21-30, 31-64, 64초과 구간으로 분할
result = np.digitize(age, bins=[20,30,64], right=True)
print(result)
print()


# In[ ]:


# 라이브러리를 임포트합니다.
import numpy as np

# 특성을 만듭니다.
age = np.array([[13],
                [30],
                [67],
                [36],
                [64],
                [24]])
# 30을 기준으로 분할
result = np.digitize(age, bins=[30])
print(result)
print()

#0-19, 20-29, 30-63, 64이상의 구간으로 분할
result = np.digitize(age, bins=[20,30,64])
print(result)
print()

#0-20, 21-30, 31-64, 64초과 구간으로 분할
result = np.digitize(age, bins=[20,30,64], right=True)
print(result)
print()


# In[ ]:


from sklearn.preprocessing import KBinsDiscretizer
# 네 개의 구간으로 나눕니다.
kb = KBinsDiscretizer(4, encode='ordinal', strategy='quantile')
print(kb.fit_transform(age))
print()

#희소행렬 리턴
kb = KBinsDiscretizer(4, encode='onehot', strategy='quantile')
print(kb.fit_transform(age))
print()

#밀집행렬 리턴
kb = KBinsDiscretizer(4, encode='onehot-dense', strategy='quantile')
print(kb.fit_transform(age))
print()

#밀집행렬 리턴
kb = KBinsDiscretizer(4, encode='onehot-dense', strategy='uniform')
print(kb.fit_transform(age))
print()

#구간의 값 확인
print(kb.bin_edges_)


# ### 군집 분석

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# 모의 특성 행렬을 생성
sample = np.array([[13, 30],
                [30, 40],
                [67, 44],
                [36, 24],
                [64, 37],
                [24, 46]])

# 데이터프레임을 생성
dataframe = pd.DataFrame(sample, columns=["feature_1", "feature_2"])
print(dataframe)
print()

# k-평균 군집 모델을 생성
clusterer = KMeans(3, random_state=0)

# 모델을 훈련
clusterer.fit(sample)

# 그룹 소속을 예측
dataframe["group"] = clusterer.predict(sample)

# 처음 몇 개의 샘플을 조회
print(dataframe.head(5))


# ## 이상치 감지

# In[ ]:


#Z 점수를 이용하는 방법
import numpy as np

def outliers_z_score(ys):
    threshold = 3

    mean_y = np.mean(ys)
    print("평균:", mean_y)
    stdev_y = np.std(ys)
    print("표준편차:", stdev_y)
    z_scores = [0.6745*(y - mean_y) / stdev_y for y in ys]
    print("z_score:", z_scores)
    return np.where(np.abs(z_scores) > threshold)

features = np.array([[10, 10, 7, 6, 4, 4, 3,3],
                     [20000, 3, 5, 9, 2, 2, 2, 2]])
print(outliers_z_score(features))


# In[ ]:


#IQR 이용하는 방법
import numpy as np

def outliers_iqr(ys):
    quartile_1, quartile_3 = np.percentile(ys, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    print("하한값:", lower_bound)
    print("상한값:", upper_bound)
    return np.where((ys > upper_bound) | (ys < lower_bound))

features = np.array([[10, 10, 7, 6, -4900],
                     [20000, 3, 5, 9, 10]])

print(outliers_iqr(features))


# In[71]:


# 일정 비율의 데이터를 이상치로 간주하기
import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.datasets import make_blobs

# 10행 2열의 데이터를 중앙점을 1.0으로 해서 랜덤하게 생성
features, _ = make_blobs(n_samples = 10,
                         n_features = 2,
                         centers = 1,
                         random_state = 1)
print(features)

# 첨번째 행의 데이터를 이상치로 수정
features[0,0] = 10000
features[0,1] = 10000

#이상치 감지 객체를 생성 - 이상치 비율을 설정
outlier_detector = EllipticEnvelope(contamination=0.1)
outlier_detector.fit(features)
#이상치로 판정되면 -1을 리턴하고 그렇지 않으면 1을 리턴
outlier_detector.predict(features)


# In[72]:


## 이상치 처리

# 라이브러리를 임포트합니다.
import numpy as np
import pandas as pd
from sklearn import preprocessing

# 데이터프레임을 만듭니다.
houses = pd.DataFrame()
houses['Price'] = [534433, 392333, 293222, 4322032]
houses['Bathrooms'] = [2, 3.5, 2, 116]
houses['Square_Feet'] = [1500, 2500, 1500, 48000]

# 불리언 조건을 기반으로 특성을 만듭니다.
houses["Outlier"] = np.where(houses["Bathrooms"] < 20, 0, 1)
# 데이터를 확인합니다.
print(houses)
print()

#특성 변환 - 로그 특성
houses["Log_Of_Square_Feet"] = [np.log(x) for x in houses["Square_Feet"]]
# 데이터를 확인합니다.
print(houses)
print()

#스케일링
df = pd.DataFrame(houses["Bathrooms"])
scaler = preprocessing.RobustScaler()
scaler.fit(df)
x_scaled = scaler.transform(df)
houses["Scale_Bathrooms"] = x_scaled
print(houses)


# ## 결측치 조회

# In[73]:


#seaborn 패키지의 titanic 데이터 가져오기
import pandas as pd
import numpy as np

# seaborn 패키지를 sns라는 이름으로 사용할 수 있도록 import
import seaborn as sns

# titanic 데이터셋 가져오기
#titanic = sns.load_dataset('titanic')
titanic = pd.read_csv('data/titanic.csv')

#데이터 확인
print(titanic.head())

#데이터 요약 정보 확인
print(titanic.info())


# In[ ]:


print(titanic["deck"].value_counts(dropna = False))


# In[ ]:


#isnull을 이용한 null 확인
print(titanic.head().isnull())
print()
#deck 가 null 인 데이터 개수 구하기
print(titanic["deck"].isnull().sum(axis=0))


# ## 결측치 삭제

# In[ ]:


import numpy as np

features = np.array([[1.1, 11.1],
                     [2.2, 22.2],
                     [3.3, 33.3],
                     [4.4, 44.4],
                     [np.nan, 55]])

#(~ 연산자를 사용하여) 누락된 값이 없는 샘플만 남김
print(features[~np.isnan(features).any(axis=1)])


# In[ ]:


#컬럼별 null 데이터의 개수 찾기
print(titanic.isnull().sum(axis=0))
print()

#NaN 값이 500개 이상인 컬럼 제거
df_thresh = titanic.dropna(axis=1, thresh=500)
print(df_thresh .columns)


# In[ ]:


#age 열의 값이 NaN인 행 삭제
df_age = df_thresh.dropna(subset=['age'], how='any', axis=0)
df_age.info()


# ## 결측치 대체

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns

# titanic 데이터셋 가져오기
titanic = sns.load_dataset('titanic')

# embark_town 열의 829행의 NaN 데이터 출력
print(titanic['embark_town'][825:831])
print()

# embark_town 열의 NaN값을 바로 앞에 있는 828행의 값으로 변경하기
titanic['embark_town'].fillna(method='ffill', inplace=True)
print(titanic['embark_town'][825:831])
print()

#가장 많이 등장하는 값으로 변경 
titanic = sns.load_dataset('titanic')
mode = titanic['embark_town'].value_counts()
print(mode)

titanic['embark_town'].fillna(mode.idxmax(), inplace=True)
print(titanic['embark_town'][825:831])


# In[ ]:


from sklearn.impute import SimpleImputer

#중간 값으로 대체하는 SimpleImputer 생성 
simple_imputer = SimpleImputer(strategy='median')
features = np.array([[100], [200], [300], [400], [500], [np.nan]])

print(features)

#중간 값으로 대체
features_simple_imputed = simple_imputer.fit_transform(features)
print(features_simple_imputed)
print("대체된 값 Imputed Value:", features_simple_imputed[5])


# In[ ]:


import numpy as np
#기본 라이브러리가 아니므로 fancyimpute를 설치
from fancyimpute import KNN

features = np.array([[100, 200], [200, 250], [300, 300], [400, 290], [500, 380], [200, np.nan]])


# 특성 행렬에 있는 누락된 값을 예측합니다.
features_knn_imputed = KNN(k=5, verbose=0).fit_transform(features)

print(features_knn_imputed)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




