from tokenize import group
import streamlit as st 
import numpy as np
import pandas as pd
import sklearn
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix,matthews_corrcoef
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from pylab import rcParams
rcParams['figure.figsize'] = 14, 8
RANDOM_SEED = 42
LABELS = ["Normal", "Fraud"]
import chart_studio.plotly as py
import plotly.graph_objs as go
import plotly
import plotly.figure_factory as ff
from plotly.offline import init_notebook_mode, iplot
from sklearn.model_selection import cross_val_score
from testFina import God 

st.title('Credit Card Fraud Detection!')

status=False
uploaded_file = st.file_uploader("Choose a CSV file",type = ['csv'])
if uploaded_file is not None:
     status=True
     data =st.cache (pd.read_csv)(uploaded_file)
     data.head()
     
     



if status==False:
    st.warning('Please Upload a Dataset')
    st.stop()



if st.sidebar.checkbox('Show what the dataframe looks like'):
    st.write(data.head(100))
    st.write('Shape of the dataframe: ',data.shape)
    st.write('Data decription: \n',data.describe())
fraud=data[data.Class==1]
valid=data[data.Class==0]


outlier_percentage=(data.Class.value_counts()[1]/data.Class.value_counts()[0])*100

if st.sidebar.checkbox('Show fraud and valid transaction details'):
    st.write('Fraudulent transactions are: %.3f%%'%outlier_percentage)
    st.write('Fraud Cases: ',len(fraud))
    st.write('Valid Cases: ',len(valid))

    
#Obtaining X (features) and y (labels)
X=data.drop(['Class'], axis=1)
y=data.Class

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
size = st.sidebar.slider('Test Set Size', min_value=0.2, max_value=0.4)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = size, random_state = 42)

#Print shape of train and test sets
if st.sidebar.checkbox('Show the shape of training and test set features and labels'):
    st.write('X_train: ',X_train.shape)
    st.write('y_train: ',y_train.shape)
    st.write('X_test: ',X_test.shape)
    st.write('y_test: ',y_test.shape)





   

Normal = data[data['Class']==0]
Fraud = data[data['Class']==1]
if st.sidebar.checkbox('Show How different are the amount of money used in different transaction classes'):
    al=['Normal','Fraud']
    classifier = st.sidebar.selectbox('Which class', al)
    if classifier=='Normal':
        st.write('Normal')
        st.write(Normal.Amount.describe())
    elif classifier == 'Fraud':
        st.write('Fraud')
        st.write(Fraud.Amount.describe())
   
   
 #Let's have a more graphical representation of the data

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

f.suptitle('Amount per transaction by class')
bins = 50
ax1.hist(Fraud.Amount, bins = bins)
ax1.set_title('Fraud')
ax2.hist(Normal.Amount, bins = bins)
ax2.set_title('Normal')
plt.xlabel('Amount ($)')
plt.ylabel('Number of Transactions')
plt.xlim((0, 20000))
plt.yscale('log')
plt.show();


#Graphical representation of the data

g, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
g.suptitle('Time of transaction vs Amount by class')
ax1.scatter(Fraud.Time, Fraud.Amount)
ax1.set_title('Fraud')
ax2.scatter(Normal.Time, Normal.Amount)
ax2.set_title('Normal')
plt.xlabel('Time (in Seconds)')
plt.ylabel('Amount')
plt.show();

count_classes = pd.value_counts(data['Class'], sort = True)
count_classes.plot(kind = 'bar', rot=0)
plt.title("Transaction Class Distribution")
plt.xticks(range(2), LABELS)
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.show();

if st.sidebar.checkbox('Graphical Reprisentation of the Classes'):
    alg=['Histogram','Scatterplot','Bar-chart']
    classifier = st.sidebar.selectbox('Which graphics?', alg)
    if classifier=='Histogram':
        st.title('Histogram')
        st.write(f)
    elif classifier=='Scatterplot':
        st.pyplot(g)
    elif classifier=='Bar-chart':
        st.bar_chart(count_classes)

#from sklearn.ensemble import IsolationForest
#from sklearn.neighbors import LocalOutlierFactor
ISO=IsolationForest()
LOF=LocalOutlierFactor()
etree=ExtraTreesClassifier(random_state=42)
rforest=RandomForestClassifier(random_state=42)


features=X_train.columns.tolist()

#Feature selection through feature importance
##################
@st.cache
def feature_sort(model,X_train,y_train):
    #feature selection
    mod=model
    # fit the model
    mod.fit(X_train, y_train)
    # get importance
    imp = mod.feature_importances_
    return imp

#Classifiers for feature importance
clf=['Extra trees','Random forest']
mod_feature = st.sidebar.selectbox('Which model for feature importance?', clf)


if mod_feature=='Extra trees':
    model=etree
    importance=feature_sort(model,X_train,y_train)
elif mod_feature=='Random forest':
    model=rforest
    importance=feature_sort(model,X_train,y_train)
   

#Plot of feature importance
if st.sidebar.checkbox('Show plot of feature importance'):
    plt.bar([x for x in range(len(importance))], importance)
    plt.title('Feature Importance')
    plt.xlabel('Feature (Variable Number)')
    plt.ylabel('Importance')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

feature_imp=list(zip(features,importance))
feature_sort=sorted(feature_imp, key = lambda x: x[1])

n_top_features = st.sidebar.slider('Number of top features', min_value=5, max_value=20)

top_features=list(list(zip(*feature_sort[-n_top_features:]))[0])

if st.sidebar.checkbox('Show selected top features'):
    st.write('Top %d features in order of importance are: %s'%(n_top_features,top_features[::-1]))

X_train_sfs=X_train[top_features]
X_test_sfs=X_test[top_features]

X_train_sfs_scaled=X_train_sfs
X_test_sfs_scaled=X_test_sfs



#Import performance metrics, imbalanced rectifiers
from sklearn.metrics import  confusion_matrix,classification_report,matthews_corrcoef
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
np.random.seed(42) #for reproducibility since SMOTE and Near Miss use randomizations

smt = SMOTE()
nr = NearMiss()


#Define the outlier detection methods

def compute_performance(model, X_train, y_train,X_test,y_test):
    
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy').mean()
    'Accuracy: ',scores
    model.fit(X_train,y_train)
    y_pred = model.fit_predict(X_test)
    cm=confusion_matrix(y_test,y_pred)
    'Confusion Matrix: ',cm  
    cr=classification_report(y_test, y_pred)
    'Classification Report: ',cr
    
    

#Run different classification models with rectifiers
if st.sidebar.checkbox('Run a credit card fraud detection model'):
    
    alg=['Isolation Forest','Local Outlier Factor']
    classifier = st.sidebar.selectbox('Which algorithm?', alg)
    
    
    if classifier=='Local Outlier Factor':
        model=LOF
        compute_performance(model, X_train_sfs_scaled, y_train,X_test_sfs_scaled,y_test)
       
    elif classifier == 'Isolation Forest':
        model=ISO
        compute_performance(model, X_train_sfs_scaled, y_train,X_test_sfs_scaled,y_test)
        
        
        