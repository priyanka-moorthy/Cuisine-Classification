from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
import json;
import pandas as pd 
from sklearn.svm import SVC


print("Reading dataset")
train = json.load(open('train.json'))
test = json.load(open('test.json'))

def text_data_generate(input_data):
    text = [" ".join(doc['ingredients']).lower() for doc in input_data]
    return text

train_text = text_data_generate(train)
test_text = text_data_generate(test)

op_target = [doc['cuisine'] for doc in train]


print("converting raw document to matrix")

tfidf = TfidfVectorizer(binary=True)
def matrix_features(txt, flag):
    if flag == "train":
        x_val = tfidf.fit_transform(txt)
    else:
        x_val = tfidf.transform(txt)
    x_val = x_val.astype('float16')
    return x_val

X_train = matrix_features(train_text, flag="train")
X_test = matrix_features(test_text, flag="test")



print("Label to target")
lb = LabelEncoder()
y_train = lb.fit_transform(op_target)




print("Training the model..")
classifier = SVC(C=100,
                 kernel='rbf',
                 degree=3,
                 gamma=1,
                 coef0=1,
                 shrinking=True,
                 tol=0.001,
	      		 probability=False,
	      		 cache_size=200, 
	      		 class_weight=None,
	      		 verbose=False,
	      		 max_iter=-1,
          		 decision_function_shape=None, 
          		 random_state=None)
model = OneVsRestClassifier(classifier, n_jobs=4)


model.fit(X_train,y_train)

print("Predicting test data using model..")
y_test = model.predict(X_test)
y_predected = lb.inverse_transform(y_test)

print("Generating output file..")
test_id = [doc['id'] for doc in test]
output_values = pd.DataFrame({'id':test_id, 'cuisine':y_predected},columns=['id','cuisine'])
output_values.to_csv('multicusine_output.csv',index=False)
