

#Importing Datasets Offline
import sklearn.datasets as skd

# Positive and Negative Category
categories = ['com.positive','com.negative']

# Training Datas of Feedback
feedback_train = skd.load_files('feedback/train',categories=categories,encoding='ISO-8859-1')

#Testing Datas of Feedback
feedback_test = skd.load_files('feedback/test',categories=categories,encoding='ISO-8859-1')

# Train and Test are formed in dict files

## feedback_test.keys()          
## feedback_train.target_names

# => ['com.negative','com.positive']

#Word Count Vectorizer 
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()

X_train_tf = count_vect.fit_transform(feedback_train.data)

#Tfidf Transformer (Term Frequency)
from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer()

X_train_tfidf = tfidf_transformer.fit_transform(X_train_tf)

#X_train_tfidf.shape


# MultinomialNB used for the features with discrete values like word count 1,2,3.
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf,feedback_train.target)


#Count Vectorizer and TfiDF for Test data
X_test_tf = count_vect.transform(feedback_test.data)
X_test_idf = tfidf_transformer.transform(X_test_tf)

#prediction value for test data
predicted = clf.predict(X_test_idf)

#Checking Accuracy by comparing test target and predicted value
from sklearn import metrics
from sklearn.metrics import accuracy_score

# print("Accuracy:",accuracy_score(feedback_test.target,predicted))

# Classification for New Feedbacks

import requests

# classified_value
classified_value = ''
# score = ''

def onRun(grabbedFeedback):
    new_feedback = [grabbedFeedback]
    #Count Vectorization
    X_new_counts = count_vect.transform(new_feedback)
    #TFIDF (Term Frequency)
    X_new_tfidf= tfidf_transformer.transform(X_new_counts)

    predicted_new=clf.predict(X_new_tfidf)

    #Gives the array value of predicted classification
    global classified_value
    if(1 in predicted_new):
        classified_value="Positive"
    else:
        classified_value="Negative"

#ParallelDots API
# import paralleldots
# paralleldots.set_api_key("xSDCayg76qWgYYbAQ0r5BtmO4lUmAKTOauhMl6dopCQ")
# lang_code = "en"


#START OF API

from flask import  Flask
from flask_restful import  Resource
# from json import  dumps
from flask import jsonify
from urllib.request import urlopen
import json

app = Flask(__name__)


@app.route('/<feedback>',methods=['GET','POST'])
def index(feedback):
    some_json = str(feedback)
    onRun(some_json)
#     response = paralleldots.sentiment(feedback,lang_code)
#     responseData = json.dumps(response)
#     positive = json.loads(responseData)['sentiment']['positive']
#     negative = json.loads(responseData)['sentiment']['negative']
#     if(positive>negative):
#         f=open('feedback/train/com.positive/newFeedback','a')
#         f.writelines(feedback.strip('favicon.ico')+"\n")
#         f.close
#         msg="Positive"
#     elif(negative>positive):
#         f=open('feedback/train/com.negative/newFeedback','a')
#         f.writelines(feedback.strip('favicon.ico')+"\n")
#         f.close
#         msg="Negative"

    result={'data':[{'classification':classified_value,"api_classification":"api exceeded"}]}

    return jsonify(result), 201

@app.route('/',methods=['GET'])
def default():
    return jsonify({'Message':'Pass Feedback at the end of the link'})

if __name__ == '__main__':
    app.run(port='5002')


#END OF API











