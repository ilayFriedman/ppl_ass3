import csv
import re

import numpy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import tree, svm
import pandas
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.tree import DecisionTreeClassifier

# helpers
REPLACE_NO_SPACE = re.compile("[;*\'&#\"()\[\]]")
REPLACE_WITH_SPACE = re.compile("(\-)|(\/)|[_.:!?,]|(&quot;)|(&amp;)|(&lt;)|(&gt;)")

class sentimentAnalysis:

    #This function reads the train and the test files
    def reading_data_csv(self, FilePath_Train, filePath_Test):
        print("...Reading Files...")
        with open(FilePath_Train, mode='r', encoding="latin-1") as csv_fileTrain:
            csv_readerTweets = csv.reader(csv_fileTrain, delimiter=',')
            self.tweetsText_Train = [item[1] for item in list(csv_readerTweets)[1:]]
            csv_fileTrain.seek(0)
            self.tweetsValue_Train = [int(item[0]) for item in list(csv_readerTweets)[1:]]
        csv_fileTrain.close()

        with open(filePath_Test, mode='r', encoding="latin-1") as csv_fileTest:
            csv_reader = csv.reader(csv_fileTest, delimiter=',')
            self.tweetsText_Test = [item[1] for item in list(csv_reader)[1:]]
        csv_fileTest.close()

    # this function cleans tweets
    def preprocceing_tweet(self,tweet):
        tweet = REPLACE_WITH_SPACE.sub(" ", tweet)
        tweet = REPLACE_NO_SPACE.sub("", tweet.lower())
        tweet = " ".join(tweet.split(" ")[1:])   ## cleaing all the user's names (starts with '@')
        # tweet = re.sub(' +', ' ', tweet)   ## cleaning all the multiply white spaces
        return tweet

    # This function making logistic Regression
    def feature_extraction_LogisticRegression(self):
        print("...creation of pipeline-LogisticRegression...")
        # 10-Cross-validation
        self.pipeLine_LogicReg = Pipeline([
            ('data',CountVectorizer(binary=True, ngram_range=(1, 2), preprocessor=self.preprocceing_tweet)),
            ('tfidf', TfidfTransformer(use_idf=True, norm='l2')),
            ('brain', LogisticRegression(C=6.6, solver='liblinear')),
        ])
        print("...start cross validation (with preprocceing)...")
        self.scores = cross_val_score(self.pipeLine_LogicReg, self.tweetsText_Train, self.tweetsValue_Train, cv=10,
                                 scoring='accuracy')
        # print(self.scores)

        print("...fit model...")
        # print("--------------------------")
        # print("...-LogisticRegression...")
        self.pipeLine_LogicReg.fit(self.tweetsText_Train, self.tweetsValue_Train)

    def feature_extraction_KNN(self):
        print("...creation of pipeline-KNN...")
        model = make_pipeline(TfidfVectorizer(), MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(15,), random_state=1))
        print("...scores...")
        scores = cross_val_score(model, self.tweetsText_Train, self.tweetsValue_Train, cv=10, scoring='accuracy')
        print(scores)
        print("...fit model...")
        model.fit(self.tweetsText_Train, self.tweetsValue_Train)

    # This function making Naive bayes
    def feature_extraction_NaiveBayes(self):
        print("...creation of pipeline-naive_bayes...")
        self.model = make_pipeline(TfidfVectorizer(), MultinomialNB())
        # print("...scores...")
        scores = cross_val_score(self.model, self.tweetsText_Train, self.tweetsValue_Train, cv=10, scoring='accuracy')
        # print(scores)
        print("...fit model...")
        self.model.fit(self.tweetsText_Train, self.tweetsValue_Train)

    # This function making desicion tree
    def feature_extraction_decision_tree (self):
        self.pipeLine_decision_tree = make_pipeline(CountVectorizer(binary=True, max_df=0.9, ngram_range=(1, 2), preprocessor=self.preprocceing_tweet),DecisionTreeClassifier(random_state=0))
        # print("...creation of pipeline-decision_tree...")
        # print("...scores...")
        self.scores = cross_val_score(self.pipeLine_decision_tree, self.tweetsText_Train, self.tweetsValue_Train, cv=10, scoring='accuracy')
        # print(self.scores)
        self.pipeLine_decision_tree.fit(self.tweetsText_Train, self.tweetsValue_Train)

    #This function making classify on giving trained model
    def Classifier(self):
        print("...START predict...")
        self.y_preds = self.pipeLine_LogicReg.predict(self.tweetsText_Train)
        # print(self.y_preds)
        print("--------------------------------------------")
        print("Accuracy, Recall & Precision Values:")
        print("macro:")
        print(precision_recall_fscore_support(self.tweetsValue_Train, self.y_preds, average='macro'))
        print("micro:")
        print(precision_recall_fscore_support(self.tweetsValue_Train, self.y_preds, average='micro'))
        print("weighted:")
        print(precision_recall_fscore_support(self.tweetsValue_Train, self.y_preds, average='weighted'))
        # self.create_output_file()

    # This function creat a output file
    def create_output_file(self):
        finalans =[]
        i=0
        finalans=["ID,Sentiment"]
        for val in numpy.nditer(self.y_preds):
            finalans.append(str(i)+","+str(val))
            i+=1
        print(finalans)
        with open("out3-10kcross.csv", "w") as f:
            wr = csv.writer(f, delimiter="\n")
            wr.writerow(finalans)

        f.close()


SA =  sentimentAnalysis()
SA.reading_data_csv("Train.csv","Test.csv")

SA.feature_extraction_LogisticRegression()
SA.Classifier()