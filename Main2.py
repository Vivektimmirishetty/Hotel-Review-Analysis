from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
import numpy as np
from tkinter import ttk
from tkinter import filedialog
import pandas as pd
from sklearn.model_selection import train_test_split
from string import punctuation
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.naive_bayes import GaussianNB
import pickle

main = Tk()
main.title("Improving Hotel Review Analysis with Machine Learning : A Focus on Supervised  and semi Supervised Model Advances in Setiment Analysis")
main.geometry("1300x1200")

sid = SentimentIntensityAnalyzer()

global filename
global X, Y
global tfidf_X_train, tfidf_X_test, tfidf_y_train, tfidf_y_test
global tfidf_vectorizer
accuracy = []

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

textdata = []
labels = []
global classifier

def cleanPost(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = ' '.join(tokens)
    return tokens

def uploadDataset():    
    global filename
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.insert(END,filename+" loaded\n")
    

def preprocess():
    textdata.clear()
    labels.clear()
    text.delete('1.0', END)
    dataset = pd.read_csv(filename)
    for i in range(len(dataset)):
        msg = dataset.get_value(i, 'Review')
        label = dataset.get_value(i, 'Label')
        msg = str(msg)
        msg = msg.strip().lower()
        labels.append(int(label))
        clean = cleanPost(msg)
        textdata.append(clean)
        text.insert(END,clean+" ==== "+str(label)+"\n")
    TFIDFfeatureEng()    

def TFIDFfeatureEng():
    global Y
    global tfidf_vectorizer
    global tfidf_X_train, tfidf_X_test, tfidf_y_train, tfidf_y_test
    stopwords= nltk.corpus.stopwords.words("english")
    tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords, use_idf=True, smooth_idf=False, norm=None, decode_error='replace', max_features=1000)
    tfidf = tfidf_vectorizer.fit_transform(textdata).toarray()        
    df = pd.DataFrame(tfidf, columns=tfidf_vectorizer.get_feature_names())
    text.insert(END,str(df))
    print(df.shape)
    df = df.values
    X = df[:, 0:1000]
    Y = np.asarray(labels)
    le = LabelEncoder()
    Y = le.fit_transform(Y)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    tfidf_X_train, tfidf_X_test, tfidf_y_train, tfidf_y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"\n\nTotal Reviews found in dataset : "+str(len(X))+"\n")
    text.insert(END,"Total records used to train machine learning algorithms : "+str(len(tfidf_X_train))+"\n")
    text.insert(END,"Total records used to test machine learning algorithms  : "+str(len(tfidf_X_test))+"\n")
    pickle.dump(tfidf_vectorizer, open("tfidf.pickle", "wb"))
    with open('tfidf.txt', 'wb') as file:
        pickle.dump(tfidf_vectorizer, file)
    file.close()

def gaussianKernelGramMatrixFull(X1, X2, sigma=0.1):
    gram_matrix = np.zeros((X1.shape[0], X2.shape[0]))
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            x1 = x1.flatten()
            x2 = x2.flatten()
            gram_matrix[i, j] = np.exp(- np.sum(np.power((x1 - x2),2)) / float( 2*(sigma**2)))
    return gram_matrix

def EMSVM():
    text.delete('1.0', END)
    accuracy.clear();
    #in  below line clf is the svm object
    clf = svm.SVC(C=0.1, kernel="precomputed")
    #clf is getting trained on train data
    clf.fit(gaussianKernelGramMatrixFull(tfidf_X_train,tfidf_X_train),tfidf_y_train)
    #clf is performing prediction on test data
    predict = clf.predict( gaussianKernelGramMatrixFull(tfidf_X_test, tfidf_X_train))
    #calculating svm accuracy on test data by performing prediction
    acc = accuracy_score(tfidf_y_test,predict)*100
    accuracy.append(acc)
    text.insert(END,"EM-SVM Accuracy : "+str(acc)+"\n")
    #creating file as emsvm
    with open('emsvm.txt', 'wb') as file:
        #using pickle saving clf object to file
        pickle.dump(clf, file)
    file.close()

def EMNaiveBayes():
    cls = GaussianNB()
    cls.fit(tfidf_X_train, tfidf_y_train)
    predict = cls.predict(tfidf_X_test)
    acc = accuracy_score(tfidf_y_test,predict)*100
    accuracy.append(acc)
    text.insert(END,"EM-Naive Bayes Accuracy : "+str(acc)+"\n")
    with open('emnb.txt', 'wb') as file:
        pickle.dump(cls, file)
    file.close()

def runSVM():
    global classifier
    cls = svm.SVC(class_weight='balanced')
    cls.fit(tfidf_X_train, tfidf_y_train)
    predict = cls.predict(tfidf_X_test)
    acc = accuracy_score(tfidf_y_test,predict)*100
    accuracy.append(acc)
    text.insert(END,"Supervised SVM Accuracy : "+str(acc)+"\n")
    classifier = cls
    with open('svm.txt', 'wb') as file:
        pickle.dump(cls, file)
    file.close()

def runNB():
    cls = MultinomialNB()
    cls.fit(tfidf_X_train, tfidf_y_train)
    predict = cls.predict(tfidf_X_test)
    acc = accuracy_score(tfidf_y_test,predict)*100
    accuracy.append(acc)
    text.insert(END,"Supervised Naive Bayes Accuracy : "+str(acc)+"\n")
    with open('nb.txt', 'wb') as file:
        pickle.dump(cls, file)
    file.close()
    
def graph():
    bars = ('EM-SVM', 'EM-Naive Bayes','Supervised SVM','Supervised Naive Bayes')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, accuracy)
    plt.xticks(y_pos, bars)
    plt.show()

def predict():
    text.delete('1.0', END)
    testfile = filedialog.askopenfilename(initialdir="Dataset")
    testData = pd.read_csv(testfile,encoding='iso-8859-1')
    positive = 0
    negative = 0
    neutral = 0
    text.delete('1.0', END)
    fakes = 0
    reals = 0
    for i in range(len(testData)):
        msg = testData.get_value(i, 'Reviews')
        review = msg.lower()
        review = review.strip().lower()
        review = cleanPost(review)
        testReview = tfidf_vectorizer.transform([review]).toarray()
        predict = classifier.predict(testReview)
        sentiment_dict = sid.polarity_scores(review)
        negative = sentiment_dict['neg']
        positive = sentiment_dict['pos']
        neutral = sentiment_dict['neu']
        compound = sentiment_dict['compound']
        print(predict)
        if predict == 0:
            fakes = fakes + 1
            text.insert(END,"Given review predicted as FAKE\n\n")
        else:
            reals = reals + 1
            text.insert(END,"Given review predicted as TRUTHFULL\n\n")
        result = ''
        if compound >= 0.05 : 
            result = 'Positive' 
  
        elif compound <= - 0.05 : 
            result = 'Negative' 
  
        else : 
            result = 'Neutral'
    
        text.insert(END,review+'\nCLASSIFIED AS '+result+"\n\n")
    print(str(fakes)+" "+str(real))    
    plt.pie([positive,negative,neutral],labels=["Positive","Negative","Neutral"],autopct='%1.1f%%')
    plt.title('Sentiment Graph')
    plt.axis('equal')
    plt.show()
    
font = ('times', 15, 'bold')
title = Label(main, text='Improving Hotel Review Analysis with Machine Learning : A Focus on Supervised  and semi Supervised Model Advances in Setiment Analysis')
#title.config(bg='powder blue', fg='olive drab')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')

uploadButton = Button(main, text="Upload Gold Standard Reviews Dataset", command=uploadDataset)
uploadButton.place(x=20,y=100)
uploadButton.config(font=ff)


processButton = Button(main, text="Preprocess Dataset", command=preprocess)
processButton.place(x=20,y=150)
processButton.config(font=ff)

emsvmButton = Button(main, text="Run EM-SVM Algorithm", command=EMSVM)
emsvmButton.place(x=20,y=200)
emsvmButton.config(font=ff)

emnbButton = Button(main, text="Run EM-Naive Bayes Algorithm", command=EMNaiveBayes)
emnbButton.place(x=20,y=250)
emnbButton.config(font=ff)

svmButton = Button(main, text="Run SVM Algorithm", command=runSVM)
svmButton.place(x=20,y=300)
svmButton.config(font=ff)

nbButton = Button(main, text="Run Naive Bayes Algorithm", command=runNB)
nbButton.place(x=20,y=350)
nbButton.config(font=ff)

graphButton = Button(main, text="Comparison Graph", command=graph)
graphButton.place(x=20,y=400)
graphButton.config(font=ff)


predictButton = Button(main, text="Upload Test Review & Predict Fake & Sentiments", command=predict)
predictButton.place(x=20,y=450)
predictButton.config(font=ff)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=85)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=550,y=100)
text.config(font=font1)

main.config(bg='OliveDrab2')
main.mainloop()
