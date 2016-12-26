# @Author Mohammed 25-12-2016

#Lets try our luck with regression first shall we?

rm(list=ls())

source("Libraries.R")

data<-read.csv("ADULT_USI_FE_Numerical.csv",header = T)
View(head(data,10))
#Oh no got a extra column by mistake fixing it
data<-data[,-1]
View(head(data,10))
#Seems good now doesnt it

#so lets split our data now in training and testing
set.seed(1025)
train<-sample(1:30162,24129)
test<--train
training_data<-data[train,]
testing_data<-data[test,]

#Cool doing

uci_pred<-knn(train=training_data[,-14],test=testing_data[,-14],cl=training_data[,14],k=3)

table(uci_pred,testing_data[,14])
#Accuracy is 75 % with basic KNN

# but i have made a fundamental mistake here i havent normalized the data
#what this means that capital gain,loss and fnlwgt will have a higher 
#impact on the predictions that we make
#we require class library

normalize<-function(x){
  num<-x-min(x)
  denom<-max(x)-min(x)
  return(num/denom)
}

#My normalized data to avoid any kind of incorrect 
#dependencies on a single feature
uci_norm<-as.data.frame(lapply(data[,-14], normalize))
summary(uci_norm)

training_data<-uci_norm[train,]
testing_data<-uci_norm[test,]

#Cool doing

uci_pred<-knn(train=training_data,test=testing_data,cl=data[train,14],k=11)

table(uci_pred,data[test,14])
#Accuracy for K=3 81%
#Accuracy for K=6 82.6%
#Accuracy for K=9 82.7%
#Accuracy for K=11 82.8%
#Accuracy for K=12 82.7%

#Hence k=11 is our optimal solution