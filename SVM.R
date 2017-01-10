# @Author Mohammed 01-01-2017

#Loading Packages
source("Libraries.R")

#Loading data
data<-read.csv("ADULT_USI_FE_Numerical_XGBOOST.csv",header=T)
View(head(data))
data<-data[,-1]
View(head(data))

train_test<-data
features = names(train_test)
for (f in features) {
  if (class(train_test[[f]])=="integer") {
    levels <- unique(train_test[[f]])
    train_test[[f]] <- as.numeric(as.integer(factor(train_test[[f]], levels=levels)))
  }
}

for(i in 1:30162){
  if(train_test[i,14]==1)
  {
    train_test[i,14]=0
  }
  else if(train_test[i,14]==2)
  {
    train_test[i,14]=1
  }
}

set.seed(999)
#Splitting data into training and testing
train<-sample(1:30162,24129,replace = F)
test<--train

training_data<-train_test[train,]
testing_data<-train_test[test,]


svm_model<-svm(training_data[,-14],training_data[,14])
summary(svm_model)

pred<- ifelse(predict(svm_model,testing_data[,-14])>=0.5,1,0)

table(pred,testing_data[,14])

#Accuracy of SVM is 81%