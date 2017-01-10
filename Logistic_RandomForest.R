# @Author Mohammed 28-12-2016

#Loading Packages
source("Libraries.R")

#Loading data
data<-read.csv("ADULT_USI_FE_Numerical.csv",header=T)
View(head(data))
data<-data[,-1]
View(head(data))

#Splitting data into training and testing
train<-sample(1:30162,24129,replace = F)
test<--train

training_data<-data[train,]
testing_data<-data[test,]

logit.fit=glm(income~.,family=binomial(logit),data=training_data)

summary(logit.fit)
#Great No multicolinearity here
vif(logit.fit)

preds<-ifelse(predict(logit.fit,newdata=testing_data[,-14],type="response")>=0.5,1,0)
table(testing_data[,14],preds)

#Accuracy is 80%

#===================================================================
################################## Random Forest
x_train[['income']] <- NULL
x_test[['income']] <- NULL

# Tuning takes factors as target variables
bestmtry <- tuneRF(training_data[,-14], as.factor(training_data[,14]), 
                   ntreeTry=100, stepFactor=1.5, improve=0.01, trace=TRUE, plot=TRUE, dobest=FALSE) 


rf.fit <- randomForest(income ~ ., data=training_data, 
                       mtry=4, ntree=1000, keep.forest=TRUE, importance=TRUE, test=x_test) 

importance(rf.fit)
varImpPlot(rf.fit)

# Confusion Matrix
preds <- predict(rf.fit, newdata=testing_data[,-14], type="response")
table(testing_data[,14], preds)
caret::confusionMatrix(testing_data[,-14], preds, mode = "prec_recall")
#86% Accuracy!!!