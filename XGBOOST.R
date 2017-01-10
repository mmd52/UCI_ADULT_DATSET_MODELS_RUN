# @Author Mohammed 28-12-2016

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

#====================================================================
######################## Preparing for xgboost

dtrain = xgb.DMatrix(as.matrix(training_data[,-14]), label=training_data[,14])
dtest = xgb.DMatrix(as.matrix(testing_data[,-14]))

xgb_param_adult = list(
  nrounds = c(700),
  eta = 0.075,#eta between(0.01-0.2)
  max_depth = 6, #values between(3-10)
  subsample = 0.7,#values between(0.5-1)
  colsample_bytree = 0.7,#values between(0.5-1)
  num_parallel_tree=1,
  objective='binary:logistic',
  min_child_weight = 1
)

res = xgb.cv(xgb_param_adult,
             dtrain,
             nrounds=700,   # changed
             nfold=3,           # changed
             early_stopping_rounds=15,
             print_every_n = 10,
             verbose= 1)

xgb.fit = xgb.train(xgb_param_adult, dtrain, 500)


# Confusion Matrix
preds <- ifelse(predict(xgb.fit, newdata=as.matrix(testing_data[,-14])) >= 0.5, 1, 0)
caret::confusionMatrix(testing_data[,14], preds, mode = "prec_recall")

#========86.29% accuracy