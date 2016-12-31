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
  nrounds = c(200),
  eta = 0.075,
  max_depth = 6, 
  subsample = 0.7,
  colsample_bytree = 0.7,
  num_parallel_tree=1,
  objective='binary:logistic',
  min_child_weight = 1
)

res = xgb.cv(xgb_param_adult,
             dtrain,
             nrounds=200,   # changed
             nfold=3,           # changed
             early_stopping_rounds=15,
             print_every_n = 10,
             verbose= 1)

xgb.viz.cv <- function(dataset) {
  dataset$iteration <- as.integer(rownames(dataset))
  p <- ggplot(dataset, aes(x = iteration)) + 
    geom_line(aes(y = train.error.mean), colour="blue") + 
    geom_line(aes(y = test.error.mean), colour = "red") + 
    #geom_line(aes(y = train.error.mean + train.error.std), colour="black") +
    #geom_line(aes(y = train.error.mean - train.error.std), colour="black") +
    ylab(label="Error (MAE)") + 
    xlab("Iteration") + 
    ggtitle("Test vs Train") +
    scale_colour_manual(name="Dataset", values=c(test="red", train="blue")) 
  return(p)
}
xgb.viz.cv(res)
xgb.fit = xgb.train(xgb_param_adult, dtrain, 490)


# Confusion Matrix
x_test[['income']] <- NULL
preds <- ifelse(predict(xgb.fit, newdata=as.matrix(testing_data[,-14])) >= 0.5, 1, 0)
caret::confusionMatrix(testing_data[,14], preds, mode = "prec_recall")

# Overfitting?
xgb.fit = xgb.train(xgb_param_adult, dtrain, 700)
preds <- ifelse(predict(xgb.fit, newdata=as.matrix(testing_data[,-14])) >= 0.5, 1, 0)
caret::confusionMatrix(testing_data[,14], preds, mode = "prec_recall")

#========85% accuracy