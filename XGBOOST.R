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


#======================================================
ntrain <- 21115
ntree <- 100
# bootstrap samples the same size as the training set
# samples <- sapply(1:ntree,
#                   FUN = function(iter){ 
#                     sample(x_train, size=1, replace=T)
#                   }
# )
# sample without replacement
sample_instance <- function(data, p=0.7) {
  idx <- sample(c(TRUE, FALSE), nrow(data), replace=TRUE, prob = c(p, 1-p))
  sampleSet <- data[idx, ]
  return(sampleSet)
}

x_train$income <- y_train
x_test$income <- y_test
# train individual trees
treelist <-lapply(1:ntree,
                  FUN=function(iter){
                    samp <- sample_instance(x_train);
                    logit.fit = glm(income ~ ., family = binomial(logit), data = samp)
                    return(logit.fit)
                  }
)
# make predictions
predict.bag <- function(treelist, newdata) {
  preds <- sapply(1:length(treelist),
                  FUN=function(iter) {
                    ifelse(predict(treelist[[iter]], newdata=newdata, type="response") >= 0.5, 1, 0)
                  }
  )
  predsums <- rowSums(preds)
  preds.frac <- predsums/length(treelist)
  preds.bag <- ifelse(preds.frac >= 0.5, 1, 0)
  return(preds.bag)
}

x_test[['income']] <- NULL
preds <- predict.bag(treelist, x_test)
caret::confusionMatrix(y_test, preds, mode = "prec_recall")


#==================================================================
################################# Stacking

### logit reg
x_train$income <- y_train
x_test$income <- y_test
logit.fit = glm(income ~ ., family = binomial(logit), data = x_train)
preds.logit.train <- ifelse(predict(logit.fit, newdata=x_train, type="response") >= 0.5, 1, 0)
preds.logit.test <- ifelse(predict(logit.fit, newdata=x_test, type="response") >= 0.5, 1, 0)

### rf
x_train[['income']] <- NULL
x_test[['income']] <- NULL
# Tuning takes factors as target variables
bestmtry <- tuneRF(x_train, as.factor(y_train), 
                   ntreeTry=100, stepFactor=1.5, improve=0.01, trace=TRUE, plot=TRUE, dobest=FALSE) 
x_train$income <- as.factor(y_train)
x_test$income <- as.factor(y_test)
rf.fit <- randomForest(income ~ ., data=x_train, 
                       mtry=3, ntree=1000, keep.forest=TRUE, importance=TRUE, test=x_test) 
preds.rf <- predict(rf.fit, newdata=x_test, type="response")


### stacked rf
# logit reg as a feature
x_train$logit_f <- preds.logit.train
x_test$logit_f <- preds.logit.test

x_train[['income']] <- NULL
x_test[['income']] <- NULL
# Tuning takes factors as target variables
bestmtry <- tuneRF(x_train, as.factor(y_train), 
                   ntreeTry=100, stepFactor=1.5, improve=0.01, trace=TRUE, plot=TRUE, dobest=FALSE) 
x_train$income <- as.factor(y_train)
x_test$income <- as.factor(y_test)
rf.fit <- randomForest(income ~ ., data=x_train, 
                       mtry=3, ntree=1000, keep.forest=TRUE, importance=TRUE, test=x_test) 
preds.rf.stacked <- predict(rf.fit, newdata=x_test, type="response")

# Compare performance
acc.logit <- caret::confusionMatrix(y_test, preds.logit, mode = "prec_recall")
acc.rf <- caret::confusionMatrix(y_test, preds.rf, mode = "prec_recall")
acc.rf.stacked <- caret::confusionMatrix(y_test, preds.rf.stacked, mode = "prec_recall")

acc.logit$overall['Accuracy']
acc.rf$overall['Accuracy']
acc.rf.stacked$overall['Accuracy']
