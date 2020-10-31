library(dplyr)
library(glmnet)

train <- read.csv("train.csv")

train = train[,-1]
train$R <-rowSums(train[,1:1024])
train$G <- rowSums(train[,1025:2048])
train$B <- rowSums(train[,2049:3072])


## 파생변수 있는 Lasso
sam<-sample(1:nrow(train),nrow(train)*0.7)
fit_lasso = glmnet::glmnet(x=as.matrix(train[sam,-(ncol(train)-3)]),y=train$label[sam],
                           family='multinomial',alpha=1,nlambda = 100)

s<-fit_lasso$lambda[length(fit_lasso$lambda)]

pr<-predict(fit_lasso,as.matrix(train[-sam,-c(ncol(train)-3)]),s=s)

sum(apply(pr,1,which.max)==train$label[-sam]-2)/dim(pr)[1]

## 파생변수 없는 Lasso
train <- read.csv("train.csv")

train = train[,-1]

sam<-sample(1:nrow(train),nrow(train)*0.7)
fit_lasso = glmnet::glmnet(x=as.matrix(train[sam,-(ncol(train))]),y=train$label[sam],
                           family='multinomial',alpha=1,nlambda = 100)

s<-fit_lasso$lambda[length(fit_lasso$lambda)]

pr<-predict(fit_lasso,as.matrix(train[-sam,-c(ncol(train))]),s=s)

sum(apply(pr,1,which.max)==train$label[-sam]-2)/dim(pr)[1]




## 파생변수 있는 XGBoost
params = list(max_depth = 6,
              min_child_weight = 6,
              subsample = 0.8,
              colsample_bytree = 0.5,
              eta = 0.3,
              objective="multi:softprob",
              eval_metric="mlogloss",
              num_class=3)
sam<-sample(1:nrow(train),nrow(train)*0.7)
model = xgboost(data = as.matrix(train[sam, -c(ncol(train)-3)]), label = train$label[sam]-3, params = params, nrounds = nrounds, early_stopping_rounds = 500, verbose = 0)
pred = predict(model, newdata = as.matrix(train[-sam, -c(ncol(train)-3)]))

pred=apply(matrix(pred,ncol=3,byrow=T),1,which.max)

sum(pred==train$label[-sam]-2)/length(pred)

## 파생변수 없는 XGBoost
train <- read.csv("train.csv")

train = train[,-1]

sam<-sample(1:nrow(train),nrow(train)*0.7)
params = list(max_depth = 6,
              min_child_weight = 6,
              subsample = 0.8,
              colsample_bytree = 0.5,
              eta = 0.3,
              objective="multi:softprob",
              eval_metric="mlogloss",
              num_class=3)
sam<-sample(1:nrow(train),nrow(train)*0.7)
model = xgboost(data = as.matrix(train[sam, -c(ncol(train))]), label = train$label[sam]-3, params = params, nrounds = nrounds, early_stopping_rounds = 500, verbose = 0)
pred = predict(model, newdata = as.matrix(train[-sam, -c(ncol(train))]))

pred=apply(matrix(pred,ncol=3,byrow=T),1,which.max)

sum(pred==train$label[-sam]-2)/length(pred)
