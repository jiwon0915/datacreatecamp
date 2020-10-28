

# Random Forest model
library(dplyr)
library(ranger)  #randomforest보다 빠르게 가능한 ranger 패키지 사용

train = read.csv('sentiment_dtm_train.csv')
test = read.csv('sentiment_dtm_test.csv')

train_x <- select(train, -c(Id))
train_y <- train$label

test_x <- select(test, -c(Id))
test_y <- test$label


#parameter ‘mtry’의 Grid search를 위한 데이터프레임 생성
rf_result <- data.frame(expand.grid(mtry = seq(10, 130, 30)),
                        "Accuracy" = rep(NA, 5))


#5-fold cv
set.seed(0)
cv_index <- createFolds(train$label, k = 5)

pb <- progress_bar$new(total = nrow(rf_result))
for(i in 1:nrow(rf_result)){
  cv_temp = c()
  for(k in 1:5){
    cv_train = train_x[-cv_index[[k]],]
    cv_test = train_x[cv_index[[k]],]
    model = ranger(label ~.,
                   data=cv_train,
                   num.trees = 300,
                   mtry=rf_result[i, "mtry"],
                   seed = 0,
                   classification = TRUE)
    pred = predict(model, data = cv_test)
    temp_acc = sum(pred$predictions == cv_test$label)/nrow(cv_test)
    
    cv_temp[k] = temp_acc
  }
  rf_result[i,"Accuracy"] = mean(cv_temp)
  
  pb$tick()
}

rf_result


#### Lasso penalty
train = train %>% select(-Id)
n = NROW(train)
K = 5
set.seed(1234)
cvf = cvTools::cvFolds(n,K=K)
grid_lasso = cbind(lambda=c(10^seq(0.2,-6,length=100),0),'ACC'=0)
for (i in 1:NROW(grid_lasso)){
  result <- numeric(5)
  for (k in 1:K)
  {
    index = cvf$subsets[cvf$which == k]
    cv_lasso <- glmnet::glmnet(x=as.matrix(train[-index,-ncol(train)]),y=train$label[-index],
                               family='binomial',alpha=1,lambda=grid_lasso[i,1])
    
    ytrue = train$label[index]
    ypred <- predict(cv_lasso,newx=as.matrix(train[index,-ncol(train)]),type='class') %>% as.numeric()
    result[k] <- mean(ytrue == ypred)
  }
  grid_lasso[i,2] = mean(result)
  # print(grid_lasso[i,])
}
grid_lasso
grid_lasso[which.max(grid_lasso[,2]),]
## lambda 0.001805368 ACC 0.787559898 

fit_lasso = glmnet::glmnet(x=as.matrix(train[-index,-ncol(train)]),y=train$label[-index],
                           family='binomial',alpha=1,lambda=grid_lasso[which.max(grid_lasso[,2]),1])
sum(fit_lasso$beta==0)


lm_fit = fit_ridge = glmnet::glmnet(x=as.matrix(train[-index,-ncol(train)]),y=train$label[-index],
                                    family='binomial',alpha=1,lambda=0)


## Adaptive Lasso
weight = 1/abs(lm_fit$beta)

n = NROW(train)
K = 5
set.seed(1234)
cvf = cvTools::cvFolds(n,K=K)
grid_Alasso = cbind(lambda=c(10^seq(0,-4,length=100),0),'ACC'=0)
for (i in 1:NROW(grid_Alasso)){
  result <- numeric(5)
  for (k in 1:K)
  {
    index = cvf$subsets[cvf$which == k]
    cv_Alasso <- glmnet::glmnet(x=as.matrix(train[-index,-ncol(train)]),y=train$label[-index],
                                family='binomial',alpha=1,lambda=grid_Alasso[i,1],penalty.factor=weight)
    ytrue = train$label[index]
    ypred <- predict(cv_Alasso,newx=as.matrix(train[index,-ncol(train)]),type='class') %>% as.numeric()
    result[k] <- mean(ytrue == ypred)
  }
  grid_Alasso[i,2] = mean(result)
  # print(grid_lasso[i,])
}

grid_Alasso

grid_Alasso[which.max(grid_Alasso[,2]),]
fit_Alasso = glmnet::glmnet(x=as.matrix(train[-index,-ncol(train)]),y=train$label[-index],
                            family='binomial',alpha=1,penalty.factor=weight,lambda=grid_Alasso[which.max(grid_Alasso[,2]),1])
fit_Alasso$beta
## Alasso 성능 0.0041320124 adaptive lasso 0.7905208
sum(fit_Alasso$beta==0)

fit_Alasso
Id = test %>% select(Id)
test = test %>% select(-Id)
y_pred = predict(fit_Alasso,newx=as.matrix(test),type='class') %>% as.numeric()
Id = test %>% select
y_pred_data = cbind(Id,y_pred)
write.csv(y_pred_data,'submission.csv',row.names=F)
