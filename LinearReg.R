library(caret)
library(dplyr)
library(xlsx)
library(ggplot2)
parkinsons <- read.csv("parkinsons.csv")
parkinsons <- read.csv("parkinsons.csv")
data <- parkinsons %>% select(motor_UPDRS,starts_with("Jitter"),starts_with("Shimmer"),
                              NHR, HNR, RPDE, DFA, PPE)

n=dim(data)[1]
set.seed(12345) 
id=sample(1:n, floor(n*0.6)) 
train=data[id,]
test=data[-id,]

scaler=preProcess(train)
trainS=predict(scaler,train)
testS=predict(scaler,test)

#Train
mdl=lm(motor_UPDRS ~.,
       data=trainS)
#Predict
TrainmdlPr=predict(mdl,
                   newdata = trainS)
TestmdlPr=predict(mdl,
                  newdata = testS)

#MSE Test and Training
MSE_train=mean((TrainmdlPr -trainS$motor_UPDRS)^2)
cat("MSE Train: ",MSE_train,"\n")
MSE_test=mean((TestmdlPr -testS$motor_UPDRS)^2)
cat("MSE Test: ",MSE_test,"\n")
p_val <- data.frame(p_value = summary(mdl)$coefficients[,4])
print(arrange(p_val%>%filter(p_value < 0.05)))


#Loglikelihood function
loglik <- function(theta,sigma){
  Y <- trainS[,1]
  X <- as.matrix(trainS[-1])
  n <- length(Y)
  result <- -((n/2)*log(2*pi*sigma^2))-(sum((X%*%(as.matrix(theta))-Y)^2)/(2*sigma^2))
  return(result)
}

#Ridge Function
ridgeFun <- function(par,lambda){
  #par:size 17(1:16 theta and 17:sigma)
  theta <- par[-length(par)]#par[-17]
  sigma <- par[length(par)]#par[17]
  result <- -loglik(theta,sigma) + (lambda * sum(theta * theta))
  return(result)
  
}

#Ridge Optimal Function
ridgeOptFun <- function(par,fn,lambda,method){
  result <- optim(par=par,fn=fn,lambda=lambda,
                  method=method)
  return(result)
}

#Degree of Freedom Function
DFFun <- function(lambda,data){
  X <- as.matrix(data[-1])
  # X <- as.matrix(trainS[-1])
  result<- X %*% solve(t(X)%*%X+lambda*diag(ncol(X)))%*%t(X)
  return(sum(diag(result)))
}



#Optimal parameter

## When lambda=1
opt_theta1<-ridgeOptFun(par=c(rep(0,dim(trainS)[2]-1),1),fn=ridgeFun,lambda=1,method="BFGS")
# opt_theta1$par

## When lambda=100
opt_theta100<-ridgeOptFun(par=c(rep(0,dim(trainS)[2]-1),1),fn=ridgeFun,lambda=100,method="BFGS")
# opt_theta100$par

## When lambda=1000
opt_theta1000<-ridgeOptFun(par=c(rep(0,dim(trainS)[2]-1),1),fn=ridgeFun,lambda=1000,method="BFGS")
# opt_theta1000$par

#MSE Calculations
#################
error_cal <- function(data,lambda){
  X <- as.matrix(data[-1])
  opt_theta <- ridgeOptFun(par=c(rep(0,dim(data)[2]-1),1),fn=ridgeFun,lambda=lambda,method="BFGS")
  
  # Y_hat=X*theta_hat
  Y_Predicted<-t(as.matrix(opt_theta$par[-length(opt_theta$par)]))%*%t(X)
  error<-data$motor_UPDRS-Y_Predicted
  return(list(error=error,Y_Predicted=Y_Predicted))
}

#Train error
error_train1<-error_cal(data=trainS,lambda=1)
error_train100<-error_cal(data=trainS,lambda=100)
error_train1000<-error_cal(data=trainS,lambda=1000)


#Test error
error_test1<-error_cal(data=testS,lambda=1)
error_test100<-error_cal(data=testS,lambda=100)
error_test1000<-error_cal(data=testS,lambda=1000)


#MSE result

#When Lambda=1
#--------------

mse_train_val1 <-mean(error_train1$error^2)
mse_test_val1  <-mean(error_test1$error^2)

#When Lambda=100
#--------------

mse_train_val100 <-mean(error_train100$error^2)
mse_test_val100  <-mean(error_test100$error^2)

#When Lambda=1000
#--------------

mse_train_val1000 <-mean(error_train1000$error^2)
mse_test_val1000  <-mean(error_test1000$error^2)

df<-data.frame(lambda=c(1,100,1000),
               MSE_Test=c(mse_test_val1,mse_test_val100,mse_test_val1000),
               MSE_Training=c(mse_train_val1,
                              mse_train_val100,mse_train_val1000))
#kable(df, caption = "Table showing The MSE")



df_plot<-data.frame(lambda=c(1,100,1000),type=c("test","test","test","training","training","training"),
                    MSE=c(mse_test_val1,mse_test_val100,mse_test_val1000,mse_train_val1,
                          mse_train_val100,mse_train_val1000))
ggplot(df_plot, aes(x=lambda, y = MSE)) +
  geom_line(aes(color = type, linetype = type))

## Test
df_te1 <- DFFun(1,testS)
df_te100 <- DFFun(100,testS)
df_te1000 <- DFFun(1000,testS)

## Training

df_tr1 <- DFFun(1,trainS)
df_tr100 <- DFFun(100,trainS)
df_tr1000 <- DFFun(1000,trainS)
df<-data.frame(lambda=c(1,100,1000),
               MSE_Test=c(mse_test_val1,mse_test_val100,mse_test_val1000),
               MSE_Training=c(mse_train_val1,
                              mse_train_val100,mse_train_val1000),
               DF_Test=c(df_te1,df_te100,df_te1000),
               DF_Training=c(df_tr1,df_tr100,df_tr1000))
#kable(df, caption = "Table showing The MSE and Degree of freedom")

