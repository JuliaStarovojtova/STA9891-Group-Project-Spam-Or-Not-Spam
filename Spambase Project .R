rm(list = ls())    #delete objects
cat("\014")        #clear console

### REQUIRED LIBRARIES ###
library(readr)
library(tidyverse)
library(modelr)
library(glmnet)
library(randomForest)
library(reshape)
library(ggthemes)
library(gridExtra)
library(caTools)

set.seed(1)

### OPERATIONS WITH DATA ###

SpamBase = read_csv("/Users/yuliastarovoytova/Desktop/Sta\ 9891\ Project/Data/spambase_csv.csv", 
                        col_names = TRUE)
SpamBase=as.data.frame(SpamBase)
str(SpamBase)
dim(SpamBase)
summary(SpamBase)
table(SpamBase$class)

n_spam = sum(SpamBase$class == 1) #number of spam class outcomes 
n_spam
n_non_spam=sum(SpamBase$class == 0) #number of non_spam class outcomes
n_non_spam

hist(SpamBase$class)

# Create Y vector
Y=SpamBase$class
Y

# Create X matrix
#X = model.matrix(Res.Var.ND~.,TomsHardware)[,-1]
X=data.matrix(SpamBase%>%select(-class))
dim(X)

### TRAIN AND TEST: LASSO, ELASTIC-NET, RIDGE, RANDOM FORREST GENERATION WITH REQUIRED CALCULATIONS ###

# 1 Obtain n (number of observations) and p (number of predictors)
n=nrow(X)
p=ncol(X)

# 2 Define Modeling Parameters including train (n.train) and test (n.test) number of observations
#d.rate=0.9 # division rate for train and test data
d.repit=50 # repeat the ongoing tasks 50 times 
n.train=round(0.9*n)
n.test=n-n.train

# 3 Prepare the following zero-filled matrices that we are going to fill as values are obtained 

# Matrices for train and test R-squared 
Auc.train.table=matrix(0,d.repit,4)
colnames(Auc.train.table)=c("Lasso","Elastic-Net","Ridge","Random Forrest")
Auc.test.table=matrix(0,d.repit,4)
colnames(Auc.test.table)=c("Lasso","Elastic-Net","Ridge","Random Forrest")

# Matrix for the time it takes to cross-validate Lasso/Elastic-Net/Ridge regression
Time.cv=matrix(0,d.repit,3)
colnames(Time.cv)=c("Lasso","Elastic-Net","Ridge")

# 4 Fit Lasso, Elastic-Net, Ridge and Random forest (50 times repition)

for (j in c(1:d.repit)) {
  cat("d.repit = ", j, "\n")
  
  shuffled_indexes =  sample(n)
  train            =  shuffled_indexes[1:n.train]
  test             =  shuffled_indexes[(1+n.train):n]
  X.train          =  X[train, ]
  y.train          =  Y[train]
  X.test           =  X[test, ]
  y.test           =  Y[test]
  
  # Fit Lasso, calculate time, AUC (both train and test), and estimated coefficients
  
  # Lasso
  
  time.start       =  Sys.time()
  Lasso = cv.glmnet(x = X.train, y=y.train, family = "binomial", alpha = 1, intercept = TRUE, standardize = FALSE, nfolds = 10, type.measure="auc")
  time.end         =  Sys.time()
  Time.cv [j,1] = time.end - time.start
  lasso_min = glmnet(x = X.train, y=y.train, lambda = Lasso$lambda[which.max(Lasso$cvm)], family = "binomial", alpha = 1, intercept = TRUE, standardize = FALSE)
  
  beta0.hat.lasso = lasso_min$a0
  beta.hat.lasso = as.vector(lasso_min$beta)
  
  seq_lasso=c(seq(0,1, by = 0.01))
  seq_len_lasso=length(seq_lasso)
  FPR_train_lasso=rep(0, seq_len_lasso)
  TPR_train_lasso=rep(0, seq_len_lasso)
  FPR_test_lasso=rep(0, seq_len_lasso)
  TPR_test_lasso=rep(0, seq_len_lasso)
  
  prob.train.lasso              =        exp(X.train %*% beta.hat.lasso +  beta0.hat.lasso  )/(1 + exp(X.train %*% beta.hat.lasso +  beta0.hat.lasso  ))
  prob.test.lasso              =        exp(X.test %*% beta.hat.lasso +  beta0.hat.lasso  )/(1 + exp(X.test %*% beta.hat.lasso +  beta0.hat.lasso  ))
  
  for (i in 1:seq_len_lasso){
    
    # training
    y.hat.train.lasso             =        ifelse(prob.train.lasso > seq_lasso[i], 1, 0) #table(y.hat.train, y.train)
    FP.train.lasso                =        sum(y.train[y.hat.train.lasso==1] == 0) # false positives = negatives in the data that were predicted as positive
    TP.train.lasso               =        sum(y.hat.train.lasso[y.train==1] == 1) # true positives = positives in the data that were predicted as positive
    P.train.lasso                 =        sum(y.train==1) # total positives in the data
    N.train.lasso                 =        sum(y.train==0) # total negatives in the data
    FPR.train.lasso              =        FP.train.lasso/N.train.lasso # false positive rate = type 1 error = 1 - specificity
    TPR.train.lasso             =        TP.train.lasso/P.train.lasso # true positive rate = 1 - type 2 error = sensitivity
    typeI.err.train.lasso         =        FPR.train.lasso
    typeII.err.train.lasso        =        1 - TPR.train.lasso
    FPR_train_lasso[i]            =        typeI.err.train.lasso
    TPR_train_lasso[i]            =        1 - typeII.err.train.lasso

    
    # test
    y.hat.test.lasso              =        ifelse(prob.test.lasso > seq_lasso[i],1,0) #table(y.hat.test, y.test)  
    FP.test.lasso               =        sum(y.test[y.hat.test.lasso==1] == 0) # false positives = negatives in the data that were predicted as positive
    TP.test.lasso                 =        sum(y.hat.test.lasso[y.test==1] == 1) # true positives = positives in the data that were predicted as positive
    P.test.lasso                  =        sum(y.test==1) # total positives in the data
    N.test.lasso                  =        sum(y.test==0) # total negatives in the data
    TN.test.lasso                 =        sum(y.hat.test.lasso[y.test==0] == 0)# negatives in the data that were predicted as negatives
    FPR.test.lasso                =        FP.test.lasso/N.test.lasso # false positive rate = type 1 error = 1 - specificity
    TPR.test.lasso                =        TP.test.lasso/P.test.lasso # true positive rate = 1 - type 2 error = sensitivity = recall
    typeI.err.test.lasso          =        FPR.test.lasso
    typeII.err.test.lasso        =        1 - TPR.test.lasso
    FPR_test_lasso[i]             =        typeI.err.test.lasso
    TPR_test_lasso[i]             =        1 - typeII.err.test.lasso
  }
  # train
  df.train.lasso = as.data.frame(FPR_train_lasso)
  df.train.lasso=df.train.lasso %>% mutate(TPR_train_lasso)%>%mutate(Set = "Train")%>% mutate(seq_lasso)
  colnames(df.train.lasso)=c("FPR","TPR", "Set", "Threshold")
  
  # test
  df.test.lasso = as.data.frame(FPR_test_lasso)
  df.test.lasso=df.test.lasso %>% mutate(TPR_test_lasso)%>%mutate(Set = "Test")%>% mutate(seq_lasso)
  colnames(df.test.lasso)=c("FPR","TPR", "Set", "Threshold")
  
  # test + train
  
  df_train_test.lasso=rbind(df.train.lasso,df.test.lasso)
  
  #Auc.train[i,1]=colAUC(prob.train.lasso, y.train, plotROC=FALSE)[1]
  Auc.train.table[j,1]=colAUC(prob.train.lasso, y.train, plotROC=FALSE)[1]
  Auc.train.table[j,1]
  #Auc.test[i,1]=colAUC(prob.test.lasso, y.test, plotROC=FALSE)[1]
  Auc.test.table[j,1]=colAUC(prob.test.lasso, y.test, plotROC=FALSE)[1]
  Auc.test.table[j,1] 
  
  # Elastic-net
  
  time.start       =  Sys.time()
  Elastic_net = cv.glmnet(x = X.train, y=y.train, family = "binomial", alpha = 0.5, intercept = TRUE, standardize = FALSE, nfolds = 10, type.measure="auc")
  time.end         =  Sys.time()
  Time.cv [j,2] = time.end - time.start
  elnet_min = glmnet(x = X.train, y=y.train, lambda = Elastic_net$lambda[which.max(Elastic_net$cvm)], family = "binomial", alpha = 1, intercept = TRUE, standardize = FALSE)
  
  beta0.hat.elnet = elnet_min$a0
  beta.hat.elnet = as.vector(elnet_min$beta)
  
  seq_elnet=c(seq(0,1, by = 0.01))
  seq_len_elnet=length(seq_elnet)
  FPR_train_elnet=rep(0, seq_len_elnet)
  TPR_train_elnet=rep(0, seq_len_elnet)
  FPR_test_elnet=rep(0, seq_len_elnet)
  TPR_test_elnet=rep(0, seq_len_elnet)
  
  prob.train.elnet              =        exp(X.train %*% beta.hat.elnet +  beta0.hat.elnet  )/(1 + exp(X.train %*% beta.hat.elnet +  beta0.hat.elnet  ))
  prob.test.elnet              =        exp(X.test %*% beta.hat.elnet +  beta0.hat.elnet  )/(1 + exp(X.test %*% beta.hat.elnet +  beta0.hat.elnet  ))
  
  for (i in 1:seq_len_elnet){
    
    # training
    y.hat.train.elnet             =        ifelse(prob.train.elnet > seq_elnet[i], 1, 0) #table(y.hat.train, y.train)
    FP.train.elnet                =        sum(y.train[y.hat.train.elnet==1] == 0) # false positives = negatives in the data that were predicted as positive
    TP.train.elnet               =        sum(y.hat.train.elnet[y.train==1] == 1) # true positives = positives in the data that were predicted as positive
    P.train.elnet                 =        sum(y.train==1) # total positives in the data
    N.train.elnet                 =        sum(y.train==0) # total negatives in the data
    FPR.train.elnet              =        FP.train.elnet/N.train.elnet # false positive rate = type 1 error = 1 - specificity
    TPR.train.elnet             =        TP.train.elnet/P.train.elnet # true positive rate = 1 - type 2 error = sensitivity
    typeI.err.train.elnet         =        FPR.train.elnet
    typeII.err.train.elnet        =        1 - TPR.train.elnet
    FPR_train_elnet[i]            =        typeI.err.train.elnet
    TPR_train_elnet[i]            =        1 - typeII.err.train.elnet
    
    
    # test
    y.hat.test.elnet              =        ifelse(prob.test.elnet > seq_elnet[i],1,0) #table(y.hat.test, y.test)  
    FP.test.elnet               =        sum(y.test[y.hat.test.elnet==1] == 0) # false positives = negatives in the data that were predicted as positive
    TP.test.elnet                 =        sum(y.hat.test.elnet[y.test==1] == 1) # true positives = positives in the data that were predicted as positive
    P.test.elnet                  =        sum(y.test==1) # total positives in the data
    N.test.elnet                  =        sum(y.test==0) # total negatives in the data
    TN.test.elnet                 =        sum(y.hat.test.elnet[y.test==0] == 0)# negatives in the data that were predicted as negatives
    FPR.test.elnet                =        FP.test.elnet/N.test.elnet # false positive rate = type 1 error = 1 - specificity
    TPR.test.elnet                =        TP.test.elnet/P.test.elnet # true positive rate = 1 - type 2 error = sensitivity = recall
    typeI.err.test.elnet          =        FPR.test.elnet
    typeII.err.test.elnet        =        1 - TPR.test.elnet
    FPR_test_elnet[i]             =        typeI.err.test.elnet
    TPR_test_elnet[i]             =        1 - typeII.err.test.elnet
  }
  # train
  df.train.elnet = as.data.frame(FPR_train_elnet)
  df.train.elnet=df.train.elnet %>% mutate(TPR_train_elnet)%>%mutate(Set = "Train")%>% mutate(seq_elnet)
  colnames(df.train.elnet)=c("FPR","TPR", "Set", "Threshold")
  
  # test
  df.test.elnet = as.data.frame(FPR_test_elnet)
  df.test.elnet=df.test.elnet %>% mutate(TPR_test_elnet)%>%mutate(Set = "Test")%>% mutate(seq_elnet)
  colnames(df.test.elnet)=c("FPR","TPR", "Set", "Threshold")
  
  # test + train
  
  df_train_test.elnet=rbind(df.train.elnet,df.test.elnet)
  
  #Auc.train[i,1]=colAUC(prob.train.elnet, y.train, plotROC=FALSE)[1]
  Auc.train.table[j,2]=colAUC(prob.train.elnet, y.train, plotROC=FALSE)[1]
  #Auc.test[i,1]=colAUC(prob.test.elnet, y.test, plotROC=FALSE)[1]
  Auc.test.table[j,2]=colAUC(prob.test.elnet, y.test, plotROC=FALSE)[1]

  # Ridge
  
  time.start       =  Sys.time()
  Ridge = cv.glmnet(x = X.train, y=y.train, family = "binomial", alpha = 0.5, intercept = TRUE, standardize = FALSE, nfolds = 10, type.measure="auc")
  time.end         =  Sys.time()
  Time.cv [j,3] = time.end - time.start
  ridge_min = glmnet(x = X.train, y=y.train, lambda = Ridge$lambda[which.max(Ridge$cvm)], family = "binomial", alpha = 1, intercept = TRUE, standardize = FALSE)
  
  beta0.hat.ridge = ridge_min$a0
  beta.hat.ridge = as.vector(ridge_min$beta)
  
  seq_ridge=c(seq(0,1, by = 0.01))
  seq_len_ridge=length(seq_ridge)
  FPR_train_ridge=rep(0, seq_len_ridge)
  TPR_train_ridge=rep(0, seq_len_ridge)
  FPR_test_ridge=rep(0, seq_len_ridge)
  TPR_test_ridge=rep(0, seq_len_ridge)
  
  prob.train.ridge              =        exp(X.train %*% beta.hat.ridge +  beta0.hat.ridge  )/(1 + exp(X.train %*% beta.hat.ridge +  beta0.hat.ridge  ))
  prob.test.ridge              =        exp(X.test %*% beta.hat.ridge +  beta0.hat.ridge  )/(1 + exp(X.test %*% beta.hat.ridge +  beta0.hat.ridge  ))
  
  for (i in 1:seq_len_ridge){
    
    # training
    y.hat.train.ridge             =        ifelse(prob.train.ridge > seq_ridge[i], 1, 0) #table(y.hat.train, y.train)
    FP.train.ridge                =        sum(y.train[y.hat.train.ridge==1] == 0) # false positives = negatives in the data that were predicted as positive
    TP.train.ridge               =        sum(y.hat.train.ridge[y.train==1] == 1) # true positives = positives in the data that were predicted as positive
    P.train.ridge                 =        sum(y.train==1) # total positives in the data
    N.train.ridge                 =        sum(y.train==0) # total negatives in the data
    FPR.train.ridge              =        FP.train.ridge/N.train.ridge # false positive rate = type 1 error = 1 - specificity
    TPR.train.ridge             =        TP.train.ridge/P.train.ridge # true positive rate = 1 - type 2 error = sensitivity
    typeI.err.train.ridge         =        FPR.train.ridge
    typeII.err.train.ridge        =        1 - TPR.train.ridge
    FPR_train_ridge[i]            =        typeI.err.train.ridge
    TPR_train_ridge[i]            =        1 - typeII.err.train.ridge
    
    
    # test
    y.hat.test.ridge              =        ifelse(prob.test.ridge > seq_ridge[i],1,0) #table(y.hat.test, y.test)  
    FP.test.ridge               =        sum(y.test[y.hat.test.ridge==1] == 0) # false positives = negatives in the data that were predicted as positive
    TP.test.ridge                 =        sum(y.hat.test.ridge[y.test==1] == 1) # true positives = positives in the data that were predicted as positive
    P.test.ridge                  =        sum(y.test==1) # total positives in the data
    N.test.ridge                  =        sum(y.test==0) # total negatives in the data
    TN.test.ridge                 =        sum(y.hat.test.ridge[y.test==0] == 0)# negatives in the data that were predicted as negatives
    FPR.test.ridge                =        FP.test.ridge/N.test.ridge # false positive rate = type 1 error = 1 - specificity
    TPR.test.ridge                =        TP.test.ridge/P.test.ridge # true positive rate = 1 - type 2 error = sensitivity = recall
    typeI.err.test.ridge          =        FPR.test.ridge
    typeII.err.test.ridge        =        1 - TPR.test.ridge
    FPR_test_ridge[i]             =        typeI.err.test.ridge
    TPR_test_ridge[i]             =        1 - typeII.err.test.ridge
  }
  # train
  df.train.ridge = as.data.frame(FPR_train_ridge)
  df.train.ridge=df.train.ridge %>% mutate(TPR_train_ridge)%>%mutate(Set = "Train")%>% mutate(seq_ridge)
  colnames(df.train.ridge)=c("FPR","TPR", "Set", "Threshold")
  
  # test
  df.test.ridge = as.data.frame(FPR_test_ridge)
  df.test.ridge=df.test.ridge %>% mutate(TPR_test_ridge)%>%mutate(Set = "Test")%>% mutate(seq_ridge)
  colnames(df.test.ridge)=c("FPR","TPR", "Set", "Threshold")
  
  # test + train
  
  df_train_test.ridge=rbind(df.train.ridge,df.test.ridge)
  
  #Auc.train[i,1]=colAUC(prob.train.ridge, y.train, plotROC=FALSE)[1]
  Auc.train.table[j,3]=colAUC(prob.train.ridge, y.train, plotROC=FALSE)[1]
  #Auc.test[i,1]=colAUC(prob.test.ridge, y.test, plotROC=FALSE)[1]
  Auc.test.table[j,3]=colAUC(prob.test.ridge, y.test, plotROC=FALSE)[1]
}


mean(Auc.train.table[,1])
mean(Auc.test.table[,1])
mean(Auc.train.table[,2])
mean(Auc.test.table[,2])
mean(Auc.train.table[,3])
mean(Auc.test.table[,3])

mean(Time.cv[,1])
mean(Time.cv[,2])
mean(Time.cv[,3])



#Side-by-Side Boxplots of AUC 
par(mfrow=c(1,2))
a=boxplot(Auc.train.table[,1], Auc.train.table[,2],Auc.train.table[,3],Auc.train.table[,4],
          main = "Auc for Train Data",
          names = c("Lasso", "Elastic-Net", "Ridge", "Random Forest"),
          col=c("blue","yellow", "green","red"),
          ylim=c(0.94,0.98))


b=boxplot(Auc.test.table[,1], Auc.test.table[,2],Auc.test.table[,3],Auc.test.table[,4],
          main = "Auc for Test Data",
          names = c("Lasso", "Elastic-Net", "Ridge", "Random Forest"),
          col=c("blue","yellow", "green","red"),
          ylim=c(0.94,0.98))


