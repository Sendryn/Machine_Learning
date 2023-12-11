#Decision Trees & Random Forests
#Regression tree

rm(list=ls())


LoadLibrairies=function(){
  #install.packages("caTools")
  library(caTools)
  library(MASS)
  #install.packages("rpart")
  library(rpart)
  #install.packages("rpart.plot")
  library(rpart.plot)
  library(ggplot2)
  library(randomForest)
  library(gbm)
  library(cowplot)
  library(caret)
  print('The librairies have been loaded.')
}

LoadLibrairies()

View(Boston)

#Questioon 1
set.seed(18)
Boston_idx = sample(1:nrow(Boston), nrow(Boston) / 2) 

Boston_train = Boston[Boston_idx,]
Boston_test  = Boston[-Boston_idx,]

View(Boston_train)
View(Boston_test)

#Question2
Boston_tree<-rpart(medv~.,Boston_train)

#Question3
plot(Boston_tree)
text(Boston_tree, pretty = 0)
title(main = "Regression Tree")

#Question4
rpart.plot(Boston_tree)
prp(Boston_tree)

#Question 5
print(Boston_tree)
summary(Boston_tree)



printcp(Boston_tree)
plotcp(Boston_tree)



#Question 6
RMSE <- function(x,y) {
  sqrt(mean((x-y)^2))
}

#Question 7
test_pred_tree <- predict(object =Boston_tree, newdata =Boston_test)
RMSE(Boston_train$medv,test_pred_tree)
# 11.72005

#Question 8
lm.fit <- lm(medv ~ ., data = Boston_train)
test_pred_lm <- predict(object =lm.fit, newdata =Boston_test)
RMSE(Boston_train$medv,test_pred_lm)
#11.41185

# linear regression beats the tree! We’ll improve on this tree by considering ensembles of trees.


data_mod <- data.frame(Predicted = predict(Boston_tree),  # Create data for ggplot2
                       Observed = Boston_train$medv)

ggplot(data_mod,                                     # Draw plot using ggplot2 package
       aes(x = Predicted,
           y = Observed)) +
  geom_point() +
  labs(title="Predicted vs Actual: Single Tree, Test Data")+
  geom_abline(intercept = 0,
              slope = 1,
              color = "blue",
              size = 2)

data_lm <- data.frame(Predicted = predict(lm.fit),  # Create data for ggplot2
                       Observed = Boston_train$medv)

ggplot(data_lm,                                     # Draw plot using ggplot2 package
       aes(x = Predicted,
           y = Observed)) +
  geom_point() +
  labs(title="Predicted vs Actual: Linear Model, Test Data")+
  geom_abline(intercept = 0,
              slope = 1,
              color = "blue",
              size = 2)

#Question 10

#install.packages("randomForest")


bag.boston <- randomForest(medv ~ ., 
                           data=Boston_train, 
                           mtry=ncol(Boston_train)-1,
                           importance=TRUE)
bag.boston



#Question 11

test_pred_bag <- predict(object =bag.boston, newdata =Boston_test)
RMSE(Boston_train$medv,test_pred_bag)
#11.8575 worse than the linear model and the simple tree


#Question 12

rf.boston <- randomForest(medv ~ ., 
                           data=Boston_train,
                           #mtry=(ncol(Boston_train)-1)/3,
                           importance=TRUE)
rf.boston

test_pred_rf <- predict(object =rf.boston, newdata =Boston_test)
RMSE(Boston_train$medv,test_pred_rf)
#11.56359 better than the simple tree and the bagging model but worse than the linear model


#Question 13
importance(rf.boston)
#The tree most important predictors are rm, lstat ans crim
#Same result when we have selected the best predictor in linear regression model


#Question 14
varImpPlot(rf.boston)

#Question 15
#install.packages("gbm")
library(gbm)
Boston_boost = gbm(medv ~ ., data = Boston_train, distribution = "gaussian", 
                   n.trees = 5000, interaction.depth = 4, shrinkage = 0.01)

test_pred_boost <- predict(object =Boston_boost, newdata =Boston_test)

RMSE(Boston_train$medv,test_pred_boost)
#12.48116 The worse


#Question 16
summary(Boston_boost)
#the most important variable is lstat



#Question 17

data_bag <- data.frame(Predicted = predict(bag.boston),  # Create data for ggplot2
                      Observed = Boston_train$medv)

data_boost <- data.frame(Predicted = predict(Boston_boost),  # Create data for ggplot2
                      Observed = Boston_train$medv)

data_rf <- data.frame(Predicted = predict(rf.boston),  # Create data for ggplot2
                         Observed = Boston_train$medv)

library(cowplot)

plot_grid(singleTree,bagging,rf,boost, ncol = 2, nrow = 2)
singleTree<-ggplot(data_mod,
       aes(x = Predicted,
           y = Observed)) +
  geom_point() +
  labs(title="Predicted vs Actual: Single Tree, Test Data")+
  geom_abline(intercept = 0,
              slope = 1,
              color = "blue",
              size = 2)

bagging<-ggplot(data_bag,
       aes(x = Predicted,
           y = Observed)) +
  geom_point() +
  labs(title="Predicted vs Actual: Bagging, Test Data")+
  geom_abline(intercept = 0,
              slope = 1,
              color = "blue",
              size = 2)

rf<-ggplot(data_rf,
       aes(x = Predicted,
           y = Observed)) +
  geom_point() +
  labs(title="Predicted vs Actual: Random Forest, Test Data")+
  geom_abline(intercept = 0,
              slope = 1,
              color = "blue",
              size = 2)

boost<-ggplot(data_boost,
       aes(x = Predicted,
           y = Observed)) +
  geom_point() +
  labs(title="Predicted vs Actual: Boosting, Test Data")+
  geom_abline(intercept = 0,
              slope = 1,
              color = "blue",
              size = 2)












#CLASSIFICATION TREE

#Loading Dataset
spam <- read.csv('C:/Fichiers R/A4/spam.csv')
View(spam)

#Response type in factor
spam$spam <- as.integer(spam$spam)
View(spam$spam)
class(spam$spam)
spam$spam <-factor(spam$spam)
class(spam$spam)

View(spam)


#Split the dataset into training and test sets 
set.seed(123)
split = sample.split(spam$spam, SplitRatio = 0.75)
# here we chose the SplitRatio to 75% of the dataset,
# and 25% for the test set.
train = subset(spam, split == TRUE)
# we use subset to split the dataset
test = subset(spam, split == FALSE)
View(train)
View(test)

ACCURACY <- function(actual,predicted) {
  tab<-table(actual,predicted)
  True_pos=tab[1]
  False_neg=tab[3]
  False_pos=tab[2]
  True_neg=tab[4]
  accuracy = (True_pos + True_neg) / (True_pos+False_neg+False_pos+True_neg)
  accuracy
}


#Logistic Regression Model
logistic_model <- glm(formula = spam ~ ., family <- binomial, data <- train)
summary(logistic_model)

test_pred_logistic <- predict(object =logistic_model, newdata =test)
View(test_pred_logistic)

#Simple Tree
spam_tree<-rpart(spam~.,train)
plot(spam_tree)
text(spam_tree, pretty = 0)
title(main = "Regression Tree")


#Bagging 
spam_bag <- randomForest(spam ~ ., 
                           data=train, 
                           mtry=ncol(train)-1,
                           importance=TRUE)
spam_bag
pred_bag <- predict(object =spam_bag, newdata =test)



#Random Forest
spam_rf <- randomForest(spam ~ ., 
                          data=train,
                          #mtry=(ncol(train)-1)/3,
                          importance=TRUE)
spam_rf

pred_rf <- predict(object =spam_rf, newdata =test)
importance(spam_rf)


#Boosting
spam_boost = gbm(spam ~ ., data = train, distribution = "gaussian", 
                   n.trees = 5000, interaction.depth = 4, shrinkage = 0.01)
spam_boost

pred_boost <- predict(object =spam_boost, newdata =test)


test_pred <- predict(object = logistic_model, test, type="response")
test_pred
test$spam = ifelse(test_pred > 0.5, 1,0) 
View(test)

#Data for ggplot2
data_logistic <- data.frame(Predicted = predict(logistic_model), 
                       Observed = train$spam)

data_tree <- data.frame(Predicted = predict(spam_tree), 
                    Observed = train$spam)

data_bag <- data.frame(Predicted = predict(spam_bag), 
                    Observed = train$spam)

data_rf <- data.frame(Predicted = predict(spam_rf), 
                    Observed = train$spam)

data_boost <- data.frame(Predicted = predict(spam_boost), 
                    Observed = train$spam)

plot_grid(singleTree,bagging,rf,boost, ncol = 2, nrow = 2)
ggplot(data_tree,
                   aes(x = Predicted,
                       y = Observed)) +
  geom_point() +
  labs(title="Predicted vs Actual: Single Tree, Test Data")+
  geom_abline(intercept = 0,
              slope = 1,
              color = "blue",
              size = 2)

ggplot(data_bag,
                aes(x = Predicted,
                    y = Observed)) +
  geom_point() +
  labs(title="Predicted vs Actual: Bagging, Test Data")+
  geom_abline(intercept = 0,
              slope = 1,
              color = "blue",
              size = 2)

rf<-ggplot(data_rf,
           aes(x = Predicted,
               y = Observed)) +
  geom_point() +
  labs(title="Predicted vs Actual: Random Forest, Test Data")+
  geom_abline(intercept = 0,
              slope = 1,
              color = "blue",
              size = 2)

boost<-ggplot(data_boost,
              aes(x = Predicted,
                  y = Observed)) +
  geom_point() +
  labs(title="Predicted vs Actual: Boosting, Test Data")+
  geom_abline(intercept = 0,
              slope = 1,
              color = "blue",
              size = 2)

table(spam_tree,predict(spam_tree))

confMat <- table(test$spam,predict(spam_rf))
