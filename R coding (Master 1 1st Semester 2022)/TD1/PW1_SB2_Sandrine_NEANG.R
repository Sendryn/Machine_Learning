##### Practical Work n°1 - Sandrine NEANG ESILV A4 SB2 #####


##### Data Preprocessing in R #####

#Question 1
#install.packages("caTools")
library(caTools)

#Question 2
dataset <- read.csv("~/Scolarité/ESILV/M1 A4/S7/Machine Learning/TDs/Datas/dataset.csv")

#Question 3
#View(dataset)

#Question 4
dataset$Age[is.na(dataset$Age)] <- mean(dataset$Age,na.rm=TRUE) #na.rm=TRUE => retire les NA dans le calcul de la moyenne
#View(dataset)

dataset$Salary[is.na(dataset$Salary)] <- mean(dataset$Salary,na.rm=TRUE)
#View(dataset)

#Question 5
dataset$Country = factor(dataset$Country, levels = c('France','Spain','Germany'), labels = c(1,2,3))

#Question 6
dataset$Purchased = factor(dataset$Purchased, levels = c('Yes','No '), labels = c(1,2))

sample <- sample.split(dataset$Purchased, SplitRatio = 0.6)
train <- subset(dataset, sample==TRUE)
test <- subset(dataset, sample==FALSE)

#View(train)
#View(test)

#Question 7
#train's normalization
train[,2]<- (train[,2] - min(train$Age))/(max(train$Age)-min(train$Age))
train[,2]<- 2*train[,2]
train[,2]<- train[,2]-1
train[,2]<- 3*train[,2]

train[,3]<- (train[,3] - min(train$Salary))/(max(train$Salary)-min(train$Salary))
train[,3]<- 2*train[,3]
train[,3]<- train[,3]-1
train[,3]<- 3*train[,3]

#test's normalization
test[,2]<- (test[,2] - min(test$Age))/(max(test$Age)-min(test$Age))
test[,2]<- 2*test[,2]
test[,2]<- test[,2]-1
test[,2]<- 3*test[,2]

test[,3]<- (test[,3] - min(test$Salary))/(max(test$Salary)-min(test$Salary))
test[,3]<- 2*test[,3]
test[,3]<- test[,3]-1
test[,3]<- 3*test[,3]

#View(train)
#View(test)

##### First Machine Learning Project in R Step-By-Step #####
#Question 1
data("iris")
View(iris)

#Question 2
#install.packages('caret')
library(caret)

validation_iris <- createDataPartition(iris$Species, p=0.80, list=FALSE)

test_iris <- iris[-validation_iris,]
View(test_iris)
train_iris <- iris[validation_iris,]
View(train_iris)

#Question 3
dim(iris)
summary(iris)
sapply(iris, class)
levels(iris$Species)

#Question 4
prop.table(iris$Sepal.Length)

#Question 5
par(mfrow=c(1,4))
for(i in 1:4) {
  boxplot(iris[,1:4][,i], main=names(iris)[i])
}

#Question 6
par(mfrow=c(1,4))
for(i in 1:4) {
  barplot(height=iris[,1:4][,i], names=names(iris)[i])
}

#Question 7
featurePlot(x=iris[,1:4], y=iris[,5], plot="box")
featurePlot(x=iris[,1:4], y=iris[,5], plot="ellipse")

#Question 8 
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"

#Question 9
#kNN
set.seed(7)
fit.knn <- train(Species~., data=iris, method="knn", metric=metric, trControl=control)

#SVM
set.seed(7)
fit.svm <- train(Species~., data=iris, method="svmRadial", metric=metric, trControl=control)

#Random Forest
set.seed(7)
fit.rf <- train(Species~., data=iris, method="rf", metric=metric, trControl=control)

#Question 10
results <- resamples(list( knn=fit.knn, svm=fit.svm, rf=fit.rf))
summary(results)

#Question 11
dotplot(results)

#Question 12
print(fit.knn)







