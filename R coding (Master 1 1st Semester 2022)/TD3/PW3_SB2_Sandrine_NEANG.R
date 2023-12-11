##### Practical Work n°3 - Sandrine NEANG ESILV A4 SB2 #####

#Question 1
library(MASS)
data("Boston")

#Question 2 : split the datatest into training and testing set
variables = c("crim", "zn", "indus", "chas", "nox", "rm","age", "dis","rad", "tax","ptratio", "black","lstat", "medv")
train = 1:400
test = -train
training_data = Boston[train, variables]
testing_data = Boston[test, variables]

#Question 3 : is there a linear relationship between the variables "medv" and "age" ?
cor(training_data$medv, training_data$age) #-0.278153 => not equals to zero so there is a linear relationship

#Question 4 : 
model1 = lm(medv~age, data = training_data)
plot(training_data$age, training_data$medv,
     xlab="Age of the house",
     ylab="Median House Value",
     col="red",
     pch=20)

abline(model1,col="blue",lwd=3)


#question 5
model2 = lm(medv ~log(lstat) + age, data=training_data)
model2

library(rgl)
options(rgl.printRglwidget = TRUE)

rgl::plot3d(log(Boston$lstat),
            Boston$age,
            Boston$medv, type = "p",
            xlab = "log(lstat)",
            ylab = "age",
            zlab = "medv", site = 5, lwd = 15)


rgl::planes3d(model2$coefficients["log(lstat)"],
              model2$coefficients["age"], -1,
              model2$coefficients["(Intercept)"], alpha = 0.3, front = "line")


#question 6
summary(model2)

#question 7
#p-value is below 2.2e-16 which means that the null hypothesis can be rejected and our results likely did not happen by chance
#therefore there is a significant association between the predictor and the outcome variables.

#question 8
plot(model2,2) 

#question 9
model3 = lm(medv ~ ., data=training_data)
model3

model4 = lm(medv ~ . + log(lstat), data=training_data)
model4











