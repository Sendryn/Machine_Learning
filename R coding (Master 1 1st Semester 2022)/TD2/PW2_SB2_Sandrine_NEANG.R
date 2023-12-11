##### Practical Work n°2 - Sandrine NEANG ESILV A4 SB2 #####

##### Example 1 #####

#Question 1
library(MASS)
data("Boston")
View(Boston)
dim(Boston)

#Question 2 : Split the dataset into train and test subset using only lstat (X) and medv (Y) variables
variables = c("lstat", "medv")
train = 1:400
test = -train
training_data = Boston[train, variables]
testing_data = Boston[test, variables]

#Question 3 : check for linearity between lstat and medv variables
plot(training_data$lstat,training_data$medv) #according to the plot, the relationship is not linear

#Question 4 : transformation of istat (the explanatory variable - X) by using the "log" function
lstatlog=log(training_data$lstat)

#Question 5 : run the linear regression model using the log transformation
model <- lm(formula = medv~log(lstat), data=training_data) #Y ~ X
model

#Question 6 : plot of the obtained regression model
plot(log(training_data$lstat), training_data$medv,
     xlab = "Log Transform of % of Lower Status of the population",
     ylab = "Median House Value",
     col = "blue",
     pch = 20)

abline(model, col = "red", lwd =3)

#Question 7 : predict what is the median values of the house with lstat = 5%
predict(model, data.frame(lstat = c(5))) #32.14277

#Question 8 : predict what is the median values of houses with lstat = 5%, 10% and 15%
predict(model, data.frame(lstat = c(5,10,15))) #32.14277 ; 23.68432 ; 18.73644 

#Question 9 : compute the mean squared error (MSE) using the test data
prediction <- predict(model, newdata = testing_data)
prediction
MSE <- mean((testing_data$medv - prediction)^2)
MSE #17.69552


##### Example 2 #####

#Question 1
library(datarium)
library(ggpubr)

#Question 2 : load and inspect the marketing data
data("marketing")
View(marketing)

#Question 3 : create a scatter plot displaying the sales units versus youtube advertising budget and add a smoothed line
ggscatter(marketing, "youtube", "sales") + geom_smooth(method=lm)

#Question 4 : Compute the correlation coefficient between sales and youtube features
cor(marketing$youtube, marketing$sales) #0.78

#Question 5 : determine the beta coefficients of the linear model sales = b0 + b1*youtube
marketingmodel = lm(sales~youtube, data=marketing)
marketingmodel #b0 = 8.43911 and b1 = 0.04754

#Question 6 : 
summary(marketingmodel)
#Residuals:
#Min       1Q   Median       3Q      Max 
#-10.0632  -2.3454  -0.2295   2.4805   8.6548 

#Coefficients:
#  Estimate Std. Error t value Pr(>|t|)    
#(Intercept) 8.439112   0.549412   15.36   <2e-16 ***
#  youtube     0.047537   0.002691   17.67   <2e-16 ***
#  ---
#  Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

#Residual standard error: 3.91 on 198 degrees of freedom
#Multiple R-squared:  0.6119,	Adjusted R-squared:  0.6099 
#F-statistic: 312.1 on 1 and 198 DF,  p-value: < 2.2e-16

#Question 7 : 
ggscatter(marketing, "youtube", "sales") + stat_smooth()

#Question 8 :
help(summary)

#Question 9 : 
#p-value is below 2.2e-16 which means that the null hypothesis can be rejected and our results likely did not happen by chance
#therefore there is a significant association between the predictor and the outcome variables.

#Question 10 :
confint(marketingmodel, level=0.95)
#                 2.5 %     97.5 %
#(Intercept) 7.35566312 9.52256140
#youtube     0.04223072 0.05284256

#Question 11 : according to summary(marketingmodel)
RSE=3.91
R2=0.6119
Fstatistic=312.1

#Question 12 : percentage error
RSE/mean(marketing$sales) #0.2323647

#Question 13 : is the F test significant in our case ?
#The F test is significant here because the Fstatistic value is 312.1 which means that it is highly significant.

#Question 14 : 
plot(marketingmodel,1)
#Residuals VS Fitted : there is a horizontal line which means that the model shows a linear relationship between Y and X

plot(marketingmodel, 2)
#Normal Q-Q : the points follow the line which means that residuals points are normally distributed

plot(marketingmodel, 3)
#Scare-Location : the points are not equally spread in the plot which means that our model is a heteroscedacity model 
#to reduce heteroscedacity we need to transform sales Y into log(sales))

#Question 15 : interpretation of the "residuals VS leverage" plot
plot(marketingmodel,5) 
#There are not any influential points in our regression model (extreme values that might influence the regression results)






