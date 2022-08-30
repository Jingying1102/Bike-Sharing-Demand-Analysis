train <- read.csv("D:/STAT5630/HW/Project/data/train.csv")
splitdatetime<-matrix(unlist(strsplit(as.character(train$datetime),split=' ')),ncol=2,byrow=T)
colnames(splitdatetime)<-c('date','time')
bikesharing<-subset(cbind(train,splitdatetime),select=-c(datetime,registered,casual))
bikesharing$date<-as.Date(bikesharing$date)
library(lubridate)
bikesharing$month<-month(bikesharing$date)
bikesharing$year<-year(bikesharing$date)
bikesharing$hour<-hour(train$datetime)

bikesharing$temp<-abs(bikesharing$temp-25)
bikesharing$atemp<-abs(bikesharing$atemp-25)

bikesharing$season<-as.factor(bikesharing$season)
bikesharing$holiday<-as.factor(bikesharing$holiday)
bikesharing$workingday<-as.factor(bikesharing$workingday)
bikesharing$weather<-as.factor(bikesharing$weather)
bikesharing$temp<-as.numeric(bikesharing$temp)
bikesharing$atemp<-as.numeric(bikesharing$atemp)
bikesharing$humidity<-as.numeric(bikesharing$humidity)
bikesharing$windspeed<-as.numeric(bikesharing$windspeed)
bikesharing$hour<-as.factor(bikesharing$hour)
bikesharing$year<-as.factor(bikesharing$year)


train<-bikesharing
set.seed(1693)
smp_size <- floor(0.8 * nrow(train))
train_ind <- sample(seq_len(nrow(train)), size = smp_size)
train.bike <- train[train_ind, ]
test.bike <- train[-train_ind, ] # random sample testing set

#a<-sample(5,length(bikesharing$date),replace=T)
#b<- a==1
#test.bike<-bikesharing[b,]
#train.bike<-bikesharing[!b,]
#trainbike.x<-subset(train.bike[,-10],select=-c(count))
#trainbike.y<-subset(train.bike[,-10],select=c(count))
#train.dropdate<-train.bike[,-10]
#train.dropdatetime<-train.dropdate[,-10]

#creat function to caculate mse
mse = function(yi, yi_pred) {
  mean((yi - yi_pred) ^ 2)
  }
#linear regression
linear.fit<-lm(count~.-date-time,data=train.bike)
#summary(linear.fit)
linear.mse<-mse(predict(linear.fit,test.bike),test.bike$count)
linear.mse
#add interaction terms
linear.fit<-lm(count~.-date-time-atemp+hour*workingday,data=train.bike)
#summary(linear.fit)
linear.int.mse<-mse(predict(linear.fit,test.bike),test.bike$count)
linear.int.mse
#log transformation
library(MASS)
lmBikeRentLog <- lm(count~.+hour*workingday, data = train.bike)
lmBikeRentLogAIC <- stepAIC(lmBikeRentLog, direction="both")
plot(lmBikeRentLog$fitted.values, lmBikeRentLog$residuals,pch='.',col=2)
AIC.mse<-mse(test.bike$count,predict(lmBikeRentLogAIC,newdata=test.bike))
AIC.mse

#knn
library(kknn)

knn.mse<-c()
for(i in 1:20){
knn.fit = kknn(count~., train = train.bike, test = test.bike,
               k = i, kernel = "rectangular")
test.pred = knn.fit$fitted.values
knn.mse[i]<-mean((test.pred - test.bike$count)^2)
}
bestk<-which.min(knn.mse)
knn.bestk<-kknn(count~., train = train.bike, test = test.bike,
                k = 2, kernel = "rectangular")
knn.pred = knn.bestk$fitted.values
knn.mse<-mean((test.pred - test.bike$count)^2)
knn.mse
#partial least square regression
library(leaps)
pls.fit<-regsubsets(count~.,data=train.bike[,-c(10,11)])
summary(pls.fit)
pls.fit.p<-regsubsets(count~.,data=train.bike[,-c(10,11)],nvmax=4)
summary(pls.fit.p)
predict.regsubsets = function(object, newdata, id, ...) {
  form = as.formula(object$call[[2]])
  mat = model.matrix(form, newdata)
  coefi = coef(object, id = id)
  mat[, names(coefi)] %*% coefi
}
pls.mse<-mse(test.bike$count,predict.regsubsets(pls.fit.p,test.bike,id=4))
pls.mse

#tree
library(gbm)
train.dropdate<-train.bike[,-10]
tree.reg<-gbm(count~.,data=train.dropdate[,-10],distribution = 'gaussian',n.trees = 1000,interaction.depth = 4)
tree.reg
par(cex.axis=0.7,cex=1,cex.lab=1,las=1,mfcol=c(1,1))
title(main='Relative Influence of Each Variable on Count')
summary(tree.reg)#effect of variables on count
title(main='Relative Influence of Each Variable on Count')
#time and workingday are the two most important variables
tree.mse<-mse(test.bike$count,predict(tree.reg,test.bike,n.trees=1000))
tree.mse
predict(tree.reg,test.bike,n.trees=1000)

grplasso(count~.,data=train.dropdate[,-10], nonpen = ~ 1, data, weights,
         subset, lambda=20, penscale = sqrt,
         model = LogReg(), center = TRUE, standardize = TRUE,
         control = grpl.control(), contrasts = NULL, ...)

# Ridge

r.bike<-train.bike[,c(-6,-9,-10,-11)]
r.test.bike<-test.bike[,c(-6,-9,-10,-11)]

ridge_glm_fit  <- cv.glmnet(data.matrix(r.bike), train.bike$count,alpha = 0, nfolds = 5)
ridge_lambda  <- ridge_glm_fit$lambda.min  # ridge lambda
ridge_lambda  # generally close to 7
fit.r <- glmnet(data.matrix(r.bike), train.bike$count, lambda = 7.202063,alpha = 0)
#coef(fit.r,s = 0.12, exact = F) # ridge model
r.mse<-mse(test.bike$count,predict(fit.r,newx = data.matrix(r.test.bike), s = 7.202063, type = c("response")))
r.mse

# Lasso

lasso_glm_fit  <- cv.glmnet(data.matrix(r.bike), train.bike$count,alpha = 1, nfolds = 5)
lasso_lambda  <- lasso_glm_fit$lambda.min  # lasso lambda
lasso_lambda  # generally close to 0.3
fit.l <- glmnet(data.matrix(r.bike), train.bike$count, lambda = 0.3584484,alpha = 1)
l.mse<-mse(test.bike$count,predict(fit.l,newx = data.matrix(r.test.bike), s = 0.3584484, type = c("response")))
l.mse