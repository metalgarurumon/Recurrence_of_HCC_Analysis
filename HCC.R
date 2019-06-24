library(tidyverse)
library(caret)
library(nnet)

accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) }
recall <- function(x){x[2, 2]/sum(x[2, ]) }
precision <- function(x){x[2, 2]/sum(x[, 2]) }

## Load the data

setwd("E:/liver cancer big data")

HCC = read.csv("hrr.csv", stringsAsFactors = F)

HCC = select(HCC, -ID, -Tip)

HCC_RecS2_0 = HCC %>% filter(RecS2==0)
HCC_RecS2_1 = HCC %>% filter(RecS2==1)

head(HCC)

str(HCC)

table(HCC$RecS2)/nrow(HCC)

res = cor(HCC, method = "spearman")

res[1,]

library(corrplot)
corrplot(res, type = "upper", order = "hclust", 
         tl.col = "black", tl.srt = 45)


glm_1 = glm(RecS2 ~ ., family = binomial(logit), data = HCC)
summary(glm_1)

step(glm_1, k = log(nrow(HCC)))  # stepwise selection based on BIC

summary(glm_1)

glm_BIC = glm(RecS2 ~ Gender + AFP + TD + TN + MVI + LC, 
              family = binomial(logit), data = HCC)

summary(glm_BIC)




##############################################
# normalize all continuous variables

continuous_vars <- c("Age", "AFP", "WBC", "PLT", 
                      "ALB", "TBIL", "GGT", "ALP", "TD","TN")

##the normalization function is created
#normal <- function(x) { (x -min(x))/(max(x)-min(x))   }

##Run nomalization on first 4 coulumns of dataset because they are the predictors
#HCC[, continuous_vars] <- lapply(HCC[, continuous_vars], normal)

summary(HCC)


##############################################
# transform all categorical variables
categorical_vars <- c("Gender", "Etiology", "RM", "RT", 
                      "OBL", "MVI", "EG", "TC", "LC", "RecS2")
HCC[, categorical_vars] <- lapply(HCC[, categorical_vars] , factor)
str(HCC)
dim(HCC)

summary(HCC)



##############################################
# Split the data into training and test set
set.seed(1)
training_samples <- HCC$RecS2 %>% 
    createDataPartition(p = 0.7, list = FALSE)
train_data <- HCC[training_samples, ]
test_data <- HCC[-training_samples, ]

dim(train_data)
dim(test_data)

train_data <- transform(train_data, RecS2 = as.factor(RecS2)) 
test_data <- transform(test_data, RecS2 = as.factor(RecS2))



#########################SVM

library(e1071)
svm.model <- svm(RecS2 ~ ., train_data)
summary(svm.model)


as.vector(predict(svm.model, train_data, type = "class") )

#training set
confusion.train.svm = table(train_data$RecS2, 
                            predict(svm.model, train_data, type = "class"))
accuracy.train.svm = sum(diag(confusion.train.svm))/sum(confusion.train.svm)
recall.train.svm = confusion.train.svm[2, 2]/sum(confusion.train.svm[2, ])
precision.train.svm = confusion.train.svm[2, 2]/sum(confusion.train.svm[, 2])

#test set
confusion.test.svm = table(test_data$RecS2,
                           predict(svm.model, test_data, type = "class"))
accuracy.test.svm = sum(diag(confusion.test.svm))/sum(confusion.test.svm)
recall.test.svm = confusion.test.svm[2, 2]/sum(confusion.test.svm[2, ])
precision.test.svm = confusion.test.svm[2, 2]/sum(confusion.test.svm[, 2])



svm_same = which(test_data$RecS2 == predict(svm.model, test_data, type = "class") )
svm_different = which(test_data$RecS2 != predict(svm.model, test_data, type = "class") )


#########################Random Forest

library(randomForest)
randomForest.model <- randomForest(RecS2 ~ ., train_data)
summary(randomForest.model)

#training set
confusion.train.randomForest = table(train_data$RecS2,
                                     predict(randomForest.model, train_data, type = "class"))
accuracy.train.randomForest = sum(diag(confusion.train.randomForest))/sum(confusion.train.randomForest)
recall.train.randomForest = confusion.train.randomForest[2, 2]/sum(confusion.train.randomForest[2, ])
precision.train.randomForest = confusion.train.randomForest[2, 2]/sum(confusion.train.randomForest[, 2])

#test set
confusion.test.randomForest = table(test_data$RecS2,
                                  predict(randomForest.model, test_data, type = "class"))
accuracy.test.randomForest = sum(diag(confusion.test.randomForest))/sum(confusion.test.randomForest)
recall.test.randomForest = confusion.test.randomForest[2, 2]/sum(confusion.test.randomForest[2, ])
precision.test.randomForest = confusion.test.randomForest[2, 2]/sum(confusion.test.randomForest[, 2])


randomForest_same = which(test_data$RecS2 == predict(randomForest.model, test_data, type = "class") )
randomForest_different = which(test_data$RecS2 != predict(randomForest.model, test_data, type = "class") )

intersect(randomForest_same, svm_same)

intersect(randomForest_different, svm_different)


#########################Single Hidden Layer Neural Network

library(nnet)
nnet.model <- nnet(RecS2 ~ ., train_data, size = 30, decay = .01)
summary(nnet.model)

#training set
confusion.train.nnet = table(train_data$RecS2, 
                             predict(nnet.model, train_data, type = "class"))
accuracy.train.nnet = sum(diag(confusion.train.nnet))/sum(confusion.train.nnet)
recall.train.nnet = confusion.train.nnet[2, 2]/sum(confusion.train.nnet[2, ])
precision.train.nnet = confusion.train.nnet[2, 2]/sum(confusion.train.nnet[, 2])

#test set
confusion.test.nnet = table(test_data$RecS2, 
                            predict(nnet.model, test_data, type = "class"))
accuracy.test.nnet = sum(diag(confusion.test.nnet))/sum(confusion.test.nnet)
recall.test.nnet = confusion.test.nnet[2, 2]/sum(confusion.test.nnet[2, ])
precision.test.nnet = confusion.test.nnet[2, 2]/sum(confusion.test.nnet[, 2])

nnet_same = which(test_data$RecS2 == predict(nnet.model, test_data, type = "class") )
nnet_different = which(test_data$RecS2 != predict(nnet.model, test_data, type = "class") )

intersect(nnet_same, intersect(svm_same, randomForest_same) )

intersect(nnet_different, intersect(svm_different, randomForest_different) )



#########################KNN

##load the package class
library(class)

train_data1 <- train_data[,-1]
test_data1 <- test_data[,-1]


accuracy_knn = numeric(99)
for (i in 2:100) {
    ##run knn function
    pr <- knn(train_data1, test_data1, cl=train_data$RecS2, k=i)
    
    ##create confusion matrix
    tab <- table(pr, test_data$RecS2)
    
    ##this function divides the correct predictions by total number of predictions that tell us how accurate teh model is.
    accuracy_knn[i] <- accuracy(tab)
}

max(accuracy_knn)
plot(accuracy_knn, type = "l")

?plot



#########################Summary
accuracy <- c(accuracy.train.svm, accuracy.test.svm, 
              accuracy.train.randomForest, accuracy.test.randomForest,
              accuracy.train.nnet, accuracy.test.nnet)
precision <- c(precision.train.svm, precision.test.svm, 
                         precision.train.randomForest, precision.test.randomForest,
                         precision.train.nnet, precision.test.nnet)
recall <- c(recall.train.svm, recall.test.svm, 
            recall.train.randomForest, recall.test.randomForest,
            recall.train.nnet, recall.test.nnet)

results_summary <- rbind(accuracy, precision, recall)

colnames(results_summary) <- paste(rep(c("svm", "Random Forest", "SHL-NN"), each=2), 
                                   rep(c("training set", "test set"), 3) )

results_summary <- round(results_summary, 3)
results_summary



confusion.train.svm
confusion.test.svm
confusion.train.randomForest
confusion.test.randomForest
confusion.train.nnet
confusion.test.nnet

