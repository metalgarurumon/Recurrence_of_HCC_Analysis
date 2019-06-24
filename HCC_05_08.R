########################
# load the data        #
########################

# Check if required packages are installed. If not, install them.

packages <- c("tidyverse", "caret", "nnet", "corrplot", "GGally", "plotROC")
if (length(setdiff(packages, rownames(installed.packages() ) ) ) > 0)   {
    install.packages(setdiff(packages, rownames(installed.packages() ) ) )  
}

lapply(packages, library, character.only = TRUE)


setwd("E:/Recurrence of HCC")

HCC = read.csv("HCCR.csv", stringsAsFactors = F)
HCC = HCC[, -1]  #remove ID column
summary(HCC)
dim(HCC)
colnames(HCC)

HCC = HCC %>%
    as.tbl() %>% 
    filter(RecS6F!=6) %>% 
    mutate(RecS1 = ifelse(RecS6F %in% c(1,2,3), 1, 0)) %>% 
    mutate(RecS1 = as.factor(RecS1) ) %>% 
    print

table(HCC$RecS1)
table(HCC$RecS2)


########################
# Descriptive Analysis #
########################

continuous_vars = c("Age", "AFP", "HBVDNA", "WBC", "PLT", 
                    "ALB", "TBIL", "GGT", "ALP", "TD", "TN")
discrete_vars = c("Gender", "ASJRPD", "Etiology", "RM", "RT", 
                  "OBL", "MVI", "EG", "TC", "SN", "LC")
log_vars = c("PLT", "WBC", "HBVDNA", "GGT", "ALP", "TBIL")

HCC_continuous = HCC[, continuous_vars]
head(HCC_continuous)
dim(HCC_continuous)
HCC_discrete = HCC[, discrete_vars]
head(HCC_discrete)
dim(HCC_discrete)
HCC_logvars = HCC[, log_vars]
head(HCC_logvars)
dim(HCC_logvars)

#ggplot(gather(HCC_logvars, cols, value), aes(x = value)) + 
#   geom_histogram(binwidth = 20) + facet_grid(. ~ cols) +
#   ggtitle("Histogram of Several Variables")


# histogram before transformation 
par(mfrow = c(2, 3))
hist(HCC$PLT, main = "ÑªÐ¡°å")
hist(HCC$WBC, main = "°×Ï¸°û")
hist(HCC$HBVDNA, main = "ÒÒ¸Î²¡¶¾DNA¶¨Á¿")
hist(HCC$GGT, main = "¹È°±õ£×ªëÄÃ¸")
hist(HCC$ALP, main = "¼îÐÔÁ×ËáÃ¸")
hist(HCC$TBIL, main = "×Üµ¨ºìËØ")
hist(HCC$Age, main = "ÄêÁä")
hist(HCC$ALB, main = "°×µ°°×")
hist(HCC$TN, main = "Ö×ÁöÊýÁ¿")
hist(HCC$AFP, main = "¼×Ì¥µ°°×")
hist(HCC$TD, main = "Ö×ÁöÖ±¾¶")
par(mfrow = c(1, 1))


# barplot of all categorical variables
 ggplot(gather(HCC_discrete, cols, value), aes(x = value)) + 
   geom_bar(stat = "bin") + facet_grid(. ~ cols) 

apply(HCC[, discrete_vars], 2, table)


########################
# Data Manipulation    #
########################

# transform all categorical variables
HCC[, discrete_vars] <- lapply(HCC[, discrete_vars], factor)
str(HCC)
summary(HCC)

# log transformation to the following continous variables
# remain as it is: Age, ALB, TD, TN
# log£ºPLT, WBC
# log10: HBVDNA, GGT, ALP, TBIL
log_vars = c("PLT", "WBC")
log10_vars = c("HBVDNA", "GGT", "ALP", "TBIL")

HCC[, log_vars] <- lapply(HCC[, log_vars], log)
HCC[, log10_vars] <- lapply(HCC[, log10_vars], log10)


# histogram after transformation
par(mfrow = c(2,3))
hist(HCC$PLT, main = "ÑªÐ¡°å")
hist(HCC$WBC, main = "°×Ï¸°û")
hist(HCC$HBVDNA, main = "ÒÒ¸Î²¡¶¾DNA¶¨Á¿")
hist(HCC$GGT, main = "¹È°±õ£×ªëÄÃ¸")
hist(HCC$ALP, main = "¼îÐÔÁ×ËáÃ¸")
hist(HCC$TBIL, main = "×Üµ¨ºìËØ")
par(mfrow = c(1,1))

# distribution of dependent variable
table(HCC$RecS2)

library(AMR)
HCC %>% freq(RecS2, row.names = FALSE)

HCC = HCC[HCC$RecS2 %in% c(0,1), ]
HCC = select(HCC, -RecS6F, -RecS2)


HCC %>% freq(RecS1, row.names = FALSE)

# Split the data into training and test set
set.seed(1)
training_samples <- HCC$RecS1 %>% 
    createDataPartition(p = 0.75, list = FALSE)
train_data <- HCC[training_samples, ]
test_data <- HCC[-training_samples, ]

dim(train_data)
dim(test_data)
table(train_data$RecS1)
table(test_data$RecS1)

library(AMR)
train_data %>% freq(RecS1, row.names = FALSE)

discrete_vars = c("Gender", "ASJRPD", "Etiology", "RM", "RT", 
                  "OBL", "MVI", "EG", "TC", "SN", "LC")
HCC %>% freq(ASJRPD, row.names = FALSE)

str(train_data)


##################################
##Rank Features By Importance
##################################

# load the library
library(mlbench)
library(caret)

# ensure results are repeatable
set.seed(1)
# prepare training scheme
control <- trainControl(method = "repeatedcv", number = 10, repeats = 3, classProbs = T)
?trainControl

all_methods <- names(getModelInfo())


full_model_train = function(traindata, testdata, method, control) {
    # train the model by using the selected method
    model <- train(RecS2 ~ ., data = train_data, method = method, trControl = control)
    # estimate variable importance
    importance <- varImp(model, scale = TRUE)
    # summarize importance
    print(importance)
    # plot importance
    plot(importance, main = "Feature Importance")
    # performance 
    model_pred = predict(model, test_data[, -23])
    confusionMatrix(model_glm_pred, test_data[, 23] )
}



#1. GLM
# ensure results are repeatable
set.seed(1)
model_glm <- train(RecS2 ~ ., data = train_data, 
                   method = "glm", 
                   trControl = control,
                   metric = "ROC"
                   )


# estimate variable importance
importance_glm <- varImp(model_glm, scale = TRUE)
# summarize importance
print(importance_glm)
# plot importance
plot(importance_glm, main = "Feature Importance in Logistic Regression")

# performance 
model_glm_pred = predict(model_glm, test_data[, -23])   #prob = 0.5 as threshold
confusionMatrix(data = model_glm_pred, reference = test_data[, 23], 
                positive = c("1") )

#getMethod("confusionMatrix")
confusionMatrix.train

df <- data.frame(predictions = model_glm_pred[, 2], labels = test_data[, 23])
head(df)

library(plotROC)
rocplot <- ggplot(df, aes(m = predictions, d = labels)) + 
    geom_roc(n.cuts = 30, labels = FALSE)

library(ROCR)
library(caTools)

model_glm_pred = predict(model_glm, test_data[, -23], type = "prob")
a = VariableThresholdMetrics(model_glm_pred[, 2], test_data[, 23])
names(a)
head(a$roc.df)

plot(a$threshold.df[,3], a$threshold.df[,4], type = "l",
     xlab = "recall", ylab = "precision")

df <- data.frame(predictions = model_glm_pred[, 2], labels = test_data[, 23])
head(df)
df$predictions
pred <- prediction(df$predictions, df$labels)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)


library(precrec)
precrec_obj <- evalmod(scores = df$predictions, labels = df$labels)
autoplot(precrec_obj)
precrec_obj2 <- evalmod(scores = df$predictions, labels = df$labels, mode="basic")
autoplot(precrec_obj2)  


library(ROCit)
## Warning: package 'ROCit' was built under R version 3.5.2
ROCit_obj <- rocit(score=df$predictions, class=df$labels)
plot(ROCit_obj)



#2. Random Forest
# ensure results are repeatable
set.seed(1)
model_rf <- train(RecS1 ~ ., data = train_data, method = "rf", trControl = control)

# estimate variable importance
importance_rf <- varImp(model_rf, scale = TRUE)
# summarize importance
print(importance_rf)
# plot importance
plot(importance_rf, main = "Feature Importance in Random Forest")

# performance 
model_rf_pred = predict(model_rf, test_data[, -23] )
confusionMatrix(model_rf_pred, test_data[, 23] )


#3. SVM
# ensure results are repeatable
set.seed(1)
model_svm <- train(RecS2 ~ ., data = train_data, method = "svmLinear", trControl = control)
?train


# estimate variable importance
importance_svm <- varImp(model_svm, scale = TRUE)
# summarize importance
print(importance_svm)
# plot importance
plot(importance_svm, main = "Feature Importance in SVM")

# performance 
model_svm_pred = predict(model_svm, test_data[, -23] )
confusionMatrix(model_svm_pred, test_data[, 23] )


#4. XgBoost
# ensure results are repeatable
set.seed(1)
model_XgBoost <- train(RecS2 ~ ., data = train_data, method = "xgbDART", trControl = control)

# estimate variable importance
importance_XgBoost <- varImp(model_XgBoost, scale = TRUE)
# summarize importance
print(importance_XgBoost)
# plot importance
plot(importance_XgBoost, main = "Feature Importance in XgBoost")

# performance 
model_XgBoost_pred = predict(model_XgBoost, test_data[, -23] )
confusionMatrix(model_XgBoost_pred, test_data[, 23] )


#5. Adaboost
# ensure results are repeatable
set.seed(1)
model_Adaboost <- train(RecS2 ~ ., data = train_data, method = "adaboost", trControl = control)

# estimate variable importance
importance_Adaboost <- varImp(model_Adaboost, scale = TRUE)
# summarize importance
print(importance_Adaboost)
# plot importance
plot(importance_Adaboost, main = "Feature Importance in Adaboost")

# performance 
model_Adaboost_pred = predict(model_Adaboost, test_data[, -23] )
confusionMatrix(model_Adaboost_pred, test_data[, 23] )



#########
##Bagging
#########
#6-1. Bagged CART
# ensure results are repeatable
set.seed(1)
model_Bagged_CART <- train(RecS2 ~ ., data = train_data, method = "treebag", trControl = control)

# estimate variable importance
importance_Bagged_CART <- varImp(model_Bagged_CART, scale = TRUE)
# summarize importance
print(importance_Bagged_CART)
# plot importance
plot(importance_Bagged_CART, main = "Feature Importance in Bagged CART")

# performance 
model_Bagged_CART_pred = predict(model_Bagged_CART, test_data[, -23] )
confusionMatrix(model_Bagged_CART_pred, test_data[, 23] )
?confusionMatrix

#6-2. Bagged Adaboost
# ensure results are repeatable
set.seed(1)
model_Bagged_Adab <- train(RecS2 ~ ., data = train_data, method = "AdaBag", trControl = control)

# estimate variable importance
importance_Bagged_Adab <- varImp(model_Bagged_Adab, scale = TRUE)
# summarize importance
print(importance_Bagged_Adab)
# plot importance
plot(importance_Bagged_Adab, main = "Feature Importance in Bagged Adaboost")

# performance 
model_Bagged_Adab_pred = predict(model_Bagged_Adab, test_data[, -23] )
confusionMatrix(model_Bagged_Adab_pred, test_data[, 23] )



###############
# Feature Selection
###############

#1. Univariate Filters (20 variables were selected£©

#1-1. Random Forest
filterCtrl_rf <- sbfControl(functions = rfSBF, method = "repeatedcv", number = 10, repeats = 3)
set.seed(1)
rfWithFilter_rf <- sbf(HCC[, -23], HCC[, 23], sbfControl = filterCtrl_rf)
rfWithFilter_rf


#1-2. Bagged Adaboost
filterCtrl_treebag <- sbfControl(functions = treebagSBF, method = "repeatedcv", repeats = 3)
set.seed(1)
rfWithFilter_treebag <- sbf(HCC[, -23], HCC[, 23], sbfControl = filterCtrl_treebag)
rfWithFilter_treebag


#1-3. Linear Discriminant Analysis
filterCtrl_lda <- sbfControl(functions = ldaSBF, method = "repeatedcv", repeats = 3)
set.seed(1)
rfWithFilter_lda <- sbf(HCC[, -23], HCC[, 23], sbfControl = filterCtrl_lda)
rfWithFilter_lda



#2. RFE(Recursive Feature Elimination)

# load the library
library(mlbench)
library(caret)


#2-1. Random Forest
# ensure the results are repeatable
set.seed(1)

# define the control using a random forest selection function
control_rf <- rfeControl(functions = rfFuncs, method = "repeatedcv", number = 10, repeats = 3)

# run the RFE algorithm
results_rf <- rfe(RecS1 ~ ., data = HCC, 
                  sizes = 1:22, rfeControl = control_rf)

# summarize the results
print(results_rf)

# list the chosen features
predictors(results_rf)

# plot the results
plot(results_rf, type=c("g", "o"), main = "RFE by Random Forest")
results_rf$optVariables



#2-2. Bagged Adaboost
# ensure the results are repeatable
set.seed(1)

# define the control using a random forest selection function
control_treebag <- rfeControl(functions = treebagFuncs, method = "repeatedcv", number = 10, repeats = 3)

# run the RFE algorithm
results_treebag <- rfe(RecS1 ~ ., data = train_data, 
                  sizes = 1:22, rfeControl = control_treebag)

# summarize the results
print(results_treebag)

# list the chosen features
predictors(results_treebag)

# plot the results
plot(results_treebag, type=c("g", "o"), main = "RFE by Bagged Adaboost")
results_treebag$optsize
results_treebag$optVariables



#2-3. Naive Bayes
# ensure the results are repeatable
set.seed(1)

# define the control using a random forest selection function
control_nb <- rfeControl(functions = nbFuncs, method = "repeatedcv", number = 10, repeats = 3)

# run the RFE algorithm
results_nb <- rfe(RecS1 ~ ., data = train_data, 
                  sizes = 1:22, rfeControl = control_nb)

# summarize the results
print(results_nb)

# list the chosen features
predictors(results_nb)

# plot the results
plot(results_nb, type=c("g", "o"), main = "RFE by Naive Bayes")
results_nb$optsize
results_nb$optVariables



# 3. Genetic Algorithm

# 3-1. Random Forest
# ensure the results are repeatable
set.seed(1)
ga_ctrl <- gafsControl(functions = rfGA, 
                       method = "cv", number = 5)

rf_ga <- gafs(x = train_data[, -23], y = train_data[, 23],
              iters = 10, gafsControl = ga_ctrl)

plot(rf_ga) + theme_bw()


# 3-2. Bagged Adaboost
# ensure the results are repeatable
set.seed(1)
ga_ctrl <- gafsControl(functions = treebagGA, 
                       method = "cv", number = 5)

treebagGA_ga <- gafs(x = train_data[, -23], y = train_data[, 23],
                     iters = 5, gafsControl = ga_ctrl)

plot(treebagGA_ga) + theme_bw()


