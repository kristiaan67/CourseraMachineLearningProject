library(dplyr)
library(caret)

set.seed(20210419)

if (!file.exists("./data/pml-training.csv")) {
    download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", 
                  "./data/pml-training.csv", method = "curl");
}
if (!file.exists("./data/pml-testing.csv")) {
    download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", 
                  "./data/pml-testing.csv", method = "curl");
}

# Data Preparation

pmlTraining <- read.csv("./data/pml-training.csv")

# Class A = correct
# other classes are wrongly executed exercise
pmlTraining <- mutate(pmlTraining, classe = as.factor(classe)) %>%
    select(classe, num_window,
           contains("roll_belt") | contains("pitch_belt") | contains("picth_belt") | contains("yaw_belt") | 
               contains("accel_belt") | contains("gyros_belt") | contains("magnet_belt") |
               contains("roll_arm") | contains("pitch_arm") | contains("picth_arm") | contains("yaw_arm") |
               contains("accel_arm") | contains("gyros_arm") | contains("magnet_arm") |
               contains("roll_dumbbell") | contains("pitch_dumbbell") | contains("picth_dumbbell") | contains("yaw_dumbbell") |
               contains("accel_dumbbell") | contains("gyros_dumbbell") | contains("magnet_dumbbell") |
               contains("roll_forearm") | contains("pitch_forearm") | contains("picth_forearm") | contains("yaw_forearm") |
               contains("accel_forearm") | contains("gyros_forearm") | contains("magnet_forearm"))

# remove columns that contain near zero variance predictors
pmlTraining <- select(pmlTraining, -nearZeroVar(pmlTraining, names = TRUE))
# remove columns that contain more than 30% NA values
pmlTraining <- select_if(pmlTraining, colSums(is.na(pmlTraining)) <= nrow(pmlTraining) * .30)

inTrain  <- createDataPartition(pmlTraining$classe, p = 0.7, list = FALSE)
pmlTraining_train <- pmlTraining[inTrain,]
pmlTraining_validate  <- pmlTraining[-inTrain,]

# cross validation
trControl <- trainControl(method = "cv", number = 5)

# 1. Classification Trees
fitCT <- train(classe ~ ., method = "rpart", data = pmlTraining_train, 
               trControl = trControl)
predCT <- predict(fitCT, newdata = pmlTraining_validate)
print("1. Classification Trees")
print(confusionMatrix(predCT, pmlTraining_validate$classe))

# 2. Random Forest
fitRF <- train(classe ~ ., method = "rf", data = pmlTraining_train, 
               trControl = trControl)
predRF <- predict(fitRF, newdata = pmlTraining_validate)
print("2. Random Forest")
print(confusionMatrix(predRF, pmlTraining_validate$classe))

# 3. Generated Boosted Model
fitGBM <- train(classe ~ ., method = "gbm", data = pmlTraining_train, 
                trControl = trControl, verbose = FALSE)
predGBM <- predict(fitGBM, newdata = pmlTraining_validate)
print("3. Generated Boosted Model")
print(confusionMatrix(predGBM, pmlTraining_validate$classe))

# 4. Model based Prediction: Linear Discriminant Analysis
fitLDA <- train(classe ~ ., method = "lda", data = pmlTraining_train, 
                trControl = trControl)
predLDA <- predict(fitLDA, newdata = pmlTraining_validate)
print("4.1 Linear Discriminant Analysis")
print(confusionMatrix(predLDA, pmlTraining_validate$classe))


bestFit <- fitRF
pmlTesting <- read.csv("./data/pml-testing.csv")
predTesting <- predict(bestFit, newdata = pmlTesting)
for (i in 1:length(predTesting)) print((paste(i, ": ", predTesting[i])))



