library(randomForest);
library(caret);
library(ISLR);

library(doParallel);
cl <- makeCluster(2);#detectCores());
registerDoParallel(cl);

data <- read.csv("D:/IJS/luka/hella/R/result", encoding="UTF-8", row.names = NULL);
data01 <- read.csv("D:/IJS/luka/hella/R/results01", encoding="UTF-8", row.names = NULL);
data001 <- read.csv("D:/IJS/luka/hella/R/results001", encoding="UTF-8", row.names = NULL);
data0001 <- read.csv("D:/IJS/luka/hella/R/results0001", encoding="UTF-8", row.names = NULL);
data03 <- read.csv("D:/IJS/luka/hella/R/results03", encoding="UTF-8", row.names = NULL);
data <- data.frame(data$Bin0, data$Bin1,data$Bin2,data$Bin3,data$Bin4,data$Bin5,data$Bin6,data$Bin7,data$Bin8,data$Bin9,
                   data$Bin10,data$Bin11,data$Bin12,data$Bin13,data$Bin14,data$Bin15,data$X428.crne.pike);
scaledData <- data.frame(scale(data));

trainIndex <- createDataPartition(data$X429.pike, p=0.75, list=FALSE);
trainData <- Carseats[trainIndex,];
testData <- Carseats[-trainIndex,];

# izmerjene vrednosti za lazje racunanje R2
observed <- testData$X429.pike;

# Model vseh spremenljivk:
modelLin = train(data.X428.crne.pike ~ data.Bin11 + data.Bin12 + data.Bin13 + data.Bin0 + data.Bin14 + data.Bin5 + data.Bin3 + data.Bin7 + data.Bin9 + data.Bin4 + data.Bin2 + data.Bin8 + data.Bin15 + data.Bin1 + data.Bin6,
                   data = scaledData, method = 'glm');
modelLin01 = train(X429.pike ~ Bin11 + Bin12 + Bin13 + Bin0 + Bin14 + Bin5 + Bin3 + Bin7 + Bin9 + Bin4 + Bin2 + Bin8 + Bin15 + Bin1 + Bin6,
                   data = data01, method = 'glm');
modelLin001 = train(X429.pike ~ Bin11 + Bin12 + Bin13 + Bin0 + Bin14 + Bin5 + Bin3 + Bin7 + Bin9 + Bin4 + Bin2 + Bin8 + Bin15 + Bin1 + Bin6,
                   data = data001, method = 'glm');
modelLin0001 = train(X429.pike ~ Bin11 + Bin12 + Bin13 + Bin0 + Bin14 + Bin5 + Bin3 + Bin7 + Bin9 + Bin4 + Bin2 + Bin8 + Bin15 + Bin1 + Bin6,
                    data = data0001, method = 'glm');

model01 = train(X429.pike ~ Bin11 + Bin12 + Bin13 + Bin0 + Bin14 + Bin5 + Bin3 + Bin7 + Bin9 + Bin4 + Bin2 + Bin8 + Bin15 + Bin1 + Bin6,
              data = data01, method='rf');
modelLin03 = train(X428.crne.pike ~ Bin11 + Bin12 + Bin13 + Bin0 + Bin14 + Bin5 + Bin3 + Bin7 + Bin9 + Bin4 + Bin2 + Bin8 + Bin15 + Bin1 + Bin6,
                   data = data03, method = 'glm');

stopCluster(cl);

pred <- predict(modelLin$finalModel, data);
plot(1:length(pred), data$data.X428.crne.pike);
points(1:length(pred),pred, col="red");

