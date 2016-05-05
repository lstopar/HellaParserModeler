library(caret);


#data03 <- read.csv("./results03", encoding="UTF-8", row.names = NULL, sep=",");
data <- read.csv("./results0001", encoding="UTF-8", row.names = NULL, sep=",");

trainIndex <- createDataPartition(data$X9.enojni.zarek, p=0.75, list=FALSE);
trainData <- data[trainIndex,];
testData <- data[-trainIndex,];

control <- trainControl(method = "boot", number = 30);

modelLin <- train(trainData$X9.enojni.zarek ~ .,data = trainData, trControl = control, method = 'glm')

coeffs <- modelLin$finalModel$coefficients[!is.na(modelLin$finalModel$coefficients)]
coeffs1 <- coeffs[abs(coeffs) > 1]
coeffs2 <- names(coeffs1);
coeffs2 <- coeffs2[coeffs2 != "(Intercept)"];
coeffs2 <- coeffs2[sub("X.*","VN",coeffs2) != "VN"];

formula <- as.formula(paste(c("trainData$X9.enojni.zarek ~", coeffs2),collapse = "+"))
# modelKnn <- train(formula, data=trainData, method ='knn', trControl = control)
modelLin1 <- train(formula, data=trainData, method ='glm', trControl = control)

pred <- predict(modelLin1, testData);
#plot(1:length(pred), data$data.X428.crne.pike, col="blue");
#points(1:length(pred),pred, col="red");
plot(1:length(pred),pred, col="red",
     ylim=c(min(pred,testData$X9.enojni.zarek),max(pred,testData$X9.enojni.zarek)));
lines(1:length(pred),pred, col="red")
points(1:length(pred), testData$X9.enojni.zarek, col="blue");
lines(1:length(pred), testData$X9.enojni.zarek, col="blue");
