library(caret);


data03 <- read.csv("./results03", encoding="UTF-8", row.names = NULL, sep=",");

trainIndex <- createDataPartition(data03$X9.enojni.zarek, p=0.75, list=FALSE);
trainData <- data03[trainIndex,];
testData <- data03[-trainIndex,];

control <- trainControl(method = "boot", number = 30);

temps <- c("Temp","Temp.1","Temp.2","Temp.3","Temp.4","Temp.5","Temp.5","Temp.6",
           "Temp.7","Temp.8","Temp.9","Temp.10","Temp.11")
formula1 <- as.formula(paste(c("trainData$X180.spojna.linija ~", "Prirobn", temps), collapse = "+"))

modelWeld <- train( formula1 ,data = trainData, trControl = control, method = 'glm')

coeffs <- modelWeld$finalModel$coefficients[!is.na(modelWeld$finalModel$coefficients)]
#coeffs1 <- coeffs[abs(coeffs) > 50]
coeffs2 <- names(coeffs);
coeffs2 <- coeffs2[coeffs2 != "(Intercept)"];

formula2 <- as.formula(paste(c("trainData$X180.spojna.linija ~", coeffs2),collapse = "+"))
modelWeld1 <- train(formula2, data=trainData, method ='glm', trControl = control)


pred1 <- predict(modelWeld1, testData);
#plot(1:length(pred), data$data.X428.crne.pike, col="blue");
#points(1:length(pred),pred, col="red");
plot(1:length(pred1),pred1, col="red", 
     ylim=c(min(pred1,testData$X180.spojna.linija),max(pred1,testData$X180.spojna.linija)));
lines(1:length(pred1),pred1, col="red")
points(1:length(pred1), testData$X180.spojna.linija, col="blue");
lines(1:length(pred1), testData$X180.spojna.linija, col="blue");

