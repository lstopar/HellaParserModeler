library(caret);

data <- read.csv("D:/IJS/luka/hella/R/result", encoding="UTF-8", row.names = NULL);
data03 <- read.csv("D:/IJS/luka/hella/R/results03", encoding="UTF-8", row.names = NULL);
data <- data03;
data <- data.frame(data$Bin0, data$Bin1,data$Bin2,data$Bin3,data$Bin4,data$Bin5,data$Bin6,data$Bin7,data$Bin8,data$Bin9,
                   data$Bin10,data$Bin11,data$Bin12,data$Bin13,data$Bin14,data$Bin15);
scaledData <- data.frame(scale(data),data03$X429.pike);

control <- trainControl(method = "boot", number = 30);

vars1 <- c();
for (i in 0:15) {
  vars1 <- c(vars1, paste("data.Bin",toString(i),sep=""));
}
vars2 <- vars1;
for (var in vars1){
  for (var1 in vars1) {
    vars2 <- c(vars2, paste(var, var1, sep="*"));
  }
}
vars <- c();
for (var in vars2) {
  vars <- c(vars, paste("I(",var,")"));
}
formula <- paste("data03.X429.pike ~ ", paste(vars, collapse="+"));
formula <- as.formula(formula);

modelLin <- train(data03.X429.pike ~ ., data = scaledData, method = 'glm', trControl = control);

coeffs <- modelLin$finalModel$coefficients[!is.na(modelLin$finalModel$coefficients)]
coeffs <- coeffs[abs(coeffs) > 100]
coeffs <- names(coeffs);
coeffs <- coeffs[coeffs != "(Intercept)"];
coeffs1 <- c()
for (coef in coeffs) {
  coeffs1 <- c(coeffs1, gsub("`","",coef))
}

formula2 <- paste("data03.X429.pike ~ ", paste(coeffs1, collapse="+"));
formula2 <- as.formula(as.formula(formula2));

modelLin2 <- train(formula2, data = scaledData, method = 'glm');

pred <- predict(modelLin2, scaledData);
#plot(1:length(pred), data$data.X428.crne.pike, col="blue");
#points(1:length(pred),pred, col="red");
plot(1:length(pred), scaledData$data03.X429.pike, col="blue");
lines(1:length(pred), scaledData$data03.X429.pike, col="blue");
points(1:length(pred),pred, col="red");
lines(1:length(pred),pred, col="red")


#points(1:length(scaledData$data.Bin0),scaledData$data.Bin3, col = "purple")
#points(1:length(data$data.Bin0),data$data.X428.crne.pike)


