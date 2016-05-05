library(caret);

data <- read.csv("./result", encoding="UTF-8", row.names = NULL);
data03 <- read.csv("./results03", encoding="UTF-8", row.names = NULL);
data001 <- read.csv("./results001", encoding="UTF-8", row.names = NULL);
data <- data03;
data <- data.frame(data$Bin0, data$Bin1,data$Bin2,data$Bin3,data$Bin4,data$Bin5,data$Bin6,data$Bin7,data$Bin8,data$Bin9,
                   data$Bin10,data$Bin11,data$Bin12,data$Bin13,data$Bin14,data$Bin15);
scaledData <- data.frame(scale(data),data03$X429.pike);
data1 <- data001;
data1 <- data.frame(data1$Bin0, data1$Bin1,data1$Bin2,data1$Bin3,data1$Bin4,data1$Bin5,data1$Bin6,data1$Bin7,data1$Bin8,data1$Bin9,
                    data1$Bin10,data1$Bin11,data1$Bin12,data1$Bin13,data1$Bin14,data1$Bin15);

scaledData1 <- data.frame(scale(data1), data001$X429.pike);

data2 <- data001;
data2 <- data.frame((data2$Bin0+ data2$Bin1+data2$Bin2+data2$Bin3+data2$Bin4+data2$Bin5+data2$Bin6+data2$Bin7+data2$Bin8+data2$Bin9+
                    data2$Bin10+data2$Bin11+data2$Bin12+data2$Bin13+data2$Bin14+data2$Bin15));

scaledData2 <- data.frame(scale(data2), data001$X429.pike);
colnames(scaledData2) <- c("Dust", "Pike");

data3 <- data03;
data3 <- data.frame(data3$Bin2, data3$Bin3 + data3$Bin4,
                    data3$Bin5 + data3$Bin6);
scaledData3 <- data.frame(scale(data3), data03$X429.pike);
colnames(scaledData3) <- c("Bin2", "Bin34","Bin56","Pike")

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
pred <- as.integer(pred);
#plot(1:length(pred), data$data.X428.crne.pike, col="blue");
#points(1:length(pred),pred, col="red");
plot(1:length(pred), scaledData$data03.X429.pike, col="blue");
lines(1:length(pred), scaledData$data03.X429.pike, col="blue");
points(1:length(pred),pred, col="red");
lines(1:length(pred),pred, col="red")
napaka <- sum(abs(pred - scaledData$data03.X429.pike))
povpNapaka <- napaka / length(pred);

#points(1:length(scaledData$data.Bin0),scaledData$data.Bin3, col = "purple")
#points(1:length(data$data.Bin0),data$data.X428.crne.pike)

colnames(scaledData1) <- c(vars1,"data03.X429.pike");
pred001 <- predict(modelLin2, scaledData1);
pred001 <- unname(tapply(pred001, (seq_along(pred001)-1) %/% 28, sum))

pred001 <- c(pred001 / 28,13);
pred001 <- as.integer(pred001);
napaka001 <- sum(abs(pred001 - scaledData$data03.X429.pike));

plot(1:length(pred001), scaledData$data03.X429.pike, col="blue");
lines(1:length(pred001), scaledData$data03.X429.pike, col="blue");
points(1:length(pred001),pred001, col="red");
lines(1:length(pred001),pred001, col="red")


#pred001 <- sum(pred001);
#pred001 <- pred001 / 30;
#scrap = sum(scaledData$data03.X429.pike);

modelLin3 <- train(Pike ~ Dust, data = scaledData2, method='glm');

pred2 <- predict(modelLin3, scaledData2);
plot(1:length(pred2), scaledData2$Pike, col="blue");
lines(1:length(pred2), scaledData2$Pike, col="blue");
points(1:length(pred2),pred2, col="red");
lines(1:length(pred2),pred2, col="red")

modelLin4 <- train(Pike ~ Bin2 + Bin34 + Bin56,
                   data=scaledData3, method = 'glm');

pred3 <- predict(modelLin4, scaledData3);
plot(1:length(pred3), scaledData3$Pike, col="blue");
lines(1:length(pred3), scaledData3$Pike, col="blue");
points(1:length(pred3),pred3, col="red");
lines(1:length(pred3),pred3, col="red")
napaka3 <- sum(abs(pred - scaledData3$Pike))
