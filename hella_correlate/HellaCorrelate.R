# Simple R code to analyse correlation and covariance in Hella's scrap 
# by Joao Pita Costa 2016

## Correlation

## pearson (default) correlation coefficient

Hella <- read.csv("Hella-molding-V3.csv", na.strings="0")
View(Hella)
Hella[is.na(Hella)]<-0
Hellacorrmat <- cor(Hella[,c(5:142)])
Hellacorrmat[is.na(Hellacorrmat)]<-0
View(Hellacorrmat)
HellaCorrelate <- Hellacorrmat[c(23:138),c(2:22)]
View(HellaCorrelate)
heatmap(HellaCorrelate)
HellaCorrelate2=HellaCorrelate[c(1:12,90:98,105:116),]
heatmap(HellaCorrelate2)

png(file="HellaCorrelate.png")
heatmap(HellaCorrelate)
dev.off()
png(file="HellaCorrelate2.png")
heatmap(HellaCorrelate2)
dev.off()
write.table(HellaCorrelate, file="HellaCorrelate.txt", row.names=FALSE, col.names=FALSE)
write.table(HellaCorrelate2, file="HellaCorrelate2.txt", row.names=FALSE, col.names=FALSE)

## spearman correlation coefficient

Hellacorrsp <- cor(Hella[,c(5:142)], method = "spearman")
Hellacorrsp[is.na(Hellacorrsp)]<-0
HellaCorrelateSp <- Hellacorrsp[c(23:138),c(2:22)]
HellaCorrelateSp2=HellaCorrelateSp[c(1:12,90:98,105:116),]
heatmap(HellaCorrelateSp2)
png(file="HellaCorrelateSp.png")
heatmap(HellaCorrelateSp2)
dev.off()
write.table(HellaCorrelateSp2, file="HellaCorrelateSp.txt", row.names=FALSE, col.names=FALSE)

## pearson correlation coefficient

Hellacorrpe <- cor(Hella[,c(5:142)], method = "pearson")
Hellacorrpe[is.na(Hellacorrpe)]<-0
HellaCorrelatePe <- Hellacorrpe[c(23:138),c(2:22)]
HellaCorrelatePe2=HellaCorrelatePe[c(1:12,90:98,105:116),]
heatmap(HellaCorrelatePe2)
png(file="HellaCorrelatePe.png")
heatmap(HellaCorrelatePe2)
dev.off()
write.table(HellaCorrelatePe2, file="HellaCorrelatePe.txt", row.names=FALSE, col.names=FALSE)


## Covariance

# The covariance of two variables x and y in a data sample measures how the two are 
# linearly related. A positive covariance would indicates a positive linear 
# relationship between the variables, and a negative covariance would indicate 
# the opposite. 

Hellacovmat <- cov(Hella[,c(5:142)])
Hellacovmat[is.na(Hellacovmat)]<-0
View(Hellacovmat)
HellaCovariance<-Hellacovmat[c(23:138),c(2:22)]
HellaCovariance2<-HellaCovariance[c(1:12,90:98,105:116),]
heatmap(HellaCovariance2)
png(file="HellaCovariance.png")
heatmap(HellaCovariance2)
dev.off()
write.table(HellaCovariance, file="HellaCovariance.txt", row.names=FALSE, col.names=FALSE)


## Variance

Hellavarmat <- cov(Hella[,c(5:142)])
Hellavarmat[is.na(Hellavarmat)]<-0
View(Hellavarmat)
HellaVariance<-Hellavarmat[c(23:138),c(2:22)]
HellaVariance2<-HellaVariance[c(1:12,90:98,105:116),]
heatmap(HellaVariance2)
png(file="HellaVariance.png")
heatmap(HellaVariance2)
dev.off()
write.table(HellaVariance, file="HellaVariance.txt", row.names=FALSE, col.names=FALSE)

