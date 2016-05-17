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
HellaCorrelate <- Hellacorrmat[c(46:138),c(2:60)]
View(HellaCorrelate)
heatmap(HellaCorrelate)
HellaCorrelate=HellaCorrelate[apply(HellaCorrelate[,-1], 1, function(x) !all(x==0)),]
heatmap(HellaCorrelate)

png(file="HellaCorrelate.png")
heatmap(HellaCorrelate)
dev.off()

png(file="HellaCorrelate2.png",width=800,height=650)
heatmap.2(HellaCorrelate)
dev.off()

write.table(HellaCorrelate, file="HellaCorrelate.txt", row.names=FALSE, col.names=FALSE)

## spearman correlation coefficient

Hellacorrsp <- cor(Hella[,c(5:142)], method = "spearman")
Hellacorrsp[is.na(Hellacorrsp)]<-0
HellaCorrelateSp <- Hellacorrsp[c(23:138),c(2:22)]
HellaCorrelateSp=HellaCorrelateSp[apply(HellaCorrelateSp[,-1], 1, function(x) !all(x==0)),]
heatmap(HellaCorrelateSp)
png(file="HellaCorrelateSp.png")
heatmap(HellaCorrelateSp)
dev.off()
write.table(HellaCorrelateSp, file="HellaCorrelateSp.txt", row.names=FALSE, col.names=FALSE)

## pearson correlation coefficient

Hellacorrpe <- cor(Hella[,c(5:142)], method = "pearson")
Hellacorrpe[is.na(Hellacorrpe)]<-0
HellaCorrelatePe <- Hellacorrpe[c(23:138),c(2:22)]
HellaCorrelatePe=HellaCorrelatePe[apply(HellaCorrelatePe[,-1], 1, function(x) !all(x==0)),]
heatmap(HellaCorrelatePe)
png(file="HellaCorrelatePe.png")
heatmap(HellaCorrelatePe)
dev.off()
write.table(HellaCorrelatePe, file="HellaCorrelatePe.txt", row.names=FALSE, col.names=FALSE)


## Covariance

# The covariance of two variables x and y in a data sample measures how the two are 
# linearly related. A positive covariance would indicates a positive linear 
# relationship between the variables, and a negative covariance would indicate 
# the opposite. 

Hellacovmat <- cov(Hella[,c(5:142)])
Hellacovmat[is.na(Hellacovmat)]<-0
View(Hellacovmat)
HellaCovariance<-Hellacovmat[c(46:138),c(2:60)]
HellaCovariance=HellaCovariance[apply(HellaCovariance[,-1], 1, function(x) !all(x==0)),]
heatmap(HellaCovariance)

png(file="HellaCovariance.png")
heatmap(HellaCovariance)
dev.off()

png(file="HellaCovariance2.png",width=800,height=650)
heatmap.2(HellaCovariance)
dev.off()

write.table(HellaCovariance, file="HellaCovariance.txt", row.names=FALSE, col.names=FALSE)


## Variance

Hellavarmat <- var(Hella[,c(5:142)])
Hellavarmat[is.na(Hellavarmat)]<-0
View(Hellavarmat)
HellaVariance<-Hellavarmat[c(23:138),c(2:22)]
HellaVariance=HellaVariance[apply(HellaVariance[,-1], 1, function(x) !all(x==0)),]
heatmap(HellaVariance)
png(file="HellaVariance.png")
heatmap(HellaVariance)
dev.off()
write.table(HellaVariance, file="HellaVariance.txt", row.names=FALSE, col.names=FALSE)


# Find Max correlated values

HellaCorrelate[HellaCorrelate == 1] <- 0
image(HellaCorrelate)
hmax <- max.col(HellaCorrelate)
print(hmax)

N <- 5
M <-ncol(HellaCorrelate)
# C<-1:M
#ndx <- order(HellaCorrelate[,C], decreasing = T)[1:N]
top<-matrix(0,M,N)

C<-1
while (C < M+1) {
  top[C,]<-order(HellaCorrelate[,C], decreasing = T)[1:N]
  C<-C+1}

Hellainfo<-top
Hellainfo[,]<-rownames(HellaCorrelate)[top[,]]
rownames(Hellainfo)<-colnames(HellaCorrelate)
View(Hellainfo)
write.csv(Hellainfo, file="Hella_top5_correlations.csv")
