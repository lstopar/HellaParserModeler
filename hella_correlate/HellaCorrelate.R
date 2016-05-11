# Simple R code to find the correlations in Hella's scrap by Joao Pita Costa 2016

Hella <- read.csv("Hella-molding-V2.csv", na.strings="0")
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
