# Simple R code to find the correlations in Hella's scrap by Joao Pita Costa 2016

Hella <- read.csv("~/Dropbox/joao (1)/Hella_joao.csv", na.strings="0")
View(Hella)
Hella[is.na(Hella)]<-0
Hellacorrmat <- cor(Hella[,c(5:142)])
Hellacorrmat[is.na(Hellacorrmat)]<-0
HellaCorrelate <- Hellacorrmat[c(23:138),c(2:22)]
heatmap(HellaCorrelate)

png(file="HellaCorrelate.png")
heatmap(HellaCorrelate)
dev.off()
write.table(HellaCorrelate, file="HellaCorrelate.txt", row.names=FALSE, col.names=FALSE)
