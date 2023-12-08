source('Niletools.R')

niledata = read.csv('nile.csv')

#normalizes the data
lfnorm = (niledata[[2]]-median(niledata[[2]]))/sd(niledata[[2]])

#uses the Lanzante 1996 method to find change points in the data
cps = Lanzante(lfnorm,niledata[[1]])

changeyrs=c()
for(i in 1:cps[[1]]){
    changeyrs[[i]] = cps[[i*3-1]]
}

#plots changepoints over the record as a pdf
pdf('NileLowFlow.pdf', width=10,height=5)
plotannual(lfnorm,niledata[[1]],changeyrs,"Nile Low Flow")
dev.off()



