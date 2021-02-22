rm(list = ls()); gc()
dat1 = read.csv('microbiome.csv', head = T, stringsAsFactors = F)
# dat1$label = as.factor(dat1$label)

nrep = 100
pe = rep(NA, nrep)
n = nrow(dat1); n.train = round(n*.5)
p = ncol(dat1) - 1
indmat.dist = matrix(NA, nrep, n.train)
top200 = matrix(NA, 200, nrep)

library("energy")
dist.m = function(mat) {
  nr = nrow(mat)
  smat <- apply(mat, 1, crossprod)
  mat1 <- matrix(smat, nrow=nr, ncol=nr)
  mat3 <- tcrossprod(mat)
  mat4 <- mat1 + t(mat1) - 2*mat3
  diag(mat4) <- 0
  mat5 <- sqrt(mat4)
  return(mat5)
}
myTest = function(XX, YY) {
  # browser()
  XX = as.matrix(XX)
  n = nrow(XX)
  M1 = dist.m(XX) ## or one can use M1 =as.matrix(dist(XX))
  M2 = dist.m(YY) ## M2 = as.matrix(dist(YY))
  DC =dcovU_stats( M1 , M2 )
  v = n*(n-3)/2
  T_R = sqrt(v-1) * DC[2]/(sqrt( 1- DC[2]^2 ) ) 
  1 - pt(T_R, v-1)
}

ind.y = which(colnames(dat1) == 'label')
for (ii in 1:nrep) {
  print(ii)
  set.seed(ii)
  ind.train = sample(1:n, n.train)
  ind.test = setdiff(1:n, ind.train)
  indmat.dist[ii,] = ind.train
  
  pvec = apply(dat1[ind.train,-ind.y], 2, myTest, YY = as.matrix(dat1[ind.train,ind.y]))
  top200[,ii] = sort(pvec, index.return=T)$ix[1:200]
}

write.csv(indmat.dist, file = 'indmat_dist_p50.csv', row.names = F)
write.csv(top200, file = 'top200_p50.csv', row.names = F)
