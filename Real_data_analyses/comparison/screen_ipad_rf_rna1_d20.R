
rm(list = ls()); gc()
# setwd('/Users/yikong/Dropbox (CSU Fullerton)/aResearch/DeepLINK/revision/screening_ipad_rf/')
setwd('/home/yinfeiko/DeepLINK/revision')
dat1 = read.csv('rna_data1.csv', head = T, stringsAsFactors = F)
source('hdknockoff_all_Factor.R')
library(glmnet); library(RSpectra); library(randomForest)
model.size = read.csv("model_size_multiple_d_rna1.csv", head = T, stringsAsFactors = F)

nrep = 100
pe.rf1 = pe.rf2 = pe.rf3 = pe.ipad = rep(NA, nrep)
n = nrow(dat1)
n.screen = round(n*.5); n.train = round(n*.4); n.test = round(n*.1)
p = ncol(dat1) - 1
indmat.dist = matrix(NA, nrep, n.screen)
top500 = matrix(NA, 500, nrep)

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
  XX = as.matrix(XX)
  n = nrow(XX)
  M1 = dist.m(XX) ## or one can use M1 =as.matrix(dist(XX))
  M2 = dist.m(YY) ## M2 = as.matrix(dist(YY))
  DC =dcovU_stats( M1 , M2 )
  v = n*(n-3)/2
  # T_R = sqrt(v-1) * DC[2]/(sqrt( 1- DC[2]^2 ) )
  # 1 - pt(T_R, v-1)
  sqrt(v-1) * DC[2]/(sqrt( 1- DC[2]^2 ) )
}

dvec = c(20,30,40,50,100,200,300,400,500)
d = 20
k = which(dvec == d)
size1 = model.size[,k]

print(paste0('d = ', d))
decrease.acc = decrease.gini = matrix(NA, d, nrep)
top100.acc = top100.gini = matrix(NA, 100, nrep)
var.ipad = vector('list', length = nrep)

ind.y = which(colnames(dat1) == 'real_y')
for (ii in 1:nrep) {
  print(ii)
  set.seed(ii)
  ind.screen = sample(1:n, n.screen)
  ind1 = setdiff(1:n, ind.screen)
  ind.train = sample(ind1, n.train)
  ind.test = setdiff(ind1, ind.train)
  indmat.dist[ii,] = ind.screen
  
  pvec = apply(dat1[ind.screen,-ind.y], 2, myTest, YY = as.matrix(dat1[ind.screen,ind.y]))
  top500[,ii] = sort(pvec, decreasing = T, index.return=T)$ix[1:500]
  
  ## 1. Randome Forest - refit RF with matched model size ##
  indvar1 = top500[1:d,ii]
  model.rf0 = randomForest(as.factor(real_y) ~ ., data = dat1[ind.train, c(ind.y, indvar1)], importance = T)
  ip = importance(model.rf0, type = 2)
  ind = sort(importance(model.rf0)[,4], index.return=T, decreasing=T)$ix[1:size1[ii]]
  
  # refit RF with model size of DeepLINK #
  vars = colnames(dat1[ind.train, c(ind.y, indvar1)])[ind+1]
  fm = paste0('as.factor(real_y) ~ ', paste(vars, collapse = ' + '))
  model.rf1 = randomForest(as.formula(fm), data = dat1[ind.train, c(ind.y, indvar1)], importance = T)
  
  pred.model.rf = predict(model.rf1, newdata = dat1[ind.test,]) #prediction
  pe.rf1[ii] = mean(pred.model.rf != dat1[ind.test,'real_y'])
  
  ## 2. Random Forest with top model size variables from screening ##
  indvar2 = top500[1:size1[ii],ii]
  model.rf2 = randomForest(as.factor(real_y) ~ ., data = dat1[ind.train, c(ind.y, indvar2)], importance = T)
  pred.model.rf2 = predict(model.rf2, newdata = dat1[ind.test,]) #prediction
  pe.rf2[ii] = mean(pred.model.rf2 != dat1[ind.test,'real_y'])
  
  ## 3. Random Forest ##
  pred.model.rf3 = predict(model.rf0, newdata = dat1[ind.test,]) #prediction
  pe.rf3[ii] = mean(pred.model.rf3 != dat1[ind.test,'real_y'])
  
  ## 4. IPAD - logit L1 ##
  r = 3 # number of factors chosen by user
  Xnew = hdknockoffX(as.matrix(dat1[ind.train, indvar1]), as.factor(dat1[,ind.y]), r, method = 'facrand')
  
  # Supp_plus contains the selected variables by IPAD
  Supp_plus = hdknockoff_plus(Xnew, as.factor(dat1[ind.train,ind.y]), 0.2, lambda = 'CV')
  var.ipad[[ii]] = Supp_plus
  
  if (length(Supp_plus) > 0) {
    obj.net = cv.glmnet(as.matrix(dat1[ind.train,names(Supp_plus)]), dat1[ind.train, 'real_y'], family='binomial', alpha = 1) 
    prob4 = predict(obj.net, newx=as.matrix(dat1[ind.test, names(Supp_plus)]), s="lambda.min", type='response')
    pred.model.ipad = as.numeric(prob4 > 0.5)
    pe.ipad[ii] = mean(pred.model.ipad != dat1[ind.test,'real_y'])
  } else {
    pred.model.ipad = rep(as.numeric(mean(dat1$real_y[ind.train]) > 0.5), length(ind.test))
    pe.ipad[ii] = mean(pred.model.ipad != dat1[ind.test,'real_y'])
  }
  #########################################################
}

mean.se = function(x, ns = 3) {
  m = format(round(mean(x), ns), nsmall = ns)
  se1 = format(round(sd(x)/sqrt(length(x)), ns), nsmall = ns)
  paste0(m, ' (', se1, ')')
}
pe = c(mean.se(pe.rf1), mean.se(pe.rf2), mean.se(pe.rf3), mean.se(pe.ipad))
names(pe) = c('RF-RF matched', 'RF matched', 'RF', 'IPAD-L1')
print(pe)

save.image(paste0('results_rna1_d', d, '.RData'))

