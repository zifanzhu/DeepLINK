
rm(list = ls()); gc()
setwd('/Users/yikong/Dropbox (CSU Fullerton)/aResearch/DeepLINK/revision/common_feature/code1')
library(dplyr)

mean.se = function(x, ns=3) {
  m = round(mean(x), ns); s = round(sd(x)/sqrt(length(x)), ns)
  paste0(format(m, nsmall=ns), ' (', format(s, nsmall=ns), ')')
}

dvec = c(20, 30, 40, 50, 100, 200)
out = as.data.frame(matrix(NA, 6, 2))
colnames(out) = c('trainingPE', 'testPE')
rownames(out) = paste0('d', dvec)

# load('variable_names.RData')
dat1 = read.csv('../microbiome_data1_common.csv', head = T, stringsAsFactors = F)

for (v in 1:6) {
  print(paste0('== v', v, ' =='))
  pe.train = read.csv(paste0("microbiome_data1_common_real_pe_train_v", v, ".csv"), head = T)
  pe.test = read.csv(paste0("microbiome_data1_common_real_pe_test_v", v, ".csv"), head = T)
  
  out[v,] = c(mean.se(pe.train$X0), mean.se(pe.test$X0))
  
  top250 = unlist(read.csv('screen_top250.csv', head=T))
  d = dvec[v]
  
  selected = read.csv(paste0('microbiome_data1_common_real_selected_v', v,'.csv'), head = T)[,-1]
  var.sel = vector('list', length=100)
  for (ii in 1:100) {
    var.sel[[ii]] = top250[1:d][which(selected[ii,] == 1)]
  }
  
  d1 = 20
  tb = sort(table(unlist(var.sel)), decreasing = T)
  ind1 = tb[1:d1] %>% names() %>% as.numeric()
  
  top.selected = cbind.data.frame(variable = colnames(dat1)[ind1], frequence = tb[1:d1] %>% as.numeric)
  write.csv(top.selected, file = paste0('top_selected_d', d,'.csv'), row.names = F)
}

write.csv(t(out), file = 'error_rate.csv')
