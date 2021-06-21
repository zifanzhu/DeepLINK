
rm(list = ls()); gc()
setwd('/Users/yikong/Dropbox (CSU Fullerton)/aResearch/DeepLINK/Documents_2020.07.04_1/real_data/rna2')
library(dplyr)

dvec = c(20, 30, 40, 50, 100, 200, 300, 400, 500)
out = matrix(NA, 100, length(dvec))
colnames(out) = paste0('d', dvec)

for (v in 1:length(dvec)) {
  selected = read.csv(paste0('rna_data2_real_selected_v', v,'.csv'), head = T)[,-1]
  out[, v] = rowSums(selected)
}
setwd('/Users/yikong/Dropbox (CSU Fullerton)/aResearch/DeepLINK/revision/screening_ipad_rf/findout_size')
write.csv(out, file = 'model_size_multiple_d_rna2.csv', row.names = F)
