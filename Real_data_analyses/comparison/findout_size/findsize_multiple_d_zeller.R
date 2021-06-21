
rm(list = ls()); gc()
setwd('/Users/yikong/Dropbox (CSU Fullerton)/aResearch/DeepLINK/Documents_2020.07.04_1/real_data/zeller/screen/')
library(dplyr)

out = matrix(NA, 100, 5)
colnames(out) = c('d20', 'd30', 'd40', 'd50', 'd100')
selected162 = read.csv(paste0('./v162/Zeller_raw_real_selected_v162.csv'), head = T)[,-1]
out[,1] = rowSums(selected162)

dvec = c(30, 40, 50, 100)
for (v in 171:174) {
  selected = read.csv(paste0('Zeller_raw_real_selected_v', v,'.csv'), head = T)[,-1]
  out[, v-169] = rowSums(selected)
}
setwd('/Users/yikong/Dropbox (CSU Fullerton)/aResearch/DeepLINK/revision/screening_ipad_rf/findout_size')
write.csv(out, file = 'model_size_multiple_d_zeller.csv', row.names = F)
