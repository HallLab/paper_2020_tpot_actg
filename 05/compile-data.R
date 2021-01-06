library(tidyverse)

cd4_rep_scores <- read_delim('../04/p_cd4difw48w4_replication_scores.txt', delim = '\t') %>% 
  mutate(dat = 'Original data', type = 'Change in Absolute CD4 Count') %>% 
  select(-Replicate)
cd4_per_scores <- read_delim('../04/p_cd4difw48w4_permutation_scores.txt', delim = '\t') %>% 
  mutate(dat = 'Permuted data', type = 'Change in Absolute CD4 Count') %>% 
  select(-Permutation)
plasma_rep_scores <- read_delim('../04/vllogdifw48w4_replication_scores.txt', delim = '\t') %>% 
  mutate(dat = 'Original data', type = 'Change in plasma HIV RNA') %>% 
  select(-Replicate)
plasma_per_scores <- read_delim('../04/vllogdifw48w4_permutation_scores.txt', delim = '\t') %>% 
  mutate(dat = 'Permuted data', type = 'Change in plasma HIV RNA') %>% 
  select(-Permutation)


cd4_feats <- read_delim('../04/p_cd4difw48w4_feature_importances_reformatted.txt', delim = '\t') %>% 
  mutate(type = 'Change in Absolute CD4 Count')
plasma_feats <- read_delim('../04/vllogdifw48w4_feature_importances_reformatted.txt', delim = '\t') %>% 
  mutate(type = 'Change in plasma HIV RNA')


bind_rows(plasma_per_scores, plasma_rep_scores,
          cd4_per_scores, cd4_rep_scores) %>% 
  write_csv('gathered_data/all_pipes.csv')

bind_rows(cd4_feats, plasma_feats) %>% 
  write_csv('gathered_data/all_feature_imps.csv')


test_r2 <- function(rep_dat, per_dat){
  res <- broom::tidy(t.test(rep_dat$`R^2 Score`, per_dat$`R^2 Score`,
                            alternative = 'greater'))
  glue::glue('t = {round(res$statistic, 3)}\n P = {sprintf("%0.3g", res$p.value)}')
}
test_mse <- function(rep_dat, per_dat){
  res <- broom::tidy(t.test(rep_dat$`MSE Score`, per_dat$`MSE Score`,
                            alternative = 'greater'))
  glue::glue('t = {round(res$statistic, 3)}\n P = {sprintf("%0.3g", res$p.value)}')
}


ttest_label <- tribble(
  ~ label_r2, ~ label_mse, ~ type, ~ y,
  test_r2(cd4_rep_scores, cd4_per_scores), test_mse(cd4_rep_scores, cd4_per_scores), 'Change in Absolute CD4 Count', -26000,
  test_r2(plasma_rep_scores, plasma_per_scores), test_mse(plasma_rep_scores, plasma_per_scores), 'Change in plasma HIV RNA', -1.095)

write_csv(ttest_label, 'gathered_data/ttest-res.csv')
