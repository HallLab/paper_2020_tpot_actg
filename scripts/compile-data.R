library(tidyverse)

cd4_rep_scores <- read_delim('pipeline_scores/p_cd4difw48w4_replication_scores.txt', delim = '\t') %>% 
  mutate(dat = 'Real data', type = 'Change in Absolute CD4 Count') %>% 
  select(-Replicate)
cd4_per_scores <- read_delim('pipeline_scores/p_cd4difw48w4_permutation_scores.txt', delim = '\t') %>% 
  mutate(dat = 'Permuted data', type = 'Change in Absolute CD4 Count') %>% 
  select(-Permutation)
bili_rep_scores <- read_delim('pipeline_scores/p_logtbilw0_replication_scores.txt', delim = '\t') %>% 
  mutate(dat = 'Real data', type = 'Log10 Biliruben (mg/dL) at Week 0') %>% 
  select(-Replicate)
bili_per_scores <- read_delim('pipeline_scores/p_logtbilw0_permutation_scores.txt', delim = '\t') %>% 
  mutate(dat = 'Permuted data', type = 'Log10 Biliruben (mg/dL) at Week 0') %>% 
  select(-Permutation)
plasma_rep_scores <- read_delim('pipeline_scores/vllogdifw48w4_replication_scores.txt', delim = '\t') %>% 
  mutate(dat = 'Real data', type = 'Log10 Change in HIV RNA Copies') %>% 
  select(-Replicate)
plasma_per_scores <- read_delim('pipeline_scores/vllogdifw48w4_permutation_scores.txt', delim = '\t') %>% 
  mutate(dat = 'Permuted data', type = 'Log10 Change in HIV RNA Copies') %>% 
  select(-Permutation)


bilirubin_feats <- readxl::read_xlsx('supplementary_data/bilirubin_feature_importances.xlsx') %>% 
  mutate(type = 'Log10 Biliruben (mg/dL) at Week 0')
cd4_feats <- readxl::read_xlsx('supplementary_data/cd4_feature_importances.xlsx') %>% 
  mutate(type = 'Change in Absolute CD4 Count')
plasma_feats <- readxl::read_xlsx('supplementary_data/viralload_feature_importances.xlsx') %>% 
  mutate(type = 'Log10 Change in HIV RNA Copies')


bind_rows(plasma_per_scores, plasma_rep_scores,
          cd4_per_scores, cd4_rep_scores,
          bili_per_scores, bili_rep_scores) %>% 
  write_csv('supplementary_data/all_pipes.csv')

bind_rows(bilirubin_feats, cd4_feats, plasma_feats) %>% 
  write_csv('supplementary_data/all_feature_imps.csv')


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
  test_r2(bili_rep_scores, bili_per_scores), test_mse(bili_rep_scores, bili_per_scores), 'Log10 Biliruben (mg/dL) at Week 0', -0.0425,
  test_r2(plasma_rep_scores, plasma_per_scores), test_mse(plasma_rep_scores, plasma_per_scores), 'Log10 Change in HIV RNA Copies', -1.095)

write_csv(ttest_label, 'supplementary_data/ttest-res.csv')
