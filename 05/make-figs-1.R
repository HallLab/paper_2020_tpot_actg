library(tidyverse)
# 'p_cd4difw48w4': 'Change in Absolute CD4 Count',
# 'p_logtbilw0': 'Log10-transformed Biliruben (mg/dL) at Week Zero',
# 'vllogdifw48w4': 'Log10-transformed Change in Plasma HIV RNA Copies'

theme_set(theme_bw())

pipe_scores <- read_csv('gathered_data/all_pipes.csv')
feat_imps <- read_csv('gathered_data/all_feature_imps.csv')
ttest_res <- read_csv('gathered_data/ttest-res.csv')

r2_plot <- pipe_scores %>%
  ggplot(aes(x = fct_rev(dat), y = `R^2 Score`, color = dat)) +
  geom_boxplot(fill = NA, color = 'grey70', outlier.size = 0) +
  ggbeeswarm::geom_beeswarm(size = 2.5, alpha = 0.8, stroke = 0) +
  scale_color_manual(values = c('#D46680', 'grey40')) +
  labs(y = bquote('Pipeline'~ R^2), x = NULL) +
  facet_wrap(~ type) +
  theme(legend.position = 'None') +
  geom_text(data = ttest_res, aes(x = 1.5, y = -0.45, label = label_r2), color = 'grey10')

ggsave('figs/r2.png', r2_plot, width = 7, height = 4)
ggsave('figs/r2.pdf', r2_plot, width = 7, height = 4)


mse_plot <- pipe_scores %>% 
  ggplot(aes(x = fct_rev(dat), y = `MSE Score`, color = dat)) +
  geom_boxplot(fill = NA, color = 'grey70', outlier.size = 0) +
  ggbeeswarm::geom_beeswarm(size = 2.5, alpha = 0.8, stroke = 0) +
  scale_color_manual(values = c('#D46680', 'grey40')) +
  labs(y = 'Negative MSE', x = NULL) +
  facet_wrap(~ type, scales = 'free_y') +
  theme(legend.position = 'None') +
  geom_text(data = ttest_res, aes(x = 1.5, y = y, label = label_mse), color = 'grey10')

ggsave('figs/mse.png', mse_plot, width = 8, height = 4)
ggsave('figs/mse.pdf', mse_plot, width = 8, height = 4)
