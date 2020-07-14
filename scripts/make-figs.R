library(tidyverse)
# 'p_cd4difw48w4': 'Change in Absolute CD4 Count',
# 'p_logtbilw0': 'Log10-transformed Biliruben (mg/dL) at Week Zero',
# 'vllogdifw48w4': 'Log10-transformed Change in Plasma HIV RNA Copies'

theme_set(theme_bw())

pipe_scores <- read_csv('supplementary_data/all_pipes.csv')
feat_imps <- read_csv('supplementary_data/all_feature_imps.csv')
ttest_res <- read_csv('supplementary_data/ttest-res.csv')

r2_plot <- pipe_scores %>%
  ggplot(aes(x = dat, y = `R^2 Score`, color = dat)) +
  geom_boxplot(fill = NA, color = 'grey70', outlier.size = 0) +
  ggbeeswarm::geom_beeswarm(size = 2.5, alpha = 0.8, stroke = 0) +
  scale_color_manual(values = c('grey40', '#D46680')) +
  labs(y = bquote('Pipeline'~ R^2), x = NULL) +
  facet_wrap(~ type) +
  theme(legend.position = 'None') +
  geom_text(data = ttest_res, aes(x = 1.5, y = -0.45, label = label_r2), color = 'grey10')

ggsave('figs/r2.png', r2_plot, width = 7, height = 4)
ggsave('figs/r2.pdf', r2_plot, width = 7, height = 4)


mse_plot <- pipe_scores %>% 
  ggplot(aes(x = dat, y = `MSE Score`, color = dat)) +
  geom_boxplot(fill = NA, color = 'grey70', outlier.size = 0) +
  ggbeeswarm::geom_beeswarm(size = 2.5, alpha = 0.8, stroke = 0) +
  scale_color_manual(values = c('grey40', '#D46680')) +
  labs(y = 'Negative MSE', x = NULL) +
  facet_wrap(~ type, scales = 'free_y') +
  theme(legend.position = 'None') +
  geom_text(data = ttest_res, aes(x = 1.5, y = y, label = label_mse), color = 'grey10')

ggsave('figs/mse.png', mse_plot, width = 8, height = 4)
ggsave('figs/mse.pdf', mse_plot, width = 8, height = 4)


importance_df <- feat_imps %>% 
  mutate(Pipeline = as.factor(`Pipeline Score Rank`)) %>% 
  add_count(type, bin) %>%
  group_by(type, Pipeline) %>% 
  mutate_at(vars(matches('Importance')), ~ dense_rank(desc(.x))) %>% 
  ungroup() %>% 
  mutate(avg_rank = miscTools::rowMedians(select(., contains('Importance'))),
         jit_avg_rank = jitter(avg_rank, amount = 0.1)) %>%
  {.}

top_genes <- importance_df %>%
  group_by(type, bin) %>%
  summarise(mean_across_pipes = mean(avg_rank), .groups = 'drop') %>%
  group_by(type) %>%
  slice_min(mean_across_pipes, n = 11) %>%
  pull(bin)

importance_p <- importance_df %>% 
  filter(bin %in% top_genes) %>% 
  ggplot(aes(x = fct_reorder(fct_reorder(bin, n), -avg_rank), 
             y = jit_avg_rank, group = bin)) +
  geom_point(aes(color = Pipeline)) +
  coord_flip() +
  labs(x = NULL, y = 'Median rank (jittered)') +
  theme(legend.position = c(0.9, 0.84),
        legend.key.height = unit(4, 'mm')) +
  scale_y_continuous(breaks = seq.int(10)) +
  scale_color_viridis_d(option = 'D', begin = 0.2) +
  facet_grid(rows = vars(type), scales = 'free_y', space = 'free_y') +
  NULL

ggsave('figs/feat_imp.png', importance_p, width = 5, height = 7)
ggsave('figs/feat_imp.pdf', importance_p, width = 5, height = 7)

