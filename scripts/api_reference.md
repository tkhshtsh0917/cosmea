## Training components / 学習用の部品

::: cosmea.DocumentFrequency
    options:
      heading_level: 3
      merge_init_into_class: true
      members: [from_dataset, from_ids, count]

::: cosmea.MixedNegativeDataCollator
    options:
      heading_level: 3

::: cosmea.LogQMultipleNegativesRankingLoss
    options:
      heading_level: 3
      merge_init_into_class: true
      members: [compute_loss_from_embeddings, get_config_dict]

::: cosmea.CachedLogQMultipleNegativesRankingLoss
    options:
      heading_level: 3
      merge_init_into_class: true
      members: [get_config_dict]

## Lower-level functions / 低レベル関数

::: cosmea.stable_id
    options:
      heading_level: 3

::: cosmea.mixed_negative_logq_loss
    options:
      heading_level: 3

::: cosmea.candidate_log_q
    options:
      heading_level: 3

::: cosmea.negative_log_partition
    options:
      heading_level: 3
