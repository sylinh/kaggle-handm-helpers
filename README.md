# kaggle-handm-helpers

Những bổ sung/sửa chính so với bản gốc:

- `pairs.py`: thêm confidence ngược cho cặp, tính lift dựa trên toàn bộ khách, giữ thống kê tổng khách mua pair.
- `candidates.py`: đưa các thống kê pairs mới vào feature, bổ sung price-fit/discount-sensitivity cho cặp customer–article, thêm rule/source flags (recent/last_weeks/pairs/age_bucket) với n_sources/best_score/rank.
- `fe.py`: thêm decay affinity + last_seen cho customer–category, thêm trend_ratio_7_28 và newness_days trong lag features, sửa `day_week_numbers` bỏ applymap (dùng chia nguyên cho cudf), cho phép `create_art_t_features` nhận 2 hoặc 3 tham số.
- Notebook `cH&M_feature_update.ipynb`: cập nhật pipeline để dùng các feature mới, tránh shelve (dùng dict), và encode cột string thành numeric để model fit.
