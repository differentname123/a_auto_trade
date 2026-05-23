import pandas as pd
import numpy as np
import scipy.stats as stats


def evaluate_fof_portfolio(df_list, date_col='净值日期', nav_col='复权净值',
                           rebalance_days=30, max_history_days=5 * 252,
                           max_mdd_limit=0.2, hurdle_rate=0.2):
    """
    第一性原理 FOF 绝对收益评估引擎 (V4.2 终极数学严谨生产版)
    """
    if not df_list or len(df_list) < 2:
        return {"error": "Need at least 2 funds", "Total_Score": 0.0}

    # ================= 1. 数据清洗与对齐 =================
    nav_series_list = []

    for i, df in enumerate(df_list):
        temp_df = df.copy()
        temp_df[date_col] = pd.to_datetime(temp_df[date_col])
        temp_df = temp_df.set_index(date_col).sort_index()
        temp_df = temp_df[~temp_df.index.duplicated(keep='last')]
        nav_series = temp_df[nav_col].rename(f'fund_{i}')
        nav_series_list.append(nav_series)

    merged_nav = pd.concat(nav_series_list, axis=1, join='outer')
    merged_nav_raw_index = merged_nav.index

    # 严禁压缩时间轴，基准日历为骨架。找到有效起止区间
    valid_mask = merged_nav.notna().all(axis=1)
    if not valid_mask.any():
        return {"error": "No overlapping data", "Total_Score": 0.0}
    valid_start = valid_mask.idxmax()
    valid_end = valid_mask[::-1].idxmax()

    merged_nav = merged_nav.loc[valid_start:valid_end]

    if len(merged_nav) > max_history_days:
        merged_nav = merged_nav.iloc[-max_history_days:]

    if len(merged_nav) < 252:
        return {"error": "Common data length < 252 days", "Total_Score": 0.0}

    # 记录缺失率用于 VETO
    missing_ratios = merged_nav.isna().sum() / len(merged_nav)
    metrics_max_missing_ratio = float(missing_ratios.max())

    valid_start = merged_nav.index[0]
    valid_end = merged_nav.index[-1]

    raw_max_date = merged_nav_raw_index[-1]
    if (raw_max_date - valid_end).days > 35:
        return {"error": f"Zombie fund detected. Cutoff at {valid_end.strftime('%Y-%m-%d')}", "Total_Score": 0.0}

    # 先用 ffill 填平缺失期，再算收益率，彻底杜绝跨期真实收益被蒸发的灾难
    merged_nav_ffilled = merged_nav.ffill()
    fund_rets = merged_nav_ffilled.pct_change().fillna(0.0)

    # 抓捕最长连续0收益天数
    zero_rets_df = (fund_rets.abs() < 1e-8).astype(int)
    max_continuous_zeros = 0
    for col in zero_rets_df.columns:
        consecutive = zero_rets_df[col].groupby((zero_rets_df[col] == 0).cumsum()).sum().max()
        max_continuous_zeros = max(max_continuous_zeros, consecutive)

    n_days = len(fund_rets)
    n_funds = len(df_list)
    fund_rets_arr = fund_rets.values

    # ================= 2 & 3. 核心评估闭包引擎 =================
    def _evaluate_single_path(current_reb_days, offset=0, w_init=None):
        synth_ret_arr = np.zeros(n_days)

        if w_init is None:
            w = np.ones(n_funds) / n_funds
        else:
            w = np.array(w_init)

        for t in range(n_days):
            daily_port_ret = np.sum(w * fund_rets_arr[t])
            synth_ret_arr[t] = daily_port_ret

            denom = 1 + daily_port_ret if (1 + daily_port_ret) > 1e-9 else 1e-9
            w = w * (1 + fund_rets_arr[t]) / denom

            if t >= current_reb_days and (t - offset) % current_reb_days == 0:
                if w_init is None:
                    w = np.ones(n_funds) / n_funds
                else:
                    w = np.array(w_init)

        synth_ret = pd.Series(synth_ret_arr, index=fund_rets.index)
        synth_eq = (1 + synth_ret).cumprod()

        metrics = {
            'Start_Date': merged_nav.index[0].strftime('%Y-%m-%d'),
            'End_Date': merged_nav.index[-1].strftime('%Y-%m-%d'),
            'Total_Days': n_days,
            'n_funds': n_funds
        }

        # --- 核心指标计算 ---
        # [致命Bug修复 1]: 直接基于隐含本金 1.0 的期末累计净值计算年化，防止抹除首日收益
        cagr = float(synth_eq.iloc[-1] ** (252 / n_days) - 1)
        metrics['CAGR'] = cagr

        # [致命Bug修复 2]: 强行锁定历史最高水位的底线为初始本金1.0，防止开局暴跌被豁免
        cum_max = synth_eq.cummax().clip(lower=1.0)
        drawdowns = (synth_eq - cum_max) / cum_max
        metrics['Max_Drawdown'] = float(drawdowns.min())

        is_drawdown = drawdowns < 0
        recovery_groups = (~is_drawdown).cumsum()
        metrics['Max_Recovery_Days'] = int(is_drawdown.groupby(recovery_groups).sum().max())

        log_eq = np.log(synth_eq.clip(lower=1e-9).values)

        window = min(252, n_days)
        rolling_r2 = []
        for st in range(0, n_days - window + 1, 21):
            y_sub = log_eq[st:st + window]
            _, _, r_sub, _, _ = stats.linregress(np.arange(window), y_sub)
            rolling_r2.append(r_sub ** 2 if pd.notna(r_sub) else 0.0)
        metrics['Worst_Rolling_1Y_R2'] = float(min(rolling_r2))

        ar1 = synth_ret.autocorr(lag=1)
        metrics['AR1_Coefficient'] = float(ar1) if pd.notna(ar1) else 1.0
        vol_annual = synth_ret.std() * np.sqrt(252)

        n_worst = max(5, int(len(synth_ret) * 0.05))
        worst_dates = synth_ret.nsmallest(n_worst).index
        worst_fund_rets = fund_rets.loc[worst_dates]

        if len(worst_fund_rets) > 3:
            corr_matrix = worst_fund_rets.corr().values
            triu_idx = np.triu_indices_from(corr_matrix, k=1)
            if len(triu_idx[0]) > 0:
                with np.errstate(invalid='ignore'):
                    max_corr = np.nanmax(corr_matrix[triu_idx])
                metrics['Downside_Correlation'] = float(max_corr) if not np.isnan(max_corr) else 0.5
            else:
                metrics['Downside_Correlation'] = 0.5
        else:
            metrics['Downside_Correlation'] = 0.5

        # --- 4. 底线判决 (Iron VETO) ---
        vetoes = {
            "VETO_Hurdle_Rate": metrics['CAGR'] < hurdle_rate,
            "VETO_Drawdown_Crash": abs(metrics['Max_Drawdown']) > max_mdd_limit,
            "VETO_Fake_Smooth": (metrics['AR1_Coefficient'] > 0.35) and (vol_annual < 0.03),
            "VETO_Endless_Bleeding": metrics['Max_Recovery_Days'] > 180,
            "VETO_Data_Distortion": (metrics_max_missing_ratio > 0.05) or (max_continuous_zeros > 10)
        }
        metrics.update(vetoes)

        if any(vetoes.values()):
            return metrics, 0.0

        # --- 第一性原理综合打分 ---
        excess_cagr = max(0.0, metrics['CAGR'] - hurdle_rate)
        adj_mdd = max(abs(metrics['Max_Drawdown']), 0.01)

        base_score = (excess_cagr / adj_mdd) * metrics['Worst_Rolling_1Y_R2']
        time_multiplier = min(1.0, np.sqrt(n_days / 756))

        if metrics['Downside_Correlation'] > 0.8:
            p_corr = 0.1
        elif metrics['Downside_Correlation'] > 0.5:
            p_corr = np.clip(1.0 - (metrics['Downside_Correlation'] - 0.5) * 1.0, 0.5, 1.0)
        else:
            p_corr = 1.0

        path_score = base_score * time_multiplier * p_corr
        return metrics, max(0.0, float(path_score))

    # ================= 5. 多维扰动验证闭环 =================
    final_metrics, score_base = _evaluate_single_path(rebalance_days, offset=0)

    if score_base == 0.0:
        final_metrics['Total_Score'] = 0.0
        final_metrics['VETO_Perturbation_Death'] = False
        return final_metrics

    metrics_w = None
    score_weight = float('inf')

    half_offset = rebalance_days // 2

    for seed_val in [1024, 2048, 4096]:
        np.random.seed(seed_val)
        shift_vector = np.random.uniform(-0.05, 0.05, n_funds)
        w_perturbed = np.ones(n_funds) / n_funds + shift_vector
        w_perturbed = np.clip(w_perturbed, 0.01, 1.0)
        w_perturbed = w_perturbed / np.sum(w_perturbed)

        m_w, s_w = _evaluate_single_path(rebalance_days, offset=half_offset, w_init=w_perturbed)
        if s_w < score_weight:
            score_weight = s_w
            metrics_w = m_w

    if score_weight == float('inf'):
        score_weight = 0.0

    final_metrics['Total_Score'] = min(score_base, score_weight)

    final_metrics['VETO_Perturbation_Death'] = final_metrics['Total_Score'] == 0.0
    if final_metrics['VETO_Perturbation_Death'] and metrics_w is not None:
        if score_weight == 0.0:
            for k, v in metrics_w.items():
                if k.startswith("VETO_") and v is True:
                    final_metrics[k + "_in_Perturb"] = True

    return final_metrics