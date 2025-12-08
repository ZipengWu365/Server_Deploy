# Analysis of Original Multi-scale STD vs. Component Ablation

This report analyzes the performance of the **Original Multi-scale STD Linear** model (referred to as `STD-Full` in the ablation study) against its individual components (`STD-Trend`, `STD-Seasonal`, `STD-Dispersion`) and a `Baseline` (no decomposition).

## 1. Verification of "Original" Model
By comparing the results in `results_STD_LINEAR.txt` (Original Multi-scale) with `std_component_ablation_comparison_details.csv` (Ablation Study), we confirmed that **`STD-Full` represents the Original Multi-scale STD model**.

**Evidence (ETTh1, H=96):**
*   **Original (results_STD_LINEAR.txt):** MSE = `0.3419`
*   **Ablation (STD-Full):** MSE = `0.3419`

This confirms that the ablation study was conducted using the optimized, dataset-specific multi-scale configurations (e.g., `[24, 168, 504, 1440]` for ETTh1).

## 2. Performance Comparison

The following table summarizes the MSE performance of the Original (Full) model versus its components across different datasets and horizons.

### ETTh1 (Hourly)
| Horizon | Baseline | **STD-Full (Original)** | STD-Trend | STD-Seasonal | STD-Dispersion | Best Component |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **96** | **0.2899** | 0.3419 | 0.2920 | 0.2921 | 0.3259 | Baseline |
| **192** | **0.3210** | 0.3568 | 0.3259 | 0.3226 | 0.3568 | Baseline |
| **336** | 0.3456 | 0.3710 | 0.3452 | **0.3430** | 0.3746 | STD-Seasonal |
| **720** | 0.3750 | 0.4085 | **0.3659** | 0.4026 | 0.3918 | STD-Trend |

*   **Observation:** For ETTh1, the **Baseline** (no decomposition) actually outperforms the Full STD model at shorter horizons (96, 192). At longer horizons, **STD-Seasonal** (336) and **STD-Trend** (720) perform best. The Full model seems to overfit or introduce noise via the extra components for this specific dataset.

### ETTh2 (Hourly)
| Horizon | Baseline | **STD-Full (Original)** | STD-Trend | STD-Seasonal | STD-Dispersion | Best Component |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **96** | 0.2532 | **0.1925** | 0.2398 | 0.2510 | 0.2441 | **STD-Full** |
| **192** | 0.3402 | 0.2893 | 0.3163 | 0.3125 | **0.2869** | STD-Dispersion |
| **336** | 0.3720 | 0.3430 | 0.3466 | 0.3727 | **0.3344** | STD-Dispersion |
| **720** | 0.4603 | 0.4053 | 0.5605 | 0.4943 | **0.3865** | STD-Dispersion |

*   **Observation:** ETTh2 shows a different pattern. The **STD-Full** model is significantly better than Baseline at H=96. However, **STD-Dispersion** (Dispersion component only) is the clear winner for longer horizons (192, 336, 720), suggesting that the dispersion features are the most predictive for this dataset.

### ETTm1 (15-min)
| Horizon | Baseline | **STD-Full (Original)** | STD-Trend | STD-Seasonal | STD-Dispersion | Best Component |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **96** | 0.2406 | 0.2662 | 0.2419 | **0.2401** | 0.2429 | STD-Seasonal |
| **192** | 0.2680 | 0.3170 | 0.2723 | **0.2665** | 0.2689 | STD-Seasonal |
| **336** | **0.2948** | 0.3345 | 0.3516 | 0.3262 | 0.2968 | Baseline |
| **720** | **0.3304** | 0.4359 | 0.3345 | 0.3322 | 0.3452 | Baseline |

*   **Observation:** Similar to ETTh1, ETTm1 favors the **Baseline** or **STD-Seasonal** components. The Full model degrades performance, likely due to the complexity of combining all components.

### ETTm2 (15-min)
| Horizon | Baseline | **STD-Full (Original)** | STD-Trend | STD-Seasonal | STD-Dispersion | Best Component |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **96** | 0.1372 | 0.1925 | **0.1369** | 0.1418 | 0.1459 | STD-Trend |
| **192** | 0.1884 | 0.2893 | 0.1783 | 0.1767 | **0.1727** | STD-Dispersion |
| **336** | 0.3093 | 0.3430 | 0.7121 | 0.3340 | **0.2721** | STD-Dispersion |
| **720** | 0.3368 | 0.4053 | 0.3489 | 0.3323 | **0.2771** | STD-Dispersion |

*   **Observation:** ETTm2 strongly favors **STD-Dispersion** at longer horizons (336, 720), similar to ETTh2. The Full model again underperforms compared to selecting the single best component.

### Exchange Rate
| Horizon | Baseline | **STD-Full (Original)** | STD-Trend | STD-Seasonal | STD-Dispersion | Best Component |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **96** | **0.0515** | 0.0604 | 0.0560 | 0.0939 | 0.0589 | Baseline |
| **192** | 0.1192 | **0.1090** | 0.1148 | 0.1092 | 0.1415 | **STD-Full** |
| **336** | 0.2584 | **0.1732** | 0.2821 | 0.2303 | 0.3973 | **STD-Full** |
| **720** | 0.3909 | **0.4757** | 0.5674 | 0.5273 | 0.4984 | Baseline |

*   **Observation:** For Exchange Rate, the **STD-Full** model is superior at medium horizons (192, 336), significantly beating the Baseline. This suggests that the combination of Trend and Seasonal components is crucial for this dataset, which is known to have complex dependencies.

## 3. Summary & Conclusion

1.  **Original vs. Components:** The "Original" Multi-scale STD model (`STD-Full`) is **not always the best performer**. In many cases (ETTh1, ETTm1), a simpler model (Baseline or Single Component) yields lower MSE.
2.  **Dispersion Importance:** For **ETTh2** and **ETTm2**, the **Dispersion** component is highly predictive, often outperforming the Full model by a large margin. This suggests that for these datasets, the volatility/variance features are more important than the trend or seasonality.
3.  **Exchange Rate Success:** The Full model shines on the **Exchange Rate** dataset (H=192, 336), where it effectively combines components to capture complex dynamics.
4.  **Recommendation:**
    *   For **ETTh1/ETTm1**: Consider using **STD-Seasonal** or reverting to **Baseline**.
    *   For **ETTh2/ETTm2**: Switch to **STD-Dispersion** only.
    *   For **Exchange Rate**: Keep **STD-Full**.

This analysis highlights that while the Multi-scale STD approach is powerful, **adaptive component selection** based on the dataset could yield significant performance gains over the fixed "Full" strategy.
