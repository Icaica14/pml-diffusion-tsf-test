"""Model wrappers (the comparison ladder, plan Part 5).

M0 naive · M1 ARIMA/ETS · M2 DeepAR · M3 TimeGrad · + toy DDPM.
Each wrapper consumes the same `ForecastDataset` and emits the same forecast
object, so evaluation is model-agnostic. Populated from M0 onward.
"""
