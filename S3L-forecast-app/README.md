Please note that this is a snippet of the work I did in my S3L internship. 
1. Essentially, given a csv file of PG&E electricity data, the program would reorganize the data to feed it into a NeuralProphet model. 
2. This model would forecast values for the rest of the dates and give the sum of the entire projected yearly usage.
3. A plot would be generated as well
4. The bottom cell is a sample of the plotting/experimentation I did (before using NeuralProphet) to better generally visualize the relationship between energy consumption and variables such as temperature. 

---

# Public demo (web app)

A Streamlit web version of the notebook above (`Full Summer Forecasting.ipynb`, unchanged). Upload PG&E usage data (and optionally temperature) and get a 183-day forecast, a month-by-month table and simple suggestions. The internship data is private and is not in this repo; the sample files in `sample_data/` are synthetic.

```
pip install -r requirements.txt
streamlit run app.py
```

## What is original vs. new

| | |
|---|---|
| **Original** | The model in `forecast_core.py`: NeuralProphet, 10 changepoints, yearly/weekly/daily seasonality, temperature as a lagged regressor, 183-day horizon. Settings unchanged. |
| **Changed (compatibility only)** | Removed an unused `trapz` import and replaced `Series.append` (both removed from current SciPy/pandas); added a PyTorch checkpoint-loading flag. |
| **Changed (behavior)** | In my original work, I exported usage data from PG&E, added temperature and organized it in a spreadsheet, then ran it through the forecasting notebook. This public version automates that preparation and adds a web interface


## Limitations

- A statistical projection from past usage; it cannot anticipate new appliances, solar, an EV, or move-ins/outs.
- Accuracy has not been validated on real data. In one test on synthetic data with a known answer (about 13 months of history), the 183-day total came out roughly 10% low.
- Suggestions are general guidance, not an energy audit. Not affiliated with PG&E.


