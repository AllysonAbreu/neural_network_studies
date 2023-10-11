from dateutil.relativedelta import relativedelta
import pandas as pd
import datetime as dt

def beteween_dates(input_date, df_date):
    date_01 = pd.to_datetime(input_date)
    date_02 = dt.date.fromordinal(df_date)
    years = relativedelta(date_01, date_02).years
    months = relativedelta(date_01, date_02).months
    if years > 0:
        return (years * 12) + months
    else:
        return months