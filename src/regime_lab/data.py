"""Synthetic research series and point-in-time CSV ingestion."""
import numpy as np
import pandas as pd

def synthetic_prices(n=1800,seed=42):
    if n<100: raise ValueError("n must be >= 100")
    rng=np.random.default_rng(seed)
    trans=np.array([[.975,.025],[.055,.945]])
    states=np.zeros(n,dtype=int)
    for t in range(1,n): states[t]=rng.choice(2,p=trans[states[t-1]])
    mean=np.array([.00065,-.0010])
    vol=np.array([.007,.022])
    ret=mean[states]+vol[states]*rng.normal(size=n)
    idx=pd.bdate_range('2017-01-02',periods=n)
    return pd.DataFrame({'close':100*np.exp(np.cumsum(ret)),'true_state':states},index=idx)

def load_prices(csv_path):
    df=pd.read_csv(csv_path)
    if not {'date','close'}.issubset(df.columns): raise ValueError('CSV requires date,close columns')
    if df['date'].isna().any() or df['close'].isna().any(): raise ValueError('missing dates or close')
    df['date']=pd.to_datetime(df['date'],errors='raise',utc=True)
    if df['date'].duplicated().any(): raise ValueError('duplicate dates')
    if not df['date'].is_monotonic_increasing: raise ValueError('dates must be strictly increasing')
    df=df.set_index('date')
    df['close']=pd.to_numeric(df['close'],errors='raise')
    if not np.isfinite(df['close']).all() or (df['close']<=0).any(): raise ValueError('close must be positive finite')
    return df[['close']]

def make_features(prices,span=20):
    """Past-only return and rolling volatility of logarithmic returns."""
    if span<3: raise ValueError('span too small')
    r=np.log(prices['close']).diff()
    v=r.rolling(span,min_periods=span).std(ddof=1)
    df=pd.DataFrame({'return':r,'volatility':v},index=prices.index).dropna()
    if len(df)<100: raise ValueError('At least 100 feature rows required')
    return df