import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import griddata
import pandas as pd 
import plotly.express as px
import yfinance as yf 
import numpy as np
from scipy.stats import norm
from scipy import stats
import seaborn as sns
import requests
import re
import matplotlib.pyplot as plt
import datetime
import numpy as np
from scipy.optimize import brentq
from datetime import timedelta

pd.options.plotting.backend = "plotly"

st.title('Implied Volatility Surface')

st.sidebar.header('Model Parameters')
st.sidebar.write('Adjust the parameters for the Black-Scholes model.')

r = st.sidebar.number_input(
    'Risk-Free Rate (e.g., 0.045 for 4.5%)',
    value=0.0425,
    format="%.4f"
)


st.sidebar.header('Ticker Symbol')
ticker= st.sidebar.text_input(
    'Enter Ticker Symbol',
    value='SPY',
    max_chars=10
).upper()


def d0(sigma, S, K, r, t):
    d1= 1/(sigma *np.sqrt(t))*(np.log(S/K)+(r+sigma**2/2)*t)
    d2= d1-sigma*np.sqrt(t)
    return d1,d2 

def Call(sigma, S, K, r, t):

    d1= 1/(sigma *np.sqrt(t))*(np.log(S/K)+(r+sigma**2/2)*t)
    d2= d1-sigma*np.sqrt(t)

    C= norm.cdf(d1) * S - norm.cdf(d2) * K * np.exp(-r*t)
    return C

def Put(sigma, S, K, r, t, d1,d2):
    P= norm.cdf(d2) * K * np.exp(-r*t) - norm.cdf(d1) * S
    return P



def is_valid_ticker(symbol):
    """Check if a stock ticker is valid using yfinance."""
    equity = yf.Ticker(symbol)
    try:
        if not equity.info or "symbol" not in equity.info:
            return False, "Invalid Ticker"
        return True,equity
    except:
        return False, "Error retrieving data"
    
    

query = is_valid_ticker(ticker)

while not query[0]:
    ticker = input("Invalid Ticker. Please enter a valid symbol: ")
    query = is_valid_ticker(ticker)

equity = query[1]
print("Stock Data:", equity)
equity= equity.history('max')
# obtain historical realized  volatility to use to create 
# custom probability density function to base our black scholes off of
equity["prev Close"] = equity["Close"].shift(1)
equity["pct change"] = (100*equity[['prev Close','Close']].pct_change(axis=1,)['Close']).round(2)

volatilityT= (equity['pct change'].rolling(window=252).std() * np.sqrt(252)).dropna()

options = query[1]

today = datetime.date.today()
#print("Today's Date:", today)

# Convert expiration dates to datetime format
expirations = [datetime.datetime.strptime(date, "%Y-%m-%d").date() for date in options.options]

# Calculate time to expiration (TTE) in days
t = [(exp - today).days for exp in expirations]

t= np.asarray(t)
t= np.delete(t,0)
t= pd.DataFrame(t, columns=["dte"])


sigma = volatilityT.iloc[-1]*.01
def NewtonIV(C, S, K, r, t):
    tol = .00001
    max_iter = 10000
    vol = sigma  # Initial guess for volatility
    T = t / 365  # Convert days to years

    Iv = []  # Store implied volatilities

    for index, k in enumerate(K):
        count = 0
        epsilon = 1

        if k <= 0:  # Ensure strike price is valid
            Iv.append(np.nan)
            continue

        while epsilon > tol:
            if count >= max_iter:
                Iv.append(np.nan)  # If it doesn't converge, return NaN
                break

            count += 1
            vol_temp = vol

            # Ensure vol is not zero
            if vol < 1e-4:
               Iv.append(np.nan)
               break

            d1, d2 = d0(vol, S, k, r, T)
            price = Call(vol, S, k, r, T) - C.iloc[index]

            # Ensure vega is not zero
            vega = max(S * norm.pdf(d1) * np.sqrt(T), .01)

            vol = vol - (price / vega) 
            # Ensure volatility stays in reasonable range
            vol = max(min(vol, 1), 0.05)

            epsilon = abs(vol - vol_temp)
            #if abs(price) <tol: 
            #    break 

           

        Iv.append(vol)

    return Iv



def plot_iv_surface(options_dataC_list, t):
    fig = plt.figure(figsize=(40, 20))
    #ax = fig.add_subplot(111, projection="3d")

    X_data = []  # Strike Prices
    Y_data = []  # Days to Expiration (DTE)
    Z_data = []  # IV Values

    # Loop through each iteration (each expiration date)
    for i, df in enumerate(options_dataC_list):
        dte = t.loc[i, "dte"]  # Get DTE for this iteration
        
        for j in range(len(df)):
            X_data.append(df["strike"].iloc[j])
            Y_data.append(dte)
            Z_data.append(df["IV"].iloc[j])

    # Convert to numpy arrays for surface plotting
    X = np.array(X_data)
    Y = np.array(Y_data)
    Z = np.array(Z_data)

    # Create a grid for plotting
    X_grid, Y_grid = np.meshgrid(np.unique(X), np.unique(Y))

    # Interpolate Z values onto the grid
    #Z_grid = np.zeros_like(X_grid, dtype=float,)
    Z_grid = griddata((X, Y), Z, (X_grid, Y_grid), method="cubic")
    for i in range(len(X)):
        xi = np.where(X_grid[0] == X[i])[0]
        yi = np.where(Y_grid[:, 0] == Y[i])[0]
        if xi.size > 0 and yi.size > 0:
            Z_grid[yi[0], xi[0]] = Z[i]

    # Plot the surface
   
   
    fig = go.Figure(data=[go.Surface(z=Z_grid, x=X_grid, y=Y_grid, colorscale="Plasma")])
    
    # Update layout for interactive rotation
    fig.update_layout(
        title=(f"Implied Volitility Surface for ticker {ticker}" ), 
        scene=dict(
            xaxis_title="Strike Price",
            yaxis_title="Days to Expiration (DTE)",
            zaxis_title="Implied Volatility (IV)",
            xaxis= dict(range= [options_dataC_list[-1]['strike'][0],options_dataC_list[-1]['strike'].iloc[-1] ]),
            zaxis= dict(range= [.06, .99])
         
         ))
    
    st.plotly_chart(fig)



S = equity['Close'].iloc[-1]  # Stock price from equity data
 # Example risk-free rate (1%)

expirations = options.options[1:]
expirations = [datetime.datetime.strptime(date, "%Y-%m-%d").date() for date in expirations]
two_years_later = today + timedelta(days=2*365)
expirations = [date for date in expirations if date <= two_years_later]

filtered_expirations = [date.strftime("%Y-%m-%d") for date in expirations]

#print (expirations)

# DataFrame to store results
options_dataC = pd.DataFrame()
options_dataC_list= []



for index1, items in enumerate(filtered_expirations): 
    call_chain = options.option_chain(items).calls

    # Ensure no NaNs before calculations
    call_chain = call_chain.dropna(subset=["bid", "ask", "strike"])
    call_chain = call_chain[(call_chain['bid'] > 0) & (call_chain['ask'] > 0)]
    # Compute mid prices
    mid_prices = (call_chain["bid"] + call_chain["ask"]) / 2

    # Compute IV using NewtonIV function
    iv_values = NewtonIV(mid_prices, S, call_chain["strike"], r, t.iloc[index1]['dte'])
    #iv_values = other_root(mid_prices, S, call_chain["strike"], r, t.loc[index1]['dte'])

    temp_df = pd.DataFrame({

        "strike": call_chain["strike"],
        "bid": call_chain["bid"],
        "ask": call_chain["ask"],
        "Mid_price": mid_prices,
        "IV": iv_values
    })

    # Append temp_df to options_dataC
    temp_df= temp_df[(temp_df['IV'] < 1) & (temp_df['IV'] > 0.05)]
    temp_df= temp_df[(temp_df['strike'] <= S*1.2) & (temp_df['strike'] >= S*0.8 )]
    options_dataC = pd.concat([options_dataC, temp_df], ignore_index=True)
    
    options_dataC_list.append(options_dataC)


plot_iv_surface(options_dataC_list,t)
    

st.write("---")
st.markdown("Created by Brandon Maier")
