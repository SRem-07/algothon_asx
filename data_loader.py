import yfinance as yf
import pandas as pd


def get_asx50_tickers():
  """
    Returns ASX 50 tickers as a list
  """
  
  asx50_tickers = [
    "ALD.AX", "ALL.AX", "ALQ.AX", "AMC.AX", "ANZ.AX",
    "APA.AX", "ASX.AX", "BHP.AX", "BSL.AX", "BXB.AX",
    "CBA.AX", "COL.AX", "CPU.AX", "CSL.AX", "EDV.AX",
    "FMG.AX", "GMG.AX", "IAG.AX", "JHX.AX", "LNW.AX",
    "MGR.AX", "MIN.AX", "MPL.AX", "MQG.AX", "NAB.AX",
    "NEM.AX", "NXT.AX", "ORG.AX", "PLS.AX", "PME.AX",
    "QAN.AX", "QBE.AX", "REA.AX", "RIO.AX", "RMD.AX",
    "SCG.AX", "SDF.AX", "SGP.AX", "SHL.AX", "STO.AX",
    "SUN.AX", "TCL.AX", "TLS.AX", "TPG.AX", "WBC.AX",
    "WDS.AX", "WES.AX", "WOR.AX", "WOW.AX", "XRO.AX"
  ]

  return asx50_tickers

# Gather ASX 50 market data
def get_asx50_data(start_date = "2012-01-01", end_date = "2026-06-01"):
  """
    Call get_asx50_tickers and then download data for ASX50
  """
  # Get list of tickers
  asx50_tickers = get_asx50_tickers()
  
  # Load data, only adjusted close price
  data = yf.download(tickers=asx50_tickers,
                     start=start_date, end=end_date, 
                     auto_adjust=True)['Close']
  
  # Rename columns to remove 'Close' prefix
  data.columns = [col for col in data.columns]
  
  return data
  
  

# Gather ASX market data
def get_asx_market_data(start_date = "2012-01-01", end_date = "2026-06-01"):
  """
    Get data on the key market indicies. Dates can be overrided when function is called. 
    Can use as a proxy for ASX futures as the spot-futures basis is usually small and slow-moving.

    ^AXTL = ASX 20 Spot Index
    ^ALFI = ASX 50 Spot Index
    ^AXJO = ASX 200 Spot Index
  """
  
  # List of tickers
  market_tickers = ["^AXTL", "^ALFI", "^AXJO"]
  
  # Load data, only adjusted close price
  data = yf.download(tickers=market_tickers,
                     start=start_date, end=end_date, 
                     auto_adjust=True)['Close']
  
  return data

# General function to gather data from yfinance
def get_market_data(tickers, start_date = "2012-01-01", end_date = "2026-06-01"):
  """
   Customisable function to gather market close data for any tickers
  """
   
  # Load data, only adjusted close price
  data = yf.download(tickers=tickers,
                     start=start_date, end=end_date, 
                     auto_adjust=True)['Close']
  
  return data
