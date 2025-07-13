import yfinance as yf

# Contoh mengambil data harian 1 bulan terakhir saham GGRM.JK
data = yf.download('GGRM.JK', period='5y', interval='1d')

print(data.head())
