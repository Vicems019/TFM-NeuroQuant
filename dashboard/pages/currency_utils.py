# dashboard/pages/currency_utils.py

# Tasas de cambio (Simplificadas)
CURRENCY_RATES = {
    "USD": {"rate": 1.0, "symbol": "$"},
    "EUR": {"rate": 0.93, "symbol": "€"},
    "GBP": {"rate": 0.79, "symbol": "£"},
    "JPY": {"rate": 155.0, "symbol": "¥"},
}

def format_price(price, currency="USD"):
    """Format a price based on the selected currency."""
    conf = CURRENCY_RATES.get(currency, CURRENCY_RATES["USD"])
    val = price * conf["rate"]
    sym = conf["symbol"]
    if val >= 1000: return f"{sym}{val:,.0f}"
    if val >= 1:    return f"{sym}{val:,.2f}"
    return f"{sym}{val:.4f}"
