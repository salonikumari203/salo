import dash
from dash import dcc, html
import dash.dependencies as dd
import plotly.express as px
import pandas as pd
import numpy as np

np.random.seed(42)
N = 1000
df = pd.DataFrame({
    "customer_id": np.arange(N),
    "age": np.random.randint(18, 65, size=N),
    "income": np.random.normal(50000, 15000, size=N).clip(20000, 120000),
    "visits": np.random.poisson(10, size=N),
    "purchase": np.random.binomial(1, 0.4, size=N),
    "region": np.random.choice(["North", "South", "East", "West"], size=N),
})

df["spend"] = df["purchase"] * np.random.uniform(20, 300, size=N)


app = dash.Dash(__name__)
app.title = "Customer Analytics Dashboard"

app.layout = html.Div([
    html.H1("📊 Customer Analytics Dashboard"),
    
    html.Div([
        html.Label("Select Region:"),
        dcc.Dropdown(
            id="region_filter",
            options=[{"label": r, "value": r} for r in df["region"].unique()],
            value=None,
            placeholder="All Regions",
            multi=True
        )
    ], style={"width": "30%", "display": "inline-block"}),

    html.Div([
        html.H3("Key Metrics"),
        html.Div(id="kpi_container", style={"display": "flex", "gap": "40px"})
    ], style={"margin-top": "20px"}),

    html.Div([
        dcc.Graph(id="income_histogram"),
        dcc.Graph(id="spend_by_age"),
    ], style={"display": "flex", "flex-wrap": "wrap"}),

    html.Div([
        dcc.Graph(id="purchase_rate_region")
    ])
])
@app.callback(
    [dd.Output("income_histogram", "figure"),
     dd.Output("spend_by_age", "figure"),
     dd.Output("purchase_rate_region", "figure"),
     dd.Output("kpi_container", "children")],
    [dd.Input("region_filter", "value")]
)
def update_dashboard(region_filter):
    dff = df.copy()
    if region_filter:
        dff = dff[dff["region"].isin(region_filter)]

    
    fig_income = px.histogram(dff, x="income", nbins=30, title="Income Distribution")

    
    fig_spend = px.scatter(dff, x="age", y="spend", color="purchase",
                           title="Spend by Age", opacity=0.6)


    region_rate = dff.groupby("region")["purchase"].mean().reset_index()
    fig_region = px.bar(region_rate, x="region", y="purchase",
                        title="Purchase Rate by Region")

    
    total_customers = len(dff)
    purchase_rate = dff["purchase"].mean()*100
    avg_spend = dff["spend"].mean()

    kpis = [
        html.Div([
            html.H4("Total Customers"),
            html.P(f"{total_customers:,}")
        ]),
        html.Div([
            html.H4("Purchase Rate"),
            html.P(f"{purchase_rate:.1f}%")
        ]),
        html.Div([
            html.H4("Avg Spend"),
            html.P(f"${avg_spend:,.2f}")
        ])
    ]

    return fig_income, fig_spend, fig_region
if __name__ == "__main__":
    app.run_server(debug=True)
