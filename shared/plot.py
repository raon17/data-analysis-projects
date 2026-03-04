import plotly.express as px

THEME = "plotly_dark"

def bar_chart(df, x, y, title):
    return px.bar(df, x=x, y=y, title=title, template=THEME)

def line_chart(df, x, y, title):
    return px.line(df, x=x, y=y, title=title, template=THEME)