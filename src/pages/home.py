from pathlib import Path

import dash
from dash import html
import dash_bootstrap_components as dbc

dash.register_page(__name__, path='/', name='Home', title='Sentiment Analyzer | Home')

import pandas as pd
try:
    from src.ml_sentiment import evaluate_model, preprocess
except ModuleNotFoundError:
    from ml_sentiment import evaluate_model, preprocess

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "train5.csv"

# Load and preprocess data for accuracy comparison
df = pd.read_csv(DATA_PATH)
df.columns = ['Sentiment', 'Text', 'Score']
df['Text'] = df['Text'].astype(str).apply(preprocess)
X = df['Text'].values
y = df['Sentiment'].values

# Evaluate both models
vader_acc, *_ = evaluate_model(X, y, 'VADER', type=1, k=5)
nb_acc, *_ = evaluate_model(X, y, 'Naive Bayes', type=0, k=5)

# Determine which is best
if nb_acc > vader_acc:
    best_model = 'Naive Bayes'
    diff = nb_acc - vader_acc
elif vader_acc > nb_acc:
    best_model = 'VADER'
    diff = vader_acc - nb_acc
else:
    best_model = 'Both models are equal'
    diff = 0

layout = dbc.Container([
    # title
    dbc.Row([
        dbc.Col([
            html.H3('Welcome!'),
            html.P(html.B('App Overview'), className='par')
        ], width=12, className='row-titles')
    ]),
    # Guidelines
    dbc.Row([
        dbc.Col([], width=2),
        dbc.Col([
            html.P([html.B('Hello! This program detects the emotional tone of text.'), html.Br(),
                    'The program will analyze the emotional sentiment of each text sample and classify it as positive, negative, or neutral.'], className='guide'),
            html.P([html.B('1) Model Evaluation'), html.Br(),
                    'View detailed performance metrics including accuracy, precision, recall, and F1 score.'], className='guide'),
            html.P([html.B('2) Model Comparison'), html.Br(),
                    'Compare our model to the pre-built sentiment analyzer VADER.'], className='guide'),
            html.P([html.B('3) Test Your Own Text'), html.Br(),
                    'Enter your own text and see how the program classifies its sentiment.'], className='guide')
        ], width=8),
        dbc.Col([], width=2)
    ]),
    # Comparison row
    dbc.Row([
        dbc.Col([], width=2),
        dbc.Col([
            html.Div([
                html.H5('Model Comparison: VADER vs Naive Bayes', className='mt-4'),
                html.P([
                    f"VADER Accuracy: {vader_acc:.2%} ", html.Br(),
                    f"Naive Bayes Accuracy: {nb_acc:.2%} ", html.Br(),
                    html.B(f"Most Accurate: {best_model}"),
                    html.Br(),
                    html.Span(f"Difference: {diff:.2%}" if diff != 0 else "No difference in accuracy.")
                ], className='guide', style={'font-size': '1.1em'})
            ])
        ], width=8),
        dbc.Col([], width=2)
    ])
])
