import json
import pickle
import torch
import torch.nn.functional as F
from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc

from bert_update import BERT


# --------------------------------------------------
# Device
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# --------------------------------------------------
# Mean Pooling
# --------------------------------------------------
def mean_pool(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / \
           torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# --------------------------------------------------
# Load Configuration
# --------------------------------------------------
with open("sbert_config.json", "r") as f:
    config = json.load(f)


# --------------------------------------------------
# Load Vocabulary
# --------------------------------------------------
with open("word2id.pkl", "rb") as f:
    word2id = pickle.load(f)


# --------------------------------------------------
# Simple Tokenizer
# --------------------------------------------------
def tokenize(text):
    return text.lower().split()


def encode(text, max_len=100):
    tokens = tokenize(text)

    ids = [word2id.get(t, word2id.get("[UNK]", 1)) for t in tokens]
    ids = ids[:max_len]

    padding_length = max_len - len(ids)

    attention_mask = [1] * len(ids) + [0] * padding_length
    ids = ids + [word2id.get("[PAD]", 0)] * padding_length

    return torch.tensor([ids]).to(device), torch.tensor([attention_mask]).to(device)


# --------------------------------------------------
# Load SBERT Encoder
# --------------------------------------------------
model = BERT(
    n_layers=config["n_layers"],
    n_heads=config["n_heads"],
    d_model=config["d_model"],
    d_ff=config["d_ff"],
    d_k=config["d_k"],
    n_segments=config["n_segments"],
    vocab_size=config["vocab_size"],
    max_len=config["max_len"],
    device=device
).to(device)

model.load_state_dict(torch.load("sbert_encoder.pth", map_location=device))
model.eval()


# --------------------------------------------------
# Load Classifier Head
# --------------------------------------------------
classifier_head = torch.nn.Linear(3 * config["d_model"], 3).to(device)
classifier_head.load_state_dict(torch.load("sbert_classifier.pth", map_location=device))
classifier_head.eval()


# --------------------------------------------------
# Label Mapping
# --------------------------------------------------
label_map = {
    0: "Entailment",
    1: "Neutral",
    2: "Contradiction"
}


# --------------------------------------------------
# Prediction Function (NO Confidence)
# --------------------------------------------------
def predict(premise, hypothesis):
    with torch.no_grad():

        ids_a, mask_a = encode(premise)
        ids_b, mask_b = encode(hypothesis)

        u = model.get_last_hidden_state(ids_a, segment_ids=torch.zeros_like(ids_a))
        v = model.get_last_hidden_state(ids_b, segment_ids=torch.zeros_like(ids_b))

        u_mean = mean_pool(u, mask_a)
        v_mean = mean_pool(v, mask_b)

        uv_abs = torch.abs(u_mean - v_mean)

        x = torch.cat([u_mean, v_mean, uv_abs], dim=-1)

        logits = classifier_head(x)

        pred = torch.argmax(logits, dim=-1).item()

    return label_map[pred]


# --------------------------------------------------
# Dash App Layout
# --------------------------------------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([

    html.H2("Sentence-BERT NLI Text Similarity App", className="text-center mt-4"),

    html.Br(),

    dbc.Row([
        dbc.Col([
            html.Label("Premise"),
            dcc.Textarea(
                id="premise-input",
                style={"width": "100%", "height": 120},
                placeholder="Enter premise sentence..."
            )
        ])
    ]),

    html.Br(),

    dbc.Row([
        dbc.Col([
            html.Label("Hypothesis"),
            dcc.Textarea(
                id="hypothesis-input",
                style={"width": "100%", "height": 120},
                placeholder="Enter hypothesis sentence..."
            )
        ])
    ]),

    html.Br(),

    dbc.Button("Predict", id="predict-btn", color="primary"),

    html.Br(),
    html.Br(),

    html.Div(
        id="output-result",
        style={"fontSize": 22, "fontWeight": "bold"}
    )

], fluid=True)


# --------------------------------------------------
# Callback
# --------------------------------------------------
@app.callback(
    Output("output-result", "children"),
    Input("predict-btn", "n_clicks"),
    State("premise-input", "value"),
    State("hypothesis-input", "value")
)
def update_output(n_clicks, premise, hypothesis):

    if not n_clicks:
        return ""

    if not premise or not hypothesis:
        return "Please enter both sentences."

    label = predict(premise, hypothesis)

    return f"Prediction: {label}"


# --------------------------------------------------
# Run Server
# --------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
