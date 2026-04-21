"""Model training and inference for the prediction backend."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.preprocessing import MinMaxScaler


@dataclass
class ModelBundle:
    """Container for trained model state and metadata."""

    name: str
    kind: str
    feature_columns: list[str] = field(default_factory=list)
    model: object | None = None
    feature_scaler: MinMaxScaler | None = None
    target_scaler: MinMaxScaler | None = None
    look_back: int | None = None
    backend: str | None = None
    warning: str | None = None


def chronological_split(
    X: np.ndarray,
    y: np.ndarray,
    val_fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split arrays into train and validation sets without shuffling."""
    if len(X) != len(y):
        raise ValueError("Feature and target lengths do not match.")
    if len(X) < 12:
        raise ValueError("Not enough samples to train and validate the model.")

    split_idx = max(int(len(X) * (1.0 - val_fraction)), 1)
    split_idx = min(split_idx, len(X) - 1)
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]


def train_xgboost_regressor(
    X: np.ndarray,
    y: np.ndarray,
    feature_columns: list[str],
    val_fraction: float = 0.2,
    random_state: int = 42,
) -> tuple[ModelBundle, np.ndarray, np.ndarray]:
    """Train XGBoost when available, otherwise fall back gracefully."""
    X_train, X_val, y_train, y_val = chronological_split(X, y, val_fraction=val_fraction)

    backend = "xgboost"
    warning = None
    try:
        from xgboost import XGBRegressor

        model = XGBRegressor(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=random_state,
        )
    except Exception as exc:
        from sklearn.ensemble import HistGradientBoostingRegressor

        model = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=6,
            max_iter=300,
            random_state=random_state,
        )
        backend = "sklearn-hist-gradient-boosting"
        warning = f"xgboost unavailable; using sklearn fallback ({exc})."

    model.fit(X_train, y_train)
    y_pred = np.asarray(model.predict(X_val), dtype=float)
    bundle = ModelBundle(
        name="XGBoost",
        kind="tabular",
        feature_columns=feature_columns,
        model=model,
        backend=backend,
        warning=warning,
    )
    return bundle, y_val, y_pred


def _scale_sequence_inputs(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler]:
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    X_train_scaled = feature_scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val_scaled = feature_scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1))
    y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1))
    return X_train_scaled, X_val_scaled, y_train_scaled, y_val_scaled, feature_scaler, target_scaler


def train_lstm_regressor(
    X_seq: np.ndarray,
    y_seq: np.ndarray,
    feature_columns: list[str],
    look_back: int,
    epochs: int = 10,
    batch_size: int = 32,
    val_fraction: float = 0.2,
    random_state: int = 42,
) -> tuple[ModelBundle, np.ndarray, np.ndarray]:
    """Train a sequence-based LSTM regressor."""
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

    tf.random.set_seed(random_state)
    X_train, X_val, y_train, y_val = chronological_split(X_seq, y_seq, val_fraction=val_fraction)
    X_train_scaled, X_val_scaled, y_train_scaled, _, feature_scaler, target_scaler = _scale_sequence_inputs(
        X_train, X_val, y_train, y_val
    )

    model = Sequential(
        [
            Input(shape=(X_train.shape[1], X_train.shape[2])),
            LSTM(64, return_sequences=True),
            Dropout(0.15),
            LSTM(32),
            Dropout(0.1),
            Dense(16, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    callbacks = [EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)]
    model.fit(
        X_train_scaled,
        y_train_scaled,
        validation_data=(X_val_scaled, target_scaler.transform(y_val.reshape(-1, 1))),
        epochs=epochs,
        batch_size=max(1, min(batch_size, len(X_train_scaled))),
        verbose=0,
        callbacks=callbacks,
    )

    y_pred_scaled = model.predict(X_val_scaled, verbose=0).reshape(-1, 1)
    y_pred = target_scaler.inverse_transform(y_pred_scaled).ravel()
    bundle = ModelBundle(
        name="LSTM",
        kind="sequence",
        feature_columns=feature_columns,
        model=model,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        look_back=look_back,
        backend="tensorflow",
    )
    return bundle, y_val, y_pred


class _TorchTransformerRegressor:
    """Minimal wrapper to keep the PyTorch model self-contained."""

    def __init__(self, input_size: int, d_model: int = 32, nhead: int = 4, num_layers: int = 2, dropout: float = 0.1):
        import torch
        import torch.nn as nn

        class TransformerModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.input_projection = nn.Linear(input_size, d_model)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dropout=dropout,
                    batch_first=True,
                    activation="gelu",
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                self.norm = nn.LayerNorm(d_model)
                self.output = nn.Linear(d_model, 1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                seq_len = x.size(1)
                positions = torch.linspace(0.0, 1.0, steps=seq_len, device=x.device)
                positions = positions.unsqueeze(0).unsqueeze(-1).expand(x.size(0), seq_len, 1)
                encoded = self.input_projection(x) + positions
                encoded = self.encoder(encoded)
                encoded = self.norm(encoded[:, -1, :])
                return self.output(encoded).squeeze(-1)

        self.network = TransformerModel()


def train_transformer_regressor(
    X_seq: np.ndarray,
    y_seq: np.ndarray,
    feature_columns: list[str],
    look_back: int,
    epochs: int = 10,
    batch_size: int = 32,
    val_fraction: float = 0.2,
    random_state: int = 42,
) -> tuple[ModelBundle, np.ndarray, np.ndarray]:
    """Train an optional Transformer encoder regressor."""
    import torch

    torch.manual_seed(random_state)
    X_train, X_val, y_train, y_val = chronological_split(X_seq, y_seq, val_fraction=val_fraction)
    X_train_scaled, X_val_scaled, y_train_scaled, _, feature_scaler, target_scaler = _scale_sequence_inputs(
        X_train, X_val, y_train, y_val
    )

    device = torch.device("cpu")
    wrapper = _TorchTransformerRegressor(input_size=X_train.shape[2])
    model = wrapper.network.to(device)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32, device=device)
    y_train_tensor = torch.tensor(y_train_scaled.ravel(), dtype=torch.float32, device=device)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    effective_batch = max(1, min(batch_size, len(X_train_tensor)))

    model.train()
    for _ in range(epochs):
        permutation = torch.randperm(len(X_train_tensor), device=device)
        for start in range(0, len(X_train_tensor), effective_batch):
            batch_idx = permutation[start:start + effective_batch]
            optimizer.zero_grad()
            predictions = model(X_train_tensor[batch_idx])
            loss = loss_fn(predictions, y_train_tensor[batch_idx])
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_val_tensor).cpu().numpy().reshape(-1, 1)

    y_pred = target_scaler.inverse_transform(y_pred_scaled).ravel()
    bundle = ModelBundle(
        name="Transformer",
        kind="sequence",
        feature_columns=feature_columns,
        model=model,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        look_back=look_back,
        backend="torch",
    )
    return bundle, y_val, y_pred


def predict_next_close(bundle: ModelBundle, feature_frame) -> float:
    """Predict the next close from the latest feature state."""
    if bundle.kind == "tabular":
        latest_row = feature_frame[bundle.feature_columns].iloc[-1].to_numpy(dtype=np.float32).reshape(1, -1)
        return float(np.asarray(bundle.model.predict(latest_row)).ravel()[0])

    if bundle.look_back is None or bundle.feature_scaler is None or bundle.target_scaler is None:
        raise ValueError(f"Sequence model {bundle.name} is missing scaling metadata.")

    latest_sequence = feature_frame[bundle.feature_columns].tail(bundle.look_back).to_numpy(dtype=np.float32)
    if len(latest_sequence) < bundle.look_back:
        raise ValueError(f"Not enough rows to forecast with {bundle.name}.")

    scaled_sequence = bundle.feature_scaler.transform(latest_sequence).reshape(1, bundle.look_back, len(bundle.feature_columns))
    if bundle.backend == "tensorflow":
        pred_scaled = bundle.model.predict(scaled_sequence, verbose=0).reshape(-1, 1)
    elif bundle.backend == "torch":
        import torch

        with torch.no_grad():
            tensor = torch.tensor(scaled_sequence, dtype=torch.float32)
            pred_scaled = bundle.model(tensor).cpu().numpy().reshape(-1, 1)
    else:
        raise ValueError(f"Unknown backend for sequence model: {bundle.backend}")

    return float(bundle.target_scaler.inverse_transform(pred_scaled)[0, 0])
