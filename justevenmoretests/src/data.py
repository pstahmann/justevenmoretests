import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer, TargetEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
from src import config

warnings.filterwarnings("ignore", category=UserWarning)


def engineer_datetime_features(df):
    """Extrahiert zyklische Features und Deltas aus Zeitspalten."""

    if 'signup_time' in df.columns and 'purchase_time' in df.columns:
        print("  -> Verarbeite signup_time & purchase_time...")
        df['signup_time'] = pd.to_datetime(df['signup_time'], errors='coerce')
        df['purchase_time'] = pd.to_datetime(df['purchase_time'], errors='coerce')

        df['time_diff_seconds'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds()

        df['purchase_hour_sin'] = np.sin(2 * np.pi * df['purchase_time'].dt.hour / 24)
        df['purchase_hour_cos'] = np.cos(2 * np.pi * df['purchase_time'].dt.hour / 24)
        df['purchase_dow_sin']  = np.sin(2 * np.pi * df['purchase_time'].dt.dayofweek / 7)
        df['purchase_dow_cos']  = np.cos(2 * np.pi * df['purchase_time'].dt.dayofweek / 7)

        df = df.drop(columns=['signup_time', 'purchase_time'])

    for time_col in ['Time', 'time', 'TransactionDT']:
        if time_col in df.columns:
            print(f"  -> Transformiere {time_col} in zyklische Features...")
            df[f'{time_col}_sin'] = np.sin(2 * np.pi * df[time_col] / 86400)
            df[f'{time_col}_cos'] = np.cos(2 * np.pi * df[time_col] / 86400)
            df = df.drop(columns=[time_col])

    return df


def prepare_data(dataset_name, model_type, seed=42, target_col="Class"):
    """
    Lädt die Daten, zieht den Split (basierend auf seed) und fittet das 
    Preprocessing strikt nur auf X_train, um Leakage zu vermeiden.
    """
    path = config.DATASET_PATHS.get(dataset_name)
    if not path:
        raise ValueError(
            f"Dataset '{dataset_name}' nicht in config.py definiert. "
            f"Verfügbar: {list(config.DATASET_PATHS.keys())}"
        )

    print(f"Lade Daten von: {path} (Split-Seed: {seed})")
    df = pd.read_csv(path)
    if len(df.columns) == 1 and ';' in df.columns[0]:
        print("  -> Semikolon-Trennzeichen erkannt. Lade Daten neu...")
        df = pd.read_csv(path, sep=';')

    # 1. Zielvariable finden
    target = target_col if target_col in df.columns else next(
        (col for col in df.columns if col.lower() in ['class', 'isfraud']), None
    )
    if not target:
        raise ValueError("Keine Zielvariable (Class/isFraud) gefunden.")

    # 2. ID-Spalten entfernen
    drop_cols = ['id', 'user_id', 'TransactionID', 'device_id']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # 3. Feature Engineering für Zeit
    df = engineer_datetime_features(df)

    y = df[target].values
    df_X = df.drop(columns=[target])
    feature_names = df_X.columns.tolist()

    # 4. Spaltentypen
    cat_cols = df_X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    num_cols = df_X.select_dtypes(include=['number']).columns.tolist()

    # Split: 60% Train / 20% Cal / 20% Test (stratifiziert)
    # WICHTIG: Nutzt nun den übergebenen Seed für sauberes Multi-Seed-Design
    X_temp, X_test, y_temp, y_test = train_test_split(
        df_X, y, test_size=0.2, stratify=y, random_state=seed
    )
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=seed
    )

    print(f"  -> Pipeline ({len(num_cols)} num., {len(cat_cols)} kat.) für {model_type.upper()}")
    print(f"  -> Split: Train={len(X_train)}, Cal={len(X_cal)}, Test={len(X_test)}")

    # --- Numerische Pipeline ---
    # Confounder-Hinweis: Um im Paper fair zu vergleichen, sollte im Methodenteil
    # erwähnt werden, warum DL-Modelle Quantile-Transformation nutzen.
    if model_type in ["mlp", "resnet", "ftt"]:
        scaler_type = "quantile"
        num_scaler = QuantileTransformer(
            n_quantiles=min(1000, len(X_train)),
            output_distribution='normal', random_state=seed
        )
    else:
        scaler_type = "standard"
        num_scaler = StandardScaler()

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', num_scaler)
    ])

    # --- Kategoriale Pipeline ---
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', TargetEncoder(target_type='binary', smooth="auto", random_state=seed)),
        ('scaler', StandardScaler())
    ])

    transformers = [('num', num_pipeline, num_cols)]
    if cat_cols:
        transformers.append(('cat', cat_pipeline, cat_cols))

    preprocessor = ColumnTransformer(transformers, remainder='drop')

    print("  -> Fitte Preprocessor auf Trainingsdaten (vermeide Leakage)...")
    X_train_proc = preprocessor.fit_transform(X_train, y_train)
    X_cal_proc   = preprocessor.transform(X_cal)
    X_test_proc  = preprocessor.transform(X_test)

    return {
        'X_train': X_train_proc.astype(np.float32),
        'y_train': y_train.astype(np.int64),
        'X_cal':   X_cal_proc.astype(np.float32),
        'y_cal':   y_cal.astype(np.int64),
        'X_test':  X_test_proc.astype(np.float32),
        'y_test':  y_test.astype(np.int64),
        'feature_names': feature_names,
        'preprocessing': {
            'num_scaler': scaler_type,
            'cat_encoder': 'target_encoding',
            'model_type': model_type,
            'n_num': len(num_cols),
            'n_cat': len(cat_cols),
        },
    }
