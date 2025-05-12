from kfp.dsl import component, Input, Output, Dataset, Model

@component(
    packages_to_install=["tensorflow", "pandas", "scikit-learn"],
    base_image="python:3.12",
)
def train_model(input_data: Input[Dataset], model_output: Output[Model]):
    import pandas as pd
    import os
    from tensorflow import keras
    from tensorflow.keras import layers
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    # Enable mixed precision training
    # keras.mixed_precision.set_global_policy('mixed_float16')

    # Read data in chunks to reduce memory usage
    chunk_size = 50000  # Increased chunk size for faster processing
    chunks = pd.read_csv(input_data.path + "/processed.csv", chunksize=chunk_size)
    
    # Process first chunk to get feature names and initialize scalers
    first_chunk = next(chunks)
    first_chunk = first_chunk.dropna()
    
    # Define feature groups
    categorical_cols = ["hexid", "dayType"]
    numerical_cols = ["traversals", "lat", "lng"]
    target_col = "price"
    
    # Initialize scaler for numerical features
    scaler = StandardScaler()
    scaler.fit(first_chunk[numerical_cols])
    
    # One-hot encode the first chunk to get the expected columns
    first_chunk_encoded = pd.get_dummies(first_chunk.drop(columns=[target_col]), columns=categorical_cols)
    expected_columns = first_chunk_encoded.columns
    
    # Initialize model with simplified architecture
    model = keras.Sequential([
        layers.Input(shape=(len(expected_columns),)),
        layers.BatchNormalization(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    
    # Use Huber loss which is more robust to outliers
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='huber',
        metrics=['mae']
    )
    
    # Prepare validation data from first chunk
    X_val = first_chunk.drop(columns=[target_col])
    X_val = pd.get_dummies(X_val, columns=categorical_cols)
    for col in expected_columns:
        if col not in X_val.columns:
            X_val[col] = 0
    X_val = X_val[expected_columns]
    X_val[numerical_cols] = scaler.transform(X_val[numerical_cols])
    X_val = X_val.astype(np.float32)
    y_val = first_chunk[target_col].values.astype(np.float32)
    
    # Train on chunks
    for chunk in chunks:
        chunk = chunk.dropna()
        y = chunk[target_col].values.astype(np.float32)
        X = chunk.drop(columns=[target_col])
        
        # Scale numerical features
        X[numerical_cols] = scaler.transform(X[numerical_cols])
        
        # One-hot encode categorical columns
        X = pd.get_dummies(X, columns=categorical_cols)
        
        # Ensure consistent columns with first chunk
        for col in expected_columns:
            if col not in X.columns:
                X[col] = 0
        X = X[expected_columns]
        
        # Convert all columns to float32
        X = X.astype(np.float32)
        
        # Convert to numpy and train
        X = X.to_numpy()
        model.fit(
            X, y,
            epochs=1,
            batch_size=128,  # Increased batch size
            validation_data=(X_val, y_val),
            verbose=0
        )

    # Save the model and scaler
    os.makedirs(model_output.path, exist_ok=True)
    model.export(model_output.path)
    
    # Save the scaler parameters
    scaler_params = {
        'mean_': scaler.mean_,
        'scale_': scaler.scale_,
        'feature_names': numerical_cols
    }
    np.save(os.path.join(model_output.path, "scaler_params.npy"), scaler_params)

    with open(os.path.join(model_output.path, "expected_columns.txt"), "w") as f:
        f.write("\n".join(expected_columns))

