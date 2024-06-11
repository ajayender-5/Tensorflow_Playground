import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from tensorflow.keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import os

# Set page config
st.set_page_config(page_title="Neural Network Playground", page_icon="🧠", layout="wide")

# Set custom style
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-image: linear-gradient(#edf0f2,#edf0f2);
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Set plot size globally
st.set_option('deprecation.showPyplotGlobalUse', False)
plt.rcParams['figure.figsize'] = [7, 4]  # Decrease the default figure size

# Specify the path to the model folder relative to the script's directory

model_file_paths = {
    "ushape": playground/resources/1.ushape.csv,
    "concerticcir1": playground/resources/2.concerticcir1.csv,
    "concertriccir2": playground/resources/3.concertriccir2.csv,
    "linearsep": playground/resources/4.linearsep.csv,
    "outlier": playground/resources/5.outlier.csv,
    "overlap": playground/resources/6.overlap.csv,
    "xor": playground/resources/7.xor.csv,
    "twospirals": playground/resources/8.twospirals.csv
}

def load_data(file_path):
    return pd.read_csv(file_path)

def plot_datasets():
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))  # Decreased figure size for datasets plot
    dataset_names = list(model_file_paths.keys())
    for i, (dataset_name, dataset_path) in enumerate(model_file_paths.items()):
        row = i // 4
        col = i % 4
        dataset = load_data(dataset_path)
        X = dataset.iloc[:, :-1].values
        Y = dataset.iloc[:, -1].values
        ax = axes[row, col]
        ax.scatter(X[:, 0], X[:, 1], c=Y, cmap='viridis', edgecolors='k')
        ax.set_title(f'Dataset: {dataset_name}')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    st.pyplot(fig)
    return st.selectbox('Select Dataset:', dataset_names)

def app():
    st.title('A Neural Network Playground')

    st.sidebar.title("Configuration")

    st.sidebar.subheader("Dataset Preview")
    dataset_choice = plot_datasets()

    num_hidden_layers = st.sidebar.slider("Number of Hidden Layers", 1, 5, 1, key='num_hidden_layers')
    epochs = st.sidebar.slider("Select number of epochs", 100, 1000, 100, key='epochs')
    lr = st.sidebar.number_input('Enter Learning Rate:', value=0.001, key='lr')
    hidden_layers = []

    for i in range(num_hidden_layers):
        st.sidebar.markdown(f"### Hidden Layer {i + 1}")
        units = st.sidebar.slider(f"Number of units for hidden layer {i + 1}", 1, 10, 1, key=f"units_{i}")
        activation = st.sidebar.selectbox(f"Activation function for hidden layer {i + 1}",
                                           ['tanh', 'sigmoid', "relu", "linear"],
                                           key=f"activation_{i}")
        hidden_layers.append((units, activation))

    if st.sidebar.button("Train Model", key='train_button'):
        dataset_path = model_file_paths[dataset_choice]

        # Load dataset
        dataset = load_data(dataset_path)

        # Split dataset into training and testing sets
        X = dataset.iloc[:, :-1].values
        Y = dataset.iloc[:, -1].values.astype(np.int_)  # Convert target variable to integers
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        def create_layer(units, activation):
            return Dense(units=units, activation=activation, use_bias=True)

        def build_model():
            input_layer = Input(shape=(2,))
            x = input_layer
            for units, activation in hidden_layers:
                x = create_layer(units, activation)(x)
            output_layer = Dense(units=1, activation='sigmoid')(x)  # Output layer for binary classification
            model = Model(inputs=input_layer, outputs=output_layer)
            model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])
            return model

        model = build_model()

        history = model.fit(X_train, Y_train, epochs=epochs, validation_data=(X_test, Y_test), verbose=0)

        # Plot training and testing loss
        st.subheader('Training and Testing Loss')
        st.line_chart(pd.DataFrame({
            'Training Loss': history.history['loss'],
            'Testing Loss': history.history['val_loss']
        }))

        # Plot the output layer decision region
        st.subheader('Decision Region for the Output Layer')
        fig, ax = plt.subplots(figsize=(6, 3))  # Decreased figure size for decision region plot
        plot_decision_regions(X, Y, clf=model, ax=ax)
        st.pyplot(fig)

        # Display the plots side by side
        st.subheader('Hidden Layer Outputs')
        hidden_layers_output = [layer.output for layer in model.layers if isinstance(layer, Dense)]

        # Extract the output of each neuron from all hidden layers
        for layer_num, layer_output in enumerate(hidden_layers_output[:-1]):  # Exclude the output layer
            num_neurons = layer_output.shape[1]
            for neuron_num in range(num_neurons):
                neuron_model = Model(inputs=model.input, outputs=layer_output[:, neuron_num])
                st.write(f"#### Hidden Layer {layer_num + 1}, Neuron {neuron_num + 1}")
                fig, ax = plt.subplots(figsize=(6, 3))  # Decreased figure size for neuron output plot
                plot_decision_regions(X, Y, clf=neuron_model, ax=ax)
                st.pyplot(fig)

if __name__ == "__main__":
    app()
