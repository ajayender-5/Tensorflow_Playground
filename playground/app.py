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

# Get the absolute path to the current script
script_path = os.path.abspath(__file__)

# Set the working directory to the script's directory
os.chdir(os.path.dirname(script_path))

# Specify the path to the model folder relative to the script's directory
model_folder = "resources"

model_file_path1 = os.path.join(model_folder, '1.ushape.csv')
model_file_path2 = os.path.join(model_folder, '2.concerticcir1.csv')
model_file_path3 = os.path.join(model_folder, '3.concertriccir2.csv')
model_file_path4 = os.path.join(model_folder, '4.linearsep.csv')
model_file_path5 = os.path.join(model_folder, '5.outlier.csv')
model_file_path6 = os.path.join(model_folder, '6.overlap.csv')
model_file_path7 = os.path.join(model_folder, '7.xor.csv')
model_file_path8 = os.path.join(model_folder, '8.twospirals.csv')

model_file_paths = {
    "ushape": model_file_path1,
    "concerticcir1": model_file_path2,
    "concertriccir2": model_file_path3,
    "linearsep": model_file_path4,
    "outlier": model_file_path5,
    "overlap": model_file_path6,
    "xor": model_file_path7,
    "twospirals": model_file_path8
}


def load_data(file_path):
    return pd.read_csv(file_path)


def plot_dataset(dataset_path):
    dataset = load_data(dataset_path)
    X = dataset.iloc[:, :-1].values
    Y = dataset.iloc[:, -1].values
    fig, ax = plt.subplots(figsize=(4,4))
    ax.scatter(X[:, 0], X[:, 1], c=Y, cmap='viridis', edgecolors='k')
    ax.set_title('Dataset')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    st.pyplot(fig)



def app():
    st.title('A Neural Network Playground')

    st.sidebar.title("Configuration")

    # Plot dataset in the sidebar
    st.sidebar.subheader("Dataset Preview")
    dataset_choice = st.sidebar.radio("Choose a dataset", list(model_file_paths.keys()))
    plot_dataset(model_file_paths[dataset_choice])

    num_hidden_layers = st.sidebar.slider("Number of Hidden Layers", 1, 5, 1, key='num_hidden_layers')
    epochs = st.sidebar.slider("Select number of epochs", 100, 1000, 100, key='epochs')
    lr = st.sidebar.number_input('Enter Learning Rate:', value=0.001, key='lr')
    hidden_layers = []

    for i in range(num_hidden_layers):
        st.sidebar.markdown(f"### Hidden Layer {i+1}")
        units = st.sidebar.slider(f"Number of units for hidden layer {i+1}", 1, 10, 1, key=f"units_{i}")
        activation = st.sidebar.selectbox(f"Activation function for hidden layer {i+1}", ['tanh', 'sigmoid', "relu", "linear"],
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
        fig, ax = plt.subplots()
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
                st.write(f"#### Hidden Layer {layer_num+1}, Neuron {neuron_num+1}")
                fig, ax = plt.subplots()
                plot_decision_regions(X, Y, clf=neuron_model, ax=ax)
                st.pyplot(fig)


if __name__ == "__main__":
    app()
