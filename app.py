import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load trained model
model = joblib.load("iris_model.pkl")

# Load Iris dataset for visualization
iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=[
        "Sepal Length",
        "Sepal Width",
        "Petal Length",
        "Petal Width"
    ]
)

df["Species"] = [
    iris.target_names[i] for i in iris.target
]

# Title
st.title("🌸 Iris Flower Prediction")

st.write("Enter the flower measurements below:")

# Input fields
sl = st.number_input("Sepal Length", min_value=0.0, value=5.1)
sw = st.number_input("Sepal Width", min_value=0.0, value=3.5)
pl = st.number_input("Petal Length", min_value=0.0, value=1.4)
pw = st.number_input("Petal Width", min_value=0.0, value=0.2)

# Prediction
if st.button("Predict"):

    pred = model.predict([[sl, sw, pl, pw]])

    flowers = ["Setosa", "Versicolor", "Virginica"]

    result = flowers[pred[0]]

    st.success("Predicted Flower: " + result)

    # Visualization of entered values
    st.subheader("Input Feature Visualization")

    input_data = pd.DataFrame({
        "Feature": [
            "Sepal Length",
            "Sepal Width",
            "Petal Length",
            "Petal Width"
        ],
        "Value": [sl, sw, pl, pw]
    })

    st.bar_chart(
        input_data.set_index("Feature")
    )

    # Extra visual prediction: show the current input against all Iris species
    st.subheader("Visual Prediction - Iris Species Scatter Plot")

    fig_prediction, ax_prediction = plt.subplots(figsize=(10, 6))

    for species in df["Species"].unique():
        species_data = df[df["Species"] == species]
        ax_prediction.scatter(
            species_data["Sepal Length"],
            species_data["Petal Length"],
            label=species.capitalize(),
            alpha=0.7,
            s=50
        )

    ax_prediction.scatter(
        sl,
        pl,
        color="red",
        marker="X",
        s=200,
        edgecolors="black",
        linewidths=1,
        label="Current Input",
        zorder=5
    )

    ax_prediction.set_xlabel("Sepal Length (cm)")
    ax_prediction.set_ylabel("Petal Length (cm)")
    ax_prediction.set_title("Iris Species Classification - Current Input")
    ax_prediction.legend()
    ax_prediction.grid(alpha=0.2)
    st.pyplot(fig_prediction, use_container_width=True)
    plt.close(fig_prediction)


# Dataset visualization
st.subheader("Iris Dataset Visualization")

# Select features
feature = st.selectbox(
    "Select feature for visualization",
    [
        "Sepal Length",
        "Sepal Width",
        "Petal Length",
        "Petal Width"
    ]
)

# Create graph
fig, ax = plt.subplots()

for species in df["Species"].unique():

    data = df[df["Species"] == species]

    ax.scatter(
        data.index,
        data[feature],
        label=species
    )

ax.set_xlabel("Sample")
ax.set_ylabel(feature)
ax.set_title(feature + " Distribution")
ax.legend()

st.pyplot(fig)