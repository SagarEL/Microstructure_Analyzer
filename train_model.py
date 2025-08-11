import numpy as np
from sklearn.linear_model import LinearRegression  # or any model you want
import joblib

# Example training data (replace with your real features and targets)
X = np.array([
    [5.0, 1.2],  # [mean_equivalent_diameter, mean_aspect_ratio]
    [7.1, 1.1],
    [6.3, 1.3],
    [8.0, 1.0]
])
y = np.array([100, 120, 110, 130])  # Example material property (replace with your data)

# Train the model
model = LinearRegression()
model.fit(X, y)

# Save the trained model as model.pkl (this creates a binary file)
joblib.dump(model, "model.pkl")
print("Model saved as model.pkl")