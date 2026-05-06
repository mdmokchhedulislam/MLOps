# load model 

import joblib
import pandas as pd

model = joblib.load("model.pkl")

# print("model pkl is", model)


new_data = pd.DataFrame({
    "hours": [15]
})

prediction = model.predict(new_data)

print("Prediction:", prediction)


# MLOps Course
# Day 6: FastAPI ML API
# MLOps Beginner to API Course Day 6 continue”







