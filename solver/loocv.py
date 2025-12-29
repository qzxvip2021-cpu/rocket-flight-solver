import numpy as np
from .models import build_height_model


def loocv_height_model(df):
    errors = []

    for i in range(len(df)):
        train = df.drop(index=i)
        test = df.iloc[i]

        beta, _, m0 = build_height_model(train)

        m_c = test["liftoff_mass"] - m0
        rho_ratio = test["rho_ratio"]

        X_test = np.array([1.0, m_c, rho_ratio])
        H_pred = float(X_test @ beta)

        errors.append(H_pred - test["apogee"])

    errors = np.array(errors)
    rmse = np.sqrt(np.mean(errors ** 2))

    return errors, rmse
