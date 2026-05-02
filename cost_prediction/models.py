from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

def get_trained_models(X_train, y_train):
    """
    Trains the models using the training set.
    """
    model1 = LinearRegression()
    model1.fit(X_train, y_train)

    model2 = DecisionTreeRegressor(max_depth=4, random_state=42)
    model2.fit(X_train, y_train)

    model3 = RandomForestRegressor(max_depth=6, random_state=42)
    model3.fit(X_train, y_train)
    
    return {
        "Linear Regression": model1,
        "Decision Tree": model2,
        "Random Forest": model3
    }

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluates and returns accuracy metrics.
    """
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred) * 100
    
    print(f"Model: {model_name}")
    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2 Score: {r2:.2f}%")
    print("-" * 30)
    
    return {"mae": mae, "rmse": rmse, "r2": r2}