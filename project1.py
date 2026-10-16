import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

df = pd.read_excel("Concrete_Data.xls")
test_ind = (500, 630)

def splitdf(data, test_index: tuple):
    return data.iloc[test_index[0]:test_index[1]], data.drop(data.index[test_index[0]:test_index[1]])

def grad_descent_train_old(data, target: int, xvars: set, learning_rate: float, num_iterations: int):
    row, col = data.shape
    num_samples = row
    for i in xvars:
        if i < 0 or i >= col or i == target: raise ValueError("Invalid Col Index in Set:Xvars")
    if target < 0 or target >= col: raise ValueError("Invalid Target Index")
    num_features = len(xvars)

    x = data.iloc[:, list(xvars)]
    y = data.iloc[:, target]

    weights = np.zeros(num_features)
    bias = 0

    r2loss_history = []

    for i in range(num_iterations):
        y_pred = x @ weights + bias

        errors = y_pred - y

        r2loss = 1 - np.sum(errors ** 2) / np.sum((y - y.mean()) ** 2)
        r2loss_history.append(r2loss)

        dw = (2 / num_samples) * (x.T @ errors)
        db = (2 / num_samples) * np.sum(errors)

        weights -= learning_rate * dw
        bias -= learning_rate * db

    return weights, bias, r2loss_history

def grad_descent_train(data, target: int, xvars: set, learning_rate: float = 0.01,
                         num_iterations: int = 3000, tolerance: float = 1e-10, patience: int = 200,
                         beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8, ):
    row, col = data.shape

    if target < 0 or target >= col:
        raise ValueError("Invalid Target Index")

    if any(i < 0 or i >= col or i == target for i in xvars):
        raise ValueError("Invalid Col Index in Set:Xvars")

    if not xvars:
        raise ValueError("xvars must contain at least one feature")

    if learning_rate <= 0 or num_iterations < 1:
        raise ValueError("learning_rate and num_iterations must be positive")

    if tolerance < 0 or patience < 1:
        raise ValueError("tolerance must be nonnegative and patience positive")

    feature_indices = sorted(xvars)
    X = data.iloc[:, feature_indices].to_numpy(dtype=float)
    y = data.iloc[:, target].to_numpy(dtype=float)

    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Data must not contain NaN or infinite values")

    num_samples, num_features = X.shape

    weights = np.zeros(num_features)
    bias = 0.0

    m_w = np.zeros(num_features)
    v_w = np.zeros(num_features)
    m_b = 0.0
    v_b = 0.0

    r2loss_history = []
    best_mse = np.inf
    best_weights = weights.copy()
    best_bias = bias
    stale_iterations = 0

    y_mean = y.mean()
    denominator = np.sum((y - y_mean) ** 2)

    for step in range(1, num_iterations + 1):

        y_pred = X @ weights + bias
        errors = y_pred - y

        mse = np.mean(errors ** 2)

        if denominator > 0:
            r2 = 1.0 - np.sum(errors ** 2) / denominator
        else:
            r2 = 1.0 if np.sum(errors ** 2) == 0 else 0.0

        r2loss_history.append(r2)

        improvement_threshold = tolerance * max(1.0, best_mse if np.isfinite(best_mse) else 1.0)

        if mse < best_mse - improvement_threshold:
            best_mse = mse
            best_weights = weights.copy()
            best_bias = bias
            stale_iterations = 0
        else:
            stale_iterations += 1

        if stale_iterations >= patience:
            break

        dw = (2.0 / num_samples) * (X.T @ errors)
        db = 2.0 * np.mean(errors)

        m_w = beta1 * m_w + (1.0 - beta1) * dw
        v_w = beta2 * v_w + (1.0 - beta2) * (dw ** 2)

        m_b = beta1 * m_b + (1.0 - beta1) * db
        v_b = beta2 * v_b + (1.0 - beta2) * (db ** 2)

        m_w_hat = m_w / (1.0 - beta1 ** step)
        v_w_hat = v_w / (1.0 - beta2 ** step)

        m_b_hat = m_b / (1.0 - beta1 ** step)
        v_b_hat = v_b / (1.0 - beta2 ** step)

        weights -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
        bias -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + epsilon)

    return best_weights, best_bias, r2loss_history

def modelstats(train_data, test_data, target:int, xvars: set, weights: np.ndarray, bias: float, precision: int = 3):
    row, col = train_data.shape
    for i in xvars:
        if i < 0 or i >= col or i == target: raise ValueError("Invalid Col Index in Set:Xvars")
    if target < 0 or target >= col: raise ValueError("Invalid Target Index")

    xtrain = train_data.iloc[:, list(xvars)]
    ytrain = train_data.iloc[:, target]
    ytrain_pred = xtrain @ weights + bias

    xtest = test_data.iloc[:, list(xvars)]
    ytest = test_data.iloc[:, target]
    ytest_pred = xtest @ weights + bias

    trainMSE = np.mean((ytrain - ytrain_pred) ** 2)
    testMSE = np.mean((ytest - ytest_pred) ** 2)

    trainR2 = 1 - (np.sum((ytrain - ytrain_pred) ** 2) / np.sum((ytrain - np.mean(ytrain)) ** 2))
    testR2 = 1 - (np.sum((ytest - ytest_pred) ** 2) / np.sum((ytest - np.mean(ytest)) ** 2))

    return (round(trainMSE, precision), round(testMSE, precision),
            round(trainR2, precision), round(testR2, precision))

def get_pvalues(data, target: int, xvars: set):
    x = data.iloc[:, list(xvars)]
    y = data.iloc[:, target]

    x = sm.add_constant(x)

    model = sm.OLS(y, x).fit()

    return model.pvalues.drop("const")

SHOW_VARIABLE_DATA = True
DO_MULTIVAR_TEST = True
DO_SINGLEVAR_TEST = True
SHOW_HISTORY_PLOT = False

if __name__ == "__main__":
    xset = {0, 1, 2, 3, 4, 5, 6, 7}
    ycol = 8

    test, train = splitdf(df, test_ind)

    if SHOW_VARIABLE_DATA:
        feature_names = ["Cement", "Slag", "Fly Ash", "Water",
                         "Superplasticizer", "Coarse Agg", "Fine Agg", "Age"]

        # Table 1: Descriptive statistics for each variable
        variable_data = df.iloc[:, sorted(xset)].copy()
        variable_data.columns = feature_names

        stats_table = variable_data.agg(
            ["mean", "std", "min", "max"]
        ).T

        stats_table.columns = ["Mean", "Stdev", "Min", "Max"]

        print("\nVariable Descriptive Statistics:")
        print(stats_table.round(4))

        # Table 2: Pairwise correlations between variables
        corr_table = variable_data.corr()

        print("\nPairwise Variable Correlations:")
        print(corr_table.round(4))

    if DO_MULTIVAR_TEST:
        multivar_iters = 5000

        w, b, lh = grad_descent_train(data=train, target=ycol, xvars=xset,
                                      learning_rate=0.01, num_iterations=multivar_iters)
        statsbase = modelstats(train_data=train, test_data=test,
                               target=ycol, xvars=xset,
                               weights=w, bias=b)
        basemodel = LinearRegression()
        basemodel.fit(train.iloc[:, list(xset)], train.iloc[:, ycol])
        optimalstatsbase = modelstats(train_data=train, test_data=test,
                                      target=ycol, xvars=xset,
                                      weights=basemodel.coef_, bias=basemodel.intercept_)
        pvalues_base = get_pvalues(data=train, target=ycol, xvars=xset)

        scaled = df.copy().astype(float)
        scaled.iloc[:, 0:8] = (scaled.iloc[:, 0:8] - scaled.iloc[:, 0:8].mean()) / scaled.iloc[:, 0:8].std()
        scaled_test, scaled_train = splitdf(scaled, test_ind)
        sw, sb, slh = grad_descent_train(data=scaled_train, target=ycol, xvars=xset,
                                         learning_rate=0.01, num_iterations=multivar_iters)
        statsscaled = modelstats(train_data=scaled_train, test_data=scaled_test,
                                 target=ycol, xvars=xset,
                                 weights=sw, bias=sb)
        scaledmodel = LinearRegression()
        scaledmodel.fit(scaled_train.iloc[:, list(xset)], scaled_train.iloc[:, ycol])
        optimalstatsscaled = modelstats(train_data=scaled_train, test_data=scaled_test,
                                        target=ycol, xvars=xset,
                                        weights=scaledmodel.coef_, bias=scaledmodel.intercept_)
        pvalues_scaled = get_pvalues(data=scaled_train, target=ycol, xvars=xset)

        log = df.copy().astype(float)
        log.iloc[:, 0:8] = np.log1p(log.iloc[:, 0:8])
        log_test, log_train = splitdf(log, test_ind)
        lw, lb, llh = grad_descent_train(data=log_train, target=ycol, xvars=xset,
                                         learning_rate=0.01, num_iterations=multivar_iters)
        statslog = modelstats(train_data=log_train, test_data=log_test,
                              target=ycol, xvars=xset,
                              weights=lw, bias=lb)
        logmodel = LinearRegression()
        logmodel.fit(log_train.iloc[:, list(xset)], log_train.iloc[:, ycol])
        optimalstatslog = modelstats(train_data=log_train, test_data=log_test,
                                     target=ycol, xvars=xset,
                                     weights=logmodel.coef_, bias=logmodel.intercept_)
        pvalues_log = get_pvalues(data=log_train, target=ycol, xvars=xset)

        tableMR = pd.DataFrame(
            data=[statsbase, optimalstatsbase, statsscaled, optimalstatsscaled, statslog, optimalstatslog],
            columns=["Training MSE", "Test MSE", "Training R2", "Test R2"],
            index=["Unmodified", "Unmodified (Opt)",
                   "Standardized", "Standardized (Opt)",
                   "Logarithmic", "Logarithmic (Opt)"]
        )
        print("\nMSE and R^2 Comparisons:")
        print(tableMR)

        feature_names = ["Cement", "Slag", "Fly Ash", "Water",
                         "Superplasticizer", "Coarse Agg", "Fine Agg", "Age"]

        tableCoef = pd.DataFrame(
            data=[
                list(w) + [b],
                list(basemodel.coef_) + [basemodel.intercept_],
                list(sw) + [sb],
                list(scaledmodel.coef_) + [scaledmodel.intercept_],
                list(lw) + [lb],
                list(logmodel.coef_) + [logmodel.intercept_]
            ],
            columns=feature_names + ["Beta"],
            index=[
                "Unmodified",
                "Unmodified (Opt)",
                "Standardized",
                "Standardized (Opt)",
                "Logarithmic",
                "Logarithmic (Opt)"
            ]
        )

        print("\nMultivariate Model Coefficients:")
        print(tableCoef.round(4))

    if DO_SINGLEVAR_TEST:
        models = []
        mcoefs = []
        optmodels = []
        optcoefs = []
        maxR2 = []
        for i in range(len(xset)):
            models.append(grad_descent_train(data=train, target=ycol, xvars={i},
                                             learning_rate=0.01, num_iterations=5000))
            svmodel = LinearRegression()
            svmodel.fit(train.iloc[:, [i]], train.iloc[:, ycol])
            optmodels.append((svmodel.coef_, svmodel.intercept_))
        for model in models:
            maxR2.append(max(model[2]))
            mcoefs.append(model[0][0])
        for model in optmodels:
            optcoefs.append(model[0][0])

        svtable = pd.DataFrame(
            data={"GD": mcoefs,
                  "Optimized": optcoefs},
            index=["Cement", "Slag", "Fly Ash", "Water",
                   "Superplasticizer", "Coarse Agg", "Fine Agg", "Age"]
        )
        print("\nSingle Variable Model Coefficients:")
        print(svtable)

    if DO_MULTIVAR_TEST and DO_SINGLEVAR_TEST:
        pvalue_table = pd.DataFrame(
            data={"Untransformed": pvalues_base.values,
                  "Standardized": pvalues_scaled.values,
                  "Logarithmic": pvalues_log.values,
                  "SingleVar R2": maxR2},
            index=["Cement", "Slag", "Fly Ash", "Water",
                   "Superplasticizer", "Coarse Agg", "Fine Agg", "Age"]
        )

        pvalue_table = pvalue_table.round(4)
        print("\nVariable P-Values:")
        print(pvalue_table)

    if DO_MULTIVAR_TEST and SHOW_HISTORY_PLOT:
        plt.plot(lh, label="Untransformed")
        plt.plot(slh, label="Standardized")
        plt.plot(llh, label="Logarithmic")
        plt.xlabel("Iteration")
        plt.ylabel(r"$R^2$")
        plt.ylim(bottom=-1, top=1)
        plt.legend()
        plt.show()
