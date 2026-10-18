# Linear Regression --- Complete Machine Learning Guide

> A deep, practical, mathematics-first guide to linear regression, from
> intuition and equations to implementation, diagnostics, and
> deployment.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![NumPy](https://img.shields.io/badge/NumPy-supported-013243)
![scikit--learn](https://img.shields.io/badge/scikit--learn-supported-F7931E)
![License](https://img.shields.io/badge/License-MIT-green)

## Table of Contents

-   [1. Learning goals](#1-learning-goals)
-   [2. What is regression?](#2-what-is-regression)
-   [3. Linear regression intuition](#3-linear-regression-intuition)
-   [4. Mathematical notation](#4-mathematical-notation)
-   [5. Simple linear regression](#5-simple-linear-regression)
-   [6. Multiple linear regression](#6-multiple-linear-regression)
-   [7. How the best-fit line is
    learned](#7-how-the-best-fit-line-is-learned)
-   [8. Ordinary least squares](#8-ordinary-least-squares)
-   [9. Cost and loss functions](#9-cost-and-loss-functions)
-   [10. Gradient descent](#10-gradient-descent)
-   [11. Evaluation metrics](#11-evaluation-metrics)
-   [12. Assumptions and diagnostics](#12-assumptions-and-diagnostics)
-   [13. Data preparation](#13-data-preparation)
-   [14. Feature engineering](#14-feature-engineering)
-   [15. Regularization](#15-regularization)
-   [16. Polynomial regression](#16-polynomial-regression)
-   [17. From-scratch implementation](#17-from-scratch-implementation)
-   [18. scikit-learn implementation](#18-scikit-learn-implementation)
-   [19. End-to-end house-price
    example](#19-end-to-end-house-price-example)
-   [20. Graphs and visual
    diagnostics](#20-graphs-and-visual-diagnostics)
-   [21. Interpreting coefficients](#21-interpreting-coefficients)
-   [22. Statistical inference](#22-statistical-inference)
-   [23. Common mistakes](#23-common-mistakes)
-   [24. Debugging guide](#24-debugging-guide)
-   [25. Practice exercises](#25-practice-exercises)
-   [26. Interview questions](#26-interview-questions)
-   [27. Quick revision sheet](#27-quick-revision-sheet)
-   [28. Glossary](#28-glossary)
-   [29. Project checklist](#29-project-checklist)
-   [30. Further reading](#30-further-reading)

## How to use this guide

Read the conceptual sections first, then reproduce every code block in a
notebook. Change the data, deliberately introduce mistakes, and inspect
how the outputs change. Do not memorize formulas without knowing what
each symbol represents.

This document is intentionally organized as a reference manual. It is
long because the goal is to explain the reasoning, implementation,
limitations, and interpretation---not just provide a short definition.

## 1. Learning goals

By the end, you should be able to:

1.  Explain regression and distinguish it from classification.
2.  Write a simple and multiple linear regression equation.
3.  Interpret slope, intercept, coefficients, predictions, and
    residuals.
4.  Derive the ordinary least squares solution.
5.  Explain MSE, RMSE, MAE, and the coefficient of determination.
6.  Implement batch gradient descent with NumPy.
7.  Train, evaluate, and inspect a model with scikit-learn.
8.  Detect common assumption violations using diagnostic plots.
9.  Prevent data leakage during preprocessing and validation.
10. Communicate model performance and limitations honestly.

## 2. What is regression?

Regression is a supervised-learning task in which a model learns a
relationship between input features and a numeric target. Examples
include predicting a house price, delivery duration, temperature, sales
amount, or energy usage.

A regression model does not have to produce a straight line.
"Regression" names the type of prediction task; "linear" describes how
the model is linear in its coefficients. Polynomial regression, for
example, can create a curved relationship while remaining linear in its
fitted coefficients.

### Regression versus classification

  -----------------------------------------------------------------------
  Property                Regression              Classification
  ----------------------- ----------------------- -----------------------
  Typical target          Continuous numeric      Category or class
                          value                   

  Example                 Predict a home's sale   Predict whether an
                          price                   email is spam

  Example output          245000.0                `spam`

  Common metrics          MAE, RMSE, R²           Accuracy, precision,
                                                  recall, F1

  Common models           Linear regression,      Logistic regression,
                          random forest regressor decision tree
                                                  classifier
  -----------------------------------------------------------------------

A numeric-looking target does not automatically make a task regression.
Predicting a postal code is usually classification because the number is
a category label rather than a quantity with meaningful numeric
distance.

## 3. Linear regression intuition

Imagine collecting data for several houses. For every house, record its
area and its selling price. A scatter plot shows each observed house as
a point. Linear regression finds a line that summarizes the relationship
between area and price.

The line is not expected to pass through every point. Real data contains
noise, omitted variables, measurement errors, and natural variation. The
model instead tries to make predictions that are good according to a
chosen objective.

For a simple model:

\[ \\hat y = b_0 + b_1 x \]

-   (x): input feature, such as area.
-   (`\hat`{=tex}y): predicted target, such as price.
-   (b_0): intercept.
-   (b_1): slope or coefficient.

If (b_1) is positive, the fitted prediction increases as (x) increases.
If it is negative, the prediction decreases as (x) increases. If it is
close to zero, the fitted linear relationship is weak in that feature's
units, though the feature may still matter in nonlinear or interaction
effects.

### An illustrative graph

The following points are made-up teaching data, not a real market
estimate. The fitted line represents the general upward trend.

``` python
import matplotlib.pyplot as plt

area = [500, 700, 900, 1100, 1300, 1500]
price = [120, 150, 185, 210, 250, 275]  # illustrative price units

plt.scatter(area, price, label="Observed houses")
plt.plot([500, 1500], [115, 280], label="Illustrative fitted line")
plt.xlabel("Area (square feet)")
plt.ylabel("Price (illustrative units)")
plt.title("Simple linear regression: area vs. price")
plt.legend()
plt.grid(alpha=0.25)
plt.show()
```

The line in this small visualization is illustrative. In a real
workflow, estimate the line from training data rather than hand-writing
its endpoints.

## 4. Mathematical notation

  Symbol                               Meaning
  ------------------------------------ ---------------------------------------------------
  \(m\) or (n)                         Number of observations
  \(p\)                                Number of input features
  (x_i)                                Input value or feature vector for observation (i)
  (y_i)                                Actual target for observation (i)
  (`\hat`{=tex}y_i)                    Prediction for observation (i)
  (b_0)                                Intercept
  (b_j)                                Coefficient for feature (j)
  (e_i) or (r_i)                       Residual, (y_i-`\hat`{=tex}y_i)
  \(X\)                                Design matrix
  (`\mathbf`{=tex}y)                   Target vector
  (`\boldsymbol`{=tex}`\beta`{=tex})   Coefficient vector
  \(J\) or (L)                         Objective/loss function
  (`\alpha`{=tex}) or (`\eta`{=tex})   Learning rate, depending on convention

Notation differs between books and libraries. In statistics,
coefficients are often written as (`\beta`{=tex}). In code, `coef_` and
`intercept_` are the usual scikit-learn attributes.

### Actual values, predictions, errors, and residuals

The actual value is observed in the dataset. The prediction is produced
by the model. A residual is the observed value minus the prediction:

\[ r_i = y_i - `\hat`{=tex}y_i \]

-   Positive residual: the model predicted too low.
-   Negative residual: the model predicted too high.
-   Zero residual: prediction exactly matches the observation.

The residual is not the same as the model's unknown true error term in a
statistical population. Residuals are calculated from the fitted sample;
error terms are theoretical quantities in a model.

## 5. Simple linear regression

Simple linear regression uses one input feature and one numeric target:

\[ `\hat`{=tex}y_i = b_0 + b_1x_i \]

### Slope

The slope describes the expected change in the fitted target for a
one-unit increase in the feature, holding the model form fixed.

If the slope is (2.5) price units per square foot, then a
100-square-foot increase corresponds to a fitted increase of (250) price
units. This is a statement about the model's prediction, not
automatically a causal claim.

### Intercept

The intercept is the prediction when (x=0). It can be useful
mathematically even when zero is outside the observed data range. If a
house with zero area is impossible, the intercept may not have a
practical interpretation.

### Example

Suppose the fitted equation is:

\[ `\hat`{=tex}y = 40 + 0.18x \]

For (x=1000):

\[ `\hat`{=tex}y = 40 + 0.18(1000) = 220 \]

The model predicts 220 target units. The result's units depend on the
target and feature units.

### What makes it linear?

The model is linear in its coefficients. The following is linear
regression:

\[ `\hat`{=tex}y = b_0 + b_1x_1 + b_2x_2 \]

This can also be linear regression:

\[ `\hat`{=tex}y = b_0 + b_1x + b_2x\^2 \]

The second equation is curved as a function of (x), but remains linear
in (b_0,b_1,b_2). It is often called polynomial regression.

## 6. Multiple linear regression

Multiple linear regression uses two or more input features:

\[ `\hat`{=tex}y = b_0 + b_1x_1 + b_2x_2 + `\cdots`{=tex}+ b_px_p \]

For house-price prediction, features could include area, number of
bedrooms, age, distance to transit, and location indicators.

### Coefficient interpretation

A coefficient is interpreted as the model's predicted change for a
one-unit increase in that feature while holding the other included
features constant. This is sometimes called a "ceteris paribus"
interpretation.

For example:

\[ `\hat{price}`{=tex}=50000+120(area)+8000(bedrooms)-900(age) \]

-   Area coefficient: +120 price units for each area unit, holding the
    other listed variables constant.
-   Bedroom coefficient: +8000 units per bedroom, holding area and age
    constant.
-   Age coefficient: -900 units per year, holding area and bedrooms
    constant.

These values are fictional and only demonstrate interpretation.
Correlated features can make coefficient estimates unstable, and a
coefficient should not be interpreted causally without an appropriate
research design and assumptions.

### Matrix representation

For (n) observations and (p) features:

\[ `\mathbf`{=tex}y = X`\boldsymbol`{=tex}`\beta`{=tex}+
`\boldsymbol`{=tex}`\epsilon`{=tex} \]

If the intercept is included, the first column of (X) is usually a
column of ones. The dimensions are:

-   (X): (n `\times`{=tex}(p+1))
-   (`\boldsymbol`{=tex}`\beta`{=tex}): ((p+1) `\times 1`{=tex})
-   (`\mathbf`{=tex}y): (n `\times 1`{=tex})

This shape check is useful when implementing the algorithm with NumPy.

## 7. How the best-fit line is learned

The model needs a rule for choosing coefficients. Ordinary least squares
chooses coefficients that minimize the sum of squared residuals:

\[ SSE = `\sum`{=tex}\_{i=1}^{n}(y_i-`\hat`{=tex}y_i)^2 \]

Squaring makes positive and negative residuals unable to cancel each
other. It also penalizes large errors more strongly than small errors.

The fitting procedure is not simply "draw a line through the points." It
is an optimization problem: select the coefficients that produce the
lowest objective on the training observations.

### Why not fit every point?

Real observations rarely lie exactly on one line. A line through every
point may not exist, and fitting noise can harm generalization. The goal
is to estimate a useful pattern, not memorize each observation.

### Training and generalization

Training performance describes fit to data used for learning. Test
performance estimates performance on held-out data from a suitable
target population. A very low training error does not guarantee low test
error.

## 8. Ordinary least squares

For simple linear regression with an intercept, the ordinary least
squares slope and intercept have closed-form expressions.

Let:

\[
`\bar`{=tex}{x}=`\frac{1}{n}`{=tex}`\sum`{=tex}*{i=1}\^{n}x_i,`\qquad`{=tex}
`\bar`{=tex}{y}=`\frac{1}{n}`{=tex}`\sum`{=tex}*{i=1}\^{n}y_i \]

Then:

\[ b_1 = `\frac{\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y})}`{=tex}
{`\sum`{=tex}\_{i=1}^{n}(x_i-`\bar`{=tex}{x})^2} \]

and:

\[ b_0 = `\bar`{=tex}{y}-b_1`\bar`{=tex}{x} \]

The slope compares how (x) and (y) vary together with how much (x)
varies. The intercept positions the line so it passes through
((`\bar`{=tex}{x},`\bar`{=tex}{y})).

### Matrix solution

When (X\^TX) is invertible, the ordinary least squares solution can be
written:

\[ `\hat{\boldsymbol\beta}`{=tex}=(X^TX)^{-1}X\^T`\mathbf`{=tex}y \]

In numerical code, do not usually compute the inverse explicitly. Use a
stable solver such as `numpy.linalg.lstsq`, QR decomposition, or
singular value decomposition. If features are collinear, (X\^TX) may be
singular or poorly conditioned.

### Important distinction

The closed-form solution is a mathematical result. Gradient descent is
an iterative optimization method. Both can minimize the same
least-squares objective, but their numerical behavior and computational
costs differ.

## 9. Cost and loss functions

A loss measures the error for one example or a collection of examples. A
cost/objective often refers to the aggregated quantity optimized during
training. Terminology varies by source.

### Mean squared error

\[ MSE =
`\frac{1}{n}`{=tex}`\sum`{=tex}\_{i=1}^{n}(y_i-`\hat`{=tex}y_i)^2 \]

MSE is nonnegative and measured in squared target units. Its derivative
is convenient, but large residuals receive disproportionately high
weight.

### Sum of squared errors

\[ SSE = `\sum`{=tex}\_{i=1}^{n}(y_i-`\hat`{=tex}y_i)^2 \]

SSE scales with sample size. MSE divides by the number of observations.
For a fixed dataset, minimizing SSE and MSE gives the same coefficients.

### Mean absolute error

\[ MAE =
`\frac{1}{n}`{=tex}`\sum`{=tex}\_{i=1}\^{n}\|y_i-`\hat`{=tex}y_i\| \]

MAE is measured in the target's original units and is less sensitive to
large residuals than MSE. Absolute loss is not differentiable at zero,
although optimization methods can handle it.

### Root mean squared error

\[ RMSE = `\sqrt{MSE}`{=tex} \]

RMSE is in the same units as the target. It remains more sensitive to
large residuals than MAE.

### Which metric should you choose?

Choose a metric based on the cost of errors and the use case. If large
mistakes are especially costly, RMSE may be informative. If typical
absolute deviation is easier to explain, MAE can be useful. Report more
than one metric when they answer different questions, and always compare
against a baseline.

## 10. Gradient descent

Gradient descent starts with coefficient values and repeatedly updates
them in the direction that reduces the objective.

For parameters (`\theta`{=tex}):

\[
`\theta`{=tex}`\leftarrow`{=tex}`\theta`{=tex}-`\alpha`{=tex}`\frac{\partial J}{\partial \theta}`{=tex}
\]

Here (`\alpha`{=tex}) is the learning rate. A learning rate that is too
small can make progress very slow. A learning rate that is too large can
overshoot the minimum or diverge.

For linear regression with MSE and an intercept:

\[ `\frac{\partial J}{\partial b_0}`{=tex}
=-`\frac{2}{n}`{=tex}`\sum`{=tex}\_i(y_i-`\hat`{=tex}y_i) \]

\[ `\frac{\partial J}{\partial b_j}`{=tex}
=-`\frac{2}{n}`{=tex}`\sum`{=tex}*i x*{ij}(y_i-`\hat`{=tex}y_i) \]

The update uses all training examples in batch gradient descent. Some
implementations define the cost as (1/(2n)) times SSE, which removes the
factor of 2 from the gradients. Both conventions are valid if used
consistently.

### Gradient descent workflow

1.  Initialize coefficients.
2.  Compute predictions.
3.  Compute residuals.
4.  Compute the objective.
5.  Compute gradients.
6.  Update coefficients.
7.  Repeat until a stopping criterion is reached.
8.  Inspect the loss history and validate on unseen data.

### Convergence checks

Common stopping conditions include a maximum number of iterations, a
small change in objective, a small gradient norm, or a small parameter
update. A fixed iteration count alone does not prove that the optimizer
converged.

## 11. Evaluation metrics

### Coefficient of determination ((R\^2))

For a test set, a common definition is:

\[ R\^2=1-`\frac{\sum_i(y_i-\hat y_i)^2}`{=tex}
{`\sum`{=tex}\_i(y_i-`\bar`{=tex}y)\^2} \]

The denominator measures variation around the target mean baseline. An
(R\^2) of 1 indicates perfect predictions on the evaluated data. An
(R\^2) of 0 means the model performs like the mean baseline under this
definition. Negative (R\^2) means it performs worse than that baseline.

**R² is not the percentage of predictions that are correct.** It is also
not proof that the model is causal, unbiased, or suitable for
deployment.

### Adjusted (R\^2)

A common formula for a model with (p) predictors and (n) observations
is:

\[ R^2\_{adj}=1-(1-R^2)`\frac{n-1}{n-p-1}`{=tex} \]

This formula assumes the usual intercept-based setting and (n\>p+1).
Adjusted (R\^2) penalizes adding predictors under its degrees-of-freedom
adjustment, but it is not a substitute for held-out evaluation.

### Baseline comparison

A `DummyRegressor` that predicts a simple training-set statistic can
show whether a complex model adds value. The baseline must be fitted
only on training data. Compare models on the same split or
cross-validation folds.

## 12. Assumptions and diagnostics

Linear regression can produce predictions without every classical
statistical assumption being true. However, assumptions matter for
interpreting coefficients, uncertainty estimates, and inference.

1.  **Linearity in parameters:** the conditional mean is represented
    adequately by the chosen features and coefficient form.
2.  **Independent errors/observations:** errors are not improperly
    dependent, given the data design.
3.  **Homoscedasticity:** error variance is constant across fitted
    values or relevant predictors.
4.  **No perfect multicollinearity:** one predictor is not an exact
    linear combination of other predictors.
5.  **Exogeneity / zero conditional mean:** predictors are not
    systematically related to the model error in a way that biases
    coefficients.
6.  **Normal errors:** primarily relevant for exact small-sample
    classical inference, not a requirement for computing least-squares
    predictions.

### Residual-versus-fitted plot

A random cloud around zero is generally more reassuring than a visible
curve or funnel. Curvature can indicate a missing nonlinear pattern. A
funnel can indicate changing variance. Clusters or runs can indicate
dependence or omitted group structure.

### Q--Q plot

A Q--Q plot compares residual quantiles with theoretical normal
quantiles. Strong tail departures may indicate non-normality or
outliers. A Q--Q plot should be interpreted alongside the sample size,
residual plot, and modeling purpose.

### Outliers and influential observations

An outlier has an unusual response or residual. A high-leverage point
has an unusual predictor configuration. An influential observation
meaningfully changes fitted results when included versus excluded. These
are related but not identical concepts.

Do not remove a point merely because it is inconvenient. First check for
data errors, explainable rare cases, changes in data collection, and
whether the point belongs to the target population.

## 13. Data preparation

A reliable model starts with a reliable dataset.

-   Define what one row represents.
-   Identify the target and candidate features.
-   Check data types and units.
-   Inspect missing values and duplicate records.
-   Check impossible values and inconsistent categories.
-   Separate training, validation, and test data.
-   Fit preprocessing steps on training data only.
-   Preserve a reproducible split or random seed.
-   Document data provenance and license.

### Data leakage

Leakage occurs when information unavailable at prediction time
influences training or evaluation. Examples include fitting a scaler on
the entire dataset before splitting, using a future outcome as a
feature, or selecting features based on the test labels.

Use a scikit-learn `Pipeline` and `ColumnTransformer` so transformations
are fitted inside each training fold.

## 14. Feature engineering

Feature engineering encodes useful information in a form the model can
learn.

-   Transform skewed variables when scientifically sensible.
-   Encode categories with one-hot encoding or another appropriate
    method.
-   Add interactions when the effect of one feature may depend on
    another.
-   Add polynomial terms when curvature is plausible.
-   Standardize features when numerical optimization or regularization
    benefits from comparable scales.
-   Keep units and transformations documented.

Feature engineering must respect prediction-time availability. It should
also be evaluated with cross-validation rather than selected based on
test-set performance.

## 15. Regularization

Regularization adds a penalty to discourage unnecessarily large
coefficients.

### Ridge regression (L2)

\[
J(`\boldsymbol`{=tex}`\beta`{=tex})=MSE+`\lambda`{=tex}`\sum`{=tex}\_{j=1}\^{p}`\beta`{=tex}\_j\^2
\]

Ridge shrinks coefficients toward zero but generally does not make them
exactly zero. The intercept is typically not penalized. Scaling is
important because the penalty depends on coefficient magnitude.

### Lasso regression (L1)

\[
J(`\boldsymbol`{=tex}`\beta`{=tex})=MSE+`\lambda`{=tex}`\sum`{=tex}\_{j=1}\^{p}\|`\beta`{=tex}\_j\|
\]

Lasso can set some coefficients exactly to zero, providing a form of
feature selection. With strongly correlated features, selected variables
can be unstable.

### Elastic Net

Elastic Net combines L1 and L2 penalties. It can be useful when there
are many correlated predictors and a sparse solution is desirable.

The parameter called `alpha` in scikit-learn's Ridge/Lasso APIs is a
regularization strength; it is not the learning rate used in the
gradient-descent equation above.

## 16. Polynomial regression

Polynomial regression expands the feature space:

\[ `\hat`{=tex}y=b_0+b_1x+b_2x^2+`\cdots`{=tex}+b_dx^d \]

The curve can fit nonlinear trends, but increasing degree increases
flexibility and the risk of overfitting. Polynomial features should be
learned within a pipeline and evaluated with validation data.

For multiple features, polynomial expansion can rapidly create many
terms, including interactions. Be careful about dimensional growth and
interpretation.

## 17. From-scratch implementation

The code below implements batch gradient descent using NumPy. It assumes
numeric, finite feature values and a one-dimensional target. Scaling
features often improves convergence.

``` python
import numpy as np

class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, n_iters=10_000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows.")
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("X and y must contain only finite values.")

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.n_iters):
            y_pred = X @ self.weights + self.bias
            errors = y_pred - y

            dw = (2 / n_samples) * (X.T @ errors)
            db = (2 / n_samples) * np.sum(errors)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            updated_pred = X @ self.weights + self.bias
            mse = np.mean((y - updated_pred) ** 2)
            self.loss_history.append(mse)

        return self

    def predict(self, X):
        if self.weights is None:
            raise ValueError("Call fit before predict.")
        X = np.asarray(X, dtype=float)
        return X @ self.weights + self.bias
```

### Example usage

``` python
X = np.array([[1], [2], [3], [4], [5]], dtype=float)
y = np.array([3, 5, 7, 9, 11], dtype=float)

model = LinearRegressionGD(learning_rate=0.01, n_iters=5000)
model.fit(X, y)

print("Weight:", model.weights)
print("Bias:", model.bias)
print("Prediction for x=6:", model.predict([[6]])[0])
```

The target data follows (y=2x+1), so a well-converged model should
estimate a slope near 2 and intercept near 1. Small differences can
occur because of the learning rate and iteration count.

### Plot loss history

``` python
import matplotlib.pyplot as plt

plt.plot(model.loss_history)
plt.xlabel("Iteration")
plt.ylabel("Training MSE")
plt.title("Gradient descent convergence")
plt.grid(alpha=0.25)
plt.show()
```

A steadily decreasing curve suggests progress. A flat curve can mean
convergence or an unsuitable setup. A rising or oscillating curve can
indicate a learning rate that is too large, poor scaling, or an
implementation error.

## 18. scikit-learn implementation

``` python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

X = np.array([[1], [2], [3], [4], [5], [6]], dtype=float)
y = np.array([3, 5, 7, 9, 11, 13], dtype=float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
print("MAE:", mean_absolute_error(y_test, predictions))
print("RMSE:", np.sqrt(mean_squared_error(y_test, predictions)))
print("R²:", r2_score(y_test, predictions))
```

For a tiny dataset, the test metrics can vary considerably with the
split. Use this as an API demonstration, not as evidence of production
performance.

## 19. End-to-end house-price example

This section shows the shape of a robust workflow. Replace the
placeholder column names with columns in your actual dataset.

``` python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("houses.csv")

target = "price"
X = df.drop(columns=[target])
y = df[target]

numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", Ridge(alpha=1.0)),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, pred))
print(f"MAE:  {mean_absolute_error(y_test, pred):,.2f}")
print(f"RMSE: {rmse:,.2f}")
print(f"R²:   {r2_score(y_test, pred):.3f}")
```

### Before using this workflow

-   Confirm the target column exists and is numeric.
-   Check whether the dataset contains time, household, or geographic
    groups that require a group-aware split.
-   Use a time-based split when the intended deployment predicts future
    observations.
-   Inspect the distribution and units of the target.
-   Compare with a baseline.
-   Evaluate on a final test set only after model choices are finished.
-   Check whether the model is appropriate for the consequences of an
    incorrect prediction.

## 20. Graphs and visual diagnostics

### Scatter plot with fitted line

For a one-feature model, plot training observations and the fitted line
over the feature range. Do not connect test observations in arbitrary
row order.

``` python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[1], [2], [3], [4], [5], [6]], dtype=float)
y = np.array([2.2, 4.1, 5.8, 8.2, 9.7, 12.1])

model = LinearRegression().fit(X, y)
x_line = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)

plt.scatter(X[:, 0], y, label="Observed")
plt.plot(x_line[:, 0], model.predict(x_line), label="Fitted line")
plt.xlabel("Feature x")
plt.ylabel("Target y")
plt.title("Observed data and fitted linear model")
plt.legend()
plt.grid(alpha=0.25)
plt.show()
```

### Residual plot

``` python
pred = model.predict(X)
residuals = y - pred

plt.scatter(pred, residuals)
plt.axhline(0, linestyle="--")
plt.xlabel("Fitted values")
plt.ylabel("Residuals (actual - predicted)")
plt.title("Residuals versus fitted values")
plt.grid(alpha=0.25)
plt.show()
```

### Actual versus predicted plot

``` python
plt.scatter(y, pred)
lower = min(y.min(), pred.min())
upper = max(y.max(), pred.max())
plt.plot([lower, upper], [lower, upper], linestyle="--")
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.title("Actual versus predicted")
plt.grid(alpha=0.25)
plt.show()
```

Points close to the diagonal indicate close predictions. This plot does
not by itself diagnose every assumption or establish generalization; use
held-out predictions for a more honest evaluation.

### Coefficient plot

For multiple features, show coefficients with feature names and, where
relevant, uncertainty intervals. Coefficients on different scales should
not be compared as measures of importance without care. Standardized
coefficients can aid comparison but change the interpretation.

## 21. Interpreting coefficients

A coefficient's meaning depends on the units, encoding, and other
features in the model.

-   If area is measured in square feet, changing the feature to square
    meters changes its numeric coefficient.
-   If the target is log-transformed, a coefficient is not a direct
    change in the original target units.
-   With one-hot encoding, each category coefficient is interpreted
    relative to a reference category, depending on the encoding.
-   With interactions, a main-effect coefficient is conditional on the
    interacting feature's reference or zero value.
-   With correlated features, individual coefficients can be unstable
    even when predictions are reasonable.

### Prediction is not causation

A model can predict well using correlations that are not causal. For
causal interpretation, one must address confounding, selection,
measurement, and study design. A fitted coefficient alone does not
establish what would happen if a feature were deliberately changed.

## 22. Statistical inference

Classical regression inference may report standard errors, confidence
intervals, and hypothesis tests for coefficients. These depend on
assumptions about the model and data collection.

-   A **standard error** describes estimated sampling variability of an
    estimate under a specified framework.
-   A **confidence interval** is produced by a procedure with a stated
    long-run coverage property under its assumptions.
-   A **p-value** measures how incompatible the data are with a null
    hypothesis under the assumed model; it is not the probability that
    the null hypothesis is true.
-   **Statistical significance** is not the same as practical
    importance.
-   **Prediction intervals** account for uncertainty about a future
    observation and are usually wider than confidence intervals for the
    mean response.

For rigorous inference, document assumptions, the sampling design,
missing-data handling, and any multiple-testing decisions. Do not use
ordinary standard errors blindly for clustered, repeated,
time-dependent, or heteroscedastic data.

## 23. Common mistakes

1.  **Evaluating only on training data.** Use held-out data or
    cross-validation.
2.  **Calling R² accuracy.** R² is not a classification accuracy
    percentage.
3.  **Ignoring units.** Coefficients and error metrics have units.
4.  **Assuming correlation proves causation.** Prediction and causal
    inference are different goals.
5.  **Scaling before splitting.** This leaks test information into
    preprocessing.
6.  **Using a random split for time series.** Future observations may
    leak into the past.
7.  **Removing outliers without investigation.** An unusual observation
    may be valid and important.
8.  **Adding every feature.** More predictors can increase variance,
    leakage, and interpretability problems.
9.  **Using gradient descent without checking convergence.** Plot the
    loss and inspect coefficients.
10. **Trusting one score.** Compare multiple metrics, baselines, and
    error slices.
11. **Ignoring extrapolation.** A line can produce implausible
    predictions far outside the training range.
12. **Treating a coefficient as causal.** Interpret it conditionally and
    in context.

## 24. Debugging guide

  --------------------------------------------------------------------------------
  Symptom            Possible cause     What to check
  ------------------ ------------------ ------------------------------------------
  Loss becomes `nan` Invalid data or    Finite values, learning rate, feature
                     exploding updates  scale

  Loss oscillates    Learning rate too  Reduce learning rate; scale features
                     large              

  Coefficients are   Units,             Feature scales, correlations, solver
  huge               collinearity, poor 
                     conditioning       

  Training score     Overfitting or     Split strategy, leakage, model complexity
  high, test score   distribution shift 
  low                                   

  Negative test R²   Worse than the     Baseline, split difficulty, target shift
                     test mean baseline 

  Predictions are    Weak signal, bad   Feature variance, coefficients, model form
  all similar        preprocessing,     
                     underfit           

  Shape mismatch     Wrong array        `X.shape`, `y.shape`, target flattening
                     dimensions         

  Category error at  Unseen category or `OneHotEncoder(handle_unknown="ignore")`
  prediction         inconsistent       
                     pipeline           

  Implausible        Extrapolation or   Feature range, target limits, alternative
  predictions        missing            model
                     constraints        

  Coefficients       Small sample or    Cross-validation, regularization,
  change across      correlated         uncertainty
  splits             features           
  --------------------------------------------------------------------------------

## 25. Practice exercises

Try to solve these without looking at the answers immediately.

1.  Given (`\hat`{=tex}y=5+3x), calculate the prediction at (x=7).
2.  If actual value is 50 and prediction is 43, calculate the residual.
3.  Why does squared error penalize a residual of 10 more than two
    residuals of 5 in total?
4.  What is the difference between SSE and MSE?
5.  Why is RMSE easier to interpret in the original target units?
6.  Can R² be negative on test data? Explain.
7.  Why can a high training R² coexist with poor test performance?
8.  What does a funnel shape in a residual plot suggest?
9.  Why is a feature scaler fitted only on training data?
10. When could a time-based split be more appropriate than a random
    split?
11. What is the difference between a residual and a theoretical error
    term?
12. Why can multicollinearity make coefficient interpretation difficult?
13. What does a learning rate do?
14. Why might standardized features help gradient descent?
15. What is the difference between Ridge and Lasso?
16. Why is polynomial regression still linear in its coefficients?
17. What does the intercept mean when zero is outside the observed
    feature range?
18. Why should a model be compared with a baseline?
19. Why is prediction not proof of causation?
20. What should you check before removing an outlier?

### Exercise answers

1.  (5+3(7)=26).
2.  Residual (=50-43=7), so the model underpredicted by 7.
3.  A residual of 10 contributes (100); two residuals of 5 contribute
    (25+25=50).
4.  SSE sums squared residuals; MSE averages them over observations.
5.  RMSE takes the square root of MSE, returning the target's units.
6.  Yes. It means the model's squared error exceeds that of the
    evaluated-set mean baseline under the standard definition.
7.  It may have overfit, leaked information, or encountered a different
    test distribution.
8.  Often changing error variance, called heteroscedasticity.
9.  To avoid using information from held-out observations during
    training.
10. When deployment predicts future observations and temporal order
    matters.
11. Residuals are computed from fitted sample predictions; error terms
    are theoretical deviations in the model.
12. Correlated predictors can trade off against each other, making
    estimates unstable.
13. It sets the step size of parameter updates.
14. Features on similar scales can improve optimization conditioning.
15. Ridge uses an L2 penalty; Lasso uses an L1 penalty and can set
    coefficients to zero.
16. The powers of (x) are treated as transformed input columns;
    coefficients enter linearly.
17. It is the model's mathematical prediction at zero, which may not be
    practically meaningful.
18. A baseline tests whether the model adds predictive value.
19. Correlations can arise from confounding or other noncausal
    structures.
20. Check data errors, context, target population, and influence on
    conclusions.

## 26. Interview questions

### Fundamentals

**Q1. What is linear regression?**\
A supervised-learning method that models a numeric target as a linear
combination of features and coefficients.

**Q2. What is the difference between simple and multiple linear
regression?**\
Simple regression uses one predictor; multiple regression uses two or
more.

**Q3. What does the slope mean?**\
It is the fitted change in target per one-unit change in a predictor,
conditional on the other predictors in a multiple regression.

**Q4. What does the intercept mean?**\
The fitted target when all predictors equal zero under the model's
encoding.

**Q5. Why use least squares?**\
It provides a tractable objective with a closed-form solution under
suitable conditions and convenient optimization properties.

### Optimization

**Q6. What is gradient descent?**\
An iterative method that updates parameters in the negative gradient
direction.

**Q7. What happens if the learning rate is too high?**\
Updates can overshoot the minimum, oscillate, or diverge.

**Q8. Why scale features?**\
Scaling can improve numerical conditioning and make optimization behave
more predictably.

**Q9. Does gradient descent always find the minimum for linear
regression?**\
For a convex least-squares objective, suitable learning rates and
conditions can lead to a global minimum; arbitrary settings do not
guarantee convergence.

### Evaluation

**Q10. What is MSE?**\
The average squared difference between actual and predicted targets.

**Q11. What is RMSE?**\
The square root of MSE, expressed in the target's units.

**Q12. What is R²?**\
A relative measure comparing residual sum of squares with variation
around the target mean baseline.

**Q13. Can R² be negative?**\
Yes, especially on held-out data when predictions are worse than the
mean baseline.

**Q14. Why is a baseline important?**\
It establishes a minimum standard for usefulness.

### Diagnostics and advanced concepts

**Q15. What is multicollinearity?**\
Strong linear relationships among predictors that can make individual
coefficient estimates unstable.

**Q16. What is heteroscedasticity?**\
Nonconstant conditional error variance.

**Q17. What is regularization?**\
Adding a penalty to the fitting objective to discourage overly large
coefficients.

**Q18. Ridge versus Lasso?**\
Ridge uses squared coefficient penalties; Lasso uses absolute-value
penalties and can produce sparse solutions.

**Q19. What is data leakage?**\
Using information during training or evaluation that would not
legitimately be available at prediction time.

**Q20. What is extrapolation?**\
Predicting outside the feature range represented in the training data,
where the learned relationship may not hold.

## 27. Quick revision sheet

  --------------------------------------------------------------------------
  Concept                        Remember
  ------------------------------ -------------------------------------------
  Simple model                   (`\hat`{=tex}y=b_0+b_1x)

  Multiple model                 (`\hat`{=tex}y=b_0+`\sum`{=tex}\_jb_jx_j)

  Residual                       Actual minus predicted

  SSE                            Sum of squared residuals

  MSE                            Mean squared residual

  RMSE                           Square root of MSE

  MAE                            Mean absolute residual

  R²                             Relative to a mean baseline; can be
                                 negative on test data

  OLS                            Minimizes squared residuals

  Gradient descent               Repeated gradient-based parameter updates

  Ridge                          L2 coefficient penalty

  Lasso                          L1 coefficient penalty

  Leakage                        Inappropriate access to held-out or future
                                 information

  Pipeline                       Fits preprocessing within training folds

  Residual plot                  Checks structure left unexplained by the
                                 fitted model
  --------------------------------------------------------------------------

## 28. Glossary

-   **Algorithm:** a defined procedure for solving a problem.
-   **Bias (intercept):** the constant term in the prediction equation.
    This is distinct from statistical bias.
-   **Coefficient:** a learned weight associated with a feature.
-   **Confounding:** a variable or structure that distorts an
    association of interest.
-   **Correlation:** a measure of association, not proof of causality.
-   **Design matrix:** matrix containing predictor values and usually an
    intercept column.
-   **Feature:** an input variable.
-   **Generalization:** performance on new data from the intended
    population.
-   **Gradient:** vector of partial derivatives of an objective.
-   **Homoscedasticity:** constant conditional error variance.
-   **Intercept:** prediction when encoded predictors are zero.
-   **Learning rate:** multiplier controlling gradient-descent step
    size.
-   **Loss:** numerical measure of prediction error.
-   **Multicollinearity:** linear dependence or strong association among
    predictors.
-   **Objective:** quantity minimized during fitting.
-   **Overfitting:** fitting sample-specific patterns that do not
    generalize.
-   **Parameter:** a quantity learned by the model.
-   **Prediction:** model output for an input.
-   **Regularization:** penalty used to constrain model complexity.
-   **Residual:** observed target minus fitted prediction.
-   **Target:** value the model is trained to predict.
-   **Training set:** data used to fit model parameters.
-   **Validation set:** data used to compare model choices.
-   **Test set:** held-out data reserved for final evaluation.
-   **Underfitting:** model too limited to capture important structure.
-   **Variance:** in the bias--variance context, sensitivity to the
    particular training sample.

## 29. Project checklist

### Problem framing

-   [ ] State the prediction target and prediction time.
-   [ ] Identify who will use the predictions.
-   [ ] Define an acceptable error level.
-   [ ] Identify potential harms from bad predictions.

### Data

-   [ ] Document source and license.
-   [ ] Inspect rows, columns, types, units, and missingness.
-   [ ] Check duplicates and impossible values.
-   [ ] Select a split strategy that matches deployment.
-   [ ] Keep test data separate from feature/model selection.

### Modeling

-   [ ] Create a baseline.
-   [ ] Build preprocessing in a pipeline.
-   [ ] Fit a simple linear model first.
-   [ ] Evaluate with suitable metrics.
-   [ ] Use cross-validation for model comparison.
-   [ ] Try regularization if useful.
-   [ ] Inspect residuals and error slices.

### Reporting and deployment

-   [ ] Explain metrics in target units.
-   [ ] Report limitations and known failure cases.
-   [ ] Save preprocessing and model together.
-   [ ] Validate input schema at prediction time.
-   [ ] Monitor data and error drift after deployment.
-   [ ] Define a retraining and rollback process.

## 30. Further reading

-   [scikit-learn: Linear
    Models](https://scikit-learn.org/stable/modules/linear_model.html)
-   [scikit-learn: Model
    Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
-   [scikit-learn: Pipelines and Composite
    Estimators](https://scikit-learn.org/stable/modules/compose.html)
-   [NumPy documentation](https://numpy.org/doc/)
-   [Matplotlib documentation](https://matplotlib.org/stable/)
-   [An Introduction to Statistical
    Learning](https://www.statlearning.com/)
-   [OpenIntro Statistics](https://www.openintro.org/book/os/)

## License and contribution

You may adapt this guide for personal learning and your own repository.
If you publish it, verify code and links in your environment, cite any
external material you add, and choose a license that matches the rights
you have to the complete repository.

------------------------------------------------------------------------

**Learning reminder:** understand the assumptions, validate on unseen
data, and interpret coefficients in context. A model is useful only when
its evaluation matches the way it will be used.

### Reference module 1: Choosing the target

**Core idea.** Define exactly what quantity the model predicts, its
unit, and when it is known.

**Why it matters.** A model is not only an equation. It is a complete
process involving data collection, preprocessing, fitting, validation,
interpretation, and use. Decisions in this area can change the estimated
coefficients and the real-world reliability of predictions.

**Common trap.** A target that is only available after the prediction
time creates leakage.

**Practical check.** Write one sentence: 'At time T, predict Y using
information available by time T.'

**Questions to ask** - What assumptions does this step make about the
data? - Could this step use information that would not be available at
prediction time? - How would I verify the decision with a plot, a
metric, or a controlled comparison? - Can I explain the consequence to
someone who does not know machine learning?

**Experiment** 1. Make a copy of a small dataset and record its initial
shape and target distribution. 2. Apply the technique or check described
above. 3. Compare the resulting features, coefficients, residuals, or
held-out metric. 4. Write down what changed and whether that change is
expected. 5. Repeat with a different random seed or split when the
experiment permits it.

**Reporting template** - Decision: describe the choice in one
sentence. - Rationale: explain why the choice fits the task. - Evidence:
name the plot, metric, or validation result used. - Limitation: describe
a way the choice could fail. - Next step: state what you would check
before deployment.

### Reference module 2: Defining one observation

**Core idea.** Decide whether one row represents a person, house, day,
transaction, or other unit.

**Why it matters.** A model is not only an equation. It is a complete
process involving data collection, preprocessing, fitting, validation,
interpretation, and use. Decisions in this area can change the estimated
coefficients and the real-world reliability of predictions.

**Common trap.** Repeated rows for one entity may make random splits
overly optimistic.

**Practical check.** Check whether rows share people, households,
devices, or time periods.

**Questions to ask** - What assumptions does this step make about the
data? - Could this step use information that would not be available at
prediction time? - How would I verify the decision with a plot, a
metric, or a controlled comparison? - Can I explain the consequence to
someone who does not know machine learning?

**Experiment** 1. Make a copy of a small dataset and record its initial
shape and target distribution. 2. Apply the technique or check described
above. 3. Compare the resulting features, coefficients, residuals, or
held-out metric. 4. Write down what changed and whether that change is
expected. 5. Repeat with a different random seed or split when the
experiment permits it.

**Reporting template** - Decision: describe the choice in one
sentence. - Rationale: explain why the choice fits the task. - Evidence:
name the plot, metric, or validation result used. - Limitation: describe
a way the choice could fail. - Next step: state what you would check
before deployment.

### Reference module 3: Sampling and representativeness

**Core idea.** The training sample should reflect the population where
predictions will be used.

**Why it matters.** A model is not only an equation. It is a complete
process involving data collection, preprocessing, fitting, validation,
interpretation, and use. Decisions in this area can change the estimated
coefficients and the real-world reliability of predictions.

**Common trap.** A sample collected from one neighborhood may not
generalize to another.

**Practical check.** Compare feature and target distributions across
training and deployment groups.

**Questions to ask** - What assumptions does this step make about the
data? - Could this step use information that would not be available at
prediction time? - How would I verify the decision with a plot, a
metric, or a controlled comparison? - Can I explain the consequence to
someone who does not know machine learning?

**Experiment** 1. Make a copy of a small dataset and record its initial
shape and target distribution. 2. Apply the technique or check described
above. 3. Compare the resulting features, coefficients, residuals, or
held-out metric. 4. Write down what changed and whether that change is
expected. 5. Repeat with a different random seed or split when the
experiment permits it.

**Reporting template** - Decision: describe the choice in one
sentence. - Rationale: explain why the choice fits the task. - Evidence:
name the plot, metric, or validation result used. - Limitation: describe
a way the choice could fail. - Next step: state what you would check
before deployment.

### Reference module 4: Missing values

**Core idea.** Missingness may be random, systematic, or informative.

**Why it matters.** A model is not only an equation. It is a complete
process involving data collection, preprocessing, fitting, validation,
interpretation, and use. Decisions in this area can change the estimated
coefficients and the real-world reliability of predictions.

**Common trap.** Dropping every incomplete row can bias the sample and
waste data.

**Practical check.** Fit imputation on training data inside a pipeline.

**Questions to ask** - What assumptions does this step make about the
data? - Could this step use information that would not be available at
prediction time? - How would I verify the decision with a plot, a
metric, or a controlled comparison? - Can I explain the consequence to
someone who does not know machine learning?

**Experiment** 1. Make a copy of a small dataset and record its initial
shape and target distribution. 2. Apply the technique or check described
above. 3. Compare the resulting features, coefficients, residuals, or
held-out metric. 4. Write down what changed and whether that change is
expected. 5. Repeat with a different random seed or split when the
experiment permits it.

**Reporting template** - Decision: describe the choice in one
sentence. - Rationale: explain why the choice fits the task. - Evidence:
name the plot, metric, or validation result used. - Limitation: describe
a way the choice could fail. - Next step: state what you would check
before deployment.

### Reference module 5: Duplicate rows

**Core idea.** Exact duplicates may be legitimate repeated events or
accidental data copies.

**Why it matters.** A model is not only an equation. It is a complete
process involving data collection, preprocessing, fitting, validation,
interpretation, and use. Decisions in this area can change the estimated
coefficients and the real-world reliability of predictions.

**Common trap.** Removing valid repeated events changes the population
represented by the data.

**Practical check.** Inspect identifiers and collection logic before
deduplicating.

**Questions to ask** - What assumptions does this step make about the
data? - Could this step use information that would not be available at
prediction time? - How would I verify the decision with a plot, a
metric, or a controlled comparison? - Can I explain the consequence to
someone who does not know machine learning?

**Experiment** 1. Make a copy of a small dataset and record its initial
shape and target distribution. 2. Apply the technique or check described
above. 3. Compare the resulting features, coefficients, residuals, or
held-out metric. 4. Write down what changed and whether that change is
expected. 5. Repeat with a different random seed or split when the
experiment permits it.

**Reporting template** - Decision: describe the choice in one
sentence. - Rationale: explain why the choice fits the task. - Evidence:
name the plot, metric, or validation result used. - Limitation: describe
a way the choice could fail. - Next step: state what you would check
before deployment.

### Reference module 6: Categorical variables

**Core idea.** A linear model needs categories represented numerically.

**Why it matters.** A model is not only an equation. It is a complete
process involving data collection, preprocessing, fitting, validation,
interpretation, and use. Decisions in this area can change the estimated
coefficients and the real-world reliability of predictions.

**Common trap.** Integer codes can imply a false ordering and equal
spacing.

**Practical check.** Use one-hot encoding for nominal categories when
appropriate.

**Questions to ask** - What assumptions does this step make about the
data? - Could this step use information that would not be available at
prediction time? - How would I verify the decision with a plot, a
metric, or a controlled comparison? - Can I explain the consequence to
someone who does not know machine learning?

**Experiment** 1. Make a copy of a small dataset and record its initial
shape and target distribution. 2. Apply the technique or check described
above. 3. Compare the resulting features, coefficients, residuals, or
held-out metric. 4. Write down what changed and whether that change is
expected. 5. Repeat with a different random seed or split when the
experiment permits it.

**Reporting template** - Decision: describe the choice in one
sentence. - Rationale: explain why the choice fits the task. - Evidence:
name the plot, metric, or validation result used. - Limitation: describe
a way the choice could fail. - Next step: state what you would check
before deployment.

### Reference module 7: Ordinal variables

**Core idea.** Ordinal categories have a natural order, but the
distances between levels may not be equal.

**Why it matters.** A model is not only an equation. It is a complete
process involving data collection, preprocessing, fitting, validation,
interpretation, and use. Decisions in this area can change the estimated
coefficients and the real-world reliability of predictions.

**Common trap.** Encoding beginner=1, intermediate=2, expert=3 imposes a
numeric structure.

**Practical check.** Compare ordinal encoding with one-hot encoding
using validation.

**Questions to ask** - What assumptions does this step make about the
data? - Could this step use information that would not be available at
prediction time? - How would I verify the decision with a plot, a
metric, or a controlled comparison? - Can I explain the consequence to
someone who does not know machine learning?

**Experiment** 1. Make a copy of a small dataset and record its initial
shape and target distribution. 2. Apply the technique or check described
above. 3. Compare the resulting features, coefficients, residuals, or
held-out metric. 4. Write down what changed and whether that change is
expected. 5. Repeat with a different random seed or split when the
experiment permits it.

**Reporting template** - Decision: describe the choice in one
sentence. - Rationale: explain why the choice fits the task. - Evidence:
name the plot, metric, or validation result used. - Limitation: describe
a way the choice could fail. - Next step: state what you would check
before deployment.

### Reference module 8: Feature units

**Core idea.** A coefficient depends on the unit in which a feature is
recorded.

**Why it matters.** A model is not only an equation. It is a complete
process involving data collection, preprocessing, fitting, validation,
interpretation, and use. Decisions in this area can change the estimated
coefficients and the real-world reliability of predictions.

**Common trap.** Changing dollars to thousands of dollars changes the
coefficient magnitude.

**Practical check.** Write the units next to important coefficients.

**Questions to ask** - What assumptions does this step make about the
data? - Could this step use information that would not be available at
prediction time? - How would I verify the decision with a plot, a
metric, or a controlled comparison? - Can I explain the consequence to
someone who does not know machine learning?

**Experiment** 1. Make a copy of a small dataset and record its initial
shape and target distribution. 2. Apply the technique or check described
above. 3. Compare the resulting features, coefficients, residuals, or
held-out metric. 4. Write down what changed and whether that change is
expected. 5. Repeat with a different random seed or split when the
experiment permits it.

**Reporting template** - Decision: describe the choice in one
sentence. - Rationale: explain why the choice fits the task. - Evidence:
name the plot, metric, or validation result used. - Limitation: describe
a way the choice could fail. - Next step: state what you would check
before deployment.

### Reference module 9: Centering features

**Core idea.** Centering subtracts a reference value, often the training
mean.

**Why it matters.** A model is not only an equation. It is a complete
process involving data collection, preprocessing, fitting, validation,
interpretation, and use. Decisions in this area can change the estimated
coefficients and the real-world reliability of predictions.

**Common trap.** Centering changes the intercept and can improve
interpretation of interactions.

**Practical check.** Use a fitted transformer to preserve the training
reference.

**Questions to ask** - What assumptions does this step make about the
data? - Could this step use information that would not be available at
prediction time? - How would I verify the decision with a plot, a
metric, or a controlled comparison? - Can I explain the consequence to
someone who does not know machine learning?

**Experiment** 1. Make a copy of a small dataset and record its initial
shape and target distribution. 2. Apply the technique or check described
above. 3. Compare the resulting features, coefficients, residuals, or
held-out metric. 4. Write down what changed and whether that change is
expected. 5. Repeat with a different random seed or split when the
experiment permits it.

**Reporting template** - Decision: describe the choice in one
sentence. - Rationale: explain why the choice fits the task. - Evidence:
name the plot, metric, or validation result used. - Limitation: describe
a way the choice could fail. - Next step: state what you would check
before deployment.

### Reference module 10: Standardization

**Core idea.** Standardization subtracts a mean and divides by a
standard deviation.

**Why it matters.** A model is not only an equation. It is a complete
process involving data collection, preprocessing, fitting, validation,
interpretation, and use. Decisions in this area can change the estimated
coefficients and the real-world reliability of predictions.

**Common trap.** A standard deviation of zero makes a feature constant
and cannot be used in the usual way.

**Practical check.** Fit the scaler on training data only.

**Questions to ask** - What assumptions does this step make about the
data? - Could this step use information that would not be available at
prediction time? - How would I verify the decision with a plot, a
metric, or a controlled comparison? - Can I explain the consequence to
someone who does not know machine learning?

**Experiment** 1. Make a copy of a small dataset and record its initial
shape and target distribution. 2. Apply the technique or check described
above. 3. Compare the resulting features, coefficients, residuals, or
held-out metric. 4. Write down what changed and whether that change is
expected. 5. Repeat with a different random seed or split when the
experiment permits it.

**Reporting template** - Decision: describe the choice in one
sentence. - Rationale: explain why the choice fits the task. - Evidence:
name the plot, metric, or validation result used. - Limitation: describe
a way the choice could fail. - Next step: state what you would check
before deployment.

### Reference module 11: Min-max scaling

**Core idea.** Min-max scaling maps training values to a chosen range.

**Why it matters.** A model is not only an equation. It is a complete
process involving data collection, preprocessing, fitting, validation,
interpretation, and use. Decisions in this area can change the estimated
coefficients and the real-world reliability of predictions.

**Common trap.** Future values can fall outside the range observed
during fitting.

**Practical check.** Do not assume scaled future values are always
between zero and one.

**Questions to ask** - What assumptions does this step make about the
data? - Could this step use information that would not be available at
prediction time? - How would I verify the decision with a plot, a
metric, or a controlled comparison? - Can I explain the consequence to
someone who does not know machine learning?

**Experiment** 1. Make a copy of a small dataset and record its initial
shape and target distribution. 2. Apply the technique or check described
above. 3. Compare the resulting features, coefficients, residuals, or
held-out metric. 4. Write down what changed and whether that change is
expected. 5. Repeat with a different random seed or split when the
experiment permits it.

**Reporting template** - Decision: describe the choice in one
sentence. - Rationale: explain why the choice fits the task. - Evidence:
name the plot, metric, or validation result used. - Limitation: describe
a way the choice could fail. - Next step: state what you would check
before deployment.

### Reference module 12: Log transformations

**Core idea.** A log transform can reduce right skew and model
multiplicative patterns.

**Why it matters.** A model is not only an equation. It is a complete
process involving data collection, preprocessing, fitting, validation,
interpretation, and use. Decisions in this area can change the estimated
coefficients and the real-world reliability of predictions.

**Common trap.** Zeros and negative values need special handling;
interpretation changes.

**Practical check.** Explain whether the target or a feature was
transformed.

**Questions to ask** - What assumptions does this step make about the
data? - Could this step use information that would not be available at
prediction time? - How would I verify the decision with a plot, a
metric, or a controlled comparison? - Can I explain the consequence to
someone who does not know machine learning?

**Experiment** 1. Make a copy of a small dataset and record its initial
shape and target distribution. 2. Apply the technique or check described
above. 3. Compare the resulting features, coefficients, residuals, or
held-out metric. 4. Write down what changed and whether that change is
expected. 5. Repeat with a different random seed or split when the
experiment permits it.

**Reporting template** - Decision: describe the choice in one
sentence. - Rationale: explain why the choice fits the task. - Evidence:
name the plot, metric, or validation result used. - Limitation: describe
a way the choice could fail. - Next step: state what you would check
before deployment.

### Reference module 13: Interactions

**Core idea.** An interaction lets one feature's relationship vary with
another feature.

**Why it matters.** A model is not only an equation. It is a complete
process involving data collection, preprocessing, fitting, validation,
interpretation, and use. Decisions in this area can change the estimated
coefficients and the real-world reliability of predictions.

**Common trap.** Main-effect coefficients alone can be misleading when
interactions are present.

**Practical check.** Include interactions only with a rationale and
validate their value.

**Questions to ask** - What assumptions does this step make about the
data? - Could this step use information that would not be available at
prediction time? - How would I verify the decision with a plot, a
metric, or a controlled comparison? - Can I explain the consequence to
someone who does not know machine learning?

**Experiment** 1. Make a copy of a small dataset and record its initial
shape and target distribution. 2. Apply the technique or check described
above. 3. Compare the resulting features, coefficients, residuals, or
held-out metric. 4. Write down what changed and whether that change is
expected. 5. Repeat with a different random seed or split when the
experiment permits it.

**Reporting template** - Decision: describe the choice in one
sentence. - Rationale: explain why the choice fits the task. - Evidence:
name the plot, metric, or validation result used. - Limitation: describe
a way the choice could fail. - Next step: state what you would check
before deployment.

### Reference module 14: Polynomial terms

**Core idea.** Powers of a feature allow curved fitted relationships.

**Why it matters.** A model is not only an equation. It is a complete
process involving data collection, preprocessing, fitting, validation,
interpretation, and use. Decisions in this area can change the estimated
coefficients and the real-world reliability of predictions.

**Common trap.** High degrees can oscillate and extrapolate badly.

**Practical check.** Use cross-validation to select complexity and
inspect the curve.

**Questions to ask** - What assumptions does this step make about the
data? - Could this step use information that would not be available at
prediction time? - How would I verify the decision with a plot, a
metric, or a controlled comparison? - Can I explain the consequence to
someone who does not know machine learning?

**Experiment** 1. Make a copy of a small dataset and record its initial
shape and target distribution. 2. Apply the technique or check described
above. 3. Compare the resulting features, coefficients, residuals, or
held-out metric. 4. Write down what changed and whether that change is
expected. 5. Repeat with a different random seed or split when the
experiment permits it.

**Reporting template** - Decision: describe the choice in one
sentence. - Rationale: explain why the choice fits the task. - Evidence:
name the plot, metric, or validation result used. - Limitation: describe
a way the choice could fail. - Next step: state what you would check
before deployment.

### Reference module 15: Splines

**Core idea.** Spline bases allow flexible piecewise smooth
relationships.

**Why it matters.** A model is not only an equation. It is a complete
process involving data collection, preprocessing, fitting, validation,
interpretation, and use. Decisions in this area can change the estimated
coefficients and the real-world reliability of predictions.

**Common trap.** Too many knots can overfit, and boundary behavior
matters.

**Practical check.** Consider splines when a straight line is too
restrictive but high-degree polynomials are unstable.

**Questions to ask** - What assumptions does this step make about the
data? - Could this step use information that would not be available at
prediction time? - How would I verify the decision with a plot, a
metric, or a controlled comparison? - Can I explain the consequence to
someone who does not know machine learning?

**Experiment** 1. Make a copy of a small dataset and record its initial
shape and target distribution. 2. Apply the technique or check described
above. 3. Compare the resulting features, coefficients, residuals, or
held-out metric. 4. Write down what changed and whether that change is
expected. 5. Repeat with a different random seed or split when the
experiment permits it.

**Reporting template** - Decision: describe the choice in one
sentence. - Rationale: explain why the choice fits the task. - Evidence:
name the plot, metric, or validation result used. - Limitation: describe
a way the choice could fail. - Next step: state what you would check
before deployment.

### Reference module 16: Train/validation/test

**Core idea.** Training fits parameters, validation supports choices,
and test estimates final performance.

**Why it matters.** A model is not only an equation. It is a complete
process involving data collection, preprocessing, fitting, validation,
interpretation, and use. Decisions in this area can change the estimated
coefficients and the real-world reliability of predictions.

**Common trap.** Repeatedly checking the test set turns it into a
validation set.

**Practical check.** Freeze the final test set until the modeling
decisions are complete.

**Questions to ask** - What assumptions does this step make about the
data? - Could this step use information that would not be available at
prediction time? - How would I verify the decision with a plot, a
metric, or a controlled comparison? - Can I explain the consequence to
someone who does not know machine learning?

**Experiment** 1. Make a copy of a small dataset and record its initial
shape and target distribution. 2. Apply the technique or check described
above. 3. Compare the resulting features, coefficients, residuals, or
held-out metric. 4. Write down what changed and whether that change is
expected. 5. Repeat with a different random seed or split when the
experiment permits it.

**Reporting template** - Decision: describe the choice in one
sentence. - Rationale: explain why the choice fits the task. - Evidence:
name the plot, metric, or validation result used. - Limitation: describe
a way the choice could fail. - Next step: state what you would check
before deployment.

### Reference module 17: Cross-validation

**Core idea.** Cross-validation repeats fitting and evaluation across
training folds.

**Why it matters.** A model is not only an equation. It is a complete
process involving data collection, preprocessing, fitting, validation,
interpretation, and use. Decisions in this area can change the estimated
coefficients and the real-world reliability of predictions.

**Common trap.** Preprocessing outside the pipeline can leak information
into folds.

**Practical check.** Put imputation, scaling, encoding, and the model
inside a pipeline.

**Questions to ask** - What assumptions does this step make about the
data? - Could this step use information that would not be available at
prediction time? - How would I verify the decision with a plot, a
metric, or a controlled comparison? - Can I explain the consequence to
someone who does not know machine learning?

**Experiment** 1. Make a copy of a small dataset and record its initial
shape and target distribution. 2. Apply the technique or check described
above. 3. Compare the resulting features, coefficients, residuals, or
held-out metric. 4. Write down what changed and whether that change is
expected. 5. Repeat with a different random seed or split when the
experiment permits it.

**Reporting template** - Decision: describe the choice in one
sentence. - Rationale: explain why the choice fits the task. - Evidence:
name the plot, metric, or validation result used. - Limitation: describe
a way the choice could fail. - Next step: state what you would check
before deployment.

### Reference module 18: Time-aware splitting

**Core idea.** Use earlier observations for training and later
observations for evaluation when predicting the future.

**Why it matters.** A model is not only an equation. It is a complete
process involving data collection, preprocessing, fitting, validation,
interpretation, and use. Decisions in this area can change the estimated
coefficients and the real-world reliability of predictions.

**Common trap.** Random shuffling can train on future patterns and test
on the past.

**Practical check.** Respect chronology and consider rolling-origin
evaluation.

**Questions to ask** - What assumptions does this step make about the
data? - Could this step use information that would not be available at
prediction time? - How would I verify the decision with a plot, a
metric, or a controlled comparison? - Can I explain the consequence to
someone who does not know machine learning?

**Experiment** 1. Make a copy of a small dataset and record its initial
shape and target distribution. 2. Apply the technique or check described
above. 3. Compare the resulting features, coefficients, residuals, or
held-out metric. 4. Write down what changed and whether that change is
expected. 5. Repeat with a different random seed or split when the
experiment permits it.

**Reporting template** - Decision: describe the choice in one
sentence. - Rationale: explain why the choice fits the task. - Evidence:
name the plot, metric, or validation result used. - Limitation: describe
a way the choice could fail. - Next step: state what you would check
before deployment.

### Reference module 19: Group-aware splitting

**Core idea.** Keep observations from the same group together when
deployment involves new groups.

**Why it matters.** A model is not only an equation. It is a complete
process involving data collection, preprocessing, fitting, validation,
interpretation, and use. Decisions in this area can change the estimated
coefficients and the real-world reliability of predictions.

**Common trap.** The same person or device in both train and test can
inflate scores.

**Practical check.** Use group-aware splitters when the unit of
generalization is a group.

**Questions to ask** - What assumptions does this step make about the
data? - Could this step use information that would not be available at
prediction time? - How would I verify the decision with a plot, a
metric, or a controlled comparison? - Can I explain the consequence to
someone who does not know machine learning?

**Experiment** 1. Make a copy of a small dataset and record its initial
shape and target distribution. 2. Apply the technique or check described
above. 3. Compare the resulting features, coefficients, residuals, or
held-out metric. 4. Write down what changed and whether that change is
expected. 5. Repeat with a different random seed or split when the
experiment permits it.

**Reporting template** - Decision: describe the choice in one
sentence. - Rationale: explain why the choice fits the task. - Evidence:
name the plot, metric, or validation result used. - Limitation: describe
a way the choice could fail. - Next step: state what you would check
before deployment.

### Reference module 20: Baseline models

**Core idea.** A baseline gives a simple performance reference.

**Why it matters.** A model is not only an equation. It is a complete
process involving data collection, preprocessing, fitting, validation,
interpretation, and use. Decisions in this area can change the estimated
coefficients and the real-world reliability of predictions.

**Common trap.** A complex model with no improvement over a baseline may
not justify its cost.

**Practical check.** Try a mean or median predictor for regression.

**Questions to ask** - What assumptions does this step make about the
data? - Could this step use information that would not be available at
prediction time? - How would I verify the decision with a plot, a
metric, or a controlled comparison? - Can I explain the consequence to
someone who does not know machine learning?

**Experiment** 1. Make a copy of a small dataset and record its initial
shape and target distribution. 2. Apply the technique or check described
above. 3. Compare the resulting features, coefficients, residuals, or
held-out metric. 4. Write down what changed and whether that change is
expected. 5. Repeat with a different random seed or split when the
experiment permits it.

**Reporting template** - Decision: describe the choice in one
sentence. - Rationale: explain why the choice fits the task. - Evidence:
name the plot, metric, or validation result used. - Limitation: describe
a way the choice could fail. - Next step: state what you would check
before deployment.

### Reference module 21: Underfitting

**Core idea.** Underfitting occurs when the model cannot capture
important structure.

**Why it matters.** A model is not only an equation. It is a complete
process involving data collection, preprocessing, fitting, validation,
interpretation, and use. Decisions in this area can change the estimated
coefficients and the real-world reliability of predictions.

**Common trap.** A simple line can miss strong curvature or
interactions.

**Practical check.** Compare training and validation errors and inspect
residual structure.

**Questions to ask** - What assumptions does this step make about the
data? - Could this step use information that would not be available at
prediction time? - How would I verify the decision with a plot, a
metric, or a controlled comparison? - Can I explain the consequence to
someone who does not know machine learning?

**Experiment** 1. Make a copy of a small dataset and record its initial
shape and target distribution. 2. Apply the technique or check described
above. 3. Compare the resulting features, coefficients, residuals, or
held-out metric. 4. Write down what changed and whether that change is
expected. 5. Repeat with a different random seed or split when the
experiment permits it.

**Reporting template** - Decision: describe the choice in one
sentence. - Rationale: explain why the choice fits the task. - Evidence:
name the plot, metric, or validation result used. - Limitation: describe
a way the choice could fail. - Next step: state what you would check
before deployment.

### Reference module 22: Overfitting

**Core idea.** Overfitting captures noise or sample-specific patterns.

**Why it matters.** A model is not only an equation. It is a complete
process involving data collection, preprocessing, fitting, validation,
interpretation, and use. Decisions in this area can change the estimated
coefficients and the real-world reliability of predictions.

**Common trap.** More features and higher polynomial degrees can
increase variance.

**Practical check.** Use held-out evaluation, regularization, and
simpler feature sets.

**Questions to ask** - What assumptions does this step make about the
data? - Could this step use information that would not be available at
prediction time? - How would I verify the decision with a plot, a
metric, or a controlled comparison? - Can I explain the consequence to
someone who does not know machine learning?

**Experiment** 1. Make a copy of a small dataset and record its initial
shape and target distribution. 2. Apply the technique or check described
above. 3. Compare the resulting features, coefficients, residuals, or
held-out metric. 4. Write down what changed and whether that change is
expected. 5. Repeat with a different random seed or split when the
experiment permits it.

**Reporting template** - Decision: describe the choice in one
sentence. - Rationale: explain why the choice fits the task. - Evidence:
name the plot, metric, or validation result used. - Limitation: describe
a way the choice could fail. - Next step: state what you would check
before deployment.

### Reference module 23: Bias-variance trade-off

**Core idea.** Prediction error can reflect systematic simplification
and sensitivity to the training sample.

**Why it matters.** A model is not only an equation. It is a complete
process involving data collection, preprocessing, fitting, validation,
interpretation, and use. Decisions in this area can change the estimated
coefficients and the real-world reliability of predictions.

**Common trap.** A lower training error alone does not imply a better
model.

**Practical check.** Use validation performance to balance flexibility
and stability.

**Questions to ask** - What assumptions does this step make about the
data? - Could this step use information that would not be available at
prediction time? - How would I verify the decision with a plot, a
metric, or a controlled comparison? - Can I explain the consequence to
someone who does not know machine learning?

**Experiment** 1. Make a copy of a small dataset and record its initial
shape and target distribution. 2. Apply the technique or check described
above. 3. Compare the resulting features, coefficients, residuals, or
held-out metric. 4. Write down what changed and whether that change is
expected. 5. Repeat with a different random seed or split when the
experiment permits it.

**Reporting template** - Decision: describe the choice in one
sentence. - Rationale: explain why the choice fits the task. - Evidence:
name the plot, metric, or validation result used. - Limitation: describe
a way the choice could fail. - Next step: state what you would check
before deployment.

### Reference module 24: MSE details

**Core idea.** MSE squares residuals before averaging.

**Why it matters.** A model is not only an equation. It is a complete
process involving data collection, preprocessing, fitting, validation,
interpretation, and use. Decisions in this area can change the estimated
coefficients and the real-world reliability of predictions.

**Common trap.** Large errors dominate because squaring magnifies them.

**Practical check.** Use it when large errors should receive a stronger
penalty.

**Questions to ask** - What assumptions does this step make about the
data? - Could this step use information that would not be available at
prediction time? - How would I verify the decision with a plot, a
metric, or a controlled comparison? - Can I explain the consequence to
someone who does not know machine learning?

**Experiment** 1. Make a copy of a small dataset and record its initial
shape and target distribution. 2. Apply the technique or check described
above. 3. Compare the resulting features, coefficients, residuals, or
held-out metric. 4. Write down what changed and whether that change is
expected. 5. Repeat with a different random seed or split when the
experiment permits it.

**Reporting template** - Decision: describe the choice in one
sentence. - Rationale: explain why the choice fits the task. - Evidence:
name the plot, metric, or validation result used. - Limitation: describe
a way the choice could fail. - Next step: state what you would check
before deployment.

### Reference module 25: MAE details

**Core idea.** MAE averages absolute residual magnitudes.

**Why it matters.** A model is not only an equation. It is a complete
process involving data collection, preprocessing, fitting, validation,
interpretation, and use. Decisions in this area can change the estimated
coefficients and the real-world reliability of predictions.

**Common trap.** It treats a one-unit error equally regardless of
direction.

**Practical check.** Use it to communicate typical absolute error in
target units.

**Questions to ask** - What assumptions does this step make about the
data? - Could this step use information that would not be available at
prediction time? - How would I verify the decision with a plot, a
metric, or a controlled comparison? - Can I explain the consequence to
someone who does not know machine learning?

**Experiment** 1. Make a copy of a small dataset and record its initial
shape and target distribution. 2. Apply the technique or check described
above. 3. Compare the resulting features, coefficients, residuals, or
held-out metric. 4. Write down what changed and whether that change is
expected. 5. Repeat with a different random seed or split when the
experiment permits it.

**Reporting template** - Decision: describe the choice in one
sentence. - Rationale: explain why the choice fits the task. - Evidence:
name the plot, metric, or validation result used. - Limitation: describe
a way the choice could fail. - Next step: state what you would check
before deployment.

### Reference module 26: RMSE details

**Core idea.** RMSE is the square root of MSE.

**Why it matters.** A model is not only an equation. It is a complete
process involving data collection, preprocessing, fitting, validation,
interpretation, and use. Decisions in this area can change the estimated
coefficients and the real-world reliability of predictions.

**Common trap.** RMSE can be larger than MAE because it gives more
influence to large errors.

**Practical check.** Report target units and state which dataset
produced the score.

**Questions to ask** - What assumptions does this step make about the
data? - Could this step use information that would not be available at
prediction time? - How would I verify the decision with a plot, a
metric, or a controlled comparison? - Can I explain the consequence to
someone who does not know machine learning?

**Experiment** 1. Make a copy of a small dataset and record its initial
shape and target distribution. 2. Apply the technique or check described
above. 3. Compare the resulting features, coefficients, residuals, or
held-out metric. 4. Write down what changed and whether that change is
expected. 5. Repeat with a different random seed or split when the
experiment permits it.

**Reporting template** - Decision: describe the choice in one
sentence. - Rationale: explain why the choice fits the task. - Evidence:
name the plot, metric, or validation result used. - Limitation: describe
a way the choice could fail. - Next step: state what you would check
before deployment.

### Reference module 27: Median absolute error

**Core idea.** Median absolute error summarizes the median absolute
prediction error.

**Why it matters.** A model is not only an equation. It is a complete
process involving data collection, preprocessing, fitting, validation,
interpretation, and use. Decisions in this area can change the estimated
coefficients and the real-world reliability of predictions.

**Common trap.** It can hide a problematic tail of very large errors.

**Practical check.** Use it as a robust complement, not the only metric.

**Questions to ask** - What assumptions does this step make about the
data? - Could this step use information that would not be available at
prediction time? - How would I verify the decision with a plot, a
metric, or a controlled comparison? - Can I explain the consequence to
someone who does not know machine learning?

**Experiment** 1. Make a copy of a small dataset and record its initial
shape and target distribution. 2. Apply the technique or check described
above. 3. Compare the resulting features, coefficients, residuals, or
held-out metric. 4. Write down what changed and whether that change is
expected. 5. Repeat with a different random seed or split when the
experiment permits it.

**Reporting template** - Decision: describe the choice in one
sentence. - Rationale: explain why the choice fits the task. - Evidence:
name the plot, metric, or validation result used. - Limitation: describe
a way the choice could fail. - Next step: state what you would check
before deployment.

### Reference module 28: MAPE limitations

**Core idea.** Mean absolute percentage error divides error by actual
values.

**Why it matters.** A model is not only an equation. It is a complete
process involving data collection, preprocessing, fitting, validation,
interpretation, and use. Decisions in this area can change the estimated
coefficients and the real-world reliability of predictions.

**Common trap.** It is undefined at zero and unstable near zero.

**Practical check.** Avoid MAPE for targets that can be zero or near
zero.

**Questions to ask** - What assumptions does this step make about the
data? - Could this step use information that would not be available at
prediction time? - How would I verify the decision with a plot, a
metric, or a controlled comparison? - Can I explain the consequence to
someone who does not know machine learning?

**Experiment** 1. Make a copy of a small dataset and record its initial
shape and target distribution. 2. Apply the technique or check described
above. 3. Compare the resulting features, coefficients, residuals, or
held-out metric. 4. Write down what changed and whether that change is
expected. 5. Repeat with a different random seed or split when the
experiment permits it.

**Reporting template** - Decision: describe the choice in one
sentence. - Rationale: explain why the choice fits the task. - Evidence:
name the plot, metric, or validation result used. - Limitation: describe
a way the choice could fail. - Next step: state what you would check
before deployment.

### Reference module 29: R-squared limitations

**Core idea.** R² compares residual variation with variation around a
mean baseline.

**Why it matters.** A model is not only an equation. It is a complete
process involving data collection, preprocessing, fitting, validation,
interpretation, and use. Decisions in this area can change the estimated
coefficients and the real-world reliability of predictions.

**Common trap.** A high R² does not guarantee calibrated predictions,
causal validity, or low practical error.

**Practical check.** Always pair R² with an error metric and a clear
split description.

**Questions to ask** - What assumptions does this step make about the
data? - Could this step use information that would not be available at
prediction time? - How would I verify the decision with a plot, a
metric, or a controlled comparison? - Can I explain the consequence to
someone who does not know machine learning?

**Experiment** 1. Make a copy of a small dataset and record its initial
shape and target distribution. 2. Apply the technique or check described
above. 3. Compare the resulting features, coefficients, residuals, or
held-out metric. 4. Write down what changed and whether that change is
expected. 5. Repeat with a different random seed or split when the
experiment permits it.

**Reporting template** - Decision: describe the choice in one
sentence. - Rationale: explain why the choice fits the task. - Evidence:
name the plot, metric, or validation result used. - Limitation: describe
a way the choice could fail. - Next step: state what you would check
before deployment.

### Reference module 30: Adjusted R-squared

**Core idea.** Adjusted R² adjusts for the number of predictors and
sample size.

**Why it matters.** A model is not only an equation. It is a complete
process involving data collection, preprocessing, fitting, validation,
interpretation, and use. Decisions in this area can change the estimated
coefficients and the real-world reliability of predictions.

**Common trap.** It is not a universal feature-selection method and does
not replace test evaluation.

**Practical check.** Use it as a descriptive statistic under suitable
assumptions.

**Questions to ask** - What assumptions does this step make about the
data? - Could this step use information that would not be available at
prediction time? - How would I verify the decision with a plot, a
metric, or a controlled comparison? - Can I explain the consequence to
someone who does not know machine learning?

**Experiment** 1. Make a copy of a small dataset and record its initial
shape and target distribution. 2. Apply the technique or check described
above. 3. Compare the resulting features, coefficients, residuals, or
held-out metric. 4. Write down what changed and whether that change is
expected. 5. Repeat with a different random seed or split when the
experiment permits it.

**Reporting template** - Decision: describe the choice in one
sentence. - Rationale: explain why the choice fits the task. - Evidence:
name the plot, metric, or validation result used. - Limitation: describe
a way the choice could fail. - Next step: state what you would check
before deployment.

### Reference module 31: Residual mean

**Core idea.** With an intercept, ordinary least squares residuals sum
to approximately zero on training data.

**Why it matters.** A model is not only an equation. It is a complete
process involving data collection, preprocessing, fitting, validation,
interpretation, and use. Decisions in this area can change the estimated
coefficients and the real-world reliability of predictions.

**Common trap.** This algebraic property does not imply errors are
independent or normally distributed.

**Practical check.** Do not use residual mean alone as a diagnostic.

**Questions to ask** - What assumptions does this step make about the
data? - Could this step use information that would not be available at
prediction time? - How would I verify the decision with a plot, a
metric, or a controlled comparison? - Can I explain the consequence to
someone who does not know machine learning?

**Experiment** 1. Make a copy of a small dataset and record its initial
shape and target distribution. 2. Apply the technique or check described
above. 3. Compare the resulting features, coefficients, residuals, or
held-out metric. 4. Write down what changed and whether that change is
expected. 5. Repeat with a different random seed or split when the
experiment permits it.

**Reporting template** - Decision: describe the choice in one
sentence. - Rationale: explain why the choice fits the task. - Evidence:
name the plot, metric, or validation result used. - Limitation: describe
a way the choice could fail. - Next step: state what you would check
before deployment.

### Reference module 32: Residual curvature

**Core idea.** Curvature in residuals can signal a missing nonlinear
pattern.

**Why it matters.** A model is not only an equation. It is a complete
process involving data collection, preprocessing, fitting, validation,
interpretation, and use. Decisions in this area can change the estimated
coefficients and the real-world reliability of predictions.

**Common trap.** Adding polynomial terms without validation can overfit.

**Practical check.** Plot residuals against important predictors and
fitted values.

**Questions to ask** - What assumptions does this step make about the
data? - Could this step use information that would not be available at
prediction time? - How would I verify the decision with a plot, a
metric, or a controlled comparison? - Can I explain the consequence to
someone who does not know machine learning?

**Experiment** 1. Make a copy of a small dataset and record its initial
shape and target distribution. 2. Apply the technique or check described
above. 3. Compare the resulting features, coefficients, residuals, or
held-out metric. 4. Write down what changed and whether that change is
expected. 5. Repeat with a different random seed or split when the
experiment permits it.

**Reporting template** - Decision: describe the choice in one
sentence. - Rationale: explain why the choice fits the task. - Evidence:
name the plot, metric, or validation result used. - Limitation: describe
a way the choice could fail. - Next step: state what you would check
before deployment.

### Reference module 33: Heteroscedasticity

**Core idea.** Residual variance may change across fitted values or
groups.

**Why it matters.** A model is not only an equation. It is a complete
process involving data collection, preprocessing, fitting, validation,
interpretation, and use. Decisions in this area can change the estimated
coefficients and the real-world reliability of predictions.

**Common trap.** Classical standard errors can be unreliable under
heteroscedasticity.

**Practical check.** Inspect residual spread and consider robust
inference or a different model.

**Questions to ask** - What assumptions does this step make about the
data? - Could this step use information that would not be available at
prediction time? - How would I verify the decision with a plot, a
metric, or a controlled comparison? - Can I explain the consequence to
someone who does not know machine learning?

**Experiment** 1. Make a copy of a small dataset and record its initial
shape and target distribution. 2. Apply the technique or check described
above. 3. Compare the resulting features, coefficients, residuals, or
held-out metric. 4. Write down what changed and whether that change is
expected. 5. Repeat with a different random seed or split when the
experiment permits it.

**Reporting template** - Decision: describe the choice in one
sentence. - Rationale: explain why the choice fits the task. - Evidence:
name the plot, metric, or validation result used. - Limitation: describe
a way the choice could fail. - Next step: state what you would check
before deployment.

### Reference module 34: Autocorrelation

**Core idea.** Time-ordered errors may be correlated with nearby errors.

**Why it matters.** A model is not only an equation. It is a complete
process involving data collection, preprocessing, fitting, validation,
interpretation, and use. Decisions in this area can change the estimated
coefficients and the real-world reliability of predictions.

**Common trap.** Ordinary random splits and standard errors may be
inappropriate for time series.

**Practical check.** Plot residuals over time and use time-aware
validation.

**Questions to ask** - What assumptions does this step make about the
data? - Could this step use information that would not be available at
prediction time? - How would I verify the decision with a plot, a
metric, or a controlled comparison? - Can I explain the consequence to
someone who does not know machine learning?

**Experiment** 1. Make a copy of a small dataset and record its initial
shape and target distribution. 2. Apply the technique or check described
above. 3. Compare the resulting features, coefficients, residuals, or
held-out metric. 4. Write down what changed and whether that change is
expected. 5. Repeat with a different random seed or split when the
experiment permits it.

**Reporting template** - Decision: describe the choice in one
sentence. - Rationale: explain why the choice fits the task. - Evidence:
name the plot, metric, or validation result used. - Limitation: describe
a way the choice could fail. - Next step: state what you would check
before deployment.

### Reference module 35: Normality

**Core idea.** Normal errors are mainly used for exact classical
small-sample inference.

**Why it matters.** A model is not only an equation. It is a complete
process involving data collection, preprocessing, fitting, validation,
interpretation, and use. Decisions in this area can change the estimated
coefficients and the real-world reliability of predictions.

**Common trap.** Normality is not required merely to calculate an OLS
fit.

**Practical check.** Inspect Q--Q plots when inferential intervals or
tests matter.

**Questions to ask** - What assumptions does this step make about the
data? - Could this step use information that would not be available at
prediction time? - How would I verify the decision with a plot, a
metric, or a controlled comparison? - Can I explain the consequence to
someone who does not know machine learning?

**Experiment** 1. Make a copy of a small dataset and record its initial
shape and target distribution. 2. Apply the technique or check described
above. 3. Compare the resulting features, coefficients, residuals, or
held-out metric. 4. Write down what changed and whether that change is
expected. 5. Repeat with a different random seed or split when the
experiment permits it.

**Reporting template** - Decision: describe the choice in one
sentence. - Rationale: explain why the choice fits the task. - Evidence:
name the plot, metric, or validation result used. - Limitation: describe
a way the choice could fail. - Next step: state what you would check
before deployment.

### Reference module 36: Multicollinearity

**Core idea.** Predictors may share overlapping information.

**Why it matters.** A model is not only an equation. It is a complete
process involving data collection, preprocessing, fitting, validation,
interpretation, and use. Decisions in this area can change the estimated
coefficients and the real-world reliability of predictions.

**Common trap.** Coefficients can vary greatly while predictions remain
similar.

**Practical check.** Inspect correlations, condition diagnostics, and
coefficient stability.

**Questions to ask** - What assumptions does this step make about the
data? - Could this step use information that would not be available at
prediction time? - How would I verify the decision with a plot, a
metric, or a controlled comparison? - Can I explain the consequence to
someone who does not know machine learning?

**Experiment** 1. Make a copy of a small dataset and record its initial
shape and target distribution. 2. Apply the technique or check described
above. 3. Compare the resulting features, coefficients, residuals, or
held-out metric. 4. Write down what changed and whether that change is
expected. 5. Repeat with a different random seed or split when the
experiment permits it.

**Reporting template** - Decision: describe the choice in one
sentence. - Rationale: explain why the choice fits the task. - Evidence:
name the plot, metric, or validation result used. - Limitation: describe
a way the choice could fail. - Next step: state what you would check
before deployment.

### Reference module 37: Influence

**Core idea.** An observation can strongly affect the fitted
coefficients.

**Why it matters.** A model is not only an equation. It is a complete
process involving data collection, preprocessing, fitting, validation,
interpretation, and use. Decisions in this area can change the estimated
coefficients and the real-world reliability of predictions.

**Common trap.** Influence is not identical to having a large residual.

**Practical check.** Review leverage, residual size, and sensitivity to
inclusion.

**Questions to ask** - What assumptions does this step make about the
data? - Could this step use information that would not be available at
prediction time? - How would I verify the decision with a plot, a
metric, or a controlled comparison? - Can I explain the consequence to
someone who does not know machine learning?

**Experiment** 1. Make a copy of a small dataset and record its initial
shape and target distribution. 2. Apply the technique or check described
above. 3. Compare the resulting features, coefficients, residuals, or
held-out metric. 4. Write down what changed and whether that change is
expected. 5. Repeat with a different random seed or split when the
experiment permits it.

**Reporting template** - Decision: describe the choice in one
sentence. - Rationale: explain why the choice fits the task. - Evidence:
name the plot, metric, or validation result used. - Limitation: describe
a way the choice could fail. - Next step: state what you would check
before deployment.

### Reference module 38: Extrapolation

**Core idea.** A model extrapolates when it predicts beyond the feature
region represented in training.

**Why it matters.** A model is not only an equation. It is a complete
process involving data collection, preprocessing, fitting, validation,
interpretation, and use. Decisions in this area can change the estimated
coefficients and the real-world reliability of predictions.

**Common trap.** Linear trends may not persist outside observed ranges.

**Practical check.** Flag out-of-range inputs and communicate
uncertainty.

**Questions to ask** - What assumptions does this step make about the
data? - Could this step use information that would not be available at
prediction time? - How would I verify the decision with a plot, a
metric, or a controlled comparison? - Can I explain the consequence to
someone who does not know machine learning?

**Experiment** 1. Make a copy of a small dataset and record its initial
shape and target distribution. 2. Apply the technique or check described
above. 3. Compare the resulting features, coefficients, residuals, or
held-out metric. 4. Write down what changed and whether that change is
expected. 5. Repeat with a different random seed or split when the
experiment permits it.

**Reporting template** - Decision: describe the choice in one
sentence. - Rationale: explain why the choice fits the task. - Evidence:
name the plot, metric, or validation result used. - Limitation: describe
a way the choice could fail. - Next step: state what you would check
before deployment.

### Reference module 39: Causality

**Core idea.** Predictive associations do not automatically identify
causal effects.

**Why it matters.** A model is not only an equation. It is a complete
process involving data collection, preprocessing, fitting, validation,
interpretation, and use. Decisions in this area can change the estimated
coefficients and the real-world reliability of predictions.

**Common trap.** Omitted confounding variables can distort coefficients.

**Practical check.** Use causal design and domain knowledge for
intervention questions.

**Questions to ask** - What assumptions does this step make about the
data? - Could this step use information that would not be available at
prediction time? - How would I verify the decision with a plot, a
metric, or a controlled comparison? - Can I explain the consequence to
someone who does not know machine learning?

**Experiment** 1. Make a copy of a small dataset and record its initial
shape and target distribution. 2. Apply the technique or check described
above. 3. Compare the resulting features, coefficients, residuals, or
held-out metric. 4. Write down what changed and whether that change is
expected. 5. Repeat with a different random seed or split when the
experiment permits it.

**Reporting template** - Decision: describe the choice in one
sentence. - Rationale: explain why the choice fits the task. - Evidence:
name the plot, metric, or validation result used. - Limitation: describe
a way the choice could fail. - Next step: state what you would check
before deployment.

### Reference module 40: OLS numerical stability

**Core idea.** Solvers such as least squares and SVD are often more
stable than explicit matrix inversion.

**Why it matters.** A model is not only an equation. It is a complete
process involving data collection, preprocessing, fitting, validation,
interpretation, and use. Decisions in this area can change the estimated
coefficients and the real-world reliability of predictions.

**Common trap.** Near-collinear columns make the problem
ill-conditioned.

**Practical check.** Use `np.linalg.lstsq` rather than manually
inverting `X.T @ X`.

**Questions to ask** - What assumptions does this step make about the
data? - Could this step use information that would not be available at
prediction time? - How would I verify the decision with a plot, a
metric, or a controlled comparison? - Can I explain the consequence to
someone who does not know machine learning?

**Experiment** 1. Make a copy of a small dataset and record its initial
shape and target distribution. 2. Apply the technique or check described
above. 3. Compare the resulting features, coefficients, residuals, or
held-out metric. 4. Write down what changed and whether that change is
expected. 5. Repeat with a different random seed or split when the
experiment permits it.

**Reporting template** - Decision: describe the choice in one
sentence. - Rationale: explain why the choice fits the task. - Evidence:
name the plot, metric, or validation result used. - Limitation: describe
a way the choice could fail. - Next step: state what you would check
before deployment.

### Reference module 41: Gradient scale

**Core idea.** Feature scales affect the geometry of the optimization
objective.

**Why it matters.** A model is not only an equation. It is a complete
process involving data collection, preprocessing, fitting, validation,
interpretation, and use. Decisions in this area can change the estimated
coefficients and the real-world reliability of predictions.

**Common trap.** A single learning rate may behave poorly when scales
differ greatly.

**Practical check.** Standardize features when appropriate and monitor
loss.

**Questions to ask** - What assumptions does this step make about the
data? - Could this step use information that would not be available at
prediction time? - How would I verify the decision with a plot, a
metric, or a controlled comparison? - Can I explain the consequence to
someone who does not know machine learning?

**Experiment** 1. Make a copy of a small dataset and record its initial
shape and target distribution. 2. Apply the technique or check described
above. 3. Compare the resulting features, coefficients, residuals, or
held-out metric. 4. Write down what changed and whether that change is
expected. 5. Repeat with a different random seed or split when the
experiment permits it.

**Reporting template** - Decision: describe the choice in one
sentence. - Rationale: explain why the choice fits the task. - Evidence:
name the plot, metric, or validation result used. - Limitation: describe
a way the choice could fail. - Next step: state what you would check
before deployment.

### Reference module 42: Learning-rate schedule

**Core idea.** A schedule changes the step size during optimization.

**Why it matters.** A model is not only an equation. It is a complete
process involving data collection, preprocessing, fitting, validation,
interpretation, and use. Decisions in this area can change the estimated
coefficients and the real-world reliability of predictions.

**Common trap.** Reducing the rate too quickly can stop meaningful
progress.

**Practical check.** Compare loss curves and document the stopping
criterion.

**Questions to ask** - What assumptions does this step make about the
data? - Could this step use information that would not be available at
prediction time? - How would I verify the decision with a plot, a
metric, or a controlled comparison? - Can I explain the consequence to
someone who does not know machine learning?

**Experiment** 1. Make a copy of a small dataset and record its initial
shape and target distribution. 2. Apply the technique or check described
above. 3. Compare the resulting features, coefficients, residuals, or
held-out metric. 4. Write down what changed and whether that change is
expected. 5. Repeat with a different random seed or split when the
experiment permits it.

**Reporting template** - Decision: describe the choice in one
sentence. - Rationale: explain why the choice fits the task. - Evidence:
name the plot, metric, or validation result used. - Limitation: describe
a way the choice could fail. - Next step: state what you would check
before deployment.

### Reference module 43: Stopping criteria

**Core idea.** Optimization can stop after a fixed iteration count or
when changes become small.

**Why it matters.** A model is not only an equation. It is a complete
process involving data collection, preprocessing, fitting, validation,
interpretation, and use. Decisions in this area can change the estimated
coefficients and the real-world reliability of predictions.

**Common trap.** A fixed count may stop too early or waste computation.

**Practical check.** Track loss, gradient size, and coefficient changes.

**Questions to ask** - What assumptions does this step make about the
data? - Could this step use information that would not be available at
prediction time? - How would I verify the decision with a plot, a
metric, or a controlled comparison? - Can I explain the consequence to
someone who does not know machine learning?

**Experiment** 1. Make a copy of a small dataset and record its initial
shape and target distribution. 2. Apply the technique or check described
above. 3. Compare the resulting features, coefficients, residuals, or
held-out metric. 4. Write down what changed and whether that change is
expected. 5. Repeat with a different random seed or split when the
experiment permits it.

**Reporting template** - Decision: describe the choice in one
sentence. - Rationale: explain why the choice fits the task. - Evidence:
name the plot, metric, or validation result used. - Limitation: describe
a way the choice could fail. - Next step: state what you would check
before deployment.

### Reference module 44: Mini-batch gradient descent

**Core idea.** Mini-batch methods update using subsets of observations.

**Why it matters.** A model is not only an equation. It is a complete
process involving data collection, preprocessing, fitting, validation,
interpretation, and use. Decisions in this area can change the estimated
coefficients and the real-world reliability of predictions.

**Common trap.** Updates are noisier than full-batch updates.

**Practical check.** Shuffle training observations where appropriate and
use reproducible seeds.

**Questions to ask** - What assumptions does this step make about the
data? - Could this step use information that would not be available at
prediction time? - How would I verify the decision with a plot, a
metric, or a controlled comparison? - Can I explain the consequence to
someone who does not know machine learning?

**Experiment** 1. Make a copy of a small dataset and record its initial
shape and target distribution. 2. Apply the technique or check described
above. 3. Compare the resulting features, coefficients, residuals, or
held-out metric. 4. Write down what changed and whether that change is
expected. 5. Repeat with a different random seed or split when the
experiment permits it.

**Reporting template** - Decision: describe the choice in one
sentence. - Rationale: explain why the choice fits the task. - Evidence:
name the plot, metric, or validation result used. - Limitation: describe
a way the choice could fail. - Next step: state what you would check
before deployment.

### Reference module 45: Closed-form versus iterative

**Core idea.** OLS can be solved directly or by iterative optimization.

**Why it matters.** A model is not only an equation. It is a complete
process involving data collection, preprocessing, fitting, validation,
interpretation, and use. Decisions in this area can change the estimated
coefficients and the real-world reliability of predictions.

**Common trap.** Direct methods and gradient descent have different
numerical and computational trade-offs.

**Practical check.** Use a trusted library solver for normal tabular
datasets unless implementing learning is the goal.

**Questions to ask** - What assumptions does this step make about the
data? - Could this step use information that would not be available at
prediction time? - How would I verify the decision with a plot, a
metric, or a controlled comparison? - Can I explain the consequence to
someone who does not know machine learning?

**Experiment** 1. Make a copy of a small dataset and record its initial
shape and target distribution. 2. Apply the technique or check described
above. 3. Compare the resulting features, coefficients, residuals, or
held-out metric. 4. Write down what changed and whether that change is
expected. 5. Repeat with a different random seed or split when the
experiment permits it.

**Reporting template** - Decision: describe the choice in one
sentence. - Rationale: explain why the choice fits the task. - Evidence:
name the plot, metric, or validation result used. - Limitation: describe
a way the choice could fail. - Next step: state what you would check
before deployment.

### Reference module 46: Ridge interpretation

**Core idea.** Ridge penalizes squared coefficients and shrinks them
toward zero.

**Why it matters.** A model is not only an equation. It is a complete
process involving data collection, preprocessing, fitting, validation,
interpretation, and use. Decisions in this area can change the estimated
coefficients and the real-world reliability of predictions.

**Common trap.** Regularization strength depends on feature scale and
objective convention.

**Practical check.** Scale features and tune the penalty using
cross-validation.

**Questions to ask** - What assumptions does this step make about the
data? - Could this step use information that would not be available at
prediction time? - How would I verify the decision with a plot, a
metric, or a controlled comparison? - Can I explain the consequence to
someone who does not know machine learning?

**Experiment** 1. Make a copy of a small dataset and record its initial
shape and target distribution. 2. Apply the technique or check described
above. 3. Compare the resulting features, coefficients, residuals, or
held-out metric. 4. Write down what changed and whether that change is
expected. 5. Repeat with a different random seed or split when the
experiment permits it.

**Reporting template** - Decision: describe the choice in one
sentence. - Rationale: explain why the choice fits the task. - Evidence:
name the plot, metric, or validation result used. - Limitation: describe
a way the choice could fail. - Next step: state what you would check
before deployment.

### Reference module 47: Lasso interpretation

**Core idea.** Lasso can yield exact zero coefficients.

**Why it matters.** A model is not only an equation. It is a complete
process involving data collection, preprocessing, fitting, validation,
interpretation, and use. Decisions in this area can change the estimated
coefficients and the real-world reliability of predictions.

**Common trap.** Selection among correlated predictors may be unstable.

**Practical check.** Validate selected features and consider domain
knowledge.

**Questions to ask** - What assumptions does this step make about the
data? - Could this step use information that would not be available at
prediction time? - How would I verify the decision with a plot, a
metric, or a controlled comparison? - Can I explain the consequence to
someone who does not know machine learning?

**Experiment** 1. Make a copy of a small dataset and record its initial
shape and target distribution. 2. Apply the technique or check described
above. 3. Compare the resulting features, coefficients, residuals, or
held-out metric. 4. Write down what changed and whether that change is
expected. 5. Repeat with a different random seed or split when the
experiment permits it.

**Reporting template** - Decision: describe the choice in one
sentence. - Rationale: explain why the choice fits the task. - Evidence:
name the plot, metric, or validation result used. - Limitation: describe
a way the choice could fail. - Next step: state what you would check
before deployment.

### Reference module 48: Elastic Net interpretation

**Core idea.** Elastic Net mixes L1 and L2 penalties.

**Why it matters.** A model is not only an equation. It is a complete
process involving data collection, preprocessing, fitting, validation,
interpretation, and use. Decisions in this area can change the estimated
coefficients and the real-world reliability of predictions.

**Common trap.** The mixing and strength parameters both need selection.

**Practical check.** Tune parameters inside cross-validation.

**Questions to ask** - What assumptions does this step make about the
data? - Could this step use information that would not be available at
prediction time? - How would I verify the decision with a plot, a
metric, or a controlled comparison? - Can I explain the consequence to
someone who does not know machine learning?

**Experiment** 1. Make a copy of a small dataset and record its initial
shape and target distribution. 2. Apply the technique or check described
above. 3. Compare the resulting features, coefficients, residuals, or
held-out metric. 4. Write down what changed and whether that change is
expected. 5. Repeat with a different random seed or split when the
experiment permits it.

**Reporting template** - Decision: describe the choice in one
sentence. - Rationale: explain why the choice fits the task. - Evidence:
name the plot, metric, or validation result used. - Limitation: describe
a way the choice could fail. - Next step: state what you would check
before deployment.

### Reference module 49: Polynomial feature explosion

**Core idea.** The number of expanded features can grow rapidly with
degree and feature count.

**Why it matters.** A model is not only an equation. It is a complete
process involving data collection, preprocessing, fitting, validation,
interpretation, and use. Decisions in this area can change the estimated
coefficients and the real-world reliability of predictions.

**Common trap.** High-dimensional expansions can overfit and increase
compute cost.

**Practical check.** Inspect generated feature counts before fitting.

**Questions to ask** - What assumptions does this step make about the
data? - Could this step use information that would not be available at
prediction time? - How would I verify the decision with a plot, a
metric, or a controlled comparison? - Can I explain the consequence to
someone who does not know machine learning?

**Experiment** 1. Make a copy of a small dataset and record its initial
shape and target distribution. 2. Apply the technique or check described
above. 3. Compare the resulting features, coefficients, residuals, or
held-out metric. 4. Write down what changed and whether that change is
expected. 5. Repeat with a different random seed or split when the
experiment permits it.

**Reporting template** - Decision: describe the choice in one
sentence. - Rationale: explain why the choice fits the task. - Evidence:
name the plot, metric, or validation result used. - Limitation: describe
a way the choice could fail. - Next step: state what you would check
before deployment.

### Reference module 50: Prediction intervals

**Core idea.** A prediction interval aims to cover a future individual
outcome.

**Why it matters.** A model is not only an equation. It is a complete
process involving data collection, preprocessing, fitting, validation,
interpretation, and use. Decisions in this area can change the estimated
coefficients and the real-world reliability of predictions.

**Common trap.** It is wider than a confidence interval for the mean
response in classical settings.

**Practical check.** State the assumptions and coverage level when
reporting intervals.

**Questions to ask** - What assumptions does this step make about the
data? - Could this step use information that would not be available at
prediction time? - How would I verify the decision with a plot, a
metric, or a controlled comparison? - Can I explain the consequence to
someone who does not know machine learning?

**Experiment** 1. Make a copy of a small dataset and record its initial
shape and target distribution. 2. Apply the technique or check described
above. 3. Compare the resulting features, coefficients, residuals, or
held-out metric. 4. Write down what changed and whether that change is
expected. 5. Repeat with a different random seed or split when the
experiment permits it.

**Reporting template** - Decision: describe the choice in one
sentence. - Rationale: explain why the choice fits the task. - Evidence:
name the plot, metric, or validation result used. - Limitation: describe
a way the choice could fail. - Next step: state what you would check
before deployment.

### Reference module 51: Confidence intervals

**Core idea.** A confidence interval quantifies uncertainty in an
estimated parameter under a procedure.

**Why it matters.** A model is not only an equation. It is a complete
process involving data collection, preprocessing, fitting, validation,
interpretation, and use. Decisions in this area can change the estimated
coefficients and the real-world reliability of predictions.

**Common trap.** It is not the probability that a fixed parameter lies
in one realized interval.

**Practical check.** Interpret intervals with the sampling design and
assumptions.

**Questions to ask** - What assumptions does this step make about the
data? - Could this step use information that would not be available at
prediction time? - How would I verify the decision with a plot, a
metric, or a controlled comparison? - Can I explain the consequence to
someone who does not know machine learning?

**Experiment** 1. Make a copy of a small dataset and record its initial
shape and target distribution. 2. Apply the technique or check described
above. 3. Compare the resulting features, coefficients, residuals, or
held-out metric. 4. Write down what changed and whether that change is
expected. 5. Repeat with a different random seed or split when the
experiment permits it.

**Reporting template** - Decision: describe the choice in one
sentence. - Rationale: explain why the choice fits the task. - Evidence:
name the plot, metric, or validation result used. - Limitation: describe
a way the choice could fail. - Next step: state what you would check
before deployment.

### Reference module 52: P-values

**Core idea.** A p-value is computed under a null hypothesis and model
assumptions.

**Why it matters.** A model is not only an equation. It is a complete
process involving data collection, preprocessing, fitting, validation,
interpretation, and use. Decisions in this area can change the estimated
coefficients and the real-world reliability of predictions.

**Common trap.** It is not the probability that the null hypothesis is
true.

**Practical check.** Report effect size and uncertainty, not only a
thresholded p-value.

**Questions to ask** - What assumptions does this step make about the
data? - Could this step use information that would not be available at
prediction time? - How would I verify the decision with a plot, a
metric, or a controlled comparison? - Can I explain the consequence to
someone who does not know machine learning?

**Experiment** 1. Make a copy of a small dataset and record its initial
shape and target distribution. 2. Apply the technique or check described
above. 3. Compare the resulting features, coefficients, residuals, or
held-out metric. 4. Write down what changed and whether that change is
expected. 5. Repeat with a different random seed or split when the
experiment permits it.

**Reporting template** - Decision: describe the choice in one
sentence. - Rationale: explain why the choice fits the task. - Evidence:
name the plot, metric, or validation result used. - Limitation: describe
a way the choice could fail. - Next step: state what you would check
before deployment.

### Reference module 53: Deployment schema

**Core idea.** Inference code should validate expected columns, types,
and units.

**Why it matters.** A model is not only an equation. It is a complete
process involving data collection, preprocessing, fitting, validation,
interpretation, and use. Decisions in this area can change the estimated
coefficients and the real-world reliability of predictions.

**Common trap.** Silently reordered columns can produce incorrect
predictions.

**Practical check.** Use a pipeline and explicit input validation.

**Questions to ask** - What assumptions does this step make about the
data? - Could this step use information that would not be available at
prediction time? - How would I verify the decision with a plot, a
metric, or a controlled comparison? - Can I explain the consequence to
someone who does not know machine learning?

**Experiment** 1. Make a copy of a small dataset and record its initial
shape and target distribution. 2. Apply the technique or check described
above. 3. Compare the resulting features, coefficients, residuals, or
held-out metric. 4. Write down what changed and whether that change is
expected. 5. Repeat with a different random seed or split when the
experiment permits it.

**Reporting template** - Decision: describe the choice in one
sentence. - Rationale: explain why the choice fits the task. - Evidence:
name the plot, metric, or validation result used. - Limitation: describe
a way the choice could fail. - Next step: state what you would check
before deployment.

### Reference module 54: Model persistence

**Core idea.** A saved pipeline can preserve preprocessing and estimator
steps.

**Why it matters.** A model is not only an equation. It is a complete
process involving data collection, preprocessing, fitting, validation,
interpretation, and use. Decisions in this area can change the estimated
coefficients and the real-world reliability of predictions.

**Common trap.** Loading pickle-based artifacts from untrusted sources
can execute unsafe code.

**Practical check.** Store artifacts securely and document library
versions.

**Questions to ask** - What assumptions does this step make about the
data? - Could this step use information that would not be available at
prediction time? - How would I verify the decision with a plot, a
metric, or a controlled comparison? - Can I explain the consequence to
someone who does not know machine learning?

**Experiment** 1. Make a copy of a small dataset and record its initial
shape and target distribution. 2. Apply the technique or check described
above. 3. Compare the resulting features, coefficients, residuals, or
held-out metric. 4. Write down what changed and whether that change is
expected. 5. Repeat with a different random seed or split when the
experiment permits it.

**Reporting template** - Decision: describe the choice in one
sentence. - Rationale: explain why the choice fits the task. - Evidence:
name the plot, metric, or validation result used. - Limitation: describe
a way the choice could fail. - Next step: state what you would check
before deployment.

### Reference module 55: Monitoring

**Core idea.** Deployed data can drift from the training distribution.

**Why it matters.** A model is not only an equation. It is a complete
process involving data collection, preprocessing, fitting, validation,
interpretation, and use. Decisions in this area can change the estimated
coefficients and the real-world reliability of predictions.

**Common trap.** A once-good model may degrade as conditions change.

**Practical check.** Monitor input distributions, errors when labels
arrive, and subgroup performance.

**Questions to ask** - What assumptions does this step make about the
data? - Could this step use information that would not be available at
prediction time? - How would I verify the decision with a plot, a
metric, or a controlled comparison? - Can I explain the consequence to
someone who does not know machine learning?

**Experiment** 1. Make a copy of a small dataset and record its initial
shape and target distribution. 2. Apply the technique or check described
above. 3. Compare the resulting features, coefficients, residuals, or
held-out metric. 4. Write down what changed and whether that change is
expected. 5. Repeat with a different random seed or split when the
experiment permits it.

**Reporting template** - Decision: describe the choice in one
sentence. - Rationale: explain why the choice fits the task. - Evidence:
name the plot, metric, or validation result used. - Limitation: describe
a way the choice could fail. - Next step: state what you would check
before deployment.

### Reference module 56: Reproducibility

**Core idea.** Seeds, package versions, data versions, and split
definitions improve reproducibility.

**Why it matters.** A model is not only an equation. It is a complete
process involving data collection, preprocessing, fitting, validation,
interpretation, and use. Decisions in this area can change the estimated
coefficients and the real-world reliability of predictions.

**Common trap.** A seed alone does not guarantee bitwise-identical
results across all environments.

**Practical check.** Record environment and data provenance.

**Questions to ask** - What assumptions does this step make about the
data? - Could this step use information that would not be available at
prediction time? - How would I verify the decision with a plot, a
metric, or a controlled comparison? - Can I explain the consequence to
someone who does not know machine learning?

**Experiment** 1. Make a copy of a small dataset and record its initial
shape and target distribution. 2. Apply the technique or check described
above. 3. Compare the resulting features, coefficients, residuals, or
held-out metric. 4. Write down what changed and whether that change is
expected. 5. Repeat with a different random seed or split when the
experiment permits it.

**Reporting template** - Decision: describe the choice in one
sentence. - Rationale: explain why the choice fits the task. - Evidence:
name the plot, metric, or validation result used. - Limitation: describe
a way the choice could fail. - Next step: state what you would check
before deployment.

### Reference module 57: Fairness and subgroup errors

**Core idea.** Aggregate metrics can hide errors for particular groups.

**Why it matters.** A model is not only an equation. It is a complete
process involving data collection, preprocessing, fitting, validation,
interpretation, and use. Decisions in this area can change the estimated
coefficients and the real-world reliability of predictions.

**Common trap.** A model may perform poorly for a minority or
underrepresented subgroup.

**Practical check.** Evaluate error slices when lawful, appropriate, and
supported by adequate data.

**Questions to ask** - What assumptions does this step make about the
data? - Could this step use information that would not be available at
prediction time? - How would I verify the decision with a plot, a
metric, or a controlled comparison? - Can I explain the consequence to
someone who does not know machine learning?

**Experiment** 1. Make a copy of a small dataset and record its initial
shape and target distribution. 2. Apply the technique or check described
above. 3. Compare the resulting features, coefficients, residuals, or
held-out metric. 4. Write down what changed and whether that change is
expected. 5. Repeat with a different random seed or split when the
experiment permits it.

**Reporting template** - Decision: describe the choice in one
sentence. - Rationale: explain why the choice fits the task. - Evidence:
name the plot, metric, or validation result used. - Limitation: describe
a way the choice could fail. - Next step: state what you would check
before deployment.

### Reference module 58: Communication

**Core idea.** Explain metrics and limitations in the language of the
application.

**Why it matters.** A model is not only an equation. It is a complete
process involving data collection, preprocessing, fitting, validation,
interpretation, and use. Decisions in this area can change the estimated
coefficients and the real-world reliability of predictions.

**Common trap.** A single headline score can hide meaningful failure
cases.

**Practical check.** State target units, test design, baseline, and
major caveats.

**Questions to ask** - What assumptions does this step make about the
data? - Could this step use information that would not be available at
prediction time? - How would I verify the decision with a plot, a
metric, or a controlled comparison? - Can I explain the consequence to
someone who does not know machine learning?

**Experiment** 1. Make a copy of a small dataset and record its initial
shape and target distribution. 2. Apply the technique or check described
above. 3. Compare the resulting features, coefficients, residuals, or
held-out metric. 4. Write down what changed and whether that change is
expected. 5. Repeat with a different random seed or split when the
experiment permits it.

**Reporting template** - Decision: describe the choice in one
sentence. - Rationale: explain why the choice fits the task. - Evidence:
name the plot, metric, or validation result used. - Limitation: describe
a way the choice could fail. - Next step: state what you would check
before deployment.

## Appendix A. Reusable evaluation function

``` python
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def regression_report(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError("Actual and predicted arrays must have the same shape.")
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mse,
        "rmse": np.sqrt(mse),
        "r2": r2_score(y_true, y_pred),
    }
```

## Appendix B. A minimal reproducible experiment

``` python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

rng = np.random.default_rng(7)
X = rng.uniform(0, 100, size=(300, 1))
noise = rng.normal(0, 12, size=300)
y = 15 + 2.4 * X[:, 0] + noise

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=7
)
model = LinearRegression().fit(X_train, y_train)
pred = model.predict(X_test)

print("Intercept:", model.intercept_)
print("Slope:", model.coef_[0])
print("MAE:", mean_absolute_error(y_test, pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, pred)))
print("R²:", r2_score(y_test, pred))

order = np.argsort(X_train[:, 0])
plt.scatter(X_train[:, 0], y_train, alpha=0.6, label="Training observations")
plt.plot(X_train[order, 0], model.predict(X_train[order]), label="Fitted line")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Synthetic data: linear trend plus noise")
plt.legend()
plt.grid(alpha=0.25)
plt.show()
```

## Appendix C. Shape and data sanity checks

``` python
assert X_train.ndim == 2
assert y_train.ndim == 1
assert X_train.shape[0] == y_train.shape[0]
assert np.isfinite(X_train).all()
assert np.isfinite(y_train).all()
```

These assertions catch common shape and non-finite-value problems. They
do not establish that the data is representative, leakage-free, or
scientifically valid.

## Appendix D. Suggested repository structure

``` text
linear-regression-guide/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_intuition_and_scatter_plot.ipynb
│   ├── 02_ols_by_hand.ipynb
│   ├── 03_gradient_descent.ipynb
│   ├── 04_metrics_and_residuals.ipynb
│   └── 05_house_price_pipeline.ipynb
├── src/
│   ├── linear_regression_gd.py
│   ├── evaluation.py
│   └── plotting.py
├── data/
│   └── README.md
├── tests/
│   ├── test_shapes.py
│   └── test_predictions.py
└── requirements.txt
```

## Appendix E. Suggested dependencies

``` text
numpy
pandas
matplotlib
scikit-learn
jupyter
```

Install with:

``` bash
python -m pip install -r requirements.txt
```

Pin tested versions for a published reproducible project. Avoid
committing private, licensed, or sensitive datasets without permission.

## Appendix F. Pull-request checklist

-   [ ] Code blocks have been run in a clean environment.
-   [ ] Equations use consistent notation.
-   [ ] Plots have axis labels and meaningful titles.
-   [ ] Metric values are labeled with the dataset split.
-   [ ] Synthetic data is clearly marked as synthetic.
-   [ ] Data source and license are documented.
-   [ ] No credentials or private datasets are committed.
-   [ ] The README does not claim causal conclusions from prediction
    alone.

### Mathematical note 1: The mean baseline

The mean predictor outputs the training target mean for every row. It is
a useful reference for squared-error evaluation.

**Caution:** The test-set mean should not be used to train the baseline
because it uses held-out target information.

**Self-check:** Can you explain this concept, show a small example, and
describe how you would verify it empirically?

### Mathematical note 2: The median baseline

The median predictor is a useful simple reference when absolute error is
the chosen objective.

**Caution:** Choose the baseline metric to match the problem and compare
on the same held-out rows.

**Self-check:** Can you explain this concept, show a small example, and
describe how you would verify it empirically?

### Mathematical note 3: Centering and intercepts

Centering changes the zero point of a feature and therefore changes the
intercept interpretation.

**Caution:** Do not compare intercepts from differently centered models
without accounting for their reference points.

**Self-check:** Can you explain this concept, show a small example, and
describe how you would verify it empirically?

### Mathematical note 4: Feature correlation

Correlation can reveal redundant predictors, but pairwise correlation
does not capture every form of dependence.

**Caution:** A low pairwise correlation does not prove the absence of
multicollinearity.

**Self-check:** Can you explain this concept, show a small example, and
describe how you would verify it empirically?

### Mathematical note 5: Conditioning

Ill-conditioned design matrices make small data perturbations capable of
changing coefficient estimates substantially.

**Caution:** Use stable numerical solvers and consider scaling or
regularization.

**Self-check:** Can you explain this concept, show a small example, and
describe how you would verify it empirically?

### Mathematical note 6: Degrees of freedom

Classical residual variance estimation accounts for the number of fitted
parameters.

**Caution:** Degrees-of-freedom formulas depend on model and inference
assumptions.

**Self-check:** Can you explain this concept, show a small example, and
describe how you would verify it empirically?

### Mathematical note 7: SSE geometry

For linear regression, squared error is a convex quadratic function of
coefficients.

**Caution:** Convexity does not make every arbitrary numerical update
stable.

**Self-check:** Can you explain this concept, show a small example, and
describe how you would verify it empirically?

### Mathematical note 8: Gradient components

Each gradient component indicates how the objective changes with one
parameter.

**Caution:** A gradient's scale depends on feature scale and objective
definition.

**Self-check:** Can you explain this concept, show a small example, and
describe how you would verify it empirically?

### Mathematical note 9: Learning curves

Learning curves compare training and validation performance as the
amount of training data changes.

**Caution:** A gap between curves may reflect variance, but
interpretation depends on data quality and distribution.

**Self-check:** Can you explain this concept, show a small example, and
describe how you would verify it empirically?

### Mathematical note 10: Residual standard error

A residual standard error estimates residual scale in classical
regression settings.

**Caution:** It is not identical to test RMSE, which measures
predictions on held-out observations.

**Self-check:** Can you explain this concept, show a small example, and
describe how you would verify it empirically?

### Mathematical note 11: Prediction distribution

Errors can differ across target ranges, locations, or subgroups.

**Caution:** Inspect error slices rather than relying only on a single
aggregate score.

**Self-check:** Can you explain this concept, show a small example, and
describe how you would verify it empirically?

### Mathematical note 12: Calibration

For regression, calibration can mean that predictions align with
observed averages across prediction ranges or groups.

**Caution:** A strong R² does not guarantee good calibration in every
slice.

**Self-check:** Can you explain this concept, show a small example, and
describe how you would verify it empirically?

### Mathematical note 13: Data drift

Drift means the deployment data distribution differs from the reference
data distribution.

**Caution:** Drift is a warning signal, not automatic proof that
predictive performance has fallen.

**Self-check:** Can you explain this concept, show a small example, and
describe how you would verify it empirically?

### Mathematical note 14: Concept drift

Concept drift means the relationship between features and target changes
over time.

**Caution:** Input-distribution monitoring alone may not detect all
concept drift.

**Self-check:** Can you explain this concept, show a small example, and
describe how you would verify it empirically?

### Mathematical note 15: Target leakage

Target leakage occurs when a feature encodes information derived from
the target or a future outcome.

**Caution:** High validation scores can be a symptom of leakage rather
than excellent modeling.

**Self-check:** Can you explain this concept, show a small example, and
describe how you would verify it empirically?

### Mathematical note 16: Selection bias

The data inclusion process can make observed relationships differ from
the target population.

**Caution:** A more complex algorithm cannot automatically repair a
biased sample.

**Self-check:** Can you explain this concept, show a small example, and
describe how you would verify it empirically?

### Mathematical note 17: Missingness indicators

A missingness indicator can sometimes encode useful information about
the data collection process.

**Caution:** Its usefulness and ethics depend on why data is missing and
how the model will be used.

**Self-check:** Can you explain this concept, show a small example, and
describe how you would verify it empirically?

### Mathematical note 18: Robustness

Robustness asks whether the model behaves sensibly under plausible
perturbations and shifts.

**Caution:** Good average test performance is not a complete robustness
evaluation.

**Self-check:** Can you explain this concept, show a small example, and
describe how you would verify it empirically?

### Mathematical note 19: Sensitivity analysis

Sensitivity analysis studies how predictions or conclusions change when
assumptions or inputs change.

**Caution:** A single fitted coefficient table does not show all forms
of model sensitivity.

**Self-check:** Can you explain this concept, show a small example, and
describe how you would verify it empirically?

### Mathematical note 20: Reproducible reporting

A good report states data source, split method, preprocessing, model,
metrics, and limitations.

**Caution:** A number without context is hard to compare or reproduce.

**Self-check:** Can you explain this concept, show a small example, and
describe how you would verify it empirically?

# Active-recall questions

Use these prompts after reading the guide. Answer from memory, then
verify your explanation against the relevant section.

1.  Define the target, one observation, and prediction time.
2.  Write the simple and multiple linear regression equations.
3.  Explain the slope, intercept, coefficient, and residual.
4.  Derive the ordinary least squares slope and intercept.
5.  Explain the dimensions of the design matrix.
6.  Compare SSE, MSE, MAE, and RMSE.
7.  Derive the gradient of MSE for the intercept and coefficients.
8.  Explain the effect of an excessively large learning rate.
9.  Explain why scaling may help gradient descent.
10. Interpret an R² of 0, 1, or a negative value on test data.
11. Explain why R² is not a percentage of correct predictions.
12. Describe residual patterns that may reveal curvature or
    heteroscedasticity.
13. Distinguish outliers, leverage, and influence.
14. Explain multicollinearity and its effect on coefficient stability.
15. Describe how to encode categorical variables.
16. Explain how preprocessing leakage occurs and how a pipeline prevents
    it.
17. Choose a split strategy for time series or grouped observations.
18. Compare Ridge, Lasso, and Elastic Net.
19. Explain why polynomial regression is linear in its coefficients.
20. Distinguish prediction intervals from confidence intervals.
21. Explain why predictive association does not establish causality.
22. Describe how to evaluate subgroup errors and monitor drift.
23. List the information needed to reproduce a regression experiment.
24. Write a brief model card with intended use, limitations, and
    evaluation results.

## Final reminder

A strong regression project does more than fit a line. It defines the
prediction task carefully, prevents leakage, evaluates against a
baseline on unseen data, inspects residuals, interprets coefficients in
context, and documents limitations.
