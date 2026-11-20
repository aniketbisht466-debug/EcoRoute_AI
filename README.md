
# EcoRoute AI

### Intelligent Freight Mobility & Fuel-Aware Route Optimization

EcoRoute AI is a machine learning and geospatial research
platform designed to estimate heavy-duty vehicle fuel
consumption, calculate fuel-derived carbon emissions,
and compare alternative driving routes.

## Core Features

- Machine learning-based fuel prediction
- Shortest, fastest and fuel-aware routing
- OpenStreetMap road network integration
- Dynamic vehicle driving simulation
- Estimated CO2 emissions
- Interactive route planning dashboard
- Fuel Prediction Laboratory
- Scientific research analytics
- Explainable machine learning
- Model-domain validation
- REST API integration

## Dataset

The project uses the Dryad Heavy-Duty Truck
Trajectories dataset.

DOI: 10.6086/D1BT3K

- Telemetry observations: 190,802
- Total trajectories: 555
- Baseline trajectories: 208
- Eco-driving trajectories: 347

## Machine Learning Models

- Dummy Regressor
- Linear Regression
- Ridge Regression
- Polynomial Ridge Regression
- Random Forest
- XGBoost

The pipeline includes trajectory-level dataset splitting,
grouped cross-validation, hyperparameter optimization,
and held-out test evaluation.

## Initial Validation Benchmark

| Model | Validation R2 |
|---|---:|
| Random Forest | 0.889065 |
| XGBoost | 0.888477 |
| Polynomial Ridge | 0.804704 |
| Linear Regression | 0.613280 |
| Ridge | 0.613279 |
| Dummy Regressor | -0.000440 |

Final optimized-model metrics are stored separately in:

reports/ml/final_test_metrics.json

## Final Model Evaluation

The optimized XGBoost model was selected for the
final EcoRoute AI inference engine.

| Evaluation Metric | Result |
|---|---:|
| Selected Model | XGBoost |
| Held-out Test R2 | 0.886252 |
| MAE | 0.000543365 L/s |
| RMSE | 0.000937933 L/s |
| Test Observations | 31,322 |
| Test Trajectories | 84 |
| Actual Integrated Fuel | 82.7003 L |
| Predicted Integrated Fuel | 82.2615 L |
| Aggregate Fuel Bias | -0.5306% |
| Trajectory MAPE | 5.1353% |
| Trajectory WAPE | 4.5909% |

## Explainable AI

- SHAP analysis: Completed
- Permutation feature importance: Completed
- Native feature importance
- Feature response profiles
- Individual prediction explanations

## System Validation

- Automated integration tests: 19/19 passed
- Dependency compatibility check: Passed

These results represent internal dataset evaluation,
not independently validated real-world navigation
or fuel consumption accuracy.


## Technology Stack

- Python
- NumPy
- Pandas
- Scikit-learn
- XGBoost
- FastAPI
- Uvicorn
- OSMnx
- NetworkX
- GeoPandas
- Folium
- Leaflet
- HTML, CSS and JavaScript
- Matplotlib
- SHAP (optional explanations)

## Project Architecture

    EcoRoute_AI/
    ├── assets/
    ├── configs/
    ├── dashboard/
    ├── data/
    ├── docs/
    ├── models/
    ├── reports/
    ├── src/
    │   ├── analysis/
    │   ├── api/
    │   ├── data/
    │   ├── emissions/
    │   ├── features/
    │   ├── inference/
    │   ├── ml/
    │   ├── models/
    │   ├── routing/
    │   └── visualization/
    ├── tests/
    ├── launch.cmd
    ├── requirements.txt
    └── README.md

## Local Installation

The original development environment uses Windows
and Python 3.13.

Create a virtual environment:

    python -m venv .venv

Install dependencies:

    .venv\Scripts\python.exe -m pip install -r requirements.txt

Before launching, restore or regenerate the required
datasets, trained model, model metadata and road network.

Large datasets and model binaries are excluded from
the default Git repository.

## Running the Application

On Windows, run launch.cmd.

Alternatively:

    .venv\Scripts\python.exe -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000

## Application Endpoints

| Module | Endpoint |
|---|---|
| Route Planner | /dashboard |
| Fuel Prediction Lab | /fuel-lab |
| Research Analytics | /analytics |
| API Documentation | /docs |
| Health Check | /health |
| Model Information | /model/info |

## Quality Assurance

The automated system integration suite passed
19 out of 19 tests in the local development environment.

Run tests:

    .venv\Scripts\python.exe -m unittest discover -s tests -p "test_system.py" -v

## Research Limitations

- Fuel predictions are based on historical truck telemetry.
- Route-level telemetry is simulated.
- Traffic conditions are not live.
- Payload and elevation are not incorporated into routing.
- Truck-specific road restrictions require further validation.
- Unsupported fuel estimates are withheld.
- Carbon emissions are estimated from diesel consumption.
- Dynamic simulation evaluates existing candidate routes.
- External generalization requires further validation.

This project is a research prototype, not a certified
commercial navigation or emissions measurement system.

## Data Attribution

Heavy-Duty Truck Trajectories Dataset — Dryad.

Road network data:
© OpenStreetMap contributors.

https://www.openstreetmap.org/copyright

## Future Development

- Live traffic integration
- Elevation-aware fuel modelling
- Truck-specific navigation
- Fleet-level optimization
- External model validation
- Dynamic fuel-aware route optimization
- Cloud deployment
## Application Preview

### Intelligent Route Planner

![Route Planner](assets/screenshots/route_planner.png)

### Fuel Prediction Laboratory

![Fuel Lab](assets/screenshots/fuel_lab.png)

### Research Analytics

![Research Analytics](assets/screenshots/research_analytics.png)

### Explainable AI

![Explainable AI](assets/screenshots/explainable_ai.png)