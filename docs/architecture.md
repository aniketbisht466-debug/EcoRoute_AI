
# EcoRoute AI — System Architecture

## Architecture Diagram

```mermaid
flowchart TD
    A[Dryad Truck Dataset] --> B[Data Engineering]
    B --> C[Feature Engineering]
    C --> D[Scientific EDA]
    C --> E[Grouped Dataset Splitting]
    E --> F[Model Training]
    F --> G[Hyperparameter Optimization]
    G --> H[Final XGBoost Model]
    H --> I[Explainable AI]
    H --> J[Fuel Prediction Engine]
    K[OpenStreetMap Network] --> L[Route Optimizer]
    J --> L
    L --> M[Dynamic Driving Simulator]
    J --> M
    J --> N[FastAPI Backend]
    L --> N
    M --> N
    D --> O[Analytics API]
    I --> O
    O --> P[Research Analytics Dashboard]
    N --> Q[Interactive Route Planner]
    N --> R[Fuel Prediction Laboratory]
```

## Architecture Components

- Data Layer: Dataset ingestion and validation
- Intelligence Layer: Machine learning and explainability
- Geospatial Layer: Road network and route optimization
- Simulation Layer: Dynamic driving profile generation
- Service Layer: FastAPI REST endpoints
- Presentation Layer: Interactive web dashboards
- Quality Layer: Automated system integration tests
