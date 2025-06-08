# utec-bank-grupo5

## Arquitectura

```mermaid
flowchart TD
    A[⚡ GitHub Actions] --> B[📁 CSV Data]
    B --> C[📤 S3 Upload]
    C --> D[🤖 SageMaker Training]
    D --> E[📊 Log Metrics]
    E --> F[📦 SageMaker Model]
    F --> G[🚀 SageMaker Endpoint]
    
    style A fill:#f0f0f0
    style B fill:#e1f5fe
    style C fill:#f8f9fa
    style D fill:#f3e5f5
    style E fill:#e8f5e8
    style F fill:#ffeaa7
    style G fill:#fff3e0
```

## Scaffolding
```
proyecto/
├── .github/workflows/
│   └── main.yml
├── src/
│   ├── pipeline_launcher.py
│   ├── train_script.py
│   ├── validate_date.py
├── data/
│   ├── ***.csv
└── requirements.txt
```

## Gitflow
```mermaid
gitGraph
    commit
    commit
    branch develop
    checkout develop
    commit
    commit
    branch feature/example
    checkout feature/example
    commit
    commit
    checkout develop
    merge feature/example
    branch release/1.0.0
    checkout release/1.0.0
    commit
    checkout main
    merge release/1.0.0
    checkout develop
    merge release/1.0.0
    branch hotfix/1.0.1
    checkout hotfix/1.0.1
    commit
    checkout main
    merge hotfix/1.0.1
    checkout develop
    merge hotfix/1.0.1

```

## CI/CD pipeline

```mermaid
flowchart TD
    A[Trigger: Push to main/develop\nor Manual Dispatch] --> B{Environment\nDetection}
    B -->|main branch| C[PROD Environment]
    B -->|develop branch| D[TEST Environment]
    C --> E[Use PROD AWS Credentials]
    D --> F[Use TEST AWS Credentials]
    E --> G[Configure AWS]
    F --> G
    G --> H[Checkout Code]
    H --> I[Setup Python 3.9]
    I --> J[Install Dependencies]
    J --> K[Validate Data]
    K --> L[Train Model on SageMaker]
```