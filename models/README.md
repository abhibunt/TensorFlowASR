# Models

This directory is for easy access trained models

Recommended structure:

```bash
└── {{model_name}}
    ├── bp
    │   ├── assets
    │   ├── keras_metadata.pb
    │   ├── saved_model.pb
    │   └── variables
    │       ├── variables.data-00000-of-00001
    │       └── variables.index
    ├── config.yml
    ├── conformer.subwords
    └── latest.h5
```