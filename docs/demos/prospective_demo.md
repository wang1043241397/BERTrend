# BERTrend Prospective Demo

## Overview

The `bertrend_apps.prospective_demo` package provides a comprehensive web application for prospective analysis using BERTrend's topic modeling and signal detection capabilities. This application allows users to monitor information sources, analyze trends and signals, and generate reports based on the analysis.

## Features

The Prospective Demo application offers the following key features:

1. **Information Source Management**
   - Configure and manage data feeds
   - Monitor data collection status

2. **Model Management**
   - Train new models for specific time periods
   - Monitor model status by information source

3. **Signal Analysis**
   - Identify and analyze weak and strong signals
   - Visualize signal trends over time

4. **Dashboard Analysis**
   - Analyze topic evolution
   - Perform multifactorial analysis

5. **Report Generation**
   - Generate comprehensive reports based on the analysis
   - Customize report content and format

## Components

The package consists of several key components:

### Main Application

The main application (`app.py`) provides a Streamlit-based web interface with multiple tabs:
- "Veilles" (Watches/Monitoring) - for configuring data sources and checking data collection status
- "Modèles" (Models) - for monitoring models by watch
- "Tendances" (Trends) - for signal analysis
- "Analyses" - for dashboard analysis
- "Génération de rapports" (Report Generation) - for creating reports

The application includes internationalization support, allowing users to switch between French (default) and English interfaces.

### Data Processing

The package includes utilities for processing new data (`process_new_data.py`):
- Loading data for specific models and users
- Training new models for specific time periods
- Generating LLM interpretations of the models

### Authentication

The application includes authentication functionality (`authentication.py`) to manage user access and personalize the experience.

### Dashboard Components

Multiple dashboard components provide specialized visualizations and interfaces:
- `dashboard_analysis.py` - For analyzing topic evolution and trends
- `dashboard_signals.py` - For analyzing weak and strong signals
- `dashboard_common.py` - Common utilities for dashboard components

### Feed Management

The package includes utilities for managing information feeds:
- `feeds_config.py` - For configuring information sources
- `feeds_data.py` - For displaying data status
- `feeds_common.py` - Common utilities for feed management

### Report Generation

The package provides functionality for generating reports:
- `report_generation.py` - For creating reports based on the analysis
- `report_utils.py` - Utilities for report generation
- `detailed_report_template.html` - HTML template for reports

## Environment (.env)

BERTrend auto-loads a repository-level .env on import when python-dotenv is installed. Before running the Prospective Demo, ensure your .env at the repo root includes relevant variables, for example:
- BERTREND_BASE_DIR: base directory for BERTrend data/models/logs
- OpenAI/LLM: OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_DEFAULT_MODEL
- Optional: email settings for report sending, provider keys, and CUDA_VISIBLE_DEVICES

GPU selection: you can set CUDA_VISIBLE_DEVICES (e.g., "0" or "0,1"). When launching via Streamlit, you may prefix the command with CUDA_VISIBLE_DEVICES=<gpu>.

## Usage

### Starting the Application

To start the Prospective Demo application:

```bash
cd bertrend_apps/prospective_demo
CUDA_VISIBLE_DEVICES=<gpu_number> streamlit run app.py
```

### Authentication

The application includes authentication functionality. Users need to log in with their credentials to access the application. Authentication can be disabled for development purposes by setting `AUTHENTIFICATION = False` in `app.py`.

### Data Configuration

Users can configure information sources in the "Veilles" tab. This includes:
- Adding new information sources
- Configuring data collection parameters
- Monitoring data collection status

### Model Training

New models can be trained using the command-line interface:

```bash
python -m bertrend_apps.prospective_demo.process_new_data train-new-model <user_name> <model_id>
```

Or through the web interface in the "Modèles" tab.

### Data Regeneration

The application provides functionality to regenerate data from scratch, which can be useful for rebuilding models or refreshing analyses. This can be done with or without LLM analysis.

#### Command-line Interface

To regenerate data using the command-line interface:

```bash
python -m bertrend_apps.prospective_demo.process_new_data regenerate <user_name> <model_id> [--with-analysis/--no-with-analysis] [--since <YYYY-MM-dd>]
```

Parameters:
- `<user_name>`: Identifier of the user
- `<model_id>`: ID of the model to be regenerated from scratch
- `--with-analysis/--no-with-analysis`: Whether to include LLM analysis (default: `--with-analysis`)
- `--since <YYYY-MM-dd>`: Optional date to be considered as the beginning of the analysis

#### Examples

Regenerate all data with LLM analysis:
```bash
python -m bertrend_apps.prospective_demo.process_new_data regenerate user1 model123
```

Regenerate data without LLM analysis (faster):
```bash
python -m bertrend_apps.prospective_demo.process_new_data regenerate user1 model123 --no-with-analysis
```

Regenerate data since a specific date:
```bash
python -m bertrend_apps.prospective_demo.process_new_data regenerate user1 model123 --since 2023-01-01
```

#### Notes

- Regenerating data with LLM analysis may take significant time depending on the volume of data.
- The regeneration process loads all data for the specified model, optionally filters by date, and rebuilds all models from scratch.
- When regenerating without LLM analysis (`--no-with-analysis`), only the topic models are rebuilt without generating topic descriptions or signal analyses.

### Analysis

The application provides multiple analysis options:
- Signal analysis in the "Tendances" tab
- Dashboard analysis in the "Analyses" tab

### Report Generation

Users can generate reports based on the analysis in the "Génération de rapports" tab.

## Configuration

The package uses several configuration settings defined in `__init__.py`:
- `CONFIG_FEEDS_BASE_PATH` - Path for user feed configurations
- `BASE_MODELS_DIR` - Path for user models
- `DEFAULT_GRANULARITY` - Default time granularity for analysis
- `DEFAULT_WINDOW_SIZE` - Default window size for analysis

## Utilities

The package includes several utility modules to enhance functionality:

### Internationalization

The `i18n.py` module provides internationalization support, allowing users to switch between French (default) and English interfaces. The language selector is available in the sidebar.

### Performance Utilities

The `perf_utils.py` module includes utilities for optimizing performance, such as a function for selecting the least used GPU on systems with multiple GPUs.

### UI Enhancements

The `streamlit_utils.py` module provides enhanced UI components for the Streamlit interface, such as clickable dataframes with interactive buttons.

## Dependencies

The Prospective Demo package depends on:
- BERTrend core functionality
- Streamlit for the web interface
- Various data processing and visualization libraries

These dependencies are included in the "apps" optional dependency group in the main package.
