# Quantitative Factor Investing with the Mainsequence SDK

Dive into quantitative factor investing with this repository! It guides you through building and analyzing factor-based strategies using the mainsequence-sdk.

Perfect for financial professionals, students, and researchers, this project helps you apply factor investing using real data and modern tools.

This repository is the starting point for an enterprise-grade, agentic quantitative factor investment engine on the Mainsequence platform. It's built to develop 12 style factors following Axioma/Barra standards.

Need better accuracy? Our [data_connectors](/mainsequence-sdk/data-connectors/tree/main/data_connectors "data_connectors") let you easily swap data sources.

What's more, the repository also includes an agentic application integrated with MainSequence Presentation Studio. This showcases how any workflow can be transformed into a powerful agentic workflow.



## Overview
Factor investing is a strategy that chooses securities based on specific attributes or "factors" that have been associated with higher returns. This repository explores the entire lifecycle of a quantitative investing process:

* Data Acquisition: Using the mainsequence-sdk to fetch clean, reliable financial data.

* Factor Research: Identifying and constructing well-known investment factors (e.g., Value, Momentum, Size, Quality).

* Data Workflow Management: Creating and managing the update processes for the financial data workflows that feed the models.

* Application: Deploying and visualizing the strategies in a user-facing application.

## How It Works
The core of this project relies on the Mainsequence SDK, which acts as the data bridge to financial markets. The SDK is used to seamlessly pull historical pricing data, company fundamentals, and other essential financial metrics.


## Repository Structure
This repository is organized into two main components:

* `/time_series`: This directory contains the core logic for data management. Its primary component is the TimeSerie class, which creates and manages the update process for the data workflows that the quantitative strategies depend on.

* `/app`: This directory contains a user-facing application. It builds upon the data workflows from the `/time_series` folder to provide an interactive way to visualize factor performance, explore data, and see the results of the investment strategies.


## Getting Started
To begin exploring these concepts, follow these steps:

### 1. Clone the Repository:

```shell
git clone https://github.com/mainsequence-sdk/QuantitativeFactorInvesting.git
cd QuantitativeFactorInvesting

```
### 2. Install Dependencies:
This project requires Python 3.8+ and several packages. The primary dependency is the mainsequence-sdk. You can install all required packages by running:

### 3. Configure the SDK:
The mainsequence-sdk requires authentication. You will need to configure it with your API credentials. Please refer to the SDK's documentation for instructions on how to set up your credentials.

### 4. Run the Application:
Navigate to the app directory and follow the instructions within its own README or main script to launch the application.

## Disclaimer
The content of this repository is for educational and research purposes only. It is not intended to be, and should not be construed as, financial, investment, or trading advice. All investment strategies and models are demonstrated for illustrative purposes, and their past performance is not indicative of future results.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.