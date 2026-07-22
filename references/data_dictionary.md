# Data Dictionary

This document describes the primary features used in the **Customer Segmentation in the US** project. While the original Survey of Consumer Finances (SCF) dataset contains many more variables, the features listed below are the ones most frequently used throughout this analysis.

| Feature | Description |
|---------|-------------|
| **ACTBUS** | Total value of actively managed business(es) (2019 USD). |
| **AGE** | Age of the household reference person. |
| **ASSET** | Total value of household assets (2019 USD). |
| **BUS** | Total value of businesses in which the household has an active or non-active ownership interest (2019 USD). |
| **DEBT** | Total household debt (2019 USD). |
| **EQUITY** | Total value of financial assets invested in stocks (2019 USD). |
| **FIN** | Total value of household financial assets (2019 USD). |
| **HBUS** | Indicator of whether the household owns an active or non-active business. |
| **HOUSES** | Total value of the household's primary residence (2019 USD). |
| **INCCAT** | Household income percentile category. |
| **INCOME** | Total annual household income (2019 USD). |
| **KGBUS** | Unrealized capital gains or losses on businesses (2019 USD). |
| **KGTOTAL** | Total unrealized capital gains or losses for the household (2019 USD). |
| **NETWORTH** | Total household net worth (2019 USD). |
| **NHNFIN** | Total non-financial assets excluding the primary residence (2019 USD). |
| **NFIN** | Total value of household non-financial assets (2019 USD). |
| **NetWorth** | Total Assets − Total Liabilities (Debts)

| AGECL Value | Age Category |
| ----------- | ------------ |
| 1           | Under 35     |
| 2           | 35–44        |
| 3           | 45–54        |
| 4           | 55–64        |
| 5           | 65–74        |
| 6           | 75 or Older  |


**RACE Encoding:**
- **1:** White/Non-Hispanic
- **2:** Black/African-American
- **3:** Hispanic
- **5:** Other

## Notes

- All monetary values are reported in **2019 U.S. dollars (USD)**.
- The variables listed above represent only a subset of the complete Survey of Consumer Finances (SCF) dataset.
- Additional variables and detailed metadata are available in the official Survey of Consumer Finances documentation.

## Official Documentation

For the complete variable glossary and documentation, refer to the **Survey Documentation and Analysis (SDA) Codebook** provided by the **University of California, Berkeley** and the official **Survey of Consumer Finances (SCF)** documentation.

---

# Data Source

This project uses data derived from the **2019 Survey of Consumer Finances (SCF)**, conducted by the **Federal Reserve Board** in collaboration with the **U.S. Department of the Treasury** and administered by the **National Opinion Research Center (NORC) at the University of Chicago**.

The version used in this project was provided as part of the **WorldQuant University Applied Data Science Lab**.

---

# Acknowledgements

Special thanks to **Professor Nicholas Cifuentes-Goodbody** and the **WorldQuant University Applied Data Science Lab** for designing the project and making the dataset available for educational purposes.

---

# References

1. Hennig, C., & Liao, T. F. (2013). *How to find an appropriate clustering for mixed-type variables with application to socio-economic stratification.* Journal of the Royal Statistical Society: Series C (Applied Statistics), 62(3), 309–369.

2. Tatsat, H., Puri, S., & Lookabaugh, B. (2020). *Machine Learning and Data Science Blueprints for Finance: From Building Trading Strategies to Robo-Advisors Using Python.* O'Reilly Media.