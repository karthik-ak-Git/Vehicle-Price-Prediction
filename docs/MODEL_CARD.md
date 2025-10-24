# Model Card: Vehicle Price Prediction System

## Model Details

**Model Name**: Vehicle Price Prediction System  
**Version**: 2.0.0  
**Date**: October 2025  
**Model Type**: Gradient Boosted Decision Trees (XGBoost/LightGBM/CatBoost ensemble)  
**License**: MIT  

### Model Architecture

- **Primary Algorithm**: XGBoost Regressor
- **Backup Models**: LightGBM, CatBoost, Random Forest, Ridge Regression
- **Feature Engineering**: Custom preprocessing pipeline with scaling and encoding
- **Hyperparameter Optimization**: RandomizedSearchCV with 3-fold cross-validation

## Intended Use

### Primary Use Cases

1. **Price Estimation**: Provide fair market price estimates for used vehicles
2. **Market Analysis**: Understand pricing trends and factors
3. **Buyer Guidance**: Help buyers make informed purchasing decisions
4. **Seller Reference**: Assist sellers in pricing their vehicles competitively

### Intended Users

- Individual car buyers and sellers
- Automotive dealerships
- Insurance companies for valuation
- Market researchers
- Data scientists and ML practitioners

### Out-of-Scope Use Cases

- Legal valuation for court cases (consult certified appraisers)
- Insurance claim settlements (requires official appraisal)
- Tax assessment purposes
- Commercial vehicle pricing
- Heavy machinery or specialty vehicles

## Training Data

### Data Sources

- **Primary**: CarDekho datasets (multiple versions)
- **Records**: ~12,283 vehicle listings
- **Time Period**: 2015-2024
- **Geography**: Primarily Indian automotive market
- **Data Split**: 70% train, 15% validation, 15% test

### Features (14 core features → 108 after encoding)

**Numeric Features**:
- `km_driven`: Kilometers driven (odometer reading)
- `engine_cc`: Engine displacement in cubic centimeters
- `max_power_bhp`: Maximum power in brake horsepower
- `age`: Vehicle age calculated from manufacturing year
- `mileage_value`: Fuel efficiency in kmpl/kmkg
- `torque_nm`: Engine torque in Newton-meters
- `torque_rpm`: RPM at which torque is delivered
- `seats`: Number of passenger seats

**Categorical Features**:
- `make`: Vehicle manufacturer (Maruti, Toyota, Honda, etc.)
- `fuel`: Fuel type (Petrol, Diesel, CNG, Electric, Hybrid, LPG)
- `transmission`: Transmission type (Manual, Automatic)
- `owner`: Ownership history (First, Second, Third, Fourth & Above)
- `seller_type`: Type of seller (Individual, Dealer, Trustmark Dealer)
- `mileage_unit`: Unit of mileage measurement

### Data Quality

- **Missing Values**: Handled via imputation (median for numeric, mode for categorical)
- **Outliers**: Retained but monitored (prices >₹50L flagged for review)
- **Duplicates**: Removed during preprocessing
- **Validation**: Manual inspection of sample records

## Model Performance

### Metrics (Test Set)

- **R² Score**: 0.908 (90.8% variance explained)
- **Mean Absolute Error (MAE)**: ₹129,795
- **Root Mean Squared Error (RMSE)**: ₹198,453
- **Median Absolute Error**: ₹59,211
- **Mean Absolute Percentage Error (MAPE)**: 18.3%

### Performance by Price Range

| Price Range | Count | MAE | R² | MAPE |
|-------------|-------|-----|-----|------|
| Under ₹5L | 35% | ₹42,150 | 0.89 | 12.5% |
| ₹5L-10L | 40% | ₹98,420 | 0.91 | 15.8% |
| ₹10L-20L | 18% | ₹215,600 | 0.88 | 21.2% |
| Above ₹20L | 7% | ₹485,300 | 0.82 | 28.4% |

**Observation**: Model performs better on budget and mid-range vehicles (majority of market).

### Feature Importance (Top 10)

1. **transmission_Manual** (17.6%) - Manual vs automatic transmission
2. **max_power_bhp** (16.9%) - Engine power rating
3. **make_None** (9.0%) - Missing/unspecified manufacturer
4. **transmission_Automatic** (5.8%) - Automatic transmission indicator
5. **owner_0** (5.1%) - First owner status
6. **age** (4.8%) - Vehicle age in years
7. **km_driven** (4.2%) - Odometer reading
8. **engine_cc** (3.9%) - Engine displacement
9. **fuel_Diesel** (3.5%) - Diesel fuel type
10. **seats** (2.8%) - Number of seats

## Limitations

### Known Limitations

1. **Geographic Scope**: Trained primarily on Indian market data; may not generalize to other regions
2. **Luxury Vehicles**: Lower accuracy for luxury/exotic cars (limited training data)
3. **Recent Models**: Less accurate for very new vehicles (<1 year old)
4. **Electric Vehicles**: Limited EV data in training set
5. **Condition Assessment**: Cannot assess physical condition, accident history, or service records
6. **Market Dynamics**: Cannot predict sudden market shifts or economic changes

### Edge Cases

- **Vintage/Classic Cars**: Not suitable for collectible vehicles
- **Modified Vehicles**: Cannot account for aftermarket modifications
- **Commercial Use**: Not optimized for commercial vehicle pricing
- **Regional Variations**: Prices may vary significantly by location within India

## Ethical Considerations

### Fairness

- **No Discriminatory Features**: Model does not use owner demographics
- **Transparency**: Feature importance and decision factors are explainable
- **Bias Monitoring**: Regular audits for systematic over/under-pricing patterns

### Privacy

- **No Personal Data**: Training data contains no personally identifiable information
- **Anonymized Sources**: All data is publicly available or anonymized

### Potential Harms

- **Over-reliance**: Users should not rely solely on model predictions for major financial decisions
- **Market Impact**: Widespread use could influence market pricing dynamics
- **Misinformation**: Predictions should be clearly labeled as estimates, not guarantees

## Recommendations

### Best Practices

1. **Cross-reference**: Compare predictions with multiple sources
2. **Physical Inspection**: Always inspect vehicle condition in person
3. **Professional Appraisal**: Seek certified appraisal for high-value transactions
4. **Market Research**: Consider local market conditions and trends
5. **Documentation**: Verify service history and ownership documents

### Update Frequency

- **Model Retraining**: Quarterly with new data
- **Feature Updates**: As needed based on market changes
- **Performance Monitoring**: Continuous via production metrics

## Maintenance & Monitoring

### Production Monitoring

- Prediction distribution tracking
- Outlier detection (predictions >3σ from mean)
- Feature drift detection
- Performance degradation alerts

### Model Updates

- Regular retraining with fresh data
- A/B testing for model improvements
- Version control and rollback capability
- User feedback integration

## Contact & Feedback

**Maintainer**: Karthik  
**Repository**: https://github.com/karthik-ak-Git/Vehicle-Price-Prediction  
**Issues**: Please report bugs or suggestions via GitHub Issues  
**Email**: karthik@example.com

## References

- XGBoost: Chen & Guestrin (2016)
- LightGBM: Ke et al. (2017)
- CatBoost: Prokhorenkova et al. (2018)
- Dataset: CarDekho (various sources)

---

**Last Updated**: October 23, 2025  
**Version**: 2.0.0
