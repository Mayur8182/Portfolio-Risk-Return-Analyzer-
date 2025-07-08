"""
Data Quality Validator - World-class data validation and quality assurance
Ensures institutional-grade data accuracy with comprehensive validation rules
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import statistics
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ValidationSeverity(Enum):
    """Validation issue severity levels"""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"

@dataclass
class ValidationIssue:
    """Structure for validation issues"""
    severity: ValidationSeverity
    category: str
    description: str
    field: str
    value: Any
    expected_range: Optional[Tuple[float, float]] = None
    suggestion: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class DataQualityReport:
    """Comprehensive data quality report"""
    symbol: str
    total_records: int
    valid_records: int
    invalid_records: int
    quality_score: float
    completeness_score: float
    accuracy_score: float
    consistency_score: float
    timeliness_score: float
    issues: List[ValidationIssue]
    recommendations: List[str]
    generated_at: datetime
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        return {
            **asdict(self),
            'issues': [asdict(issue) for issue in self.issues],
            'severity_enum_values': {sev.name: sev.value for sev in ValidationSeverity}
        }

class DataQualityValidator:
    """
    World-class data quality validator with institutional standards
    Features:
    - Comprehensive validation rules
    - Real-time quality scoring
    - Anomaly detection
    - Data consistency checks
    - Completeness analysis
    - Accuracy verification
    """
    
    def __init__(self):
        # Quality thresholds
        self.quality_thresholds = {
            'min_volume': 1000,
            'max_daily_change': 0.50,  # 50% daily change threshold
            'min_price': 0.01,
            'max_price': 100000,
            'max_bid_ask_spread': 0.10,  # 10% spread threshold
            'min_data_points_per_day': 1,
            'max_missing_days': 5,
            'outlier_z_score': 4.0,
            'min_quality_score': 0.80
        }
        
        # Market hours and trading days
        self.market_open_hour = 9
        self.market_close_hour = 16
        self.trading_days = [0, 1, 2, 3, 4]  # Monday to Friday
        
        # Data freshness requirements
        self.max_data_age_hours = 24
        self.real_time_delay_threshold = 300  # 5 minutes
        
        # Statistical validation parameters
        self.confidence_level = 0.95
        self.rolling_window = 30  # days for rolling statistics
        
    def validate_market_data(self, data: pd.DataFrame, symbol: str) -> DataQualityReport:
        """
        Comprehensive validation of market data
        """
        try:
            issues = []
            recommendations = []
            
            # Basic data structure validation
            structure_issues = self._validate_data_structure(data, symbol)
            issues.extend(structure_issues)
            
            # Price validation
            price_issues = self._validate_prices(data, symbol)
            issues.extend(price_issues)
            
            # Volume validation
            volume_issues = self._validate_volume(data, symbol)
            issues.extend(volume_issues)
            
            # Temporal validation
            temporal_issues = self._validate_temporal_consistency(data, symbol)
            issues.extend(temporal_issues)
            
            # Statistical validation
            statistical_issues = self._validate_statistical_properties(data, symbol)
            issues.extend(statistical_issues)
            
            # Completeness validation
            completeness_issues = self._validate_completeness(data, symbol)
            issues.extend(completeness_issues)
            
            # Cross-validation with market norms
            market_issues = self._validate_market_norms(data, symbol)
            issues.extend(market_issues)
            
            # Calculate quality scores
            quality_scores = self._calculate_quality_scores(data, issues)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(issues, data)
            
            # Create comprehensive report
            report = DataQualityReport(
                symbol=symbol,
                total_records=len(data),
                valid_records=len(data) - len([i for i in issues if i.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.HIGH]]),
                invalid_records=len([i for i in issues if i.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.HIGH]]),
                quality_score=quality_scores['overall'],
                completeness_score=quality_scores['completeness'],
                accuracy_score=quality_scores['accuracy'],
                consistency_score=quality_scores['consistency'],
                timeliness_score=quality_scores['timeliness'],
                issues=issues,
                recommendations=recommendations,
                generated_at=datetime.now()
            )
            
            logger.info(f"Data quality validation completed for {symbol}: Quality Score = {quality_scores['overall']:.3f}")
            return report
            
        except Exception as e:
            logger.error(f"Error in data quality validation for {symbol}: {e}")
            return self._create_error_report(symbol, str(e))
    
    def _validate_data_structure(self, data: pd.DataFrame, symbol: str) -> List[ValidationIssue]:
        """Validate basic data structure and required columns"""
        issues = []
        
        try:
            # Check if data is empty
            if data.empty:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    category="Structure",
                    description="Dataset is empty",
                    field="data",
                    value=len(data),
                    suggestion="Ensure data source is providing valid data"
                ))
                return issues
            
            # Required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    category="Structure",
                    description=f"Missing required columns: {missing_columns}",
                    field="columns",
                    value=missing_columns,
                    suggestion="Ensure all OHLCV columns are present"
                ))
            
            # Check data types
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col in data.columns and not pd.api.types.is_numeric_dtype(data[col]):
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.HIGH,
                        category="Structure",
                        description=f"Column {col} is not numeric",
                        field=col,
                        value=str(data[col].dtype),
                        suggestion=f"Convert {col} to numeric type"
                    ))
            
            # Check index (should be datetime)
            if not isinstance(data.index, pd.DatetimeIndex):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.HIGH,
                    category="Structure",
                    description="Index is not datetime type",
                    field="index",
                    value=str(type(data.index)),
                    suggestion="Convert index to datetime"
                ))
            
        except Exception as e:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category="Structure",
                description=f"Error validating data structure: {e}",
                field="structure",
                value=str(e)
            ))
        
        return issues
    
    def _validate_prices(self, data: pd.DataFrame, symbol: str) -> List[ValidationIssue]:
        """Validate price data for accuracy and consistency"""
        issues = []
        
        try:
            price_columns = ['open', 'high', 'low', 'close']
            available_price_columns = [col for col in price_columns if col in data.columns]
            
            if not available_price_columns:
                return issues
            
            for idx, row in data.iterrows():
                # Check for negative or zero prices
                for col in available_price_columns:
                    if col in row and (pd.isna(row[col]) or row[col] <= 0):
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.CRITICAL,
                            category="Price",
                            description=f"Invalid {col} price: {row[col]}",
                            field=col,
                            value=row[col],
                            expected_range=(self.quality_thresholds['min_price'], self.quality_thresholds['max_price']),
                            suggestion="Check data source for price accuracy"
                        ))
                
                # Check OHLC relationships
                if all(col in row for col in ['open', 'high', 'low', 'close']):
                    if not (row['low'] <= row['open'] <= row['high'] and 
                           row['low'] <= row['close'] <= row['high']):
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.HIGH,
                            category="Price",
                            description=f"Invalid OHLC relationship at {idx}",
                            field="ohlc",
                            value=f"O:{row['open']}, H:{row['high']}, L:{row['low']}, C:{row['close']}",
                            suggestion="Verify OHLC data integrity"
                        ))
                
                # Check for extreme price movements
                if 'open' in row and 'close' in row and row['open'] > 0:
                    daily_change = abs(row['close'] - row['open']) / row['open']
                    if daily_change > self.quality_thresholds['max_daily_change']:
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.MEDIUM,
                            category="Price",
                            description=f"Extreme price movement: {daily_change:.2%}",
                            field="daily_change",
                            value=daily_change,
                            expected_range=(0, self.quality_thresholds['max_daily_change']),
                            suggestion="Verify if this is a valid price movement or data error"
                        ))
            
            # Statistical outlier detection
            if 'close' in data.columns:
                returns = data['close'].pct_change().dropna()
                if len(returns) > 10:
                    z_scores = np.abs(stats.zscore(returns))
                    outliers = returns[z_scores > self.quality_thresholds['outlier_z_score']]
                    
                    for idx, outlier_return in outliers.items():
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.MEDIUM,
                            category="Price",
                            description=f"Statistical outlier detected: {outlier_return:.4f}",
                            field="return_outlier",
                            value=outlier_return,
                            suggestion="Review this data point for accuracy"
                        ))
        
        except Exception as e:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.HIGH,
                category="Price",
                description=f"Error validating prices: {e}",
                field="price_validation",
                value=str(e)
            ))
        
        return issues
    
    def _validate_volume(self, data: pd.DataFrame, symbol: str) -> List[ValidationIssue]:
        """Validate volume data"""
        issues = []
        
        try:
            if 'volume' not in data.columns:
                return issues
            
            # Check for negative volumes
            negative_volumes = data[data['volume'] < 0]
            if not negative_volumes.empty:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    category="Volume",
                    description=f"Negative volume detected in {len(negative_volumes)} records",
                    field="volume",
                    value=len(negative_volumes),
                    suggestion="Volume cannot be negative"
                ))
            
            # Check for zero volumes (suspicious for liquid stocks)
            zero_volumes = data[data['volume'] == 0]
            if not zero_volumes.empty and len(zero_volumes) > len(data) * 0.05:  # More than 5%
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.MEDIUM,
                    category="Volume",
                    description=f"High number of zero volume days: {len(zero_volumes)}",
                    field="volume",
                    value=len(zero_volumes),
                    suggestion="Verify if stock is actively traded"
                ))
            
            # Check for extremely low volumes
            low_volumes = data[data['volume'] < self.quality_thresholds['min_volume']]
            if not low_volumes.empty and len(low_volumes) > len(data) * 0.10:  # More than 10%
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.LOW,
                    category="Volume",
                    description=f"Many low volume days: {len(low_volumes)}",
                    field="volume",
                    value=len(low_volumes),
                    expected_range=(self.quality_thresholds['min_volume'], None),
                    suggestion="Consider liquidity implications for analysis"
                ))
            
            # Volume outlier detection
            if len(data) > 30:
                volume_median = data['volume'].median()
                extreme_volumes = data[data['volume'] > volume_median * 10]  # 10x median
                
                if not extreme_volumes.empty:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        category="Volume",
                        description=f"Extreme volume spikes detected: {len(extreme_volumes)} days",
                        field="volume_spikes",
                        value=len(extreme_volumes),
                        suggestion="Review for corporate actions or news events"
                    ))
        
        except Exception as e:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.HIGH,
                category="Volume",
                description=f"Error validating volume: {e}",
                field="volume_validation",
                value=str(e)
            ))
        
        return issues
    
    def _validate_temporal_consistency(self, data: pd.DataFrame, symbol: str) -> List[ValidationIssue]:
        """Validate temporal aspects of the data"""
        issues = []
        
        try:
            if not isinstance(data.index, pd.DatetimeIndex):
                return issues
            
            # Check for duplicate timestamps
            duplicates = data.index.duplicated()
            if duplicates.any():
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.HIGH,
                    category="Temporal",
                    description=f"Duplicate timestamps found: {duplicates.sum()}",
                    field="timestamps",
                    value=duplicates.sum(),
                    suggestion="Remove or consolidate duplicate timestamps"
                ))
            
            # Check for data gaps
            if len(data) > 1:
                date_diffs = data.index.to_series().diff().dt.days
                large_gaps = date_diffs[date_diffs > self.quality_thresholds['max_missing_days']]
                
                if not large_gaps.empty:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.MEDIUM,
                        category="Temporal",
                        description=f"Large data gaps found: {len(large_gaps)} gaps > {self.quality_thresholds['max_missing_days']} days",
                        field="data_gaps",
                        value=len(large_gaps),
                        suggestion="Fill missing data or adjust analysis period"
                    ))
            
            # Check data freshness
            if not data.empty:
                latest_date = data.index.max()
                data_age = datetime.now() - latest_date.to_pydatetime()
                
                if data_age.total_seconds() > self.max_data_age_hours * 3600:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.MEDIUM,
                        category="Temporal",
                        description=f"Data is stale: {data_age.days} days old",
                        field="data_freshness",
                        value=data_age.days,
                        suggestion="Update data source for more recent data"
                    ))
            
            # Check for weekend/holiday data (suspicious for stock data)
            if hasattr(data.index, 'weekday'):
                weekend_data = data[data.index.weekday >= 5]  # Saturday = 5, Sunday = 6
                if not weekend_data.empty:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.LOW,
                        category="Temporal",
                        description=f"Weekend data found: {len(weekend_data)} records",
                        field="weekend_data",
                        value=len(weekend_data),
                        suggestion="Verify if weekend trading data is expected"
                    ))
        
        except Exception as e:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.HIGH,
                category="Temporal",
                description=f"Error validating temporal consistency: {e}",
                field="temporal_validation",
                value=str(e)
            ))
        
        return issues
    
    def _validate_statistical_properties(self, data: pd.DataFrame, symbol: str) -> List[ValidationIssue]:
        """Validate statistical properties of the data"""
        issues = []
        
        try:
            if 'close' not in data.columns or len(data) < 30:
                return issues
            
            returns = data['close'].pct_change().dropna()
            
            if len(returns) < 10:
                return issues
            
            # Check for constant prices (no variation)
            if returns.std() == 0:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.HIGH,
                    category="Statistical",
                    description="No price variation detected (constant prices)",
                    field="price_variation",
                    value=0,
                    suggestion="Verify if the stock is actively traded"
                ))
            
            # Check for extreme skewness
            skewness = stats.skew(returns)
            if abs(skewness) > 3:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.MEDIUM,
                    category="Statistical",
                    description=f"Extreme skewness detected: {skewness:.3f}",
                    field="skewness",
                    value=skewness,
                    suggestion="Review for data quality issues or unusual market events"
                ))
            
            # Check for extreme kurtosis
            kurtosis = stats.kurtosis(returns)
            if kurtosis > 10:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.MEDIUM,
                    category="Statistical",
                    description=f"Extreme kurtosis detected: {kurtosis:.3f}",
                    field="kurtosis",
                    value=kurtosis,
                    suggestion="High kurtosis indicates fat tails - review for outliers"
                ))
            
            # Check for autocorrelation (may indicate data issues)
            if len(returns) > 50:
                autocorr = returns.autocorr(lag=1)
                if abs(autocorr) > 0.3:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.LOW,
                        category="Statistical",
                        description=f"High autocorrelation detected: {autocorr:.3f}",
                        field="autocorrelation",
                        value=autocorr,
                        suggestion="High autocorrelation may indicate data smoothing or errors"
                    ))
        
        except Exception as e:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.HIGH,
                category="Statistical",
                description=f"Error validating statistical properties: {e}",
                field="statistical_validation",
                value=str(e)
            ))
        
        return issues
    
    def _validate_completeness(self, data: pd.DataFrame, symbol: str) -> List[ValidationIssue]:
        """Validate data completeness"""
        issues = []
        
        try:
            # Check for missing values
            for column in data.columns:
                missing_count = data[column].isna().sum()
                missing_percentage = missing_count / len(data)
                
                if missing_percentage > 0.05:  # More than 5% missing
                    severity = ValidationSeverity.HIGH if missing_percentage > 0.20 else ValidationSeverity.MEDIUM
                    issues.append(ValidationIssue(
                        severity=severity,
                        category="Completeness",
                        description=f"High missing data in {column}: {missing_percentage:.1%}",
                        field=column,
                        value=missing_percentage,
                        suggestion=f"Address missing data in {column} column"
                    ))
            
            # Check expected data frequency
            if isinstance(data.index, pd.DatetimeIndex) and len(data) > 1:
                expected_business_days = pd.bdate_range(start=data.index.min(), end=data.index.max())
                actual_days = len(data)
                expected_days = len(expected_business_days)
                completeness_ratio = actual_days / expected_days
                
                if completeness_ratio < 0.90:  # Less than 90% complete
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.MEDIUM,
                        category="Completeness",
                        description=f"Data completeness: {completeness_ratio:.1%} of expected business days",
                        field="data_completeness",
                        value=completeness_ratio,
                        suggestion="Fill missing trading days or adjust analysis methodology"
                    ))
        
        except Exception as e:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.HIGH,
                category="Completeness",
                description=f"Error validating completeness: {e}",
                field="completeness_validation",
                value=str(e)
            ))
        
        return issues
    
    def _validate_market_norms(self, data: pd.DataFrame, symbol: str) -> List[ValidationIssue]:
        """Validate against market norms and expectations"""
        issues = []
        
        try:
            if 'close' not in data.columns or len(data) < 30:
                return issues
            
            returns = data['close'].pct_change().dropna()
            annual_volatility = returns.std() * np.sqrt(252)
            
            # Check for unrealistic volatility
            if annual_volatility > 2.0:  # 200% annual volatility
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.HIGH,
                    category="Market Norms",
                    description=f"Extremely high volatility: {annual_volatility:.1%}",
                    field="volatility",
                    value=annual_volatility,
                    suggestion="Verify data accuracy for such high volatility"
                ))
            elif annual_volatility < 0.05:  # 5% annual volatility
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.MEDIUM,
                    category="Market Norms",
                    description=f"Unusually low volatility: {annual_volatility:.1%}",
                    field="volatility",
                    value=annual_volatility,
                    suggestion="Verify if this is a low-volatility instrument"
                ))
            
            # Check for price level consistency
            if 'close' in data.columns:
                price_range = data['close'].max() - data['close'].min()
                avg_price = data['close'].mean()
                
                if price_range / avg_price > 5:  # Price range > 5x average price
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.MEDIUM,
                        category="Market Norms",
                        description=f"Very wide price range: {price_range/avg_price:.1f}x average price",
                        field="price_range",
                        value=price_range/avg_price,
                        suggestion="Review for stock splits or other corporate actions"
                    ))
        
        except Exception as e:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.HIGH,
                category="Market Norms",
                description=f"Error validating market norms: {e}",
                field="market_norms_validation",
                value=str(e)
            ))
        
        return issues
    
    def _calculate_quality_scores(self, data: pd.DataFrame, issues: List[ValidationIssue]) -> Dict[str, float]:
        """Calculate comprehensive quality scores"""
        try:
            # Weight issues by severity
            severity_weights = {
                ValidationSeverity.CRITICAL: 1.0,
                ValidationSeverity.HIGH: 0.7,
                ValidationSeverity.MEDIUM: 0.4,
                ValidationSeverity.LOW: 0.2,
                ValidationSeverity.INFO: 0.1
            }
            
            # Calculate weighted issue score
            total_weight = sum(severity_weights[issue.severity] for issue in issues)
            max_possible_weight = len(data) * 0.1  # Assume max 10% issues
            
            # Overall quality score
            overall_score = max(0, 1 - (total_weight / max(max_possible_weight, 1)))
            
            # Completeness score
            missing_data_ratio = data.isna().sum().sum() / (len(data) * len(data.columns))
            completeness_score = 1 - missing_data_ratio
            
            # Accuracy score (based on critical and high severity issues)
            critical_high_issues = [i for i in issues if i.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.HIGH]]
            accuracy_score = max(0, 1 - (len(critical_high_issues) / max(len(data) * 0.05, 1)))
            
            # Consistency score (based on statistical and temporal issues)
            consistency_issues = [i for i in issues if i.category in ['Statistical', 'Temporal']]
            consistency_score = max(0, 1 - (len(consistency_issues) / max(len(data) * 0.03, 1)))
            
            # Timeliness score (based on data freshness)
            timeliness_issues = [i for i in issues if 'freshness' in i.field or 'stale' in i.description.lower()]
            timeliness_score = 1 - (len(timeliness_issues) * 0.2)
            
            return {
                'overall': min(1.0, max(0.0, overall_score)),
                'completeness': min(1.0, max(0.0, completeness_score)),
                'accuracy': min(1.0, max(0.0, accuracy_score)),
                'consistency': min(1.0, max(0.0, consistency_score)),
                'timeliness': min(1.0, max(0.0, timeliness_score))
            }
            
        except Exception as e:
            logger.error(f"Error calculating quality scores: {e}")
            return {
                'overall': 0.5,
                'completeness': 0.5,
                'accuracy': 0.5,
                'consistency': 0.5,
                'timeliness': 0.5
            }
    
    def _generate_recommendations(self, issues: List[ValidationIssue], data: pd.DataFrame) -> List[str]:
        """Generate actionable recommendations based on issues"""
        recommendations = []
        
        try:
            # Group issues by category
            issue_categories = {}
            for issue in issues:
                if issue.category not in issue_categories:
                    issue_categories[issue.category] = []
                issue_categories[issue.category].append(issue)
            
            # Generate category-specific recommendations
            if 'Price' in issue_categories:
                recommendations.append("Review price data source for accuracy and implement additional validation checks")
            
            if 'Volume' in issue_categories:
                recommendations.append("Consider liquidity implications and potentially filter low-volume periods")
            
            if 'Temporal' in issue_categories:
                recommendations.append("Implement data gap filling methodology and ensure regular data updates")
            
            if 'Statistical' in issue_categories:
                recommendations.append("Review for outliers and consider robust statistical methods")
            
            if 'Completeness' in issue_categories:
                recommendations.append("Implement missing data handling strategy (interpolation, forward-fill, etc.)")
            
            # Critical issues recommendations
            critical_issues = [i for i in issues if i.severity == ValidationSeverity.CRITICAL]
            if critical_issues:
                recommendations.append("URGENT: Address critical data quality issues before proceeding with analysis")
            
            # General recommendations
            if len(issues) > len(data) * 0.1:  # More than 10% of records have issues
                recommendations.append("Consider alternative data sources or enhanced data cleaning procedures")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return ["Review data quality issues and implement appropriate data cleaning procedures"]
    
    def _create_error_report(self, symbol: str, error_message: str) -> DataQualityReport:
        """Create error report when validation fails"""
        return DataQualityReport(
            symbol=symbol,
            total_records=0,
            valid_records=0,
            invalid_records=0,
            quality_score=0.0,
            completeness_score=0.0,
            accuracy_score=0.0,
            consistency_score=0.0,
            timeliness_score=0.0,
            issues=[ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category="System",
                description=f"Validation failed: {error_message}",
                field="system",
                value=error_message
            )],
            recommendations=["Fix system error and retry validation"],
            generated_at=datetime.now()
        )
