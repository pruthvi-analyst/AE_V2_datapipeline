import pandas as pd
import numpy as np
import itertools
from typing import Dict, List, Tuple
import warnings
from sqlalchemy import create_engine, text
import traceback
warnings.filterwarnings('ignore')

# ==================================================
# DATABASE CONFIGURATION
# ==================================================
DB_CONFIG = {
    'host': '172.16.134.154',
    'port': 5432,
    'database': 'postgres',
    'username': 'biadmin',
    'password': 'biadmin'
}

# Create database connection
def create_db_engine():
    """Create SQLAlchemy engine for PostgreSQL connection"""
    connection_string = f"postgresql+psycopg2://{DB_CONFIG['username']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    return create_engine(connection_string, isolation_level="AUTOCOMMIT")

# ==================================================
# ANALYTICS CONFIGURATION
# ==================================================
# Define ALL possible rating categories
RATING_CATEGORIES = {
    'MAJOR': ['major', 'major issue', 'major issues', 'major problem'],
    'MINOR': ['minor', 'minor issue', 'minor issues', 'minor problem'],
    'NO_ISSUE': ['no', 'no issue', 'no issues', 'none', 'not applicable', 'na', 'n/a'],
    'PARTIAL': ['partial', 'partial issue', 'some issues'],
    'GOOD': ['good', 'excellent', 'perfect', 'great']
}

# Main rubrics to analyze
MAIN_RUBRICS = [
    "Instruction Following",
    "Writing Style", 
    "Code Quality",
    "Code Output Quality",
    "Tool Selection"
]

# Target analytical tables
AGREEMENT_TABLE = "bronze.batch_analytics_ae_v2_agreement"
QUALITY_TABLE = "bronze.batch_analytics_ae_v2_quality"

# ==================================================
# UTILITY FUNCTIONS
# ==================================================
def normalize_rating(raw_value) -> str:
    """
    Convert any rating value to standardized categories.
    Returns: 'MAJOR', 'MINOR', 'NO_ISSUE', 'PARTIAL', 'GOOD', or 'UNKNOWN'
    """
    if pd.isna(raw_value):
        return np.nan
    
    value = str(raw_value).strip().lower()
    
    # Check each category
    for category, patterns in RATING_CATEGORIES.items():
        if any(pattern in value for pattern in patterns):
            return category
    
    # If no match, try to infer
    if value == '':
        return np.nan
    elif 'issue' in value or 'problem' in value:
        return 'MAJOR' if 'major' in value else 'MINOR'
    
    return 'UNKNOWN'

def rating_to_numeric(rating: str) -> int:
    """Convert rating category to numeric value for calculations."""
    mapping = {
        'GOOD': 0,
        'NO_ISSUE': 1,
        'PARTIAL': 2,
        'MINOR': 3,
        'MAJOR': 4,
        'UNKNOWN': np.nan
    }
    return mapping.get(rating, np.nan)

def calculate_pairwise_disagreement(values: List[int]) -> float:
    """Calculate observed disagreement (Do) for a set of numeric ratings."""
    if len(values) < 2:
        return np.nan
    
    pairs = list(itertools.combinations(values, 2))
    if not pairs:
        return np.nan
    
    disagreements = sum(1 for a, b in pairs if a != b)
    return disagreements / len(pairs)

def calculate_krippendorff_alpha(df_norm: pd.DataFrame, rubric: str, question_col: str) -> Dict:
    """
    Calculate Krippendorff's Alpha for ordinal data.
    Follows the standard formula: α = 1 - (Do / De)
    """
    # Filter for this rubric
    rubric_data = df_norm[df_norm['Rubric'] == rubric].copy()
    
    if rubric_data.empty or rubric_data['Numeric_Rating'].isna().all():
        return {
            'Rubric': rubric,
            'K_Alpha': np.nan,
            'Observed_Disagreement_D0': np.nan,
            'Expected_Disagreement_De': np.nan,
            'Num_Questions': 0,
            'Num_Ratings': 0
        }
    
    # Remove questions with < 2 raters
    question_counts = rubric_data.groupby(question_col).size()
    valid_questions = question_counts[question_counts >= 2].index
    valid_data = rubric_data[rubric_data[question_col].isin(valid_questions)]
    
    if len(valid_questions) == 0:
        return {
            'Rubric': rubric,
            'K_Alpha': np.nan,
            'Observed_Disagreement_D0': np.nan,
            'Expected_Disagreement_De': np.nan,
            'Num_Questions': 0,
            'Num_Ratings': 0
        }
    
    # Calculate observed disagreement (Do)
    do_values = []
    for qid, grp in valid_data.groupby(question_col):
        ratings = grp['Numeric_Rating'].dropna().tolist()
        if len(ratings) >= 2:
            do = calculate_pairwise_disagreement(ratings)
            if not np.isnan(do):
                do_values.append(do)
    
    if not do_values:
        return {
            'Rubric': rubric,
            'K_Alpha': np.nan,
            'Observed_Disagreement_D0': np.nan,
            'Expected_Disagreement_De': np.nan,
            'Num_Questions': len(valid_questions),
            'Num_Ratings': len(valid_data)
        }
    
    D0 = np.mean(do_values)
    
    # Calculate expected disagreement (De)
    all_ratings = valid_data['Numeric_Rating'].dropna().tolist()
    if not all_ratings:
        De = np.nan
    else:
        # For ordinal data, use interval-level metric
        # De = 1 - Σ(p_c²) where p_c is proportion of each rating category
        rating_counts = pd.Series(all_ratings).value_counts()
        total = rating_counts.sum()
        proportions = rating_counts / total
        De = 1 - np.sum(proportions ** 2)
    
    # Calculate Krippendorff's Alpha
    if De > 0 and not np.isnan(D0):
        k_alpha = 1 - (D0 / De)
    else:
        k_alpha = np.nan
    
    return {
        'Rubric': rubric,
        'K_Alpha': round(k_alpha, 4) if not np.isnan(k_alpha) else np.nan,
        'Observed_Disagreement_D0': round(D0, 4) if not np.isnan(D0) else np.nan,
        'Expected_Disagreement_De': round(De, 4) if not np.isnan(De) else np.nan,
        'Num_Questions': len(valid_questions),
        'Num_Ratings': len(valid_data)
    }

def get_metadata_for_bug(engine, bug_id: str) -> Dict:
    """Fetch metadata for a specific bug_id from metadata table"""
    query = """
    SELECT 
        bug_date,
        workstream,
        bugtype,
        bug_version,
        task,
        reps,
        total_tasks_performed,
        bug_id
    FROM bronze.batch_analytics_metadata
    WHERE bug_id = %(bug_id)s
    """
    
    try:
        meta_df = pd.read_sql(query, engine, params={"bug_id": bug_id})
        if not meta_df.empty:
            return meta_df.iloc[0].to_dict()
    except Exception as e:
        print(f"   ❌ Error fetching metadata for bug_id {bug_id}: {e}")
    
    return None

def clear_analytics_tables(engine):
    """Clear existing data from analytics tables"""
    print("📋 Step 2: Clearing old analytics data...")
    try:
        with engine.connect() as conn:
            conn.execute(text(f"TRUNCATE TABLE {AGREEMENT_TABLE}"))
            conn.execute(text(f"TRUNCATE TABLE {QUALITY_TABLE}"))
            conn.commit()
        print("✅ Cleared analytics tables")
    except Exception as e:
        print(f"⚠️ Could not clear tables: {e}")

def analyze_bug_data(df_raw: pd.DataFrame, bug_id: str, metadata: Dict, engine):
    """
    Perform comprehensive analysis on bug-level data
    Returns: Tuple of (agreement_df, quality_df) for insertion
    """
    print(f"   📊 Processing bug_id: {bug_id}")
    print(f"   📋 Data shape: {df_raw.shape}")
    
    # ==================================================
    # 1. IDENTIFY RUBRIC COLUMNS
    # ==================================================
    print("   📊 Step 1: Identifying rubric columns...")
    
    df = df_raw.copy()
    rubric_columns = {}
    
    for col in df.columns:
        col_lower = str(col).lower().strip()
        
        # Check for main rubrics
        for rubric in MAIN_RUBRICS:
            if rubric.lower() in col_lower:
                rubric_columns[rubric] = col
                break
        
        # Also catch common variations
        if 'instr' in col_lower and 'follow' in col_lower:
            rubric_columns['Instruction Following'] = col
        elif 'writing' in col_lower or 'style' in col_lower:
            rubric_columns['Writing Style'] = col
        elif 'code quality' in col_lower:
            rubric_columns['Code Quality'] = col
        elif 'output' in col_lower and 'quality' in col_lower:
            rubric_columns['Code Output Quality'] = col
        elif 'tool' in col_lower and ('select' in col_lower or 'choice' in col_lower):
            rubric_columns['Tool Selection'] = col
    
    print(f"   ✅ Found {len(rubric_columns)} main rubric columns")
    
    # ==================================================
    # 2. NORMALIZE RATINGS
    # ==================================================
    print("   📊 Step 2: Normalizing ratings...")
    
    normalized_rows = []
    
    for rubric, col_name in rubric_columns.items():
        if col_name not in df.columns:
            print(f"   ⚠️  Column {col_name} not found for rubric {rubric}")
            continue
        
        # Process each row
        for idx, row in df.iterrows():
            question_id = row.get("Question ID", row.get("question id", row.get("Question_ID", np.nan)))
            rater_id = row.get("Rater", row.get("rater", row.get("Rater_ID", f"Rater_{idx}")))
            raw_rating = row[col_name]
            
            normalized = normalize_rating(raw_rating)
            
            normalized_rows.append({
                'Question_ID': question_id,
                'Rater_ID': rater_id,
                'Rubric': rubric,
                'Raw_Rating': raw_rating,
                'Normalized_Rating': normalized,
                'Numeric_Rating': rating_to_numeric(normalized)
            })
    
    df_normalized = pd.DataFrame(normalized_rows)
    
    if df_normalized.empty:
        print(f"   ⚠️  No valid data to analyze for bug_id: {bug_id}")
        return None, None
    
    # ==================================================
    # 3. AGREEMENT ANALYSIS
    # ==================================================
    print("   📊 Step 3: Performing agreement analysis...")
    
    agreement_rows = []
    
    for rubric in rubric_columns.keys():
        rubric_data = df_normalized[df_normalized['Rubric'] == rubric].copy()
        
        if rubric_data.empty:
            continue
        
        # Group by question
        question_disagreements = []
        
        for qid, grp in rubric_data.groupby('Question_ID'):
            ratings = grp['Numeric_Rating'].dropna().tolist()
            
            if len(ratings) >= 2:
                disagreement = calculate_pairwise_disagreement(ratings)
                question_disagreements.append(disagreement)
        
        if question_disagreements:
            total_questions = len(question_disagreements)
            agreement_count = sum(1 for d in question_disagreements if d == 0)
            agreement_rate = agreement_count / total_questions if total_questions > 0 else 0
            avg_disagreement = np.mean([d for d in question_disagreements if not np.isnan(d)])
        else:
            total_questions = 0
            agreement_count = 0
            agreement_rate = 0
            avg_disagreement = 0
        
        # Add metadata to agreement row
        agreement_row = {
            'bug_date': metadata.get('bug_date'),
            'workstream': metadata.get('workstream'),
            'bugtype': metadata.get('bugtype'),
            'bug_version': metadata.get('bug_version'),
            'task': metadata.get('task'),
            'reps': metadata.get('reps'),
            'total_tasks_performed': metadata.get('total_tasks_performed'),
            'rubric': rubric,
            'total_questions': total_questions,
            'agreement_count': agreement_count,
            'agreement_rate': round(agreement_rate, 4),
            'avg_disagreement': round(avg_disagreement, 4)
        }
        
        agreement_rows.append(agreement_row)
    
    df_agreement = pd.DataFrame(agreement_rows)
    
    # ==================================================
    # 4. QUALITY ANALYSIS (KRIPPENDORFF'S ALPHA)
    # ==================================================
    print("   📊 Step 4: Calculating Krippendorff's Alpha...")
    
    quality_rows = []
    
    for rubric in rubric_columns.keys():
        kalpha_result = calculate_krippendorff_alpha(df_normalized, rubric, 'Question_ID')
        
        # Add metadata to quality row
        quality_row = {
            'bug_date': metadata.get('bug_date'),
            'workstream': metadata.get('workstream'),
            'bugtype': metadata.get('bugtype'),
            'bug_version': metadata.get('bug_version'),
            'task': metadata.get('task'),
            'reps': metadata.get('reps'),
            'total_tasks_performed': metadata.get('total_tasks_performed'),
            'rubric': rubric,
            'k_alpha': kalpha_result['K_Alpha'],
            'observed_disagreement_d0': kalpha_result['Observed_Disagreement_D0'],
            'expected_disagreement_de': kalpha_result['Expected_Disagreement_De'],
            'num_questions': kalpha_result['Num_Questions'],
            'num_ratings': kalpha_result['Num_Ratings']
        }
        
        quality_rows.append(quality_row)
    
    df_quality = pd.DataFrame(quality_rows)
    
    return df_agreement, df_quality

def insert_analytics_data(engine, df_agreement, df_quality):
    """Insert analytics data into database tables"""
    try:
        print("   💾 Inserting analytics data into database...")
        
        # Insert agreement data
        if not df_agreement.empty:
            df_agreement.to_sql(
                AGREEMENT_TABLE.split('.')[1],  # Table name without schema
                engine,
                schema=AGREEMENT_TABLE.split('.')[0],  # Schema name
                if_exists='append',
                index=False,
                method='multi'
            )
            print(f"   ✅ Inserted {len(df_agreement)} rows into {AGREEMENT_TABLE}")
        
        # Insert quality data
        if not df_quality.empty:
            df_quality.to_sql(
                QUALITY_TABLE.split('.')[1],  # Table name without schema
                engine,
                schema=QUALITY_TABLE.split('.')[0],  # Schema name
                if_exists='append',
                index=False,
                method='multi'
            )
            print(f"   ✅ Inserted {len(df_quality)} rows into {QUALITY_TABLE}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error inserting data: {e}")
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("BUG-LEVEL QUALITY METRICS ANALYSIS - DATABASE PIPELINE")
    print("=" * 60)
    
    # ==================================================
    # 1. INITIALIZE DATABASE CONNECTION
    # ==================================================
    print("\n📋 Step 1: Initializing database connection...")
    try:
        engine = create_db_engine()
        print("✅ Database connection established")
    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
        return
    
    # ==================================================
    # 2. CLEAR EXISTING ANALYTICS DATA
    # ==================================================
    clear_analytics_tables(engine)
    
    # ==================================================
    # 3. FETCH ALL BUG IDs FROM METADATA
    # ==================================================
    print("\n📋 Step 3: Fetching bug IDs from metadata...")
    try:
        bug_query = """
        SELECT DISTINCT bug_id 
        FROM bronze.batch_analytics_metadata 
        WHERE bug_id IS NOT NULL 
        ORDER BY bug_id
        """
        bug_ids_df = pd.read_sql(bug_query, engine)
        bug_ids = bug_ids_df['bug_id'].tolist()
        print(f"✅ Found {len(bug_ids)} bug IDs to process")
    except Exception as e:
        print(f"❌ Error fetching bug IDs: {e}")
        return
    
    if not bug_ids:
        print("⚠️ No bug IDs found. Exiting.")
        return
    
    # ==================================================
    # 4. PROCESS EACH BUG INDIVIDUALLY
    # ==================================================
    print(f"\n📋 Step 4: Processing {len(bug_ids)} bugs...")
    print("="*60)
    
    success_count = 0
    fail_count = 0
    
    for idx, bug_id in enumerate(bug_ids, 1):
        print(f"\n[{idx}/{len(bug_ids)}] Processing bug_id: {bug_id}")
        
        try:
            # ==================================================
            # 4a. FETCH METADATA FOR THIS BUG
            # ==================================================
            metadata = get_metadata_for_bug(engine, bug_id)
            if not metadata:
                print(f"   ⚠️ No metadata found for bug_id: {bug_id}, skipping...")
                fail_count += 1
                continue
            
            # ==================================================
            # 4b. FETCH RAW DATA FOR THIS BUG
            # ==================================================
            print(f"   📥 Fetching raw data for bug_id: {bug_id}")
            
            # Try to fetch data from the main analytics table
            data_query = """
            SELECT * 
            FROM bronze.batch_analytics_ae_v2_data 
            WHERE bug_id = %(bug_id)s
            """
            
            df_raw = pd.read_sql(data_query, engine, params={"bug_id": bug_id})
            
            if df_raw.empty:
                print(f"   ⚠️ No data found in analytics table for bug_id: {bug_id}")
                
                # Try alternative table name
                print(f"   🔍 Trying alternative table: bronze.batch_analytics_ae_v2_data")
                alt_query = """
                SELECT * 
                FROM bronze.batch_analytics_ae_v2_data 
                WHERE bug_id = %(bug_id)s
                """
                
                try:
                    df_raw = pd.read_sql(alt_query, engine, params={"bug_id": bug_id})
                except:
                    pass
            
            if df_raw.empty:
                print(f"   ⚠️ No data found for bug_id: {bug_id}, skipping...")
                fail_count += 1
                continue
            
            print(f"   ✅ Fetched {len(df_raw)} rows of data")
            
            # Show column names on first iteration
            if idx == 1:
                print(f"   📋 Available columns in data table:")
                for col in df_raw.columns:
                    print(f"      - {col}")
            
            # ==================================================
            # 4c. PERFORM ANALYTICS
            # ==================================================
            df_agreement, df_quality = analyze_bug_data(df_raw, bug_id, metadata, engine)
            
            if df_agreement is None or df_quality is None:
                print(f"   ⚠️ Could not generate analytics for bug_id: {bug_id}")
                fail_count += 1
                continue
            
            # ==================================================
            # 4d. INSERT RESULTS INTO DATABASE
            # ==================================================
            if insert_analytics_data(engine, df_agreement, df_quality):
                success_count += 1
                
                # Print summary for this bug
                print(f"   📊 Analytics summary for {bug_id}:")
                print(f"      • Agreement metrics: {len(df_agreement)} rubrics")
                print(f"      • Quality metrics: {len(df_quality)} rubrics")
                
                # Show sample quality metrics
                if not df_quality.empty:
                    print(f"      • Sample K-alpha values:")
                    for _, row in df_quality.iterrows():
                        if not pd.isna(row['k_alpha']):
                            print(f"        - {row['rubric']}: {row['k_alpha']}")
            else:
                fail_count += 1
                
        except Exception as e:
            print(f"   ❌ Error processing bug_id {bug_id}: {e}")
            traceback.print_exc()
            fail_count += 1
    
    # ==================================================
    # 5. VERIFICATION AND SUMMARY
    # ==================================================
    print("\n" + "="*60)
    print("📊 PIPELINE COMPLETE - VERIFICATION")
    print("="*60)
    
    print(f"\n✅ Successful: {success_count}")
    print(f"❌ Failed: {fail_count}")
    print(f"📊 Total processed: {len(bug_ids)}")
    
    # ==================================================
    # 6. FINAL VERIFICATION OF INSERTED DATA
    # ==================================================
    print("\n🔍 Verifying inserted data...")
    
    try:
        # Check agreement table
        agreement_check = pd.read_sql(f"""
            SELECT 
                COUNT(*) as total_rows,
                COUNT(DISTINCT bug_date || '-' || workstream || '-' || bugtype) as unique_bugs,
                COUNT(DISTINCT rubric) as unique_rubrics
            FROM {AGREEMENT_TABLE}
        """, engine)
        
        print(f"\n📈 Agreement Table Summary:")
        print(f"   Total rows: {agreement_check['total_rows'].iloc[0]}")
        print(f"   Unique bugs: {agreement_check['unique_bugs'].iloc[0]}")
        print(f"   Unique rubrics: {agreement_check['unique_rubrics'].iloc[0]}")
        
        # Check quality table
        quality_check = pd.read_sql(f"""
            SELECT 
                COUNT(*) as total_rows,
                COUNT(DISTINCT bug_date || '-' || workstream || '-' || bugtype) as unique_bugs,
                COUNT(DISTINCT rubric) as unique_rubrics,
                AVG(k_alpha) as avg_k_alpha
            FROM {QUALITY_TABLE}
            WHERE k_alpha IS NOT NULL
        """, engine)
        
        print(f"\n📈 Quality Table Summary:")
        print(f"   Total rows: {quality_check['total_rows'].iloc[0]}")
        print(f"   Unique bugs: {quality_check['unique_bugs'].iloc[0]}")
        print(f"   Unique rubrics: {quality_check['unique_rubrics'].iloc[0]}")
        print(f"   Average K-alpha: {quality_check['avg_k_alpha'].iloc[0]:.4f}")
        
        # Show sample data
        print(f"\n📊 Sample from Agreement Table (first 3 rows):")
        sample_agreement = pd.read_sql(f"""
            SELECT 
                bug_date, workstream, bugtype, rubric, 
                agreement_rate, avg_disagreement
            FROM {AGREEMENT_TABLE}
            ORDER BY bug_date DESC, rubric
            LIMIT 3
        """, engine)
        print(sample_agreement.to_string(index=False))
        
        print(f"\n📊 Sample from Quality Table (first 3 rows):")
        sample_quality = pd.read_sql(f"""
            SELECT 
                bug_date, workstream, bugtype, rubric, 
                k_alpha, observed_disagreement_d0, expected_disagreement_de
            FROM {QUALITY_TABLE}
            WHERE k_alpha IS NOT NULL
            ORDER BY bug_date DESC, rubric
            LIMIT 3
        """, engine)
        print(sample_quality.to_string(index=False))
        
    except Exception as e:
        print(f"⚠️ Could not verify tables: {e}")
    
    print("\n" + "="*60)
    print("🎯 BUG-LEVEL QUALITY METRICS ANALYSIS COMPLETE!")
    print("="*60)
    print(f"\n✅ Analytics exported to:")
    print(f"   • {AGREEMENT_TABLE}")
    print(f"   • {QUALITY_TABLE}")

# ==================================================
# RUN THE ANALYSIS
# ==================================================
if __name__ == "__main__":
    main()