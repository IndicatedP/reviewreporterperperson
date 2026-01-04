import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import altair as alt
import numpy as np
import pdfplumber
import re
import io
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Apartment Grades Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent

def extract_booking_data_from_pdf(pdf_file):
    """Extract apartment data from Booking.com PDF."""
    data = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                for row in table:
                    if row and len(row) >= 3:
                        # Look for rows with apartment data
                        ligne = str(row[0]).strip() if row[0] else ''
                        nom = str(row[1]).strip() if row[1] else ''
                        note = str(row[2]).strip() if row[2] else ''
                        comments = str(row[3]).strip() if len(row) > 3 and row[3] else '0'

                        # Skip header rows
                        if ligne.lower() in ['ligne', 'line', ''] or nom.lower() in ['nom', 'name', '']:
                            continue

                        # Try to parse note as float
                        try:
                            note_val = float(note.replace(',', '.')) if note and note != 'X' else None
                        except:
                            note_val = None

                        try:
                            comments_val = int(re.sub(r'[^\d]', '', comments)) if comments else 0
                        except:
                            comments_val = 0

                        if nom and nom != 'None':
                            data.append({
                                'Ligne': ligne,
                                'Nom': nom,
                                'Note': note_val,
                                'Comments': comments_val
                            })
    return pd.DataFrame(data)

def extract_airbnb_data_from_pdf(pdf_file):
    """Extract apartment data from Airbnb PDF."""
    data = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                for row in table:
                    if row and len(row) >= 3:
                        ligne = str(row[0]).strip() if row[0] else ''
                        nom = str(row[1]).strip() if row[1] else ''
                        note = str(row[2]).strip() if row[2] else ''
                        comments = str(row[3]).strip() if len(row) > 3 and row[3] else '0'

                        if ligne.lower() in ['ligne', 'line', ''] or nom.lower() in ['nom', 'name', '']:
                            continue

                        try:
                            note_val = float(note.replace(',', '.')) if note and note != 'X' else None
                        except:
                            note_val = None

                        try:
                            comments_val = int(re.sub(r'[^\d]', '', comments)) if comments else 0
                        except:
                            comments_val = 0

                        if nom and nom != 'None':
                            data.append({
                                'Ligne': ligne,
                                'Nom': nom,
                                'Note': note_val,
                                'Comments': comments_val
                            })
    return pd.DataFrame(data)

def create_comparison_df(df_before, df_after, before_label, after_label):
    """Create comparison dataframe from two period dataframes."""
    # Merge on Nom (apartment name)
    merged = pd.merge(
        df_before[['Ligne', 'Nom', 'Note', 'Comments']],
        df_after[['Nom', 'Note', 'Comments']],
        on='Nom',
        how='outer',
        suffixes=(f' {before_label}', f' {after_label}')
    )

    # Fill missing Ligne from after df if needed
    if 'Ligne' not in merged.columns:
        merged['Ligne'] = ''

    before_col = f'Note {before_label}'
    after_col = f'Note {after_label}'

    # Calculate difference
    def calc_diff(row):
        if pd.isna(row[before_col]) and pd.notna(row[after_col]):
            return None  # New rating
        elif pd.notna(row[before_col]) and pd.notna(row[after_col]):
            return round(row[after_col] - row[before_col], 2)
        return None

    merged['Difference'] = merged.apply(calc_diff, axis=1)

    # Determine evolution
    def get_evolution(row):
        if pd.isna(row[before_col]) and pd.notna(row[after_col]):
            return '★ New Rating'
        elif pd.isna(row['Difference']):
            return 'N/A'
        elif row['Difference'] > 0:
            return '↑ Improved'
        elif row['Difference'] < 0:
            return '↓ Degraded'
        else:
            return '→ Stable'

    merged['Evolution'] = merged.apply(get_evolution, axis=1)

    return merged, before_col, after_col

def parse_difference(val):
    """Parse difference values, handling +/- prefixes and text values."""
    if pd.isna(val):
        return None
    val_str = str(val).strip()
    # Handle text values
    if val_str in ['NEW', 'N/A', '-', '']:
        return None
    # Remove + prefix if present (- is handled by float())
    val_str = val_str.lstrip('+')
    try:
        return float(val_str)
    except ValueError:
        return None

@st.cache_data
def load_booking_data():
    """Load and process Booking.com comparison data."""
    df = pd.read_csv(SCRIPT_DIR / "BOOKING_comparison_Oct22_vs_Dec29.csv")

    # Convert note columns to numeric, handling 'X' values
    df['Note Oct 22'] = pd.to_numeric(df['Note Oct 22'], errors='coerce')
    df['Note Dec 29'] = pd.to_numeric(df['Note Dec 29'], errors='coerce')
    # Parse Difference column with custom function
    df['Difference'] = df['Difference'].apply(parse_difference)

    return df

@st.cache_data
def load_airbnb_data():
    """Load and process Airbnb comparison data."""
    df = pd.read_csv(SCRIPT_DIR / "AIRBNB_comparison_Oct23_vs_Dec28.csv")

    # Convert note columns to numeric, handling 'X' values
    df['Note Oct 23'] = pd.to_numeric(df['Note Oct 23'], errors='coerce')
    df['Note Dec 28'] = pd.to_numeric(df['Note Dec 28'], errors='coerce')
    # Parse Difference column with custom function
    df['Difference'] = df['Difference'].apply(parse_difference)

    return df

def get_evolution_color(evolution):
    """Return color based on evolution type."""
    if '↑' in str(evolution) or 'Improved' in str(evolution):
        return 'green'
    elif '↓' in str(evolution) or 'Degraded' in str(evolution):
        return 'red'
    elif '★' in str(evolution) or 'New' in str(evolution):
        return 'blue'
    else:
        return 'gray'

def create_summary_metrics(df, oct_col, dec_col):
    """Create summary metrics for the dashboard."""
    total = len(df)

    # Ensure Difference is numeric
    df = df.copy()
    df['Difference'] = pd.to_numeric(df['Difference'], errors='coerce')

    # Count by evolution
    improved = len(df[df['Difference'] > 0])
    degraded = len(df[df['Difference'] < 0])
    stable = len(df[df['Difference'] == 0])
    new_ratings = len(df[df['Evolution'].str.contains('★|New', na=False, regex=True)])

    # With valid ratings
    with_oct = df[oct_col].notna().sum()
    with_dec = df[dec_col].notna().sum()

    # Total points gained and lost
    positive_diffs = df[df['Difference'] > 0]['Difference'].sum()
    negative_diffs = df[df['Difference'] < 0]['Difference'].sum()
    total_points_up = round(positive_diffs, 2) if pd.notna(positive_diffs) else 0
    total_points_down = round(abs(negative_diffs), 2) if pd.notna(negative_diffs) else 0
    net_change = round(total_points_up - total_points_down, 2)

    # Comment statistics
    comments_oct_col = 'Comments Oct'
    comments_dec_col = 'Comments Dec'

    # Convert comments to numeric (handle 'X' values)
    if comments_oct_col in df.columns:
        df[comments_oct_col] = pd.to_numeric(df[comments_oct_col], errors='coerce')
        total_comments_oct = int(df[comments_oct_col].sum()) if df[comments_oct_col].notna().any() else 0
    else:
        total_comments_oct = 0

    if comments_dec_col in df.columns:
        df[comments_dec_col] = pd.to_numeric(df[comments_dec_col], errors='coerce')
        total_comments_dec = int(df[comments_dec_col].sum()) if df[comments_dec_col].notna().any() else 0
    else:
        total_comments_dec = 0

    new_comments = total_comments_dec - total_comments_oct

    return {
        'total': total,
        'improved': improved,
        'degraded': degraded,
        'stable': stable,
        'new_ratings': new_ratings,
        'with_oct': with_oct,
        'with_dec': with_dec,
        'total_points_up': total_points_up,
        'total_points_down': total_points_down,
        'net_change': net_change,
        'total_comments_oct': total_comments_oct,
        'total_comments_dec': total_comments_dec,
        'new_comments': new_comments
    }

def create_top_changes_chart_altair(df, oct_col, dec_col, title, top_n=10, improvements=True):
    """Create comparison bar chart showing Oct vs Dec ratings."""
    # Make a copy and ensure Difference is numeric
    valid_df = df.copy()
    valid_df['Difference'] = pd.to_numeric(valid_df['Difference'], errors='coerce')

    # Filter for valid differences
    valid_df = valid_df[valid_df['Difference'].notna()]

    # Only show actual improvements or degradations
    if improvements:
        valid_df = valid_df[valid_df['Difference'] > 0]  # Only positive changes
        if len(valid_df) == 0:
            return None
        sorted_df = valid_df.nlargest(min(top_n, len(valid_df)), 'Difference')
    else:
        valid_df = valid_df[valid_df['Difference'] < 0]  # Only negative changes
        if len(valid_df) == 0:
            return None
        sorted_df = valid_df.nsmallest(min(top_n, len(valid_df)), 'Difference')

    if len(sorted_df) == 0:
        return None

    # Prepare data for grouped bar chart
    chart_data = []
    for _, row in sorted_df.iterrows():
        chart_data.append({
            'Apartment': row['Nom'][:40] + '...' if len(row['Nom']) > 40 else row['Nom'],
            'Full Name': row['Nom'],
            'Period': 'October',
            'Rating': row[oct_col] if pd.notna(row[oct_col]) else 0,
            'Difference': row['Difference']
        })
        chart_data.append({
            'Apartment': row['Nom'][:40] + '...' if len(row['Nom']) > 40 else row['Nom'],
            'Full Name': row['Nom'],
            'Period': 'December',
            'Rating': row[dec_col] if pd.notna(row[dec_col]) else 0,
            'Difference': row['Difference']
        })

    chart_df = pd.DataFrame(chart_data)

    # Sort order for y-axis
    apt_order = sorted_df['Nom'].apply(lambda x: x[:40] + '...' if len(x) > 40 else x).tolist()
    if not improvements:
        apt_order = apt_order[::-1]

    chart = alt.Chart(chart_df).mark_bar().encode(
        y=alt.Y('Apartment:N', sort=apt_order, title=''),
        x=alt.X('Rating:Q', title='Rating'),
        color=alt.Color('Period:N', scale=alt.Scale(
            domain=['October', 'December'],
            range=['#888888', 'green' if improvements else 'red']
        )),
        yOffset='Period:N',
        tooltip=['Full Name:N', 'Period:N', 'Rating:Q', 'Difference:Q']
    ).properties(
        title=title,
        width=400,
        height=max(300, len(sorted_df) * 35)
    )

    return chart

def create_scatter_plot(df, oct_col, dec_col, title, max_rating):
    """Create scatter plot comparing October vs December ratings."""
    valid_df = df[df[oct_col].notna() & df[dec_col].notna()].copy()

    if len(valid_df) == 0:
        return None

    # Add color based on evolution
    valid_df['Color'] = valid_df['Difference'].apply(
        lambda x: 'Improved' if x > 0 else ('Degraded' if x < 0 else 'Stable')
    )

    fig = px.scatter(
        valid_df,
        x=oct_col,
        y=dec_col,
        color='Color',
        hover_data=['Nom', 'Difference'],
        title=title,
        color_discrete_map={
            'Improved': 'green',
            'Degraded': 'red',
            'Stable': 'gray'
        }
    )

    # Add diagonal line (no change line)
    fig.add_trace(
        go.Scatter(
            x=[0, max_rating],
            y=[0, max_rating],
            mode='lines',
            name='No Change',
            line=dict(dash='dash', color='black', width=1)
        )
    )

    fig.update_layout(
        xaxis_title=f"Rating {oct_col.split()[-1]}",
        yaxis_title=f"Rating {dec_col.split()[-1]}",
        height=500
    )

    return fig

def create_distribution_chart(df, title):
    """Create histogram of rating changes distribution."""
    valid_df = df[df['Difference'].notna()]

    if len(valid_df) == 0:
        return None

    fig = px.histogram(
        valid_df,
        x='Difference',
        nbins=30,
        title=title,
        color_discrete_sequence=['steelblue']
    )

    fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="No Change")

    fig.update_layout(
        xaxis_title="Rating Change",
        yaxis_title="Number of Apartments",
        height=400
    )

    return fig

def display_alerts(df, dec_col, threshold, platform):
    """Display alerts for critical apartments."""
    st.subheader("⚠️ Alerts - Apartments Requiring Attention")

    # Critical low ratings
    critical = df[df[dec_col].notna() & (df[dec_col] < threshold)].copy()
    critical = critical.sort_values(dec_col)

    if len(critical) > 0:
        st.error(f"**Critical: {len(critical)} apartments with rating below {threshold}**")
        for _, row in critical.head(10).iterrows():
            st.write(f"• **{row['Nom']}**: {row[dec_col]}")

    # Biggest drops
    biggest_drops = df[df['Difference'].notna()].nsmallest(5, 'Difference')
    if len(biggest_drops) > 0 and biggest_drops.iloc[0]['Difference'] < -0.3:
        st.warning("**Biggest Rating Drops:**")
        for _, row in biggest_drops.iterrows():
            if row['Difference'] < -0.3:
                st.write(f"• **{row['Nom']}**: {row['Difference']:+.2f}")

def load_account_manager_mapping(file_path=None, uploaded_file=None):
    """Load account manager to apartment mapping from Excel file."""
    try:
        if uploaded_file is not None:
            df = pd.read_excel(uploaded_file, header=None)
        elif file_path and Path(file_path).exists():
            df = pd.read_excel(file_path, header=None)
        else:
            return None

        # Row 1 has manager names, Row 2+ has apartments
        managers = df.iloc[1].tolist()
        mapping = {}

        for i, manager in enumerate(managers):
            if pd.notna(manager):
                # Get apartment codes for this manager
                apts = df.iloc[2:, i].dropna().tolist()
                # Extract just the number prefix from each apartment code
                apt_numbers = []
                for apt in apts:
                    apt_str = str(apt).strip()
                    match = re.match(r'^(\d+)', apt_str)
                    if match:
                        apt_numbers.append(match.group(1))
                    else:
                        apt_numbers.append(apt_str)
                mapping[str(manager).strip()] = apt_numbers

        return mapping
    except Exception as e:
        return None

def filter_by_manager(df, manager_mapping, selected_manager):
    """Filter dataframe to show only apartments managed by selected manager."""
    if selected_manager == "All Managers" or not manager_mapping:
        return df

    apt_numbers = manager_mapping.get(selected_manager, [])
    if not apt_numbers:
        return df

    # Normalize apartment numbers (remove leading zeros for comparison)
    normalized_apt_numbers = [str(int(n)) if n.isdigit() else n for n in apt_numbers]

    # Filter by Ligne column (extract number prefix and match)
    def matches_manager(ligne):
        if pd.isna(ligne):
            return False
        ligne_str = str(ligne).strip()
        match = re.match(r'^(\d+)', ligne_str)
        if match:
            # Normalize the ligne number too (remove leading zeros)
            ligne_num = str(int(match.group(1)))
            return ligne_num in normalized_apt_numbers
        return ligne_str in apt_numbers

    return df[df['Ligne'].apply(matches_manager)]

def extract_base_apartment_number(ligne):
    """Extract base apartment number from Ligne (e.g., '07 A' -> '07', '157 B' -> '157')."""
    if pd.isna(ligne):
        return None
    ligne_str = str(ligne).strip()
    # Extract just the numeric part (remove A, B, C suffixes)
    match = re.match(r'^(\d+)', ligne_str)
    if match:
        return match.group(1)
    return ligne_str

def aggregate_by_apartment(df, oct_col, dec_col):
    """Aggregate multiple ads per apartment by averaging their ratings."""
    df = df.copy()

    # Extract base apartment number
    df['BaseApt'] = df['Ligne'].apply(extract_base_apartment_number)

    # Convert comment columns to numeric
    if 'Comments Oct' in df.columns:
        df['Comments Oct'] = pd.to_numeric(df['Comments Oct'], errors='coerce')
    if 'Comments Dec' in df.columns:
        df['Comments Dec'] = pd.to_numeric(df['Comments Dec'], errors='coerce')

    # Build aggregation dict
    agg_dict = {
        'Nom': 'first',  # Take first name as representative
        oct_col: 'mean',
        dec_col: 'mean',
        'Ligne': lambda x: ', '.join(x.dropna().astype(str).unique())  # Combine all ligne values
    }

    # Add comment columns if they exist
    if 'Comments Oct' in df.columns:
        agg_dict['Comments Oct'] = 'sum'
    if 'Comments Dec' in df.columns:
        agg_dict['Comments Dec'] = 'sum'

    # Determine groupby columns - include Platform if it exists (for combined view)
    groupby_cols = ['BaseApt']
    if 'Platform' in df.columns:
        groupby_cols.append('Platform')
        agg_dict['Platform'] = 'first'

    # Group by base apartment number (and platform if combined) and aggregate
    aggregated = df.groupby(groupby_cols).agg(agg_dict).reset_index()

    # Recalculate difference and evolution
    aggregated['Difference'] = aggregated.apply(
        lambda row: round(row[dec_col] - row[oct_col], 2) if pd.notna(row[oct_col]) and pd.notna(row[dec_col]) else None,
        axis=1
    )

    def get_evolution(row):
        if pd.isna(row[oct_col]) and pd.notna(row[dec_col]):
            return '★ New Rating'
        elif pd.isna(row['Difference']):
            return 'N/A'
        elif row['Difference'] > 0:
            return '↑ Improved'
        elif row['Difference'] < 0:
            return '↓ Degraded'
        else:
            return '→ Stable'

    aggregated['Evolution'] = aggregated.apply(get_evolution, axis=1)

    # Update Nom to show it's averaged
    aggregated['Nom'] = aggregated.apply(
        lambda row: f"[Avg] Apt {row['BaseApt']} ({row['Ligne']})" if ',' in str(row['Ligne']) else row['Nom'],
        axis=1
    )

    return aggregated

def filter_dataframe(df, search_term, show_only):
    """Filter dataframe based on user selections."""
    filtered = df.copy()

    if search_term:
        filtered = filtered[filtered['Nom'].str.contains(search_term, case=False, na=False)]

    if show_only == "Improvements":
        filtered = filtered[filtered['Difference'] > 0]
    elif show_only == "Degradations":
        filtered = filtered[filtered['Difference'] < 0]
    elif show_only == "Stable":
        filtered = filtered[filtered['Difference'] == 0]
    elif show_only == "New Ratings":
        filtered = filtered[filtered['Evolution'].str.contains('★|New', na=False, regex=True)]

    return filtered

# Main App
def main():
    st.title("🏠 Apartment Grades Dashboard")

    # Sidebar
    st.sidebar.title("Data Source")

    # Cache clear button
    if st.sidebar.button("🔄 Clear Cache & Reload"):
        st.cache_data.clear()
        st.rerun()

    data_source = st.sidebar.radio("Choose data source", ["Upload PDF Files", "Use Sample Data (Oct vs Dec 2025)"])

    st.sidebar.markdown("---")
    st.sidebar.title("Platform")
    platform = st.sidebar.radio("Select Platform", ["Booking.com", "Airbnb", "Both Platforms"])

    # Initialize variables
    df = None
    oct_col = None
    dec_col = None
    if platform == "Booking.com":
        max_rating = 10
        threshold = 5.0
    elif platform == "Airbnb":
        max_rating = 5
        threshold = 3.0
    else:  # Both Platforms
        max_rating = 10  # Use larger scale for combined view
        threshold = 5.0
    period_label = ""

    if data_source == "Upload PDF Files":
        st.markdown("### Upload your PDF files to compare")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📅 Period 1 (Before)")
            before_label = st.text_input("Label for Period 1", value="Before", key="before_label")
            before_file = st.file_uploader(
                f"Upload {platform} PDF for Period 1",
                type=['pdf'],
                key="before_pdf"
            )

        with col2:
            st.subheader("📅 Period 2 (After)")
            after_label = st.text_input("Label for Period 2", value="After", key="after_label")
            after_file = st.file_uploader(
                f"Upload {platform} PDF for Period 2",
                type=['pdf'],
                key="after_pdf"
            )

        if before_file and after_file:
            with st.spinner("Processing PDF files..."):
                try:
                    # Extract data from PDFs
                    if platform == "Booking.com":
                        df_before = extract_booking_data_from_pdf(before_file)
                        df_after = extract_booking_data_from_pdf(after_file)
                    else:
                        df_before = extract_airbnb_data_from_pdf(before_file)
                        df_after = extract_airbnb_data_from_pdf(after_file)

                    if len(df_before) == 0 or len(df_after) == 0:
                        st.error("Could not extract data from one or both PDFs. Please check the file format.")
                    else:
                        # Create comparison
                        df, oct_col, dec_col = create_comparison_df(df_before, df_after, before_label, after_label)
                        period_label = f"{before_label} vs {after_label}"
                        st.success(f"Successfully loaded {len(df_before)} apartments from Period 1 and {len(df_after)} from Period 2")

                except Exception as e:
                    st.error(f"Error processing PDFs: {str(e)}")
        else:
            st.info("Please upload both PDF files to generate the comparison.")

    else:
        # Use sample data
        period_label = "October 2025 vs December 2025"
        if platform == "Booking.com":
            try:
                df = load_booking_data()
                oct_col = "Note Oct 22"
                dec_col = "Note Dec 29"
            except:
                st.error("Sample Booking data not found. Please upload your own files.")
        elif platform == "Airbnb":
            try:
                df = load_airbnb_data()
                oct_col = "Note Oct 23"
                dec_col = "Note Dec 28"
            except:
                st.error("Sample Airbnb data not found. Please upload your own files.")
        else:
            # Both Platforms - combine data
            try:
                df_booking = load_booking_data()
                df_airbnb = load_airbnb_data()

                # Normalize column names for combining
                # Booking: Note Oct 22, Note Dec 29 -> Note Oct, Note Dec
                df_booking = df_booking.rename(columns={
                    'Note Oct 22': 'Note Oct',
                    'Note Dec 29': 'Note Dec'
                })
                df_booking['Platform'] = 'Booking.com'

                # Airbnb: Note Oct 23, Note Dec 28 -> Note Oct, Note Dec
                # Normalize Airbnb from 0-5 scale to 0-10 scale (multiply by 2)
                df_airbnb = df_airbnb.rename(columns={
                    'Note Oct 23': 'Note Oct',
                    'Note Dec 28': 'Note Dec'
                })
                df_airbnb['Note Oct'] = df_airbnb['Note Oct'] * 2
                df_airbnb['Note Dec'] = df_airbnb['Note Dec'] * 2
                df_airbnb['Difference'] = df_airbnb['Difference'] * 2  # Scale difference too
                df_airbnb['Platform'] = 'Airbnb'

                # Combine both dataframes
                df = pd.concat([df_booking, df_airbnb], ignore_index=True)
                oct_col = "Note Oct"
                dec_col = "Note Dec"
            except Exception as e:
                st.error(f"Error loading combined data: {str(e)}")

    # Only show dashboard if we have data
    if df is None or len(df) == 0:
        st.markdown("---")
        st.markdown("### 👆 Upload your PDF files or select sample data to get started")
        return

    st.markdown(f"### Comparison: {period_label}")
    st.markdown("---")

    # Account Manager section
    st.sidebar.markdown("---")
    st.sidebar.title("Account Manager")

    # Try to load default mapping file
    default_mapping_path = SCRIPT_DIR / "acctmanager.xlsx"
    manager_mapping = load_account_manager_mapping(file_path=default_mapping_path)

    # Option to upload custom mapping
    upload_mapping = st.sidebar.checkbox("Upload custom manager file", value=False)
    if upload_mapping:
        mapping_file = st.sidebar.file_uploader("Upload manager mapping (xlsx)", type=['xlsx'])
        if mapping_file:
            manager_mapping = load_account_manager_mapping(uploaded_file=mapping_file)

    # Manager selector
    if manager_mapping:
        manager_options = ["All Managers"] + sorted(manager_mapping.keys())
        selected_manager = st.sidebar.selectbox("Select Account Manager", manager_options)
        st.sidebar.caption(f"{len(manager_mapping.get(selected_manager, []))} apartments" if selected_manager != "All Managers" else "")
    else:
        selected_manager = "All Managers"
        st.sidebar.info("No manager mapping file found. Add 'acctmanager.xlsx' or upload one.")

    # Filters section
    st.sidebar.markdown("---")
    st.sidebar.title("Filters")

    # Get list of all apartments for multi-select
    all_apartments = sorted(df['Nom'].dropna().unique().tolist())

    # Apartment selection
    st.sidebar.subheader("Select Apartments")
    select_all = st.sidebar.checkbox("Select All Apartments", value=True)

    if select_all:
        selected_apartments = all_apartments
    else:
        selected_apartments = st.sidebar.multiselect(
            "Choose apartments to display",
            options=all_apartments,
            default=[],
            help="Select one or more apartments"
        )

    search_term = st.sidebar.text_input("🔍 Search within selection")
    show_only = st.sidebar.selectbox(
        "Show only",
        ["All", "Improvements", "Degradations", "Stable", "New Ratings"]
    )

    st.sidebar.markdown("---")
    st.sidebar.title("Chart Options")
    show_top_n = st.sidebar.slider("Number of apartments in Top Changes", min_value=5, max_value=100, value=10, step=5)

    st.sidebar.markdown("---")
    st.sidebar.title("Data Options")
    average_duplicates = st.sidebar.checkbox(
        "Average multiple ads per apartment",
        value=False,
        help="Combines apartments like 07 A and 07 B into a single averaged entry"
    )

    # Apply manager filter first
    working_df = df.copy()
    if selected_manager != "All Managers" and manager_mapping:
        working_df = filter_by_manager(working_df, manager_mapping, selected_manager)

    # Apply averaging if enabled
    if average_duplicates:
        working_df = aggregate_by_apartment(working_df, oct_col, dec_col)
        # Update apartment list after aggregation
        all_apartments = sorted(working_df['Nom'].dropna().unique().tolist())

    # Apply filters
    filtered_df = working_df.copy()

    # Filter by selected apartments
    if not select_all and len(selected_apartments) > 0:
        filtered_df = filtered_df[filtered_df['Nom'].isin(selected_apartments)]
    elif not select_all and len(selected_apartments) == 0:
        filtered_df = filtered_df.head(0)  # Empty dataframe

    # Apply additional filters
    filtered_df = filter_dataframe(filtered_df, search_term, show_only)

    # Summary metrics
    metrics = create_summary_metrics(filtered_df, oct_col, dec_col)

    # Show manager name in header if filtered
    if selected_manager != "All Managers":
        st.subheader(f"📊 {platform} Summary - {selected_manager}")
    else:
        st.subheader(f"📊 {platform} Summary")

    # Note about different scales for combined view
    if platform == "Both Platforms":
        st.caption("ℹ️ Airbnb ratings normalized to 0-10 scale (×2) for comparison with Booking.com")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Apartments", metrics['total'])
    with col2:
        st.metric("Improved", metrics['improved'], delta=f"+{metrics['improved']}", delta_color="normal")
    with col3:
        st.metric("Degraded", metrics['degraded'], delta=f"-{metrics['degraded']}", delta_color="inverse")
    with col4:
        st.metric("Stable", metrics['stable'])
    with col5:
        st.metric("New Ratings", metrics['new_ratings'])

    # Performance metrics row
    st.markdown("#### Performance Score")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Points Gained", f"+{metrics['total_points_up']}", delta=f"{metrics['improved']} apartments", delta_color="normal")
    with col2:
        st.metric("Total Points Lost", f"-{metrics['total_points_down']}", delta=f"{metrics['degraded']} apartments", delta_color="inverse")
    with col3:
        net = metrics['net_change']
        st.metric("Net Change", f"{'+' if net >= 0 else ''}{net}", delta="Overall" if net >= 0 else "Needs attention", delta_color="normal" if net >= 0 else "inverse")

    # Comments metrics row
    st.markdown("#### Reviews / Comments")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Comments (October)", metrics['total_comments_oct'])
    with col2:
        st.metric("Comments (December)", metrics['total_comments_dec'])
    with col3:
        new_comments = metrics['new_comments']
        st.metric("New Comments", f"{'+' if new_comments >= 0 else ''}{new_comments}",
                  delta="Growth" if new_comments > 0 else ("No change" if new_comments == 0 else "Decrease"),
                  delta_color="normal" if new_comments >= 0 else "inverse")

    st.markdown("---")

    # Tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Top Changes", "🎯 Scatter Plot", "📊 Distribution", "📋 Data Table"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            chart = create_top_changes_chart_altair(filtered_df, oct_col, dec_col, f"Top {show_top_n} Improvements (Oct → Dec)", top_n=show_top_n, improvements=True)
            if chart:
                st.altair_chart(chart, use_container_width=True)
                st.caption("Gray = October rating, Green = December rating")
            else:
                st.info("No improvements to display")

        with col2:
            chart = create_top_changes_chart_altair(filtered_df, oct_col, dec_col, f"Top {show_top_n} Degradations (Oct → Dec)", top_n=show_top_n, improvements=False)
            if chart:
                st.altair_chart(chart, use_container_width=True)
                st.caption("Gray = October rating, Red = December rating")
            else:
                st.info("No degradations to display")

    with tab2:
        valid_df = filtered_df[filtered_df[oct_col].notna() & filtered_df[dec_col].notna()].copy()

        if len(valid_df) > 0:
            # Prepare data for Altair
            valid_df['Status'] = valid_df['Difference'].apply(
                lambda x: 'Improved' if x > 0 else ('Degraded' if x < 0 else 'Stable')
            )
            valid_df['Oct Rating'] = valid_df[oct_col]
            valid_df['Dec Rating'] = valid_df[dec_col]

            # Create diagonal line data
            line_df = pd.DataFrame({'x': [0, max_rating], 'y': [0, max_rating]})

            # Scatter plot with hover
            scatter = alt.Chart(valid_df).mark_circle(size=80, opacity=0.7).encode(
                x=alt.X('Oct Rating:Q', title='October Rating', scale=alt.Scale(domain=[0, max_rating + 0.5])),
                y=alt.Y('Dec Rating:Q', title='December Rating', scale=alt.Scale(domain=[0, max_rating + 0.5])),
                color=alt.Color('Status:N', scale=alt.Scale(
                    domain=['Improved', 'Degraded', 'Stable'],
                    range=['green', 'red', 'gray']
                )),
                tooltip=['Nom:N', 'Oct Rating:Q', 'Dec Rating:Q', 'Difference:Q', 'Status:N']
            ).properties(
                title=f"{platform}: October vs December Ratings",
                width=700,
                height=500
            )

            # Diagonal line (no change)
            line = alt.Chart(line_df).mark_line(strokeDash=[5, 5], color='black', opacity=0.5).encode(
                x='x:Q',
                y='y:Q'
            )

            chart = scatter + line
            st.altair_chart(chart, use_container_width=True)
            st.caption("Hover over points to see apartment details. Points above diagonal = improved.")
        else:
            st.info("Not enough data for scatter plot")

    with tab3:
        valid_df = filtered_df[filtered_df['Difference'].notna()].copy()

        if len(valid_df) > 0:
            # Create Altair histogram with hover
            histogram = alt.Chart(valid_df).mark_bar(opacity=0.7).encode(
                x=alt.X('Difference:Q', bin=alt.Bin(maxbins=30), title='Rating Change'),
                y=alt.Y('count()', title='Number of Apartments'),
                tooltip=[alt.Tooltip('Difference:Q', bin=alt.Bin(maxbins=30), title='Rating Change Range'),
                         alt.Tooltip('count()', title='Count')]
            ).properties(
                title="Distribution of Rating Changes",
                width=700,
                height=400
            )

            # Vertical line at 0
            rule = alt.Chart(pd.DataFrame({'x': [0]})).mark_rule(color='red', strokeDash=[5, 5]).encode(
                x='x:Q'
            )

            chart = histogram + rule
            st.altair_chart(chart, use_container_width=True)
            st.caption("Red dashed line = no change. Bars to the right = improvements, left = degradations.")
        else:
            st.info("Not enough data for distribution chart")

    with tab4:
        st.subheader("Full Data Table")

        # Format the dataframe for display
        display_df = filtered_df.copy()

        # Add styling
        st.dataframe(
            display_df,
            use_container_width=True,
            height=600,
            column_config={
                "Nom": st.column_config.TextColumn("Apartment Name", width="large"),
                "Difference": st.column_config.NumberColumn("Change", format="%.2f"),
                oct_col: st.column_config.NumberColumn(f"Oct Rating", format="%.1f"),
                dec_col: st.column_config.NumberColumn(f"Dec Rating", format="%.1f"),
            }
        )

        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download filtered data as CSV",
            data=csv,
            file_name=f"{platform.lower().replace('.', '')}_filtered_data.csv",
            mime="text/csv"
        )

    st.markdown("---")

    # Alerts section
    display_alerts(filtered_df, dec_col, threshold, platform)

    # Footer
    st.markdown("---")
    st.caption("Dashboard generated on December 31, 2025 | Data: October 22-23 vs December 28-29, 2025")

if __name__ == "__main__":
    main()
