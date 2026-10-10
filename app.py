import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import shap
import os

# Set page config
st.set_page_config(
    page_title="Customer Insights Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    :root {
        --primary-blue: #38bdf8;
        --secondary-blue: #818cf8;
        --bg-color: #0f172a;
        --card-bg: #1e293b;
        --text-main: #f8fafc;
        --text-muted: #94a3b8;
        --border-color: #334155;
    }
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif !important;
    }
    
    .stApp {
        background-color: var(--bg-color);
        color: var(--text-main);
    }
    
    .metric-card {
        background-color: var(--card-bg);
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid var(--border-color);
        border-left: 4px solid var(--secondary-blue);
        transition: transform 0.2s ease-in-out;
        min-height: 105px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .metric-title {
        color: var(--text-muted);
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.02em;
        margin-bottom: 4px;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--text-main);
        white-space: nowrap;
    }

    .metric-value-segment {
        font-size: 1.2rem !important;
        font-weight: 600;
        color: var(--text-main);
        white-space: normal !important;
        word-break: normal !important;
        line-height: 1.25 !important;
    }

    /* Prevent text truncation in any Streamlit native metric widgets */
    div[data-testid="stMetricValue"] {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
    }

    div[data-testid="stMetricValue"] > div {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
        word-break: break-word !important;
        font-size: 1.4rem !important;
        line-height: 1.25 !important;
    }

    div[data-testid="stMetricLabel"] {
        white-space: normal !important;
        overflow: visible !important;
    }
    
    h1, h2, h3 {
        color: var(--text-main);
        font-weight: 600;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        border-bottom: 1px solid var(--border-color);
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 0;
        border-bottom: 2px solid transparent;
        color: var(--text-muted);
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        color: var(--primary-blue);
        border-bottom: 2px solid var(--primary-blue);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('models/current_customers.csv')
        df = df.rename(columns={'Unnamed: 0': 'Customer ID'})
        df['Customer ID'] = df['Customer ID'].astype(int)
        return df
    except FileNotFoundError:
        return None

def main():
    st.title("Customer Insights Platform")
    st.markdown("### Enterprise Analytics Dashboard")
    
    corp_colors = ['#38bdf8', '#818cf8', '#c084fc', '#f472b6', '#2dd4bf']
    
    df = load_data()
    
    if df is None:
        st.warning("Customer data not found. Please run the model building pipeline (`build_models.py`) first.")
        return
        
    # Sidebar Filters
    st.sidebar.header("Filters")
    
    segment_filter = st.sidebar.multiselect(
        "Customer Segment",
        options=df['Segment'].unique(),
        default=df['Segment'].unique()
    )
    
    clv_range = st.sidebar.slider(
        "Predicted CLV Range (₹)",
        min_value=float(df['Predicted_CLV'].min()),
        max_value=float(df['Predicted_CLV'].max()),
        value=(float(df['Predicted_CLV'].min()), float(df['Predicted_CLV'].max()))
    )
    
    churn_range = st.sidebar.slider(
        "Churn Probability Range (%)",
        min_value=0.0,
        max_value=100.0,
        value=(0.0, 100.0)
    )
    
    # Apply filters
    active_segments = segment_filter if segment_filter else df['Segment'].unique()
    
    filtered_df = df[
        (df['Segment'].isin(active_segments)) &
        (df['Predicted_CLV'] >= clv_range[0]) &
        (df['Predicted_CLV'] <= clv_range[1]) &
        (df['Churn_Probability'] * 100 >= churn_range[0]) &
        (df['Churn_Probability'] * 100 <= churn_range[1])
    ]
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Executive Summary", 
        "Customer Profiles", 
        "Retention Analysis", 
        "Lifetime Value", 
        "Customer Search"
    ])
    
    with tab1:
        st.header("Executive Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Total Customers</div>
                    <div class="metric-value">{len(filtered_df):,}</div>
                </div>
            """, unsafe_allow_html=True)
            
        with col2:
            avg_churn = filtered_df['Churn_Probability'].mean() * 100
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Avg Churn Risk</div>
                    <div class="metric-value">{avg_churn:.1f}%</div>
                </div>
            """, unsafe_allow_html=True)
            
        with col3:
            avg_clv = filtered_df['Predicted_CLV'].mean()
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Avg Predicted CLV</div>
                    <div class="metric-value">₹{avg_clv:,.0f}</div>
                </div>
            """, unsafe_allow_html=True)
            
        with col4:
            at_risk = len(filtered_df[filtered_df['Segment'].str.contains('At Risk')])
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Customers At Risk</div>
                    <div class="metric-value">{at_risk:,}</div>
                </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            # Segment Distribution
            segment_counts = filtered_df['Segment'].value_counts().reset_index()
            segment_counts.columns = ['Segment', 'Count']
            fig_seg = px.pie(segment_counts, names='Segment', values='Count', 
                            title='Customer Segments Distribution',
                            hole=0.4,
                            color_discrete_sequence=corp_colors)
            fig_seg.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#f8fafc')
            st.plotly_chart(fig_seg, width='stretch')
            
        with c2:
            # Value vs Risk Scatter
            fig_scatter = px.scatter(
                filtered_df, x='Churn_Probability', y='Predicted_CLV',
                color='Segment', size='Monetary',
                hover_data=['Customer ID', 'Frequency'],
                title='Customer Value vs Churn Risk',
                labels={'Churn_Probability': 'Churn Probability', 'Predicted_CLV': 'Predicted CLV (₹)', 'Monetary': 'Monetary (₹)'},
                color_discrete_sequence=corp_colors
            )
            fig_scatter.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#f8fafc')
            st.plotly_chart(fig_scatter, width='stretch')
            
    with tab2:
        st.header("Customer Profiles")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            fig_r = px.histogram(filtered_df, x='Recency', nbins=30, title='Recency Distribution',
                                color_discrete_sequence=['#38bdf8'])
            fig_r.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#f8fafc')
            st.plotly_chart(fig_r, width='stretch')
            
        with col2:
            fig_f = px.histogram(filtered_df[filtered_df['Frequency'] < 50], x='Frequency', nbins=30, title='Frequency Distribution (Zoomed)',
                                color_discrete_sequence=['#818cf8'])
            fig_f.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#f8fafc')
            st.plotly_chart(fig_f, width='stretch')
            
        with col3:
            fig_m = px.histogram(filtered_df[filtered_df['Monetary'] < 10000], x='Monetary', nbins=30, title='Monetary Distribution (Zoomed)',
                                labels={'Monetary': 'Monetary (₹)'},
                                color_discrete_sequence=['#c084fc'])
            fig_m.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#f8fafc')
            st.plotly_chart(fig_m, width='stretch')

    with tab3:
        st.header("Retention Analysis")
        
        c1, c2 = st.columns(2)
        with c1:
            fig_churn_hist = px.histogram(filtered_df, x='Churn_Probability', nbins=20, 
                                        title='Churn Probability Distribution',
                                        color='Segment',
                                        color_discrete_sequence=corp_colors)
            fig_churn_hist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#f8fafc')
            st.plotly_chart(fig_churn_hist, width='stretch')
            
        with c2:
            fig_recency_churn = px.scatter(filtered_df, x='Recency', y='Churn_Probability', 
                                         color='Segment',
                                         title='Recency vs Churn Risk',
                                         color_discrete_sequence=corp_colors)
            fig_recency_churn.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#f8fafc')
            st.plotly_chart(fig_recency_churn, width='stretch')
            
        st.subheader("Top High-Risk Customers")
        high_risk = filtered_df.sort_values(by='Churn_Probability', ascending=False).head(10)
        st.dataframe(high_risk[['Customer ID', 'Churn_Probability', 'Segment', 'Recency', 'Frequency', 'Monetary']].style.format({
            'Churn_Probability': '{:.1%}',
            'Monetary': '₹{:.2f}'
        }))

    with tab4:
        st.header("Lifetime Value")
        
        c1, c2 = st.columns(2)
        with c1:
            fig_clv_hist = px.histogram(filtered_df[filtered_df['Predicted_CLV'] < 10000], x='Predicted_CLV', nbins=30, 
                                        title='Predicted CLV Distribution (Zoomed)',
                                        labels={'Predicted_CLV': 'Predicted CLV (₹)'},
                                        color_discrete_sequence=['#38bdf8'])
            fig_clv_hist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#f8fafc')
            st.plotly_chart(fig_clv_hist, width='stretch')
            
        with c2:
            fig_hist_pred = px.scatter(filtered_df, x='Monetary', y='Predicted_CLV', 
                                     color='Segment',
                                     title='Historical Revenue vs Predicted CLV',
                                     labels={'Monetary': 'Historical Revenue (₹)', 'Predicted_CLV': 'Predicted CLV (₹)'},
                                     color_discrete_sequence=corp_colors)
            fig_hist_pred.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#f8fafc')
            st.plotly_chart(fig_hist_pred, width='stretch')
            
        st.subheader("Top High-Value Customers")
        high_value = filtered_df.sort_values(by='Predicted_CLV', ascending=False).head(10)
        st.dataframe(high_value[['Customer ID', 'Predicted_CLV', 'Segment', 'Monetary', 'Frequency', 'Tenure']].style.format({
            'Predicted_CLV': '₹{:.2f}',
            'Monetary': '₹{:.2f}'
        }))

    with tab5:
        st.header("Individual Customer Analysis")
        
        # Segment filter for Customer Search
        available_segments = sorted(filtered_df['Segment'].unique().tolist())
        
        col_seg, col_cust = st.columns([1, 2])
        with col_seg:
            seg_filter_options = ["All Selected Segments"] + available_segments
            chosen_seg = st.selectbox("Filter by Segment", options=seg_filter_options)
            
        search_pool = filtered_df if chosen_seg == "All Selected Segments" else filtered_df[filtered_df['Segment'] == chosen_seg]
        
        with col_cust:
            cust_options = search_pool['Customer ID'].unique()
            if len(cust_options) == 0:
                st.warning("No customers found for the selected segment.")
                cust_id = None
            else:
                cust_id = st.selectbox(
                    f"Search Customer ID ({len(cust_options):,} available)", 
                    options=cust_options
                )
        
        if cust_id is not None:
            customer = search_pool[search_pool['Customer ID'] == cust_id].iloc[0]
            
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Predicted CLV</div>
                        <div class="metric-value">₹{customer['Predicted_CLV']:,.2f}</div>
                    </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Churn Probability</div>
                        <div class="metric-value">{customer['Churn_Probability']*100:.1f}%</div>
                    </div>
                """, unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Customer Segment</div>
                        <div class="metric-value metric-value-segment">{customer['Segment']}</div>
                    </div>
                """, unsafe_allow_html=True)
            with c4:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Total Spending</div>
                        <div class="metric-value">₹{customer['Monetary']:,.2f}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # SHAP Explanation Waterfall Chart (Mockup using Plotly)
            st.subheader("Churn Risk Explainability (SHAP)")
            
            features_cols = ['Recency', 'Frequency', 'Monetary', 'Total_Items', 'Average_Order_Value', 'Tenure', 'Product_Diversity', 'Purchase_Freq_Rate', 'Recency_vs_Tenure']
            shap_cols = [f"{c}_SHAP" for c in features_cols]
            
            shap_vals = customer[shap_cols].values
            base_val = customer['SHAP_Base_Value']
            
            # Sort by absolute SHAP value
            shap_df = pd.DataFrame({'Feature': features_cols, 'SHAP Value': shap_vals})
            shap_df['Abs SHAP'] = shap_df['SHAP Value'].abs()
            shap_df = shap_df.sort_values('Abs SHAP', ascending=True)
            
            fig_shap = go.Figure(go.Bar(
                x=shap_df['SHAP Value'],
                y=shap_df['Feature'],
                orientation='h',
                marker_color=['#f87171' if val > 0 else '#34d399' for val in shap_df['SHAP Value']],
                text=[f"+{val:.2f}" if val > 0 else f"{val:.2f}" for val in shap_df['SHAP Value']],
                textposition='auto'
            ))
            
            fig_shap.update_layout(
                title='Factors Influencing Churn Probability',
                xaxis_title='SHAP Value',
                yaxis_title='Feature',
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)', 
                font_color='#f8fafc',
                height=400
            )
            
            st.plotly_chart(fig_shap, width='stretch')

if __name__ == "__main__":
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    import sys
    import os
    
    if get_script_run_ctx() is None:
        # User clicked the "Run" button instead of using `streamlit run app.py`
        os.system(f'"{sys.executable}" -m streamlit run "{__file__}"')
    else:
        main()
