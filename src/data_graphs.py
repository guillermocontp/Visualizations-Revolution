import pandas as pd
import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import math
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('birds_entact.csv')

print(df.head())

def load_tree_data(tree_file_path='Tree.xlsx', sheet_name='TreeI', header_row=0):
    """
    Load tree data from Excel file and return the dataframe.
    
    Parameters:
    - tree_file_path: Path to the Excel file (default: 'Tree.xlsx')
    - sheet_name: Name of the sheet to read (default: 'TreeI')
    - header_row: Row index containing column headers (default: 0)
    
    Returns:
    - DataFrame containing the tree data or None if file not found
    """
    # Initialize an empty dataframe
    df_combined = pd.DataFrame()
    
    # Check if file exists to avoid unnecessary errors
    if not os.path.exists(tree_file_path):
        print(f"Error: File not found at {tree_file_path}")
        return None
    
    try:
        # Read the tree file with headers at the specified row
        df_combined = pd.read_excel(tree_file_path, sheet_name=sheet_name, header=header_row)
        print(f"Successfully loaded {len(df_combined)} rows from: {tree_file_path}")
        print(f"Found columns: {list(df_combined.columns)}")
        return df_combined
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    

def create_tree_height_violin(df, height_col='Height_m', status_col='Status', 
                              colors={'Intact': '#358600', 'Degraded': '#C08552'}):
    """
    Create an interactive violin plot comparing tree heights by status using Plotly.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the tree data
    height_col : str
        Column name for tree height (default: 'Height_m')
    status_col : str
        Column name for status (default: 'Status')
    colors : dict
        Dictionary mapping status values to colors
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The interactive violin plot figure
    """
     
    # Convert height column to numeric
    df[height_col] = pd.to_numeric(df[height_col], errors='coerce')
    
    # Check if we have valid height data
    if not df[height_col].notna().any():
        print(f"No valid data found in {height_col} column")
        return None
    
    # Calculate summary statistics for annotation
    stats = df.groupby(status_col)[height_col].agg(['mean', 'median']).round(2)
    
    # Create the violin plot with plotly express
    fig = px.violin(
        df, 
        y=height_col,
        x=status_col,
        color=status_col,
        points="all",  # Show all points
        color_discrete_map=colors,
        
        labels={height_col: "Height (m)", status_col: "Forest Condition"},
        template="plotly_white",  # Clean white background with light grid
        
    )
    
    # Customize layout for a more striking appearance
    fig.update_layout(
        
        
        title=dict(
            text='<b>Tree Height Comparison:</b> Trees in <span style="color:brown;">degraded</span> areas are smaller in average</b>',
            font=dict(size=24), # Increased font size
            x=0.5, # Center the title
            xanchor='center',
            y=0.95, # Adjust vertical position if needed
            yanchor='top'
        ),          
        legend_title="<b>Forest Condition</b>",
        yaxis_title="<b>Height (meters)</b>",
        xaxis_title="<b>Forest Condition</b>",
        violingap=0.2,  # Gap between violins
          # 'overlay' for semi-transparent violins
         
    )
    
          
    
    # Add a subtle note about environmental impact
    fig.add_annotation(
        text="Trees in degraded areas show reduced height, indicating decreased forest health and carbon storage capacity.",
        x=0.5, y=-1,
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=13, color="darkgray")
    )
            
    return fig


import plotly.graph_objects as go # Make sure go is imported
# ... other imports ...

def create_tree_height_histogram(df, height_col='Height_m', status_col='Status',
                                colors={'Intact': '#2e8b57', 'Degraded': '#8b4513'}
                                ):
    """
    Create an interactive overlaid histogram comparing tree heights by status using Plotly (raw counts).

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the tree data
    height_col : str
        Column name for tree height (default: 'Height_m')
    status_col : str
        Column name for status (default: 'Status')
    colors : dict
        Dictionary mapping status values to colors
    title : str
        Title for the plot

    Returns:
    --------
    plotly.graph_objects.Figure
        The interactive histogram figure
    """


    # Convert height column to numeric
    df[height_col] = pd.to_numeric(df[height_col], errors='coerce')

    # Check if we have valid height data
    if not df[height_col].notna().any():
        print(f"No valid data found in {height_col} column")
        return None

    # Extract data for each group
    status_values = df[status_col].unique()
    valid_statuses = [s for s in status_values if s in colors]

    # Create the figure object
    fig = go.Figure()

    # Add a histogram trace for each status
    for status in valid_statuses:
        status_heights = df[df[status_col] == status][height_col].dropna()

        if len(status_heights) > 0:
            fig.add_trace(go.Histogram(
                x=status_heights,
                name=status, # This will appear in the legend
                marker_color=colors.get(status, '#888888'),
                opacity=0.75, # Make bars semi-transparent for overlap visibility
                histnorm=None, # Use raw counts
                xbins=dict( # Optional: Define binning explicitly if needed
                    # start=all_heights.min(),
                    # end=all_heights.max(),
                    # size=bin_size # Calculated earlier, or let Plotly decide
                )
            ))

    # Set y-axis title for raw counts
    yaxis_title_text = "<b>Count</b>"


    # Customize layout
    fig.update_layout(
        title={
            'text': '<b><span style="color:green;">Intact</span> areas have more species with different <span style="color:green;">heights</span></b>',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#333333'}
        },
        xaxis_title={
            'text': "<b>Tree Height (meters)</b>",
            'font': {'size': 16}
        },
        yaxis_title={
            'text': yaxis_title_text, # Use fixed title for counts
            'font': {'size': 16}
        },
        barmode='overlay', # Overlay the histograms
        template='plotly_white',
        legend_title_text='Forest Condition', # Add legend title
        margin=dict(l=50, r=50, t=100, b=50),

        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_family="Arial"
        ),
    )

    return fig


def create_bird_activity_heatmap(df, date_col='Date', time_col='Time_Stamp', 
                                 
                                title='Bird Activity:  <span style="color:steelblue;">The dawn chorus</span> and  <span style="color:steelblue;">The evening chorus</span>'):
    """
    Create an interactive heatmap of bird activity by day of week and hour using Plotly.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the bird sightings data
    date_col : str
        Column name for date (default: 'Date')
    time_col : str
        Column name for time (default: 'Time_Stamp')
    colors : list
        List of colors for the heatmap gradient (default: blues)
    title : str
        Title for the plot
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The interactive heatmap figure
    """
    data = df.copy()
    
    # Process the date column
    if date_col in data.columns:
        # Try to convert the date column to datetime
        try:
            if data[date_col].dtype != 'datetime64[ns]':
                # Try common date formats
                for date_format in ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y', '%m-%d-%Y']:
                    try:
                        data[date_col] = pd.to_datetime(data[date_col], format=date_format, errors='raise')
                        break
                    except:
                        continue
                
                # If still not converted, try pandas auto-detection
                if data[date_col].dtype != 'datetime64[ns]':
                    data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
            
            # Extract day of week
            data['Day_of_Week'] = data[date_col].dt.day_name()
        except Exception as e:
            print(f"Error processing date data: {e}")
            return None
    else:
        print(f"Date column '{date_col}' not found in the dataset")
        return None
    
    # Process the time column
    if time_col in data.columns:
        try:
            # Handle different time formats
            if data[time_col].dtype == 'object':
                # Try parsing as time strings
                try:
                    data['Hour'] = pd.to_datetime(data[time_col], format='%H:%M:%S', errors='coerce').dt.hour
                except:
                    # Try parsing as datetime strings
                    try:
                        data['Hour'] = pd.to_datetime(data[time_col], errors='coerce').dt.hour
                    except:
                        # Last resort: manual extraction from string
                        data['Hour'] = data[time_col].str.extract(r'(\d+):', expand=False).astype(float)
            elif pd.api.types.is_datetime64_dtype(data[time_col]):
                # Already datetime type
                data['Hour'] = data[time_col].dt.hour
            else:
                # Numeric type - assume it's already the hour
                data['Hour'] = data[time_col]
            
            # Ensure hours are numeric and in the correct range
            data['Hour'] = pd.to_numeric(data['Hour'], errors='coerce')
            data['Hour'] = data['Hour'].fillna(0).astype(int) % 24
        except Exception as e:
            print(f"Error processing time data: {e}")
            return None
    else:
        print(f"Time column '{time_col}' not found in the dataset")
        return None
    
    # Remove any rows with missing processed data
    data = data.dropna(subset=['Day_of_Week', 'Hour'])
    
    if len(data) == 0:
        print("No valid data after processing dates and times")
        return None
    
    # Order days of week properly
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Count sightings by hour and day of week
    heatmap_data = data.groupby(['Day_of_Week', 'Hour']).size().reset_index(name='Count')
    
    # Pivot data for the heatmap
    pivot_data = heatmap_data.pivot(
        index='Day_of_Week', 
        columns='Hour', 
        values='Count'
    ).fillna(0)
    
    # Reorder days
    pivot_data = pivot_data.reindex(days_order)
    
    # Create the heatmap using Plotly
    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=[f"{hour:02d}:00" for hour in range(24)],
        y=pivot_data.index,
        colorscale='ylgnbu',
        hoverongaps=False,
        hovertemplate='Day: %{y}<br>Hour: %{x}<br>Bird sightings: %{z}<extra></extra>',
        colorbar=dict(
            title='Number of<br>Bird Sightings',
          
        )
    ))
    
    # Calculate peak activity times for annotations
    total_by_hour = data.groupby('Hour').size()
    peak_hour = total_by_hour.idxmax()
    peak_hour_count = total_by_hour.max()
    
    total_by_day = data.groupby('Day_of_Week').size()
    day_order_dict = {day: i for i, day in enumerate(days_order)}
    sorted_days = sorted(total_by_day.index, key=lambda x: day_order_dict.get(x, 99))
    peak_day = total_by_day.idxmax()
    peak_day_count = total_by_day.max()
    
    # Customize layout
    fig.update_layout(
        title={
            'text': f"<b>{title}</b>",
            'y': 0.97,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#333333'}
        },
        xaxis={
            'title': "<b>Hour of Day</b>",
            
            'tickfont': {'size': 12},
            'tickangle': -30,
            'dtick': 2,  # Show every 2 hours for clarity
        },
        yaxis={
            'title': "<b>Day of Week</b>",
            
            'tickfont': {'size': 14},
            'categoryorder': 'array',
            'categoryarray': days_order
        },
        margin=dict(l=80, r=50, t=100, b=80),
        plot_bgcolor='rgba(248, 250, 255, 0.9)',
        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_family="Arial"
        ),
        
    )
    
    return fig


def create_bird_activity_polar(df, date_col='Date', time_col='Time_Stamp', 
                             title='Bird Activity:  <span style="color:#FF3C8E;">The dawn chorus</span> and  <span style="color:#FF3C8E;">The evening chorus</span>',
                             colorscale=None):
    """
    Create an interactive polar chart showing bird activity by day of week and hour.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the bird sightings data
    date_col : str
        Column name for date (default: 'Date')
    time_col : str
        Column name for time (default: 'Time_Stamp')
    title : str
        Title for the plot
    colorscale : list
        List of colors for the intensity scale (default is yellow to purple)
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The interactive polar chart figure
    """
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    
    # Create a copy of the dataframe to avoid modifying the original
    data = df.copy()
    
    # Process the date column to extract day of week
    if date_col in data.columns:
        try:
            if data[date_col].dtype != 'datetime64[ns]':
                # Try common date formats
                for date_format in ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y', '%m-%d-%Y']:
                    try:
                        data[date_col] = pd.to_datetime(data[date_col], format=date_format, errors='raise')
                        break
                    except:
                        continue
                
                # If still not converted, try pandas auto-detection
                if data[date_col].dtype != 'datetime64[ns]':
                    data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
            
            # Extract day of week
            data['Day_of_Week'] = data[date_col].dt.day_name()
        except Exception as e:
            print(f"Error processing date data: {e}")
            return None
    else:
        print(f"Date column '{date_col}' not found in the dataset")
        return None
    
    # Process the time column to extract hour
    if time_col in data.columns:
        try:
            # Handle different time formats
            if data[time_col].dtype == 'object':
                # Try parsing as time strings
                try:
                    data['Hour'] = pd.to_datetime(data[time_col], format='%H:%M:%S', errors='coerce').dt.hour
                except:
                    # Try parsing as datetime strings
                    try:
                        data['Hour'] = pd.to_datetime(data[time_col], errors='coerce').dt.hour
                    except:
                        # Last resort: manual extraction from string
                        data['Hour'] = data[time_col].str.extract(r'(\d+):', expand=False).astype(float)
            elif pd.api.types.is_datetime64_dtype(data[time_col]):
                # Already datetime type
                data['Hour'] = data[time_col].dt.hour
            else:
                # Numeric type - assume it's already the hour
                data['Hour'] = data[time_col]
            
            # Ensure hours are numeric and in the correct range
            data['Hour'] = pd.to_numeric(data['Hour'], errors='coerce')
            data['Hour'] = data['Hour'].fillna(0).astype(int) % 24
        except Exception as e:
            print(f"Error processing time data: {e}")
            return None
    else:
        print(f"Time column '{time_col}' not found in the dataset")
        return None
    
    # Remove any rows with missing processed data
    data = data.dropna(subset=['Day_of_Week', 'Hour'])
    
    if len(data) == 0:
        print("No valid data after processing dates and times")
        return None
    
    # Define days of week and hour ranges
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    hours = list(range(24))
    
    # Count sightings by day and hour
    activity_counts = data.groupby(['Day_of_Week', 'Hour']).size().reset_index(name='Count')
    
    # Prepare data for the polar chart
    chart_data = []
    for day in days:
        for hour in hours:
            # Find count for this day and hour
            filtered = activity_counts[(activity_counts['Day_of_Week'] == day) & 
                                      (activity_counts['Hour'] == hour)]
            
            count = filtered['Count'].values[0] if not filtered.empty else 0
            
            chart_data.append({
                'day': day,
                'hour': hour,
                'count': count
            })
    
    # Create a DataFrame with the prepared data
    df_polar = pd.DataFrame(chart_data)
    
    # Map days to radius values (Monday=1, Tuesday=2, etc.)
    day_to_radius = {day: i+1 for i, day in enumerate(days)}
    df_polar['r1'] = df_polar['day'].map(day_to_radius)
    df_polar['r2'] = df_polar['r1'] + 1
    
    # Map hours to angles (0=midnight, starting at top, clockwise)
    hour_to_angle_start = {hour: hour * 15 for hour in hours}
    hour_to_angle_end = {hour: (hour + 1) * 15 for hour in hours}
    df_polar['theta_start'] = df_polar['hour'].map(hour_to_angle_start)
    df_polar['theta_end'] = df_polar['hour'].map(hour_to_angle_end)
    
    # Find max count for normalization
    max_count = df_polar['count'].max()
    if max_count > 0:
        df_polar['normalized_count'] = df_polar['count'] / max_count
    else:
        df_polar['normalized_count'] = 0
    
    # Define a colorscale (yellow to purple)
    if colorscale is None:
        colorscale = [
            '#FFFF99', '#FFED6F', '#FFE14D', '#FFD52C', '#FFC914', '#FFBA10', 
            '#FF9D32', '#FF7E4D', '#FF5E68', '#FF3C8E', '#F033C3', '#C340E0',
            '#9D51F3', '#6872FE', '#00A6FF', '#00CCFF'
        ]
    
    # Function to map normalized count to the colorscale
    def get_color(normalized_value, colors):
        index = int(normalized_value * (len(colors) - 1))
        return colors[index]
    
    # Create the figure
    fig = go.Figure()
    
    for index, row in df_polar.iterrows():
        r1 = row['r1']
        r2 = row['r2']
        t1 = row['theta_start']
        t2 = row['theta_end']
        
        color = get_color(row['normalized_count'], colorscale)
        hover_text = f"Day: {row['day']}<br>Hour: {row['hour']:02d}:00<br>Sightings: {row['count']}"
        
        fig.add_trace(go.Scatterpolar(
            theta=[t1, t2, t2, t1, t1],
            r=[r1, r1, r2, r2, r1],
            mode='lines',
            fill='toself',
            fillcolor=color,
            line=dict(color='white', width=0.7),
            name=f"{row['day']} {row['hour']}:00",
            showlegend=False,
            hoverinfo='text',
            text=hover_text
        ))
    
    fig.update_layout(
    # ... (your existing polar dict here) ...
    polar=dict(
         radialaxis=dict(
             visible=False,
             showticklabels=False,
             showgrid=False,
             showline=False
         ),
         angularaxis=dict(
             tickvals=[(i + 0.5) * 15 for i in range(24)],
             ticktext=[f"{i:02d}:00" for i in range(24)],
             direction="clockwise",
             period=360,
             rotation=90,
             showgrid=False, # Keep False
             showline=False, # Keep False
             visible=True    # Keep True to show tick labels (hours)
         )
    ),
    # --- Add these lines ---
    xaxis=dict(
        visible=False,
        showgrid=False,
        zeroline=False
    ),
    yaxis=dict(
        visible=False,
        showgrid=False,
        zeroline=False
    ),
    # --- End of added lines ---
    title={
            'text': f"<b>{title}</b>",
            'y': 0.97,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#333333'}
        },
    showlegend=False,
    template="plotly_white", # Or try template=None
    margin=dict(t=100, b=50, l=50, r=50)
)
    
    # Add colorbar to show intensity scale
    z = np.linspace(0, 1, 100)
    colorbar_colors = [get_color(val, colorscale) for val in z]
    
    # Add a hidden trace to show the colorbar
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(
            colorscale=colorbar_colors,
            showscale=True,
            cmin=0,
            cmax=max_count,
            colorbar=dict(
                title="Bird Sightings"
                
            )
        ),
        hoverinfo='none'
    ))
    
    
    
    return fig


def create_bird_conservation_plot(df, iucn_col='IUCN_Status', status_col='Status', 
                                 title='All vulnerable <span style="color:green;">species of birds</span> are only seen in <span style="color:green;">Intact areas</span>   ',
                                 colors=None):
    """
    Create an interactive stacked bar chart showing bird conservation status distribution
    in intact vs degraded areas.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the bird data
    iucn_col : str
        Column name for IUCN conservation status (default: 'IUCN_Status')
    status_col : str
        Column name for habitat status (default: 'Status')
    title : str
        Title for the plot
    colors : dict
        Dictionary mapping status values to colors
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The interactive stacked bar chart figure
    """
    import pandas as pd
    import plotly.graph_objects as go
    import numpy as np
    
    # Create a copy to avoid modifying the original dataframe
    birds_df = df.copy()
    
    # Check for required columns
    required_columns = [iucn_col, status_col]
    if not all(col in birds_df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in birds_df.columns]
        print(f"Error: Missing required columns: {missing}")
        return None
    
    # Fill missing values with 'Unknown' to avoid NaN issues
    birds_df[iucn_col] = birds_df[iucn_col].fillna('Unknown')
    birds_df[status_col] = birds_df[status_col].fillna('Unknown')
    
    # Filter for just Intact and Degraded statuses
    valid_statuses = ['Intact', 'Degraded']
    birds_df = birds_df[birds_df[status_col].isin(valid_statuses)]
    
    if len(birds_df) == 0:
        print(f"No data after filtering for {valid_statuses}")
        return None
    
    # Group by IUCN_Status and Status, then count
    iucn_counts = birds_df.groupby([iucn_col, status_col]).size().reset_index(name='Count')
    
    # Pivot the data for easier processing
    pivot_counts = iucn_counts.pivot(index=iucn_col, columns=status_col, values='Count').fillna(0)
    
    # Ensure both status columns exist
    for status in valid_statuses:
        if status not in pivot_counts.columns:
            pivot_counts[status] = 0
    
    # Calculate total counts per IUCN status for sorting
    pivot_counts['Total'] = pivot_counts['Intact'] + pivot_counts['Degraded']
    
    # Define IUCN status order from least to most concern
    iucn_order = ['Unknown', 'Least Concern', 'Near Threatened', 'Vulnerable', 
                 'Endangered', 'Critically Endangered', 'Extinct in the Wild', 'Extinct']
    
    # Filter for statuses that exist in our data
    iucn_order = [status for status in iucn_order if status in pivot_counts.index]
    
    # Reindex using the order
    pivot_counts = pivot_counts.reindex(iucn_order)
    
    # Calculate percentages for each IUCN status
    percentages = pivot_counts.copy()
    for idx in percentages.index:
        total = percentages.loc[idx, 'Total']
        if total > 0:
            percentages.loc[idx, 'Intact'] = (percentages.loc[idx, 'Intact'] / total) * 100
            percentages.loc[idx, 'Degraded'] = (percentages.loc[idx, 'Degraded'] / total) * 100
    
    # Define vibrant colors if not provided
    if colors is None:
        colors = {
            'Intact': '#20B2AA',  # Bright green
            'Degraded': '#CD5C5C'  # Vibrant orange/red
        }
    
    # Create the figure
    fig = go.Figure()
    
    # Add traces for each status (in reverse order for proper stacking)
    for status in reversed(valid_statuses):
        fig.add_trace(go.Bar(
            y=percentages.index,
            x=percentages[status],
            name=status,
            
            orientation='h',
            marker=dict(
                color=colors.get(status, '#999999'),
                line=dict(color='white', width=1)
            ),
            text=[f"{val:.1f}%" if val > 5 else "" for val in percentages[status]],
            textposition='inside',
            insidetextanchor='middle',
            textfont=dict(color='white', size=12, family='Arial Black'),
            hovertemplate='<b>%{y}</b><br>' +
                          f'{status}: %{{x:.1f}}%<br>' +
                          'Count: %{customdata}<extra></extra>',
            customdata=pivot_counts[status]
        ))
    
    # Update layout with improved styling
    fig.update_layout(
        title={
            'text': f"<b>{title}</b>",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#333333'}
        },
        xaxis={
            'title': "<b>Percentage of Bird Sightings (%)</b>",
            
            'tickfont': {'size': 14},
            'range': [0, 100],
            'gridcolor': 'rgba(220, 220, 220, 0.8)',
        },
        yaxis={
            'title': "<b>IUCN Conservation Status</b>",
            
            'tickfont': {'size': 14},
            'categoryorder': 'array',
            'categoryarray': iucn_order,
        },
        barmode='stack',
        
        margin=dict(l=20, r=20, t=80, b=20),
        height=600,
        plot_bgcolor='rgba(245, 245, 245, 0.8)',
        
        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_family="Arial"
        ),
    )
   
  
   
    
    # Add horizontal line at y=0
    fig.add_shape(
        type="line",
        x0=0,
        y0=-0.5,
        x1=100,
        y1=-0.5,
        line=dict(color="#CCCCCC", width=2)
    )
    
    return fig


def create_tree_mass_comparison(df, common_name_col='Common_Name', height_col='Height_m', 
                               diameter_col='Diameter_cm', status_col='Status', 
                               density_col='Density', default_wood_density=0.6, top_n=15):
    """
    Create interactive treemaps comparing tree species mass between intact and degraded areas,
    using species-specific wood density values when available.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the tree data
    common_name_col : str
        Column name for tree common name
    height_col : str
        Column name for tree height in meters
    diameter_col : str
        Column name for tree diameter in centimeters
    status_col : str
        Column name for tree status (Intact/Degraded)
    density_col : str
        Column name for wood density in g/cm³
    default_wood_density : float
        Default wood density value in g/cm³ to use when species-specific values are not available
    top_n : int
        Number of top species to display in each treemap
        
    Returns:
    --------
    dict
        Dictionary containing the Plotly figure, summary statistics, and top species data
    """
    
    
    # Check if we have the required columns
    required_cols = [common_name_col, height_col, diameter_col, status_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    result = {
        "figure": None,
        "statistics": {},
        "top_species": {"intact": [], "degraded": []},
        "error": None,
        "density_used": "species-specific" if density_col in df.columns else "default"
    }
    
    if missing_cols:
        result["error"] = f"Missing required columns: {missing_cols}"
        return result
    
    try:
        # Make a copy to avoid modifying the original dataframe
        df_trees = df.copy()
        
        # Convert height and diameter to numeric, handling errors
        df_trees[height_col] = pd.to_numeric(df_trees[height_col], errors='coerce')
        df_trees[diameter_col] = pd.to_numeric(df_trees[diameter_col], errors='coerce')
        
        # Check if wood density column exists and convert to numeric
        has_density_col = density_col in df_trees.columns
        if has_density_col:
            df_trees[density_col] = pd.to_numeric(df_trees[density_col], errors='coerce')
            # Create a mapping of common names to their average wood density
            density_map = df_trees.groupby(common_name_col)[density_col].mean().to_dict()
            # For missing values, use species average or default
            df_trees[density_col] = df_trees.apply(
                lambda row: row[density_col] if pd.notnull(row[density_col]) 
                else density_map.get(row[common_name_col], default_wood_density), 
                axis=1
            )
        
        # Drop rows with missing values in key columns
        df_trees = df_trees.dropna(subset=[common_name_col, height_col, diameter_col, status_col])
        
        # Fill any missing Common_Name values with 'Unknown'
        df_trees[common_name_col] = df_trees[common_name_col].fillna('Unknown')
        
        # Calculate tree mass with species-specific wood density when available
        if has_density_col:
            # Mass ≈ π × (diameter/2)² × height × specific wood density
            df_trees['Mass_kg'] = (np.pi * ((df_trees[diameter_col]/2)**2) * 
                                  df_trees[height_col] * df_trees[density_col]) / 1000  # Convert g to kg
            result["density_used"] = "species-specific"
        else:
            # Use default wood density
            df_trees['Mass_kg'] = (np.pi * ((df_trees[diameter_col]/2)**2) * 
                                  df_trees[height_col] * default_wood_density) / 1000  # Convert g to kg
            result["density_used"] = "default"
        
        # Filter for valid statuses
        valid_statuses = ['Intact', 'Degraded']
        df_trees = df_trees[df_trees[status_col].isin(valid_statuses)]
        
        # Group by species and status to get total mass and average density
        if has_density_col:
            mass_by_species = df_trees.groupby([common_name_col, status_col]).agg({
                'Mass_kg': 'sum',
                density_col: 'mean'
            }).reset_index()
        else:
            mass_by_species = df_trees.groupby([common_name_col, status_col])['Mass_kg'].sum().reset_index()
        
        # Create separate dataframes for intact and degraded
        intact_df = mass_by_species[mass_by_species[status_col] == 'Intact']
        degraded_df = mass_by_species[mass_by_species[status_col] == 'Degraded']
        
        # Sort by mass and limit to top N species for readability
        intact_df = intact_df.sort_values('Mass_kg', ascending=False).head(top_n)
        degraded_df = degraded_df.sort_values('Mass_kg', ascending=False).head(top_n)
        
        # Create a figure with two treemaps side by side
        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[1, 1],
            subplot_titles=(
                f'Tree Species Mass in Intact Areas (Top {len(intact_df)})',
                f'Tree Species Mass in Degraded Areas (Top {len(degraded_df)})'
            ),
            specs=[[{"type": "treemap"}, {"type": "treemap"}]]
        )
        
        # Define color scales for the two treemaps
        intact_colorscale = [
            [0, '#c5e8c5'],          # Light green
            [0.5, '#5cb85c'],         # Medium green
            [1, '#2e7d32']           # Dark green
        ]
        
        degraded_colorscale = [
            [0, '#ffe0b2'],          # Light orange/tan
            [0.5, '#ff9800'],         # Medium orange
            [1, '#e65100']           # Dark orange/brown
        ]
        
        # Create treemap for intact trees if data exists
        if not intact_df.empty:
            # Format hover text with density info if available
            if has_density_col:
                intact_df['text'] = intact_df.apply(lambda row: 
                    f"{row[common_name_col]}<br>Mass: {row['Mass_kg']:.1f} kg<br>Density: {row[density_col]:.2f} g/cm³", axis=1)
            else:
                intact_df['text'] = intact_df.apply(lambda row: 
                    f"{row[common_name_col]}<br>Mass: {row['Mass_kg']:.1f} kg", axis=1)
            
            # Calculate percentages of total mass for labeling
            total_intact_mass = intact_df['Mass_kg'].sum()
            intact_df['percent'] = (intact_df['Mass_kg'] / total_intact_mass * 100).round(1)
            
            # Add treemap trace for intact trees
            fig.add_trace(go.Treemap(
                labels=intact_df[common_name_col],
                values=intact_df['Mass_kg'],
                parents=[""] * len(intact_df),
                text=intact_df['text'],
                hovertemplate='<b>%{label}</b><br>Mass: %{value:.1f} kg<br>Percentage: %{percentRoot:.1f}%<extra></extra>',
                texttemplate='<b>%{label}</b><br>',
                marker=dict(
                    colorscale=intact_colorscale,
                    colors=intact_df['Mass_kg'],
                    line=dict(width=1, color='white')
                ),
                branchvalues='total',
                textposition="middle center",
                name="Intact"
            ), row=1, col=1)
        
        # Create treemap for degraded trees if data exists
        if not degraded_df.empty:
            # Format hover text with density info if available
            if has_density_col:
                degraded_df['text'] = degraded_df.apply(lambda row: 
                    f"{row[common_name_col]}<br>Mass: {row['Mass_kg']:.1f} kg<br>Density: {row[density_col]:.2f} g/cm³", axis=1)
            else:
                degraded_df['text'] = degraded_df.apply(lambda row: 
                    f"{row[common_name_col]}<br>Mass: {row['Mass_kg']:.1f} kg", axis=1)
            
            # Calculate percentages of total mass for labeling
            total_degraded_mass = degraded_df['Mass_kg'].sum()
            degraded_df['percent'] = (degraded_df['Mass_kg'] / total_degraded_mass * 100).round(1)
            
            # Add treemap trace for degraded trees
            fig.add_trace(go.Treemap(
                labels=degraded_df[common_name_col],
                values=degraded_df['Mass_kg'],
                parents=[""] * len(degraded_df),
                text=degraded_df['text'],
                hovertemplate='<b>%{label}</b><br>Mass: %{value:.1f} kg<br>Percentage: %{percentRoot:.1f}%<extra></extra>',
                texttemplate='<b>%{label}</b><br>',
                marker=dict(
                    colorscale=degraded_colorscale,
                    colors=degraded_df['Mass_kg'],
                    line=dict(width=1, color='white')
                ),
                branchvalues='total',
                textposition="middle center",
                name="Degraded"
            ), row=1, col=2)
        
        # Calculate summary statistics
        total_intact_mass = intact_df['Mass_kg'].sum() if not intact_df.empty else 0
        total_degraded_mass = degraded_df['Mass_kg'].sum() if not degraded_df.empty else 0
        
        # Calculate percent difference
        if total_intact_mass > 0 and total_degraded_mass > 0:
            pct_diff = ((total_intact_mass - total_degraded_mass) / 
                       ((total_intact_mass + total_degraded_mass) / 2)) * 100
            pct_diff_abs = abs(pct_diff)
            diff_direction = 'higher in intact areas' if pct_diff > 0 else 'higher in degraded areas'
        else:
            pct_diff_abs = 0
            diff_direction = 'N/A'
        
        # Prepare summary statistics for return
        result["statistics"] = {
            "total_intact_mass": round(total_intact_mass, 1),
            "total_degraded_mass": round(total_degraded_mass, 1),
            "percent_difference": round(pct_diff_abs, 1) if total_intact_mass > 0 and total_degraded_mass > 0 else 0,
            "difference_direction": diff_direction
        }
        
        # Get top 5 species for each area with density info if available
        if has_density_col:
            for i, row in intact_df.head(5).iterrows():
                result["top_species"]["intact"].append({
                    "name": row[common_name_col],
                    "mass": round(row['Mass_kg'], 1),
                    "density": round(row[density_col], 2)
                })
                
            for i, row in degraded_df.head(5).iterrows():
                result["top_species"]["degraded"].append({
                    "name": row[common_name_col],
                    "mass": round(row['Mass_kg'], 1),
                    "density": round(row[density_col], 2)
                })
        else:
            for i, row in intact_df.head(5).iterrows():
                result["top_species"]["intact"].append({
                    "name": row[common_name_col],
                    "mass": round(row['Mass_kg'], 1)
                })
                
            for i, row in degraded_df.head(5).iterrows():
                result["top_species"]["degraded"].append({
                    "name": row[common_name_col],
                    "mass": round(row['Mass_kg'], 1)
                })
        
        # Update layout
        density_note = "species-specific wood density values" if has_density_col else f"default wood density value of {default_wood_density} g/cm³"
        
        fig.update_layout(
            title={
                'text': 'There is 47.6% more tree mass in <span style="color:green;">Intact areas</span>  ',
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 24, 'color': '#333333', 'family': 'Arial, sans-serif'}
            },
            margin=dict(t=100, l=10, r=10, b=25),
            height=700,
            font=dict(
                family="Arial, sans-serif",
                size=13,
                color="#333333"
            ),
            paper_bgcolor='rgba(240, 240, 240, 0.1)',
            plot_bgcolor='rgba(240, 240, 240, 0.1)',
            
            annotations=[
                
                dict(
                    x=0.5, y=-0.05,
                    xref='paper', yref='paper',
                    text=f'Note: Tree mass calculated using formula: π × (diameter/2)² × height × wood density<br>' +
                         f'Using {density_note}',
                    showarrow=False,
                    font=dict(size=11, color='gray', style='italic'),
                    align='center'
                )
            ]
        )
        
        # Store the figure in the result dictionary
        result["figure"] = fig
        
        return result
    
    except Exception as e:
        result["error"] = str(e)
        return result
    

import pandas as pd
import numpy as np
import plotly.graph_objects as go

def create_plotly_waffle_chart(birds_df):
    """
    Creates a Plotly waffle chart visualizing bird sightings by status (Intact vs. Degraded).

    Args:
        birds_df (pd.DataFrame): DataFrame containing bird sighting data with a 'Status' column.

    Returns:
        plotly.graph_objects.Figure: The Plotly figure object, or None if data is invalid.
    """
    try:
        # Check for required columns
        if 'Status' not in birds_df.columns:
            print(f"Error: Required column 'Status' not found in the dataset.")
            print(f"Available columns: {list(birds_df.columns)}")
            return None

        # --- Data Processing (similar to Matplotlib version) ---
        # Fill missing values with 'Unknown'
        birds_df_copy = birds_df.copy() # Work on a copy to avoid modifying original df
        birds_df_copy['Status'] = birds_df_copy['Status'].fillna('Unknown')

        # Filter for just Intact and Degraded statuses
        valid_statuses = ['Intact', 'Degraded']
        birds_df_copy = birds_df_copy[birds_df_copy['Status'].isin(valid_statuses)]

        # Count sightings by status
        status_counts = birds_df_copy['Status'].value_counts()

        # Ensure both statuses exist
        for status in valid_statuses:
            if status not in status_counts.index:
                status_counts[status] = 0

        # Get total sightings for each area
        intact_total = status_counts.get('Intact', 0)
        degraded_total = status_counts.get('Degraded', 0)
        total_sightings = intact_total + degraded_total

        if total_sightings == 0:
            print("No sightings found for Intact or Degraded status. Cannot create chart.")
            return None

        # Calculate proportions
        intact_proportion = intact_total / total_sightings
        degraded_proportion = degraded_total / total_sightings

        print(f"\nBird sightings by area:")
        print(f"Intact areas: {intact_total} sightings ({intact_proportion*100:.1f}% of total)")
        print(f"Degraded areas: {degraded_total} sightings ({degraded_proportion*100:.1f}% of total)")
        print(f"Total sightings: {total_sightings}")

        # --- Prepare data for Plotly Waffle ---
        total_squares = 100
        intact_squares = int(round(intact_proportion * total_squares))
        degraded_squares = total_squares - intact_squares

        # Create lists for x, y coordinates and status labels
        x_coords = []
        y_coords = []
        status_labels = []
        colors = []
        color_map = {'Intact': 'forestgreen', 'Degraded': '#D2B48C'}

        waffle_width = 10
        waffle_height = 10
        count = 0
        for r in range(waffle_height):
            for c in range(waffle_width):
                x_coords.append(c)
                y_coords.append(waffle_height - 1 - r) # Plotly y-axis goes upwards
                if count < intact_squares:
                    status_labels.append('Intact')
                    colors.append(color_map['Intact'])
                else:
                    status_labels.append('Degraded')
                    colors.append(color_map['Degraded'])
                count += 1

        # --- Create Plotly Figure ---
        fig = go.Figure()

        # Add squares as scatter points
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='markers',
            marker=dict(
                color=colors,
                size=46, # Adjust size as needed
                symbol='square',
                line=dict(width=4, color='black') # Grid lines via marker outline
            ),
            
            showlegend=False # We will create a custom legend annotation
        ))

        # --- Configure Layout ---
        fig.update_layout(
            title=dict(
                text='<b>The Cost of Degradation: Bird Populations Show <span style="color:#D2B48C;"> Only 30% Activity in Degraded Areas</span></b>',
                font=dict(size=20),
                y=0.95,
                x=0.5,
                xanchor='center',
                yanchor='top'
            ),
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=[-0.5, waffle_width - 0.5] # Ensure squares are centered
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=[-0.5, waffle_height - 0.5],
                scaleanchor="x", # Make squares square
                scaleratio=1
            ),
            width=600, # Adjust figure size
            height=700,
            plot_bgcolor='white',
            margin=dict(t=100, b=100, l=50, r=50), # Adjust margins for title/legend
            # Add border
            xaxis_showline=False, yaxis_showline=False,
            
        )

        # Add custom legend-like annotations
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.5, y=-0.1, # Position below the chart
            showarrow=False,
            text=(f"<span style='color:forestgreen; font-size: 15px;'>■</span> Intact: {intact_total} sightings ({intact_proportion*100:.1f}%)   "
                  f"<span style='color:#D2B48C; font-size: 15px;'>■</span> Degraded: {degraded_total} sightings ({degraded_proportion*100:.1f}%)"),
            align="center",
            font=dict(size=14)
        )

        # Add explanation text (optional, similar to matplotlib version)
        # fig.add_annotation(
        #     xref="paper", yref="paper",
        #     x=0.5, y=-0.18,
        #     showarrow=False,
        #     text="<i>Each square represents 1% of all bird sightings.</i>",
        #     align="center",
        #     font=dict(size=12)
        # )

        return fig

    except Exception as e:
        print(f"An error occurred while creating the Plotly waffle chart: {e}")
        return None
    

df = pd.read_csv('birds_entact.csv')
# standardize date format
def standardize_date_format(date_str):
    day, month, year = date_str.split('/')
    return f"{int(day)}/{int(month)}/{year}"

# Apply function to df
df['Date'] = df['Date'].apply(standardize_date_format)
# drop columns
df = df.drop(['ID', 'Date', 'IUCN_Status'], axis=1)
# convert Time_Stamp to datetime
df['Time_Stamp'] = pd.to_datetime(df['Time_Stamp'], format='%H:%M:%S', errors='coerce')

# time to seconds
df['Time_Seconds'] = df['Time_Stamp'].dt.hour * 3600 + df['Time_Stamp'].dt.minute * 60 + df['Time_Stamp'].dt.second

# round Duration to nearest second
df['Duration_sec'] = df['Duration_sec'].round().astype(int)

# group by Latin_Name and calculate mean
summary_df = df.groupby('Latin_Name').agg({
    'Time_Seconds': 'mean',
    'Duration_sec': 'mean'
}).reset_index()

# Convert seconds to rounded-up
def seconds_to_time_rounded_up(seconds):
    seconds = int(math.ceil(seconds / 60.0)) * 60  # round up to full minute
    time = timedelta(seconds=seconds)
    hours, remainder = divmod(time.seconds, 3600)
    minutes = remainder // 60
    return f"{hours:02d}:{minutes:02d}"

summary_df['Mean_Time_Stamp'] = summary_df['Time_Seconds'].apply(seconds_to_time_rounded_up)

# Round duration mean to whole seconds
summary_df['Mean_Duration_sec'] = summary_df['Duration_sec'].round().astype(int)

# Drop intermediate
summary_df = summary_df.drop(columns=['Time_Seconds', 'Duration_sec'])


# Convert Time_Stamp to datetime if not already
df['Time_Stamp'] = pd.to_datetime(df['Time_Stamp'], format='%H:%M:%S', errors='coerce')

# Convert time to seconds
df['Time_Seconds'] = df['Time_Stamp'].dt.hour * 3600 + df['Time_Stamp'].dt.minute * 60 + df['Time_Stamp'].dt.second

# Round Duration to nearest second
df['Duration_sec'] = df['Duration_sec'].round().astype(int)

# Define the tolerance windows
time_tolerance_seconds = 3 * 60
duration_tolerance_seconds = 5

# Function to assign IDs based on time and duration proximity
def assign_ids(df, time_tolerance, duration_tolerance):
    df['ID'] = None
    next_id = 1
    processed = [False] * len(df)

    for i in range(len(df)):
        if not processed[i]:
            df.loc[i, 'ID'] = next_id
            processed[i] = True
            time_i = df.loc[i, 'Time_Seconds']
            duration_i = df.loc[i, 'Duration_sec']

            for j in range(i + 1, len(df)):
                if not processed[j]:
                    time_j = df.loc[j, 'Time_Seconds']
                    duration_j = df.loc[j, 'Duration_sec']

                    if (abs(time_i - time_j) <= time_tolerance) and (abs(duration_i - duration_j) <= duration_tolerance):
                        df.loc[j, 'ID'] = next_id
                        processed[j] = True
            next_id += 1
    return df

# Apply the ID assignment function
df = assign_ids(df.copy(), time_tolerance_seconds, duration_tolerance_seconds)

# Group by Latin_Name and calculate mean
summary_df = df.groupby('Latin_Name').agg({
    'Time_Seconds': 'mean',
    'Duration_sec': 'mean',
    'ID': 'first'  # Keep the first assigned ID for the group
}).reset_index()

# Convert seconds to rounded-up HH:MM
def seconds_to_time_rounded_up(seconds):
    seconds = int(math.ceil(seconds / 60.0)) * 60  # round up to full minute
    time = timedelta(seconds=seconds)
    hours, remainder = divmod(time.seconds, 3600)
    minutes = remainder // 60
    return f"{hours:02d}:{minutes:02d}"

summary_df['Mean_Time_Stamp'] = summary_df['Time_Seconds'].apply(seconds_to_time_rounded_up)

# Round duration mean to whole seconds
summary_df['Mean_Duration_sec'] = summary_df['Duration_sec'].round().astype(int)

# Drop intermediate columns
summary_df = summary_df.drop(columns=['Time_Seconds', 'Duration_sec'])
# tolerance definition
time_tolerance_seconds = 3 * 60
duration_tolerance_seconds = 5

# create function to assign IDs based on time and duration 
def assign_ids(df, time_tolerance, duration_tolerance):
    df['ID'] = None
    next_id = 1
    processed = [False] * len(df)

    for i in range(len(df)):
        if not processed[i]:
            df.loc[i, 'ID'] = next_id
            processed[i] = True
            time_i = df.loc[i, 'Time_Seconds']
            duration_i = df.loc[i, 'Duration_sec']

            for j in range(i + 1, len(df)):
                if not processed[j]:
                    time_j = df.loc[j, 'Time_Seconds']
                    duration_j = df.loc[j, 'Duration_sec']

                    if (abs(time_i - time_j) <= time_tolerance) and (abs(duration_i - duration_j) <= duration_tolerance):
                        df.loc[j, 'ID'] = next_id
                        processed[j] = True
            next_id += 1
    return df

# Apply the function
df = assign_ids(df.copy(), time_tolerance_seconds, duration_tolerance_seconds)

# Group by Latin_Name and calculate mean
# keep first assigend ID for the group
summary_df = df.groupby('Latin_Name').agg({
    'Time_Seconds': 'mean',
    'Duration_sec': 'mean',
    'ID': 'first'
}).reset_index()

# Convert seconds to rounded-up HH:MM
# and round up to full minute
def seconds_to_time_rounded_up(seconds):
    seconds = int(math.ceil(seconds / 60.0)) * 60
    time = timedelta(seconds=seconds)
    hours, remainder = divmod(time.seconds, 3600)
    minutes = remainder // 60
    return f"{hours:02d}:{minutes:02d}"

summary_df['Mean_Time_Stamp'] = summary_df['Time_Seconds'].apply(seconds_to_time_rounded_up)

# Round duration mean to whole seconds
summary_df['Mean_Duration_sec'] = summary_df['Duration_sec'].round().astype(int)

# Drop intermediate columns
summary_df = summary_df.drop(columns=['Time_Seconds', 'Duration_sec'])
# Count the occurrences of each ID
id_counts = summary_df['ID'].value_counts()

# source IDs that appear only once
unique_ids = id_counts[id_counts == 1].index.tolist()

# Drop rows where the ID is in the list of unique IDs
filtered_summary_df = summary_df[~summary_df['ID'].isin(unique_ids)]


# converting Mean_Time_Stamp to numeric (seconds since midnight)
def time_to_seconds(time_str):
    """Converts HH:MM time string to seconds since midnight."""
    hours, minutes = map(int, time_str.split(':'))
    return hours * 3600 + minutes * 60

filtered_summary_df['Time_Seconds'] = filtered_summary_df['Mean_Time_Stamp'].apply(
    time_to_seconds
)

# grouping DataFrame by 'ID'
grouped_df = (
    filtered_summary_df.groupby('ID')
    .agg(
        Mean_Time_Seconds=('Time_Seconds', 'mean'),
        Mean_Duration_sec=('Mean_Duration_sec', 'mean'),
        Latin_Names=('Latin_Name', list),
    )
    .reset_index()
)


# creating the scatter plot
plt.figure(figsize=(12, 6))
sns.scatterplot(
    data=grouped_df,
    x='Mean_Time_Seconds',
    y='Mean_Duration_sec',
    hue='ID',
    palette='tab20',
    size=1000, 
    alpha=0.8,
    legend=False
)

for index, row in grouped_df.iterrows():
    species_text = '\n'.join(row['Latin_Names'])
    plt.text(
        row['Mean_Time_Seconds'],
        row['Mean_Duration_sec'],
        species_text,
        ha='center',
        va='center',
        color='black',
        fontsize=12,
    )


# x-axis time
def format_time_ticks(seconds):
    """Formats seconds since midnight to HH:MM."""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    return f"{int(hours):02d}:{int(minutes):02d}"


plt.xticks(
    ticks=plt.xticks()[0],  # Get the current tick locations
    labels=[
        format_time_ticks(x) for x in plt.xticks()[0]
    ],  
) 

plt.xlabel('Mean Arrival Time (HH:MM)')
plt.ylabel('Mean Duration (seconds)')
plt.title('Bird Arrival Time and Duration by Group')
plt.grid(True)


plt.tight_layout()  # Adjust layout to prevent labels from overlapping
plt.show()
