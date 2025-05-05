# creating the streamlit frontend for the visualizations 

import streamlit as st
from src.data_graphs import (create_tree_height_violin, 
                             load_tree_data,
                             create_tree_height_histogram,
                             create_bird_activity_heatmap,
                             create_bird_activity_polar,
                             create_bird_conservation_plot,
                             create_tree_mass_comparison,
                             create_plotly_waffle_chart,
                             create_bird_arrival_duration_plot,
                             create_sankey_diagram,
                             create_sankey_diagram_reversed,

)

# Set page configuration for a wider layout
st.set_page_config(
    page_title="Bird Conservation Data",
    page_icon="🦜",
    layout="wide"
)

# Create title and subtitle
st.title("🦜 Bird Conservation Status Analysis")
st.subheader("Comparing Bird Populations in Intact vs. Degraded Forest Areas")
st.markdown("---")

# Load the data
# Assuming df_combined is a DataFrame containing the necessary data
df_trees = load_tree_data(tree_file_path='Tree.xlsx', sheet_name='TreeI', header_row=0)
df_birds = load_tree_data(tree_file_path='Tree.xlsx', sheet_name='Birds_entact', header_row=0)

# Container for centered content
container = st.container()

# Use columns to center the graphs (3 columns, with graphs in the middle)
with container:
    # Graph 1
    st.markdown("### IUCN Status Distribution")
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        # Placeholder for Graph 1
        
        
        # Create the violin plot
        fig = create_tree_height_violin(df_trees)

        # Display the plot
        st.plotly_chart(fig, use_container_width=True)

        # To save the plot to an HTML file that can be opened in a browser
        # fig.write_html("tree_height_comparison.html")
    st.markdown("---")


    # Graph 2
    

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        # Placeholder for Graph 1
        
        
        # Create the violin plot
        fig = create_tree_height_histogram(df_trees)

        # Display the plot
        st.plotly_chart(fig, use_container_width=True)

        # To save the plot to an HTML file that can be opened in a browser
        # fig.write_html("tree_height_comparison.html")
    st.markdown("---")
        
    # Graph 3
    st.markdown("### Heatmap of Bird sightings")

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        # Placeholder for Graph 1
        
        
        # Create the violin plot
        fig = create_bird_activity_heatmap(df_birds)

        # Display the plot
        st.plotly_chart(fig, use_container_width=True)

        # To save the plot to an HTML file that can be opened in a browser
        # fig.write_html("tree_height_comparison.html")
    st.markdown("---")
        
    # Graph 4
    st.markdown("### Polar map of Bird sightings")

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        # Placeholder for Graph 1
        
        
        # Create the violin plot
        fig = create_bird_activity_polar(df_birds)

        # Display the plot
        st.plotly_chart(fig, use_container_width=True)

        # To save the plot to an HTML file that can be opened in a browser
        # fig.write_html("tree_height_comparison.html")
    st.markdown("---")

# Graph 5
    st.markdown("### Polar map of Bird sightings")

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        # Placeholder for Graph 1
        
        
        # Create the violin plot
        fig = create_bird_conservation_plot(df_birds)

        # Display the plot
        st.plotly_chart(fig, use_container_width=True)

        # To save the plot to an HTML file that can be opened in a browser
        # fig.write_html("tree_height_comparison.html")
    st.markdown("---")


    # Graph 6
    st.markdown("### Polar map of Bird sightings")

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        # Placeholder for Graph 1
        
        
        # Create the violin plot
        result = create_tree_mass_comparison(df_trees)

        # Display the plot
        st.plotly_chart(result['figure'], use_container_width=True)

        # To save the plot to an HTML file that can be opened in a browser
        # fig.write_html("tree_height_comparison.html")
    st.markdown("---")

    # Graph 7
    st.markdown("### Polar map of Bird sightings")

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        # Placeholder for Graph 1
        
        
        # Create the violin plot
        fig = create_plotly_waffle_chart(df_birds)

        # Display the plot
        st.plotly_chart(fig, use_container_width=True)

        # To save the plot to an HTML file that can be opened in a browser
        # fig.write_html("tree_height_comparison.html")
    st.markdown("---")


     # Graph 8
    st.markdown("### Polar map of Bird sightings")

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        # Placeholder for Graph 1
        
        
        # Create the violin plot
        fig =  create_bird_arrival_duration_plot(df_birds)

        # Display the plot
        st.pyplot(fig)

        # To save the plot to an HTML file that can be opened in a browser
        # fig.write_html("tree_height_comparison.html")
    st.markdown("---")

     # Graph 9
    st.markdown("### Polar map of Bird sightings")

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        # Placeholder for Graph 1
        
        
        # Create the violin plot
        fig =  create_sankey_diagram(df_birds)

        # Display the plot
        st.plotly_chart(fig, use_container_width=True)

        # To save the plot to an HTML file that can be opened in a browser
        # fig.write_html("tree_height_comparison.html")
    st.markdown("---")

       # Graph 10
    st.markdown("### Polar map of Bird sightings")

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        # Placeholder for Graph 1
        
        
        # Create the violin plot
        fig =  create_sankey_diagram_reversed(df_birds)

        # Display the plot
        st.plotly_chart(fig, use_container_width=True)

        # To save the plot to an HTML file that can be opened in a browser
        # fig.write_html("tree_height_comparison.html")
    st.markdown("---")
    

# Add a footer
st.markdown("---")
st.markdown("**Data Source**: Field observations collected in forest research areas")
st.markdown("**Contact**: research@conservationproject.org")
