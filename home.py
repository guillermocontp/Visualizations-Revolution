# creating the streamlit frontend for the visualizations 

import streamlit as st
from src.data_graphs import (create_tree_height_violin, 
                             load_tree_data,
                             create_tree_height_histogram,
                             create_bird_activity_heatmap,
                             create_bird_activity_polar,
                             create_bird_conservation_plot,
                             create_tree_volume_comparison,
                             create_plotly_waffle_chart,
                             create_bird_arrival_duration_plot,
                             create_sankey_diagram,
                             create_sankey_diagram_reversed,

)

# Set page configuration for a wider layout
st.set_page_config(
    page_title="Tracking biodiversity",
    page_icon="🌴🦜🌳",
    layout="wide",  
    initial_sidebar_state="collapsed"
)

# Create title and subtitle
# Create a single CSS block at the beginning of your app
st.markdown("""
    <style>
        /* Title styles */
        .infographic-header {
            color: white !important;
            font-size: 40px !important;
            font-weight: bold !important;
            font-family: 'open sans', sans-serif !important;
            line-height: 2 !important;
            margin-bottom: 2px !important;
            display: block !important;
            text-align: center !important;
        }
        
        /* Section header styles */
        .paragraph-header {
            color: white !important;
            font-size: 28px !important;
            font-weight: bold !important;
            font-family: 'open sans', sans-serif !important;
            line-height: 1.2 !important;
            margin-bottom: 10px !important;
            display: block !important;
            text-align: left !important;
        }
        
        /* Paragraph text styles */
        .paragraph {
            color: white !important;
            font-size: 22px !important;
            font-family: 'open sans', sans-serif !important;
            line-height: 1.5 !important;
            margin-bottom: 15px !important;
            display: block !important;
            text-align: left !important;
        }
        
        /* Highlighted text */
        .highlight-text-green {
            color: #358600 !important; 
            font-weight: bold !important;
        }
            .highlight-text-brown {
            color: #C08552 !important; 
            font-weight: bold !important;
        }
                 
        
        
    </style>
""", unsafe_allow_html=True)

# Then apply these styles in your content sections without redefining them

# Create container for intro section
intro_container = st.container(border=None)

with intro_container:
    col1, col2, col3 = st.columns([2, 3, 2])
    with col2:
        st.markdown("<p class='infographic-header'>Tracking biodiversity: Understanding its importance for <span class='highlight-text-green'>restoration and research</span></p>", unsafe_allow_html=True)

        st.image('forest_aerial.jpg', 
                    use_container_width=True)

st.markdown("""
                    
                    
                    """)
st.markdown("""
                    
                    
                    """)

paragraph_container = st.container(border=None)  
with paragraph_container:
    col1, col2, col3 = st.columns([2, 2.5, 2])
    with col2:
                    
        st.markdown("""
        <p class='paragraph-header'>Digital reality: Taking biodiversity monitoring to the next level</p>
        <p class='paragraph'>R-evolution has developed the Green Cubes Nature Methodology, a science-based, peer-reviewed approach to forest monitoring. </p>
        
        <p class='paragraph'>The data collected can help:
        <ul class='paragraph'>
        <li>More accurate monitoring for reforest efforts</li>
        <li>Correlation between tree species and animal activity</li>
        <li>Comparison between areas that have been damaged (degraded) and intact areas</li>
        </ul>
        </p>
        
        
        """, unsafe_allow_html=True)
st.markdown("""
             
            """)
st.markdown("""
            
            """)
st.markdown("""
        
        """)
st.markdown("""
        
        """)
st.markdown("""
        
        """)
            


# Load the data
# Assuming df_combined is a DataFrame containing the necessary data
df_trees = load_tree_data(tree_file_path='Tree.xlsx', sheet_name='TreeI', header_row=0)
df_birds = load_tree_data(tree_file_path='Tree.xlsx', sheet_name='Birds_entact', header_row=0)

# Creating different containers for the different content sections

jungle_container = st.container(border=None)




st.markdown('<div class="fullwidth-container">', unsafe_allow_html=True)
with jungle_container:
     # Full-width container starts   
    
    col1, col2, col3, col4, col5 = st.columns([2, 3, .5, 3, 2])
    with col2:
        st.markdown("""
        
        """)
        st.markdown("""

        """)
        st.markdown("""
        
        """)
        st.markdown("""
        
        """)
        st.markdown("""
        
        """)
        st.markdown("""
        
        """)
        st.markdown("""
        
        """)
        
        st.markdown("""
        <p class='paragraph'>Through metric analysis, we identify and measure the ecological footprint of human activities by comparing the current state of a <span class='highlight-text-brown'>degraded forest area</span> with a <span class='highlight-text-green'> intact reference zone. </span>
        </p>
        """, unsafe_allow_html=True)
        st.markdown("""
        
        """)
        st.markdown("""
        
        """)
        st.markdown("""
        <p class='paragraph'>Firstly we will dive into the diversity of tree species and its characteristics. What can we tell about a <span class='highlight-text-brown'>degraded area?  </span>
        </p>
        """, unsafe_allow_html=True)
        
    with col4:
        
        st.image('atlantic_forest_view.jpg', 
                 use_container_width=True, caption="Our example uses data from a site in Brazil, which is measuring an area of the Atlantic forest")
st.markdown('</div>', unsafe_allow_html=True)        
    # Close full-width container
    
# Create a container for your two-column layout
tree_container = st.container(border=None)

with tree_container:
    col1, col2, col3 = st.columns([2, 2.5, 2])
    with col2:
        st.markdown("""
        <p class='paragraph-header'>Impact of Degradation on Forest Vertical Structure: <span><i>A Look at Tree Heights</i></span></p>
        <p class='paragraph'>This graph, known as a violin plot, offers a deep look into the height structure of trees</p>
        
        <p class='paragraph'>What you're seeing:</p>
        <ul class='paragraph'>
            <li>The width of each violin at any given height indicates the density or concentration of trees at that specific height</li>
            <li>Notice the distinct shapes. The violin for the intact area typically extends higher and often shows a broader distribution at taller heights, signifying a healthier canopy with a greater presence of mature, tall trees.</li>
            
        </ul>
        
        
        """, unsafe_allow_html=True)
    

    
    
    # First graph - takes full width of col2
        
        
        # Create the first plot
        fig1 = create_tree_height_violin(df_trees)
        
        # Display the first plot
        st.plotly_chart(fig1, use_container_width=True)
        
        # Small spacing between graphs
        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
        
        st.markdown("""
        
        <p class='paragraph'>A histogram provides another perspective on the tree height distribution, complementing the violin plot by showing the frequency of trees within specific height classes for both the <span class='highlight-text-green'><span><i>intact</i></span></span> and <span class='highlight-text-brown'><span><i>degraded</i></span></span> areas.</p>
        <p class='paragraph'>For the <span class='highlight-text-green'><span><i>intact</i></span></span> area, we observe a more robust representation in the taller height classes, indicating that more species are present. In contrast, the distribution for the <span class='highlight-text-brown'><span><i>degraded</i></span></span> area is skewed towards shorter height classes or younger trees that were planted in the reforstation effort</p>                
        
        """, unsafe_allow_html=True)
        
        
        # Create the second plot
        fig2 = create_tree_height_histogram(df_trees)
        
        # Display the second plot
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("""
        <p class='paragraph-header'>Overal distribution of trees in Volume by species</i></span></p>
                    """, unsafe_allow_html=True)
        # Create the violin plot
        result = create_tree_volume_comparison(df_trees)

        # Display the plot
        st.plotly_chart(result['figure'], use_container_width=True)

st.markdown("<div style='margin-top: 200px;'></div>", unsafe_allow_html=True)
st.markdown("""
        <p class='infographic-header'>A look on bird species activity</p>
        
               
        """, unsafe_allow_html=True)
st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)

bird_container = st.container(border=None)

with bird_container:
    col1, col2, col3 = st.columns([2, 2.5, 2])
    with col2:
        # Create the sankey diagram
        fig = create_plotly_waffle_chart(df_birds)

        # Display the plot
        st.plotly_chart(fig, use_container_width=True)


       
bird_container2 = st.container(border=None)
with bird_container2:

    

    col1, col2, col3, col4, col5 = st.columns([2, 3, .5, 4, 1])
    with col2:
        st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)
        st.markdown("""
        
        <p class='paragraph'>Using AI to identify bird species, we can understand better bird activity through its sightings in different areas</p>
               
        """, unsafe_allow_html=True)
        st.markdown("<div style='margin-top: 600px;'></div>", unsafe_allow_html=True)
        st.markdown("""
        
        <p class='paragraph'>We can construct a flow where birds are organized by their UICN status: The IUCN Red List is a critical indicator of the health of the world’s biodiversity. We can understand behavioral patterns, and how some species avoid degraded areas</p>
               
        """, unsafe_allow_html=True)

  


    with col4:   
        
        st.image('bird_listening.png', 
                 use_container_width=True, caption="Our example uses data from a site in Brazil, which is measuring an area of the Atlantic forest")
        st.markdown("<div style='margin-top: 100px;'></div>", unsafe_allow_html=True)
        
        # Create the sankey diagram
        fig = create_bird_conservation_plot(df_birds)

        # Display the plot
        st.plotly_chart(fig, use_container_width=True)

        # To save the plot to an HTML file that can be opened in a browser
        # fig.write_html("tree_height_comparison.html")


bird_container3 = st.container(border=None)

with bird_container3:
    col1, col2, col3 = st.columns([2, 4, 2])
    with col2:

        fig = create_sankey_diagram_reversed(df_birds)
        st.plotly_chart(fig, use_container_width=True)






 

  

