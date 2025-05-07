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
                             
                             create_sankey_diagram_reversed,

)

# Set page configuration for a wider layout
st.set_page_config(
    page_title="Tracking biodiversity",
    page_icon="🌳",
    layout="wide",  
    initial_sidebar_state="collapsed",)



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
            text-align: center !important;
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
        st.markdown("<p class='infographic-header'>Forest Health Report: Understanding Biodiversity's Role in<span class='highlight-text-green'>Restoration and Research</span></p>", unsafe_allow_html=True)

        st.image('forest_aerial.jpg', 
                    use_container_width=True)

st.markdown("<div style='margin-top: 100px;'></div>", unsafe_allow_html=True)

paragraph_container = st.container(border=None)  
with paragraph_container:
    col1, col2, col3 = st.columns([2, 2.5, 2])
    with col2:
                    
        st.markdown("""
        <p class='paragraph-header'>Advanced Monitoring: A Deeper Insight into Nature's Complexity</p>
        <p class='paragraph'>How can we accurately assess a forest's condition, or measure the success of efforts to restore a damaged area? It requires more than observation alone. We employ scientific tools and methodologies (like the Green Cubes Nature Methodology developed by R-evolution) to obtain a comprehensive understanding.</p>
        
        <p class='paragraph'>This data provides a detailed "health report" for the forest, enabling us to:
        <ul class='paragraph'>
        <li>Effectively monitor if reforestation efforts are leading to genuine ecosystem recovery</li>
        <li>Investigate correlations between tree species composition and the presence of animal activity.</li>
        <li>Clearly compare areas impacted by human activity ("degraded") with untouched, healthy ("intact") forest sections.</li>
        </ul>
        </p>
        
        
        """, unsafe_allow_html=True)
        st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)   
        st.markdown("""
        <p class='paragraph'>By analyzing an area of forest that has been disturbed <span class='highlight-text-brown'>degraded </span> alongside a nearby healthy <span class='highlight-text-green'> intact </span>zone, we can measure the ecological effects of human activity. 
        </p>
        """, unsafe_allow_html=True)

         


# Load the data
# Assuming df_combined is a DataFrame containing the necessary data
df_trees = load_tree_data(tree_file_path='Tree.xlsx', sheet_name='TreeI', header_row=0)
df_birds = load_tree_data(tree_file_path='Tree.xlsx', sheet_name='Birds_entact', header_row=0)

# Creating different containers for the different content sections

jungle_container = st.container(border=None)




st.markdown('<div class="fullwidth-container">', unsafe_allow_html=True)
with jungle_container:
     # Full-width container starts   
    
    col1, col2, col3, col4, col5 = st.columns([2, 3, .1, 3, 2])
    with col2:
        
        st.image('forest_degraded.jpg', 
                 use_container_width=True,caption="Our example uses data from a site in Brazil, which is measuring an area of the Atlantic forest") 
        
    with col4:
        
        st.image('atlantic_forest_view.jpg', 
                 use_container_width=True, caption="Through different metrics, we can better understand and track the complexity of the forest in order to improve restoration efforts")
st.markdown('</div>', unsafe_allow_html=True)        
st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
    
# Create a container for your two-column layout
tree_container = st.container(border=None)

with tree_container:
    col1, col2, col3 = st.columns([2, 2.5, 2])
    with col2:
                  

           st.markdown("""
        <p class='infographic-header'>First, Let's Look at Tree Composition: What Species Are Present and How Abundant Are They?</i></span></p>
                    """, unsafe_allow_html=True)
        # Create the violin plot
           result = create_tree_volume_comparison(df_trees)

        # Display the plot
           st.plotly_chart(result['figure'], use_container_width=True)


           st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)

           st.markdown("""
            <p class='paragraph-header'>Next: Assessing Forest Height – Are We Seeing Mature Canopies or Younger Growth?
            

            <p class='paragraph'>This graph, known as a violin plot, offers a deep look into the height structure of trees</p>

            <p class='paragraph'>What you're seeing:</p>
            <ul class='paragraph'>
                <li>The width of each violin at any given height indicates the density or concentration of trees at that specific height</li>
                <li>Notice the distinct shapes. The violin for the intact area typically extends higher and often shows a broader distribution at taller heights, signifying a healthier canopy with a greater presence of mature, tall trees.</li>
                
            </ul>


            """, unsafe_allow_html=True)


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
        fig_waffle = create_plotly_waffle_chart(df_birds)

        # Display the plot
        st.plotly_chart(fig_waffle, use_container_width=True)


       
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
                 use_container_width=True)
        st.markdown("<div style='margin-top: 100px;'></div>", unsafe_allow_html=True)
        
        # Create the sankey diagram
        fig_stacked = create_bird_conservation_plot(df_birds)

        # Display the plot
        st.plotly_chart(fig_stacked, use_container_width=True)

        # To save the plot to an HTML file that can be opened in a browser
        # fig.write_html("tree_height_comparison.html")


bird_container3 = st.container(border=None)

with bird_container3:
    col1, col2, col3 = st.columns([2, 4, 2])
    with col2:
        

        fig_sankey = create_sankey_diagram_reversed(df_birds)
        st.plotly_chart(fig_sankey, use_container_width=True)



        fig_heatmap = create_bird_activity_heatmap(df_birds)
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
st.markdown("<div style='margin-top: 200px;'></div>", unsafe_allow_html=True)


bird_container4 = st.container(border=None)
with bird_container4:

    col1, col2, col3, col4, col5 = st.columns([1, 4, 0.5, 4, 1])
    with col2:

        st.image('dawn.jpg', 
                 use_container_width=True)
        st.markdown("<div style='margin-top: 100px;'></div>", unsafe_allow_html=True)

        


    with col4:   
        
        st.markdown("<div style='margin-top: 40px;'></div>", unsafe_allow_html=True)
        fig_polar_morning = create_bird_activity_polar(df_birds, time_filter='morning')
        st.plotly_chart(fig_polar_morning, use_container_width=True)

        
        
bird_container5 = st.container(border=None)
with bird_container5:

    col1, col2, col3, col4, col5 = st.columns([1, 4, 0.5, 4, 1])
    with col2:

       

        st.image('forest_sunset.jpg', 
                 use_container_width=True)
        st.markdown("<div style='margin-top: 100px;'></div>", unsafe_allow_html=True)


    with col4:   
        
        

        st.markdown("<div style='margin-top: 40px;'></div>", unsafe_allow_html=True)
        
        fig_polar_evening = create_bird_activity_polar(df_birds, time_filter='evening')
        st.plotly_chart(fig_polar_evening, use_container_width=True)








 

  

