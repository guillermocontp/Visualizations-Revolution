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
             
             .highlight-text-vul {
            color: #F9E814 !important; 
            font-weight: bold !important;
        }
             .highlight-text-nt {
            color: #CCE226 !important; 
            font-weight: bold !important;
        }
              
        
    </style>
""", unsafe_allow_html=True)

# Then apply these styles in your content sections without redefining them

# Create container for intro section
intro_container = st.container(border=None)

with intro_container:
    col1, col2, col3 = st.columns([1.4, 3.5, 1.4])
    with col2:
        st.markdown("<p class='infographic-header'>Forest Health Report: Understanding Biodiversity's Role in <span class='highlight-text-green'>Restoration and Research</span></p>", unsafe_allow_html=True)

        st.image('forest_aerial.jpg', 
                    use_container_width=True)

st.markdown("<div style='margin-top: 100px;'></div>", unsafe_allow_html=True)

paragraph_container = st.container(border=None)  
with paragraph_container:
    col1, col2, col3 = st.columns([1.4, 3.5, 1.4])
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
        st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)   
        st.markdown("""
        <p class='paragraph'>By analyzing an area of forest that has been disturbed <span class='highlight-text-brown'>degraded </span> alongside a nearby healthy <span class='highlight-text-green'> intact </span>zone, we can measure the ecological effects of human activity. 
        </p>
        """, unsafe_allow_html=True)
st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True) 
         


# Load the data
# Assuming df_combined is a DataFrame containing the necessary data
df_trees = load_tree_data(tree_file_path='Tree.xlsx', sheet_name='TreeI', header_row=0)
df_birds = load_tree_data(tree_file_path='Tree.xlsx', sheet_name='Birds_entact', header_row=0)

# Creating different containers for the different content sections

jungle_container = st.container(border=None)
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
    col1, col2, col3 = st.columns([1.5, 3.5, 1.5])
    with col2:
                  

           st.markdown("""
        <p class='paragraph-header'>First, Let's Look at Tree Composition: What Species Are Present and How Abundant Are They?</i></span></p>
                    """, unsafe_allow_html=True)
        # Create the violin plot
           result = create_tree_volume_comparison(df_trees)

        # Display the plot
           st.plotly_chart(result['figure'], use_container_width=True)
           st.markdown("""
        <p class='paragraph'>These charts are called treemaps. Each colored rectangle represents a different tree species. The size of the rectangle corresponds to the 'volume' (or total biomass/abundance) of that species within that specific forest area.</i></span></p>
        <p class='paragraph'>The total volume in the degraded area has been reduced to <b>half</b> of what we see in an intact area. There are only <span class='highlight-text-brown'>4 species present in the degraded area.</p></i></span></p>            
                       """, unsafe_allow_html=True)

           st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)

           st.markdown("""
            <p class='paragraph-header'>Next: Assessing Forest Height – Are We Seeing Mature Canopies or Younger Growth?</p>
            """, unsafe_allow_html=True)
           
             # Create the first plot
           fig1 = create_tree_height_violin(df_trees)

            # Display the first plot
           st.plotly_chart(fig1, use_container_width=True)
           

           st.markdown("""
            <p class='paragraph'>This graph, known as a violin plot, offers a deep look into the height structure of trees</p>

            <p class='paragraph'>What you're seeing:</p>
            <ul class='paragraph'>
                <li>The width of each violin at any given height indicates the density or concentration of trees at that specific height</li>
                <li>Notice the distinct shapes. The violin for the intact area extends higher and shows a broader distribution at taller heights, hinting a greater presence of mature, tall trees.</li>
                
            </ul>


            """, unsafe_allow_html=True)

          
            # Create the second plot
           fig2 = create_tree_height_histogram(df_trees)

            # Display the second plot
           st.plotly_chart(fig2, use_container_width=True)

            # Small spacing between graphs
           st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

           st.markdown("""

            <p class='paragraph'>This histogram offers another perspective on tree height distribution by showing the number of trees within defined height classes for both the <span class='highlight-text-green'>intact</span> and <span class='highlight-text-brown'>degraded</span> areas.</p>
            <p class='paragraph'>The distribution for the <span class='highlight-text-brown'>degraded</span> area shows a higher concentration of trees in shorter height classes. This might indicate a prevalence of younger trees (perhaps from recent reforestation efforts) or reflect the general absence of larger, older trees typically found in an undisturbed forest.</p>                

            """, unsafe_allow_html=True)

           st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)
           st.markdown("""
                    <p class='paragraph-header'> Examining the Link Between Forest Structure and Birdlife:</p>
                    <p class='paragraph-header'> Monitoring Bird Populations and Activity</p>
                    """, unsafe_allow_html=True)
           st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)
           st.image('birds/bird1.png',
                            use_container_width=True)
           st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)


bird_container = st.container(border=None)

with bird_container:
    col1, col2, col3 = st.columns([1.4, 3.5, 1.4])
    with col2:

        st.markdown("""
        <p class='paragraph'>Having assessed the forest structure, let's explore the bird communities these habitats support. We utilize advanced methods, including AI-assisted identification of bird species from sensor data, to understand their activity.</p>
        
        """, unsafe_allow_html=True)
        # Create the sankey diagram
        fig_waffle = create_plotly_waffle_chart(df_birds)

        # Display the plot
        st.plotly_chart(fig_waffle, use_container_width=True)

        st.markdown("""
        <p class='paragraph'>This waffle chart provides a visual summary of total bird sightings. Each segment or square can represent a specific number of detected bird presences.</p>
        <ul class='paragraph'>
            <li>By comparing the number of highlighted segments for the<span class='highlight-text-green'> intact</span> forest versus the <span class='highlight-text-brown'> degraded</span> one, we can get an initial indication of overall bird activity levels. Significant differences often suggest that healthier forests support more active bird populations.</li>  
        </ul>
        """, unsafe_allow_html=True)
        st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True
                    )
        st.image('birds/bird2.png', 
                 use_container_width=True)
        st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True
                    )

        st.markdown("""
        <p class='paragraph'>So, we see there can be a difference in overall bird presence. But can we dig deeper and see if different types of birds, especially those of conservation concern, prefer one area over the other? This is where we can look at their distribution in a really visual way.</p>
        """, unsafe_allow_html=True)
        st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True
                    )
        st.markdown("""
        <p class='paragraph-header'>Where Do Different Bird Groups Spend Their Time?</p>
        """, unsafe_allow_html=True)

        fig_sankey = create_sankey_diagram_reversed(df_birds)
        st.plotly_chart(fig_sankey, use_container_width=True)

        st.markdown("""
            <ul class='paragraph'>
                <li>We've grouped all our bird sightings based on each species' conservation status. This comes from the IUCN Red List, which is like a global health check for different species, telling us if they are of 'Least Concern', 'Near Threatened', 'Vulnerable', and so on. Each colored band starting on the left represents one of these conservation groups.</li>
                <li>The bands then connect to show us where those sightings occurred: either in the <span class= highlight-text-green>'Intact Forest'</span> or the <span class= highlight-text-brown>'Degraded Forest'.</span></li>
            </ul>
            """, unsafe_allow_html=True)

        st.markdown("""
        <p class='paragraph'>As you look at the graph, try to follow the bands. For instance, do you see thicker bands from the <span class= highlight-text-vul>'Vulnerable'</span> or <span class= highlight-text-nt>'Near Threatened'</span> bird groups flowing mostly towards the <span class= highlight-text-green>'Intact Forest'?</span> This could strongly suggest that these more sensitive species are relying more heavily on the healthier, undisturbed habitat and might be avoiding the <span class= highlight-text-brown>degraded</span> areas.
                    </p>
        """, unsafe_allow_html=True)
        st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True
                    )
        st.markdown("""
        <p class='paragraph-header'>For a deeper look:
                    </p>
        """, unsafe_allow_html=True)
        st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True
                    )
        fig_stacked = create_bird_conservation_plot(df_birds)

        # Display the plot
        st.plotly_chart(fig_stacked, use_container_width=True)

        st.markdown("<div style='margin-top: 70px;'></div>", unsafe_allow_html=True
                    )
       

        st.markdown("""
        <p class='paragraph'>Beyond total activity, <i>which bird species are most frequently observed?</i></p>
        """, unsafe_allow_html=True)
        st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True
                    )
        



bird_container2 = st.container(border=None)
with bird_container2:

    col1, col2, col3, col4, col5 = st.columns([0.5, 4, 0.2, 1.9, 0.5])
    with col2:

        st.image('bird_listening.png', 
                 use_container_width=True)
        st.markdown("<div style='margin-top: 100px;'></div>", unsafe_allow_html=True)

    with col4:
        st.image('birds/bird3.png',
                 use_container_width=True, caption='and the most observed bird is the Atlantic royal flycatcher')   
 
st.markdown("<div style='margin-top: 100px;'></div>", unsafe_allow_html=True)


bird_container6 = st.container(border=None)

with bird_container6:
    col1, col2, col3 = st.columns([1.4, 3.5, 1.4])
    with col2:
        st.markdown("""
        <p class='paragraph-header'><i>The Daily Pulse of the Forest: Overall Bird Activity by Hour</i></p>
        """, unsafe_allow_html=True)
        st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)
        st.markdown("""
        <p class='paragraph'>You'll typically see distinct peaks in activity. Many bird communities exhibit a <span style="color:#FF3C8E;"><b>'dawn chorus'</b></span> – a surge in songs and movement as birds greet the new day and begin foraging. Often, there's another, sometimes smaller, peak of activity in the <span style= "color:#1F5DD0;"><b>late afternoon or evening</b></span> as they feed again before settling down for the night.</p>
        """, unsafe_allow_html=True)

st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)

bird_container4 = st.container(border=None)
with bird_container4:

    col1, col2, col3, col4, col5 = st.columns([.5, 4, 0.5, 4.5, .5])
    with col2:

        st.image('dawn.jpg', 
                 use_container_width=True)


    with col4:   
        
        fig_polar_morning = create_bird_activity_polar(df_birds, time_filter='morning')
        st.plotly_chart(fig_polar_morning, use_container_width=True)

        
        
bird_container5 = st.container(border=None)
with bird_container5:

    col1, col2, col3, col4, col5 = st.columns([.5, 4, 0.5, 4.5, .5])
    with col2:

        st.image('forest_sunset.jpg', 
                 use_container_width=True)

    with col4:   
        
        fig_polar_evening = create_bird_activity_polar(df_birds, time_filter='evening')
        st.plotly_chart(fig_polar_evening, use_container_width=True)


st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)
bird_container7 = st.container(border=None)

with bird_container7:
    col1, col2, col3 = st.columns([1.4, 3.5, 1.4])
    with col2:
        
        
        st.markdown("""
        <p class='paragraph'>By integrating these diverse metrics—from tree species and height distributions to bird population characteristics and activity patterns—we obtain a comprehensive assessment of forest health. This detailed ecological information is invaluable for companies aiming to effectively monitor their reforestation and land restoration projects, and for all stakeholders interested in understanding and preserving biodiversity.</p>
        """, unsafe_allow_html=True)



 

  

