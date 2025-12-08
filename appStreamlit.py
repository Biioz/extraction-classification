import streamlit as st

st.title('Document Extraction and Classification')

# Reload button
col1, col2 = st.columns([6, 1])
with col2:
    if st.button('🔄 Reload', help="Clear all data and restart"):
        st.session_state.clear()
        st.rerun()

# Initialize session state
if 'current_file' not in st.session_state:
    st.session_state.current_file = None
if 'file_source' not in st.session_state:
    st.session_state.file_source = None
if 'camera_mode' not in st.session_state:
    st.session_state.camera_mode = False
if 'captured_picture' not in st.session_state:
    st.session_state.captured_picture = None

# File upload section
st.subheader("Upload a picture")
uploaded_file = st.file_uploader("Choose a file", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    st.success("File uploaded successfully!")
    st.session_state.current_file = uploaded_file
    st.session_state.file_source = "upload"
    st.write(f"**Filename:** {uploaded_file.name}")
    st.write(f"**File size:** {uploaded_file.size} bytes")

# Camera section
st.subheader("Take a Picture")
show_camera = st.button('📷 Take a Picture')

# Show camera input if button clicked or if we're already in camera mode
if show_camera or st.session_state.camera_mode:
    st.session_state.camera_mode = True
    picture = st.camera_input("Take a picture",width=300)
    
    if picture is not None and picture != st.session_state.captured_picture:
        st.success("Picture taken successfully!")
        st.session_state.captured_picture = picture
        st.session_state.current_file = picture
        st.session_state.file_source = "camera"
        st.session_state.camera_mode = False  # Hide camera after taking picture
        st.rerun()  # Refresh to show the captured image

# Processing section
if st.session_state.current_file is not None:
    st.subheader("🔍 Process Document")
    
    # Display current file info
    if st.session_state.file_source == "upload":
        st.info(f"Current file: {st.session_state.current_file.name} (uploaded)")
    elif st.session_state.file_source == "camera":
        st.info("Current file: Camera image")
        st.image(st.session_state.current_file, caption="Captured image", width=300)
    
    # Extract and classify button
    if st.button('🚀 Extract and Classify', type="primary"):
        with st.spinner("Processing document..."):
            st.write("Extraction and Classification started...")
            

            # TODO : implementer la logique d'extraction et peut être afficher les différente étape

            st.success("Process completed!")
            
            st.subheader("📊 Results")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Document Type", "Invoice")  # Example result
            with col2:
                st.metric("Confidence", "95%")  # Example confidence score
