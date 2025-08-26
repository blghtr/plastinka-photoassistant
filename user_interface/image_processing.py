import streamlit as st
from PIL import Image
from photoassist import PipelineConfig, Pipeline
import numpy as np
import io
import zipfile
import cv2
import gc
from datetime import datetime
from .my_logging import get_logger
from copy import copy
from collections import OrderedDict


logger = get_logger(__name__)


def get_pipeline():
    """Get (or create) the pipeline instance from session state."""
    if 'pipeline' not in st.session_state:
        logger.info(f'Creating pipeline with config: {st.session_state.config_path}')
        config = PipelineConfig(st.session_state.config_path)
        st.session_state.pipeline = Pipeline(config)
    return st.session_state.pipeline


def get_debug_pipeline():
    """Get (or create) the debug pipeline instance with debug_config.yaml."""
    if 'debug_pipeline' not in st.session_state:
        debug_config_path = 'configs/debug_config.yaml'
        logger.info(f'Creating debug pipeline with config: {debug_config_path}')
        config = PipelineConfig(debug_config_path)
        st.session_state.debug_pipeline = Pipeline(config, logger=get_logger("DebugPipeline"))
    return st.session_state.debug_pipeline


def process_all_debug_images():
    """
    Process all uploaded images at once and store complete results.
    No memory limitations - process everything in one batch.
    """
    try:
        pipeline = get_debug_pipeline()  # с debug_config.yaml
        
        # Подготавливаем данные для debug обработки
        debug_input = []
        for item in st.session_state.debug_uploaded_images:
            debug_item = item.copy()
            # Инициализируем intermediate_outputs для каждого изображения
            debug_item['intermediate_outputs'] = OrderedDict()
            debug_input.append(debug_item)
        
        # Обрабатываем все изображения одним вызовом (БЕЗ реверса!)
        all_results = pipeline(debug_input)
        
        # Verify order consistency after processing
        if len(all_results) == len(st.session_state.debug_uploaded_images):
            for i, (result, original) in enumerate(zip(all_results, st.session_state.debug_uploaded_images)):
                if 'exc_tb' not in result and result['name'] != original['name']:
                    logger.warning(f"Order mismatch after processing at index {i}: original={original['name']}, result={result['name']}")
        
        # Сохраняем полные результаты в session_state
        st.session_state.debug_results = all_results
        st.session_state.debug_processing_complete = True
        
        successful = len([r for r in all_results if 'exc_tb' not in r])
        failed = len([r for r in all_results if 'exc_tb' in r])
        
        if successful > 0:
            st.success(f"✅ Successfully processed {successful} images")
            # Отладочная информация о промежуточных результатах
            if all_results:
                first_result = next((r for r in all_results if 'exc_tb' not in r), None)
                if first_result and 'intermediate_outputs' in first_result:
                    stages = list(first_result['intermediate_outputs'].keys())
                    st.info(f"🔧 Found intermediate stages: {', '.join(stages)}")
                else:
                    st.warning("⚠️ No intermediate outputs found in results")
        if failed > 0:
            st.error(f"❌ Failed to process {failed} images")
            
    except Exception as e:
        st.error(f"Pipeline error: {str(e)}")
        logger.error(f"Debug processing error: {e}", exc_info=True)


def render_debug_navigation():
    """Enhanced navigation with all images in memory."""
    if not st.session_state.debug_results:
        return
        
    total_images = len(st.session_state.debug_results)
    current_idx = st.session_state.debug_current_index
    
    col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
    
    with col1:
        if st.button("⏮️ First", disabled=(current_idx == 0)):
            st.session_state.debug_current_index = 0
            st.rerun()
    
    with col2:
        if st.button("⬅️ Previous", disabled=(current_idx == 0)):
            st.session_state.debug_current_index -= 1
            st.rerun()
    
    with col3:
        # Dropdown для быстрого перехода к любому изображению
        image_options = [f"{i+1}. {result['name']}" for i, result in enumerate(st.session_state.debug_results)]
        selected = st.selectbox(
            "Jump to image:",
            options=range(len(image_options)),
            format_func=lambda x: image_options[x],
            index=current_idx,
            key="debug_nav_select"
        )
        if selected != current_idx:
            st.session_state.debug_current_index = selected
            st.rerun()
    
    with col4:
        if st.button("Next ➡️", disabled=(current_idx >= total_images - 1)):
            st.session_state.debug_current_index += 1
            st.rerun()
    
    with col5:
        if st.button("Last ⏭️", disabled=(current_idx >= total_images - 1)):
            st.session_state.debug_current_index = total_images - 1
            st.rerun()


def render_thumbnail_gallery():
    """Show thumbnails of all images for quick navigation."""
    if not st.session_state.debug_results:
        return
        
    cols = st.columns(min(10, len(st.session_state.debug_results)))
    
    for i, result in enumerate(st.session_state.debug_results):
        with cols[i % len(cols)]:
            if st.button(f"📷 {i+1}", key=f"thumb_{i}"):
                st.session_state.debug_current_index = i
                st.rerun()


def render_stage_filter():
    """Allow users to show/hide specific pipeline stages."""
    if not st.session_state.debug_results:
        return []

    # Aggregate stage names across all successful results
    all_stages_set = set()
    for r in st.session_state.debug_results:
        inter = r.get('intermediate_outputs')
        if isinstance(inter, dict) and len(inter):
            all_stages_set.update(inter.keys())

    all_stages = sorted(all_stages_set)
    if not all_stages:
        st.sidebar.info("No intermediate stages available to display.")
        return []

    st.sidebar.subheader("Show Stages:")
    selected_stages = st.sidebar.multiselect(
        "Select stages to display:",
        options=all_stages,
        default=all_stages,
        key="debug_stage_filter"
    )
    if not selected_stages:
        st.sidebar.warning("No stages selected. Enable one or more to see images.")

    return selected_stages


def render_processing_stats():
    """Show processing statistics and timing."""
    if not st.session_state.debug_results:
        return
        
    st.sidebar.subheader("Processing Stats:")
    successful = [r for r in st.session_state.debug_results if 'exc_tb' not in r]
    failed = [r for r in st.session_state.debug_results if 'exc_tb' in r]
    
    st.sidebar.write(f"Total images: {len(st.session_state.debug_results)}")
    st.sidebar.write(f"✅ Successful: {len(successful)}")
    st.sidebar.write(f"❌ Failed: {len(failed)}")
    
    if failed:
        with st.sidebar.expander("❌ Failed Images", expanded=False):
            for fail in failed:
                st.write(f"• {fail.get('name', 'Unknown')}")


def show_debug_image_with_stages(index, selected_stages):
    """
    Instantly display any image from pre-processed results.
    All data is already in memory - no loading delays.
    """
    if not st.session_state.debug_results or index >= len(st.session_state.debug_results):
        st.error("No results available or invalid index")
        return
        
    result = st.session_state.debug_results[index]
    
    # Check if this result has an error
    if 'exc_tb' in result:
        st.error(f"❌ Failed to process image: {result.get('name', 'Unknown')}")
        st.code('\n'.join(result['exc_tb']), language='text')
        return
    
    # Show original image
    if 'image' in result:
        # Get original from uploaded images
        original_img = None
        if index < len(st.session_state.debug_uploaded_images):
            original_img = st.session_state.debug_uploaded_images[index]['image']
            # Verify order consistency
            original_name = st.session_state.debug_uploaded_images[index]['name']
            result_name = result['name']
            if original_name != result_name:
                st.warning(f"⚠️ Order mismatch detected! Original: {original_name}, Result: {result_name}")
                logger.warning(f"Order mismatch at index {index}: original={original_name}, result={result_name}")
        
        if original_img:
            st.subheader(f"📷 Original: {result['name']}")
            st.image(original_img, caption="Original Image", use_column_width=True)
        
        # Show intermediate stages vertically
        if 'intermediate_outputs' in result and result['intermediate_outputs']:
            st.subheader("🔧 Processing Stages")
            st.caption(f"Available stages: {list(result['intermediate_outputs'].keys())}")
            
            for module_name, intermediate_img in result['intermediate_outputs'].items():
                if module_name in selected_stages:
                    try:
                        # Convert BGR to RGB for display
                        converted_img = cv2.cvtColor(intermediate_img, cv2.COLOR_BGR2RGB)
                        st.image(converted_img, caption=f"After {module_name}", use_column_width=True)
                    except Exception as e:
                        st.error(f"Error displaying {module_name}: {str(e)}")
        else:
            st.warning("⚠️ No intermediate outputs available for this image")
        
        # Show final result
        st.subheader("✅ Final Result")
        try:
            final_img = cv2.cvtColor(result['image'], cv2.COLOR_BGR2RGB)
            st.image(final_img, caption="Final Processed Image", use_column_width=True)
        except Exception as e:
            st.error(f"Error displaying final result: {str(e)}")


def next_image():
    """Go to the next uploaded image (if any)."""
    if st.session_state.current_image_index < len(st.session_state.uploaded_images) - 1:
        st.session_state.current_image_index += 1


def create_and_download_debug_archive():
    """Create ZIP with original images, final results, and intermediates."""
    if not st.session_state.debug_results:
        st.error("No results to download")
        return
        
    try:
        # Prepare data for archive creation
        archive_data = []
        
        for i, result in enumerate(st.session_state.debug_results):
            if 'exc_tb' not in result:  # Only successful results
                # Add final result
                final_img = cv2.cvtColor(result['image'], cv2.COLOR_BGR2RGB)
                archive_data.append({
                    'image': Image.fromarray(final_img.astype(np.uint8)),
                    'name': result['name']
                })
        
        if archive_data:
            progress_bar = st.progress(0)
            archive = create_zip(archive_data, progress_bar)
            
            st.download_button(
                label="📥 Download Debug Results",
                data=archive,
                file_name=f"debug_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip"
            )
            st.success(f"Archive ready for download with {len(archive_data)} images")
        else:
            st.warning("No successful results to download")
            
    except Exception as e:
        st.error(f"Error creating archive: {str(e)}")
        logger.error(f"Debug archive error: {e}", exc_info=True)


def clear_debug_session():
    """Clear all debug session data."""
    keys_to_clear = [
        'debug_current_index',
        'debug_uploaded_images', 
        'debug_results',
        'debug_processing_complete',
        'debug_pipeline'  # Очищаем также debug pipeline
    ]
    
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    
    gc.collect()
    st.success("🗑️ Debug session cleared")
    st.rerun()


def prev_image():
    """Go to the previous uploaded image (if any)."""
    if st.session_state.current_image_index > 0:
        st.session_state.current_image_index -= 1


def debug_mode():
    """
    Enhanced debug mode - process all images at once, navigate instantly.
    
    Features:
    - Batch processing of all uploaded images
    - All results stored in memory for instant navigation
    - Vertical display of all intermediate stages
    - Advanced navigation with thumbnails and quick jump
    - Complete archive download
    - Processing statistics
    
    Memory approach: Store everything in session_state for fast access
    """
    st.title("🔧 Debug Mode - Pipeline Visualization")
    st.info("ℹ️ Using debug_config.yaml with intermediate outputs enabled")
    
    # Initialize session state
    if 'debug_current_index' not in st.session_state:
        st.session_state.debug_current_index = 0
    if 'debug_uploaded_images' not in st.session_state:
        st.session_state.debug_uploaded_images = []
    if 'debug_results' not in st.session_state:
        st.session_state.debug_results = []
    if 'debug_processing_complete' not in st.session_state:
        st.session_state.debug_processing_complete = False
    
    # Authentication
    st.session_state._authenticator.login(location='unrendered')

    # Upload section
    uploaded_files = st.file_uploader(
        "Choose images for debug processing...",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key='debug_uploader'
    )

    if uploaded_files:
        st.session_state.debug_uploaded_images = [
            {'image': Image.open(f), 'name': f.name} 
            for f in uploaded_files
        ]
        
        st.success(f"Loaded {len(uploaded_files)} images")
        
        if st.button('🚀 Process All Images (Debug Mode)', type="primary"):
            with st.spinner('Processing all images with debug pipeline...'):
                process_all_debug_images()
    
    # Results section
    if st.session_state.debug_processing_complete and st.session_state.debug_results:
        
        # Navigation controls
        render_debug_navigation()
        
        # Optional thumbnail gallery
        with st.expander("📷 Thumbnail Gallery", expanded=False):
            render_thumbnail_gallery()
        
        # Stage filter
        selected_stages = render_stage_filter()
        
        # Processing stats
        render_processing_stats()
        
        # Main image display
        current_index = st.session_state.debug_current_index
        show_debug_image_with_stages(current_index, selected_stages)
        
        # Download section
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Download All Results"):
                create_and_download_debug_archive()
        
        with col2:
            if st.button("🗑️ Clear Results"):
                clear_debug_session()


def create_zip(data, progress_bar):
    """
    Create a ZIP archive from a list of PIL images with progress updates.
    
    Args:
        data: List of dicts with 'image' (PIL.Image) and 'name' (str) keys
        progress_bar: Streamlit progress bar for updates
    
    Returns:
        bytes: ZIP archive data
    """
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=5) as zip_file:
        total_images = len(data)
        
        for i, item in enumerate(data):
            img, img_name = item['image'], item['name']
            
            # Create image buffer
            img_buffer = io.BytesIO()
            img.save(img_buffer, format="JPEG", quality=95, optimize=True)
            
            # Add to ZIP
            zip_file.writestr(img_name, img_buffer.getvalue())
            
            # Update progress
            progress = (i + 1) / total_images
            progress_bar.progress(progress, f"📦 Packaging {i + 1}/{total_images} images...")
    
    return zip_buffer.getvalue()


def _initialize_process_session_state():
    """Initialize session state variables for main processing."""
    defaults = {
        'results': [],
        'uploaded_images': [],
        'uploader_key': 0,
        'errors': [],
        'datetime': datetime.now().strftime('%Y%m%d_%H%M%S')
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def _reset_uploader():
    """Reset uploader state and clean up memory."""
    st.session_state.results = []
    st.session_state.uploaded_images = []
    st.session_state.uploader_key += 1
    gc.collect()


def _prepare_uploaded_files(uploaded_files):
    """Convert uploaded files to pipeline input format."""
    return [
        {'image': Image.open(f), 'name': f.name} 
        for f in uploaded_files
    ]


def _process_pipeline_results(all_results):
    """Separate successful results from errors and convert images."""
    successful_results = [r for r in all_results if 'exc_tb' not in r]
    error_results = [r for r in all_results if 'exc_tb' in r]
    
    # Convert successful results to PIL Images with RGB format
    processed_results = []
    for result in successful_results:
        try:
            rgb_image = cv2.cvtColor(result['image'], cv2.COLOR_BGR2RGB)
            processed_results.append({
                'image': Image.fromarray(rgb_image.astype(np.uint8)),
                'name': result['name']
            })
        except Exception as e:
            logger.error(f"Error converting image {result['name']}: {e}")
            # Move to error results if conversion fails
            error_results.append({
                'name': result['name'],
                'module': 'ImageConversion',
                'exc_tb': [str(e)]
            })
    
    return processed_results, error_results


def _handle_processing_errors(errors):
    """Log and display processing errors."""
    if not errors:
        return
        
    st.error(f"❌ {len(errors)} image(s) failed to process")
    
    with st.expander("View Error Details", expanded=False):
        for err in errors:
            name = err.get('name', 'Unknown')
            module = err.get('module', 'Unknown')
            exc_tb = err.get('exc_tb', ['No details available'])
            
            st.code(f"Image: {name}\nModule: {module}\nError: {exc_tb[0]}")
            
            # Log detailed error
            err_message = '\n'.join([
                f'Error processing image {name} with {module}:',
                *exc_tb
            ])
            logger.error(err_message)


def process_images():
    """
    Main user flow: upload images, run pipeline, download results as a ZIP.
    
    Features:
    - Clean, modern UI with progress tracking
    - Comprehensive error handling and reporting
    - Automatic memory cleanup
    - Timestamped archive downloads
    """
    st.title("📸 Image Processing")
    st.info("ℹ️ Upload images to process them through the production pipeline")
    
    # Initialize session state
    _initialize_process_session_state()

    st.session_state._authenticator.login(location='unrendered')

    # Upload section
    with st.form('uploader_form', clear_on_submit=True):
        st.subheader("📁 Upload Images")
        uploaded_files = st.file_uploader(
            "Choose images to process...",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            key=f'uploader_{st.session_state.uploader_key}',
            help="Supported formats: JPG, JPEG, PNG, WebP"
        )
        
        col1, col2 = st.columns([3, 1])
        with col2:
            submitted = st.form_submit_button(
                '🚀 Start Processing', 
                type="primary",
                use_container_width=True
            )

    # Processing section
    if submitted and uploaded_files:
        st.divider()
        st.subheader("⚙️ Processing Images")
        
        # Show upload info
        st.success(f"📤 Loaded {len(uploaded_files)} image(s) for processing")
        
        # Initialize progress bar
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        
        try:
            # Prepare input data
            status_text.text("📋 Preparing images...")
            st.session_state.uploaded_images = _prepare_uploaded_files(uploaded_files)
            
            # Run pipeline
            status_text.text("🔄 Running pipeline...")
            pipeline = get_pipeline()
            pipeline.set_callback('progress_tracker', progress_bar.progress)
            all_results = pipeline(st.session_state.uploaded_images)
            
            # Process results
            status_text.text("📊 Processing results...")
            processed_results, error_results = _process_pipeline_results(all_results)
            
            st.session_state.results = processed_results
            st.session_state.errors = error_results
            
            # Show results summary
            progress_bar.progress(1.0)
            status_text.empty()
            
            if processed_results:
                st.success(f"✅ Successfully processed {len(processed_results)} image(s)")
                
                # Create and offer download
                progress_bar.progress(0.0, "📦 Creating archive...")
                archive = create_zip(st.session_state.results, progress_bar)
                
                # Generate filename with timestamp
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"processed_images_{timestamp}.zip"
                
                st.download_button(
                    label="📥 Download Processed Images",
                    data=archive,
                    file_name=filename,
                    mime="application/zip",
                    on_click=_reset_uploader,
                    type="primary",
                    use_container_width=True
                )
                
                # Clean up archive from memory
                del archive
            else:
                st.error("❌ No images were successfully processed")
            
            # Handle errors
            _handle_processing_errors(st.session_state.errors)
            st.session_state.errors = []  # Clear errors after displaying
            
        except Exception as e:
            st.error(f"💥 Pipeline error: {str(e)}")
            logger.error(f"Processing pipeline error: {e}", exc_info=True)
            
    elif submitted and not uploaded_files:
        st.warning('⚠️ Please upload images before starting processing')
