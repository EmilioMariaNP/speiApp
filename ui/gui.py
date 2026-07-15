import streamlit as st
import json
import os
import subprocess
import glob

# Page layout and title
st.set_page_config(
    page_title="SPEI Pipeline Configuration Editor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a more premium look
st.markdown("""
<style>
    .reportview-container {
        background-color: #0f1116;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #161b22;
        border-radius: 4px 4px 0px 0px;
        color: #c9d1d9;
        font-weight: 600;
        padding: 0 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #21262d;
        color: #58a6ff;
        border-bottom: 2px solid #58a6ff;
    }
</style>
""", unsafe_allow_html=True)

st.title("SPEI Pipeline Control Panel & Configuration Editor")

# Locate directories
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(project_root, "data")
schema_path = os.path.join(data_dir, "configs", "ui_config.json")

# Load UI schema
if not os.path.exists(schema_path):
    st.error(f"UI Configuration schema not found at {schema_path}")
    st.stop()

with open(schema_path, "r") as f:
    schema = json.load(f)

# Sidebar: Display global info
with st.sidebar:
    st.header("Pipeline Control")
    st.info("Select or create your active configuration in the 'Paths' tab. You can save your changes and trigger the pipeline execution directly from the footer action buttons.")

# Initialize active config path in session state
if "active_config_path" not in st.session_state:
    config_files = glob.glob(os.path.join(data_dir, "*.json"))
    config_files = [f for f in config_files if os.path.basename(f) != "ui_config.json"]
    config_files.sort()
    st.session_state.active_config_path = config_files[0] if config_files else ""

selected_config_path = st.session_state.active_config_path

# Initialize session state for current config dict
if "last_selected_config" not in st.session_state or st.session_state.last_selected_config != selected_config_path:
    st.session_state.last_selected_config = selected_config_path
    if os.path.exists(selected_config_path):
        with open(selected_config_path, "r") as f:
            st.session_state.current_config = json.load(f)
    else:
        st.session_state.current_config = {}
    
    # Clear all text and widget keys from session state to load new file values
    for k in list(st.session_state.keys()):
        if k.startswith("text_") or k.startswith("w_") or k.startswith("expanded_") or k.startswith("browse_path_"):
            del st.session_state[k]

# Ensure logs state exists
if "logs" not in st.session_state:
    st.session_state.logs = ""
if "running" not in st.session_state:
    st.session_state.running = False

# File/Folder Picker Helper with Native Dialog and Custom Fallback
def local_file_picker(label, current_value, key, is_dir=False, extensions=['.csv']):
    expanded_key = f"expanded_{key}"
    if expanded_key not in st.session_state:
        st.session_state[expanded_key] = False
        
    browse_path_key = f"browse_path_{key}"
    text_key = f"text_{key}"
    
    # Initialize session state value if not set
    if text_key not in st.session_state:
        st.session_state[text_key] = current_value if current_value is not None else ""
        
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        typed_val = st.text_input(label, value=st.session_state[text_key], key=f"input_widget_{key}")
        st.session_state[text_key] = typed_val
    with col_btn:
        st.write("##")
        if st.button("Browse...", key=f"btn_{key}"):
            # 1. Try Native Dialog (using tkinter)
            try:
                import tkinter as tk
                from tkinter import filedialog
                
                root = tk.Tk()
                root.withdraw()
                root.wm_attributes('-topmost', 1)
                
                # Resolve starting dir
                init_path = typed_val if typed_val else os.getcwd()
                if not os.path.isabs(init_path):
                    home_dir_val = st.session_state.current_config.get("home_dir", "")
                    if home_dir_val:
                        init_path = os.path.join(home_dir_val, init_path)
                if not os.path.exists(init_path):
                    init_path = os.getcwd()
                    
                if is_dir:
                    res_path = filedialog.askdirectory(initialdir=init_path, title="Select Directory")
                else:
                    file_types = []
                    for ext in extensions:
                        file_types.append((f"{ext.upper()[1:]} Files", f"*{ext}"))
                    file_types.append(("All Files", "*.*"))
                    res_path = filedialog.askopenfilename(
                        initialdir=os.path.dirname(init_path) if os.path.isfile(init_path) else init_path, 
                        title="Select File", 
                        filetypes=file_types
                    )
                root.destroy()
                
                if res_path:
                    st.session_state[text_key] = res_path
                    st.session_state[expanded_key] = False
                    st.rerun()
            except Exception as e:
                # 2. Fall back to custom inline browser
                st.session_state[expanded_key] = not st.session_state[expanded_key]
                init_path = typed_val if typed_val else os.getcwd()
                if not os.path.isabs(init_path):
                    home_dir_val = st.session_state.current_config.get("home_dir", "")
                    if home_dir_val:
                        init_path = os.path.join(home_dir_val, init_path)
                if not os.path.exists(init_path):
                    init_path = os.getcwd()
                st.session_state[browse_path_key] = init_path
                st.rerun()
            
    if st.session_state[expanded_key]:
        curr_dir = st.session_state.get(browse_path_key, os.getcwd())
        if os.path.isfile(curr_dir):
            curr_dir = os.path.dirname(curr_dir)
        elif not os.path.isdir(curr_dir):
            curr_dir = os.getcwd()
            
        with st.container():
            st.markdown(f"📂 *Browsing:* `{curr_dir}`")
            
            col_nav1, col_nav2 = st.columns(2)
            with col_nav1:
                if st.button("Parent Folder ⬆️", key=f"up_{key}"):
                    st.session_state[browse_path_key] = os.path.dirname(curr_dir)
                    st.rerun()
            with col_nav2:
                if is_dir:
                    if st.button("Select Current Folder ✔️", key=f"sel_{key}"):
                        st.session_state[expanded_key] = False
                        st.session_state[text_key] = curr_dir
                        st.rerun()
                        
            try:
                items = os.listdir(curr_dir)
                dirs = []
                files = []
                for item in items:
                    full_p = os.path.join(curr_dir, item)
                    if os.path.isdir(full_p):
                        dirs.append(item)
                    elif os.path.isfile(full_p):
                        if not is_dir:
                            if any(item.lower().endswith(ext.lower()) for ext in extensions):
                                files.append(item)
                dirs.sort()
                files.sort()
                
                # Navigate subfolders
                if dirs:
                    selected_subdir = st.selectbox(
                        "Navigate into folder:",
                        options=["-- Keep Current Folder --"] + dirs,
                        key=f"sel_dir_{key}"
                    )
                    if selected_subdir != "-- Keep Current Folder --":
                        st.session_state[browse_path_key] = os.path.join(curr_dir, selected_subdir)
                        st.rerun()
                else:
                    st.info("No subfolders in this directory.")
                    
                # Select files
                if not is_dir:
                    if files:
                        selected_file = st.selectbox(
                            "Select a file:",
                            options=["-- Choose File --"] + files,
                            key=f"sel_file_{key}"
                        )
                        if selected_file != "-- Choose File --":
                            chosen_path = os.path.join(curr_dir, selected_file)
                            st.session_state[expanded_key] = False
                            st.session_state[text_key] = chosen_path
                            st.rerun()
                    else:
                        st.info("No matching files in this directory.")
            except Exception as e:
                st.error(f"Error listing folder contents: {e}")
                
    return st.session_state[text_key]

# Render dynamic configuration form
st.markdown("### Configuration Parameters")
categories = list(schema.keys())
tabs = st.tabs(categories)

edited_values = {}

for tab, category in zip(tabs, categories):
    with tab:
        # Special Active Config File selection and Creation logic inside Paths tab
        if category == "Paths":
            st.markdown("#### Active Configuration")
            col_active, col_new = st.columns([4, 2])
            with col_active:
                active_config_val = local_file_picker(
                    "Active Configuration File (.json)",
                    selected_config_path,
                    "active_config_path_picker",
                    is_dir=False,
                    extensions=['.json']
                )
                if active_config_val and active_config_val != selected_config_path:
                    st.session_state.active_config_path = active_config_val
                    st.rerun()
            with col_new:
                st.write("##")
                create_btn = st.button("Create New Config...", key="btn_create_new_config", use_container_width=True)
                
            if create_btn:
                new_file = None
                try:
                    import tkinter as tk
                    from tkinter import filedialog
                    root = tk.Tk()
                    root.withdraw()
                    root.wm_attributes('-topmost', 1)
                    template_dir = os.path.dirname(selected_config_path) if selected_config_path else os.getcwd()
                    new_file = filedialog.asksaveasfilename(
                        initialdir=template_dir,
                        title="Create New Config File",
                        defaultextension=".json",
                        filetypes=[("JSON Files", "*.json")]
                    )
                    root.destroy()
                except Exception as e:
                    # Native dialog failed, show fallback UI
                    st.session_state.show_create_fallback = True
                    st.rerun()
                    
                if new_file:
                    template_path = os.path.join(data_dir, "configs", "analysis_config_template.json")
                    try:
                        with open(template_path, "r") as tf:
                            template_data = json.load(tf)
                        with open(new_file, "w") as nf:
                            json.dump(template_data, nf, indent=4)
                        st.session_state.active_config_path = new_file
                        st.success(f"Successfully created configuration at {os.path.basename(new_file)}!")
                        st.rerun()
                    except Exception as err:
                        st.error(f"Error creating configuration: {err}")
                        
            # Show fallback inline UI if activated
            if st.session_state.get("show_create_fallback", False):
                with st.expander("Create New Configuration File (Fallback)", expanded=True):
                    f_folder = local_file_picker("Select Destination Folder", os.path.dirname(selected_config_path) or os.getcwd(), "new_config_folder", is_dir=True)
                    f_name = st.text_input("File Name", value="new_config.json", key="new_config_filename")
                    
                    c_fb_1, c_fb_2 = st.columns(2)
                    with c_fb_1:
                        if st.button("Create File", key="btn_create_fallback_confirm"):
                            if not f_name.endswith(".json"):
                                f_name += ".json"
                            new_file_fb = os.path.join(f_folder, f_name)
                            template_path = os.path.join(data_dir, "configs", "analysis_config_template.json")
                            try:
                                with open(template_path, "r") as tf:
                                    template_data = json.load(tf)
                                with open(new_file_fb, "w") as nf:
                                    json.dump(template_data, nf, indent=4)
                                st.session_state.active_config_path = new_file_fb
                                st.session_state.show_create_fallback = False
                                st.success(f"Successfully created configuration at {f_name}!")
                                st.rerun()
                            except Exception as err:
                                st.error(f"Error creating configuration: {err}")
                    with c_fb_2:
                        if st.button("Cancel", key="btn_create_fallback_cancel"):
                            st.session_state.show_create_fallback = False
                            st.rerun()
            st.markdown("---")
            
        fields = schema[category]
        # Group fields into two columns for layout clarity
        cols = st.columns(2)
        field_names = list(fields.keys())
        
        for idx, key in enumerate(field_names):
            col = cols[idx % 2]
            field_meta = fields[key]
            
            label = field_meta.get("label", "")
            if not label:
                label = key
                
            info = field_meta.get("info", "")
            field_type = field_meta.get("type", "string")
            
            # Resolve default values from schema if missing from config
            default_val_from_schema = field_meta.get("default", None)
            current_val = st.session_state.current_config.get(key, None)
            if current_val is None or current_val == "":
                current_val = default_val_from_schema
                
            with col:
                # Check for options sub-key first
                options = field_meta.get("options", None)
                if options is not None and isinstance(options, list):
                    try:
                        default_idx = options.index(current_val)
                    except ValueError:
                        default_idx = 0
                    val = st.selectbox(label, options=options, index=default_idx, help=info, key=f"w_{key}")
                    edited_values[key] = val
                    
                # Path category pickers
                elif category == "Paths":
                    is_dir = (key == "home_dir" or key.endswith("_dir") or "folder" in label.lower())
                    is_csv = (key.endswith("_csv") or "csv" in label.lower())
                    
                    if is_dir:
                        val = local_file_picker(label, current_val, key, is_dir=True)
                        edited_values[key] = val
                    elif is_csv:
                        val = local_file_picker(label, current_val, key, is_dir=False, extensions=['.csv'])
                        edited_values[key] = val
                    else:
                        val = st.text_input(label, value=str(current_val) if current_val is not None else "", help=info, key=f"w_{key}")
                        edited_values[key] = val
                        
                elif field_type == "bool":
                    # Checkbox
                    default_val = bool(current_val) if current_val is not None else False
                    val = st.checkbox(label, value=default_val, help=info, key=f"w_{key}")
                    edited_values[key] = val
                    
                elif field_type == "integer":
                    # Year fields check (remove decimals)
                    is_year = "year" in key.lower()
                    try:
                        default_val = int(float(current_val)) if current_val is not None else 0
                    except (ValueError, TypeError):
                        default_val = 0
                        
                    if is_year:
                        val = st.number_input(label, value=default_val, step=1, format="%d", help=info, key=f"w_{key}")
                        edited_values[key] = int(val)
                    else:
                        try:
                            default_val_num = float(current_val) if current_val is not None else 0.0
                        except (ValueError, TypeError):
                            default_val_num = 0.0
                        val = st.number_input(label, value=default_val_num, help=info, key=f"w_{key}")
                        if current_val is not None and isinstance(current_val, int):
                            edited_values[key] = int(val)
                        else:
                            edited_values[key] = val
                            
                elif field_type == "list of integers":
                    # Comma-separated integer list
                    if isinstance(current_val, list):
                        default_val = ", ".join(str(x) for x in current_val)
                    elif current_val is not None:
                        default_val = str(current_val)
                    else:
                        default_val = ""
                    val_str = st.text_input(label, value=default_val, help=info + " (Comma-separated integers)", key=f"w_{key}")
                    
                    # Parse back to list
                    parsed_list = []
                    if val_str.strip():
                        try:
                            parsed_list = [int(x.strip()) for x in val_str.split(",") if x.strip()]
                        except ValueError:
                            st.error(f"Field '{label}' must be a comma-separated list of integers.")
                    edited_values[key] = parsed_list
                    
                elif field_type == "list of strings":
                    # Comma-separated string list
                    if isinstance(current_val, list):
                        default_val = ", ".join(str(x) for x in current_val)
                    elif current_val is not None:
                        default_val = str(current_val)
                    else:
                        default_val = ""
                    val_str = st.text_input(label, value=default_val, help=info + " (Comma-separated strings)", key=f"w_{key}")
                    
                    # Parse back to list
                    parsed_list = []
                    if val_str.strip():
                        parsed_list = [x.strip() for x in val_str.split(",") if x.strip()]
                    edited_values[key] = parsed_list
                    
                else: # string or fallback
                    default_val = str(current_val) if current_val is not None else ""
                    val = st.text_input(label, value=default_val, help=info, key=f"w_{key}")
                    edited_values[key] = val

def save_current_config(edited_vals, selected_path):
    # Update config dict in session state
    for k, v in edited_vals.items():
        st.session_state.current_config[k] = v
        
    # Verify in each CSV if home_dir is included, if so remove it
    home_dir_val = st.session_state.current_config.get("home_dir", "").strip()
    if home_dir_val:
        for k in list(st.session_state.current_config.keys()):
            if k.endswith("_csv"):
                val = st.session_state.current_config[k]
                if isinstance(val, str) and val.strip():
                    val = val.strip()
                    if home_dir_val in val:
                        rel_path = os.path.relpath(val, home_dir_val)
                        if not rel_path.startswith(".."):
                            st.session_state.current_config[k] = rel_path
                            
    # Save back to selected file path
    with open(selected_path, "w") as f:
        json.dump(st.session_state.current_config, f, indent=4)

# Actions bar
st.markdown("---")
c_actions_1, c_actions_2, _ = st.columns([1, 1, 4])

with c_actions_1:
    if st.button("Save Configuration", type="primary", use_container_width=True):
        try:
            save_current_config(edited_values, selected_config_path)
            st.success(f"Config successfully saved to {os.path.basename(selected_config_path)}!")
            st.rerun()
        except Exception as e:
            st.error(f"Error saving file: {e}")

with c_actions_2:
    run_btn = st.button("Run Pipeline", type="secondary", use_container_width=True, disabled=st.session_state.running)

# Log streaming container
log_container = st.empty()
if st.session_state.logs:
    with log_container.container():
        st.markdown("### Execution Console Logs")
        st.code(st.session_state.logs, language="text")

if run_btn:
    st.session_state.running = True
    st.session_state.logs = ""
    
    # Save first to make sure app.py gets the latest settings and cleaned paths
    save_current_config(edited_values, selected_config_path)
        
    app_script = os.path.join(project_root, "ui", "app.py")
    
    # Spawn pipeline process
    try:
        process = subprocess.Popen(
            ["python3", app_script, selected_config_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=dict(os.environ, PYTHONPATH=project_root)
        )
        
        status_info = st.info("Pipeline execution in progress...")
        
        # Read output in real-time
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                st.session_state.logs += line
                with log_container.container():
                    st.markdown("### Execution Console Logs")
                    st.code(st.session_state.logs, language="text")
                    
        rc = process.poll()
        status_info.empty()
        if rc == 0:
            st.success("Pipeline executed successfully!")
        else:
            st.error(f"Pipeline failed with exit code {rc}")
            
    except Exception as e:
        st.error(f"Error running pipeline: {e}")
        
    st.session_state.running = False
    st.rerun()
