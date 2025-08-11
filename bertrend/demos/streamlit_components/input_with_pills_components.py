import streamlit as st
from typing import Callable, Optional, List

from bertrend.demos.demos_utils.icons import PERSON_ADD_ICON


@st.fragment
def input_with_pills(
    label: str,
    key: str,
    placeholder: str = "",
    validate_fn: Optional[Callable[[str], bool]] = None,
    label_visibility="hidden",
) -> List[str]:
    """
    A simple Streamlit component that auto-adds valid input to pills.
    Dialog-safe version that doesn't use st.rerun()
    """

    # Initialize session state
    all_items_key = f"{key}_all_items"
    selected_items_key = f"{key}_selected_items"
    input_key = f"{key}_input"

    if all_items_key not in st.session_state:
        st.session_state[all_items_key] = []
    if selected_items_key not in st.session_state:
        st.session_state[selected_items_key] = []

    # Create columns: input on left, pills on right
    col1, col2 = st.columns([3, 3])

    with col1:
        # Create a form to handle Enter key press without rerun
        with st.form(key=f"{key}_form", clear_on_submit=True, border=False):
            current_input = st.text_input(
                label,
                placeholder=placeholder,
                key=input_key,
                label_visibility=label_visibility,
            )
            # Submit button is required but will be hidden
            submitted = st.form_submit_button(PERSON_ADD_ICON)

            if submitted and current_input.strip():
                is_valid = True
                error_msg = ""

                if validate_fn and not validate_fn(current_input.strip()):
                    is_valid = False
                    error_msg = "Invalid format"
                elif current_input.strip() in st.session_state[all_items_key]:
                    is_valid = False
                    error_msg = "Already exists"

                if is_valid:
                    # Add to both lists
                    st.session_state[all_items_key].append(current_input.strip())
                    st.session_state[selected_items_key].append(current_input.strip())
                else:
                    st.error(error_msg)

    with col2:
        # Display pills
        if st.session_state[all_items_key]:
            st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)

            # Use a callback to handle selection changes
            def on_pills_change():
                selected = st.session_state[f"{key}_pills_widget"]
                st.session_state[selected_items_key] = selected

            selected_pills = st.pills(
                "Items",
                st.session_state[all_items_key],
                selection_mode="multi",
                key=f"{key}_pills_widget",
                label_visibility="collapsed",
                default=st.session_state[selected_items_key],
                on_change=on_pills_change,
            )

    return st.session_state[selected_items_key]
