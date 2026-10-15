"""Streamlit entry point for AI Car Screening Jabodetabek."""

from dotenv import load_dotenv
import streamlit as st

load_dotenv()


def main() -> None:
    st.set_page_config(
        page_title="AI Car Screening Jabodetabek",
        page_icon=":material/directions_car:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    from src.ui import render_app

    render_app()


if __name__ == "__main__":
    main()
