import streamlit as st
import pandas as pd

example_data = pd.read_csv('gym_members_exercise_tracking.csv').head(10)
print(example_data)
